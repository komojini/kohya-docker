import runpod
import toml
import requests
import os
import pathlib
import re
import glob
import torch
import accelerate
from PIL import Image
from typing import Optional, Tuple, Literal
from urllib.parse import urlparse, unquote
from pathlib import Path
import zipfile
import subprocess
import multiprocessing
import logging
import uuid
import boto3
from boto3 import session
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


logger = logging.getLogger("runpod upload utility")
FMT = "%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FMT, handlers=[logging.StreamHandler()])


print("Accelerate device:", accelerate.Accelerator().device)

# Directories
ROOT_DIR = "/"
TRAIN_DATA_DIR = "/fine_tune/train_data"
TRAINING_DIR = os.path.join(ROOT_DIR, "fine_tune")
MODEL_PATH = "/sd-models/sd_xl_base_1.0.safetensors"
REPO_DIR = os.path.join(ROOT_DIR, "kohya_ss")
FINE_TUNE_DIR = os.path.join(REPO_DIR, "fine_tune")
ACCELERATE_CONFIG = "/accelerate.yaml"
OUTPUT_DIR = "/outputs"
TMP_DIR = "/tmp"
LOGGING_DIR = "/outputs/log"
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]

logger.info(f"\ntorch.cuda.is_avaliable(): {torch.cuda.is_available()}\n")


def prepare_directories():
    directories = [
        TRAIN_DATA_DIR,
        TRAINING_DIR,
        REPO_DIR,
        FINE_TUNE_DIR,
        OUTPUT_DIR,
        LOGGING_DIR
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def set_bucket_creds(bucket_creds):
    os.environ["BUCKET_ENDPOINT_URL"] = bucket_creds["endpointUrl"]
    os.environ["BUCKET_ACCESS_KEY_ID"] = bucket_creds["accessId"]
    os.environ["BUCKET_SECRET_ACCESS_KEY"] = bucket_creds["accessSecret"]

def prepare_directory():
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    print(f"Your train data directory: {TRAIN_DATA_DIR}")


def extract_and_return_images_parent_dir(zip_file):
    tmp_dir = f"{TMP_DIR}/images"
    
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)
    for root, dirs, files in os.walk(tmp_dir):
        for file in files:
            if pathlib.Path(file).suffix in ALLOWED_EXTENSIONS:
                return root
    return None

def extract_region_from_url(endpoint_url):
    """
    Extracts the region from the endpoint URL.
    """
    parsed_url = urlparse(endpoint_url)
    # AWS/backblaze S3-like URL
    if '.s3.' in endpoint_url:
        return endpoint_url.split('.s3.')[1].split('.')[0]

    # DigitalOcean Spaces-like URL
    if parsed_url.netloc.endswith('.digitaloceanspaces.com'):
        return endpoint_url.split('.')[1].split('.digitaloceanspaces.com')[0]

    return None

def extract_bucket_name_from_url(endpoint_url):
    parsed_url = urlparse(endpoint_url)
    if '.s3.' in endpoint_url:
        return endpoint_url.split('.s3.')[0].split('/')[-1]
    
    return None

# --------------------------- S3 Bucket Connection --------------------------- #
def get_boto_client(
        bucket_creds: Optional[dict] = None) -> Tuple[boto3.client, TransferConfig]:  # pragma: no cover # pylint: disable=line-too-long
    '''
    Returns a boto3 client and transfer config for the bucket.
    '''
    print('Start getting boto client.\nBUCKET_ENDPOINT_URL:', os.environ.get('BUCKET_ENDPOINT_URL'))

    bucket_session = session.Session()

    boto_config = Config(
        signature_version='s3v4',
        #retries={
        #    'max_attempts': 3,
        #    'mode': 'standard'
        #}
    )

    transfer_config = TransferConfig(
        multipart_threshold=1024 * 25,
        max_concurrency=multiprocessing.cpu_count(),
        multipart_chunksize=1024 * 25,
        use_threads=True
    )

    if bucket_creds:
        endpoint_url = bucket_creds['endpointUrl']
        access_key_id = bucket_creds['accessId']
        secret_access_key = bucket_creds['accessSecret']
    else:
        endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL', None)
        access_key_id = os.environ.get('BUCKET_ACCESS_KEY_ID', None)
        secret_access_key = os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)

    logger.info(f"\nendpoint_url: {endpoint_url}\naccess_key_id: {access_key_id[:4]}...\nsecret_access_key: {secret_access_key[:4]}...\n")
    if endpoint_url and access_key_id and secret_access_key:
        # Extract region from the endpoint URL
        region = extract_region_from_url(endpoint_url)
        logger.info(f"\nregion: {region}\n")
        boto_client = bucket_session.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=boto_config,
            region_name=region
        )
    else:
        boto_client = None

    return boto_client, transfer_config

def download_file_s3(bucket_path, file_path):
    boto_client, _ = get_boto_client()
    bucket_name = extract_bucket_name_from_url(os.environ.get('BUCKET_ENDPOINT_URL', None))

    #bucket_path = urljoin(base=os.environ.get('BUCKET_ENDPOINT_URL'), url=bucket_path, allow_fragments=True)
    
    logger.info(f"\nStart downloading file\nbucket_name: {bucket_name}\nbucket_path: {bucket_path}\nfile_path: {file_path}")
    boto_client.download_file(
        bucket_name,
        bucket_path,
        file_path
    )
    return file_path

def parse_args(config):
    args = []

    for k, v in config.items():
        if k.startswith("_"):
            args.append(f"{v}")
        elif isinstance(v, str) and v is not None:
            args.append(f'--{k}={v}')
        elif isinstance(v, bool) and v:
            args.append(f"--{k}")
        elif isinstance(v, float) and not isinstance(v, bool):
            args.append(f"--{k}={v}")
        elif isinstance(v, int) and not isinstance(v, bool):
            args.append(f"--{k}={v}")

    return args


def aria2_download(dir, filename, url, token):
    user_header = f"Authorization: Bearer {token}"

    aria2_config = {
        "console-log-level"         : "error",
        "summary-interval"          : 10,
        "header"                    : user_header if "huggingface.co" in url else None,
        "continue"                  : True,
        "max-connection-per-server" : 16,
        "min-split-size"            : "1M",
        "split"                     : 16,
        "dir"                       : dir,
        "out"                       : filename,
        "_url"                      : url,
    }
    aria2_args = parse_args(aria2_config)
    subprocess.run(["aria2c", *aria2_args])
                    
def download(zip_path, output_dir):
    output_path = Path(output_dir) / f"{uuid.uuid4()}.zip"
    downloaded_path = download_file_s3(zip_path, output_path)
    return downloaded_path

def upload_model(model_path, save_path):
    s3_client, _ = get_boto_client()
    bucket_name = extract_bucket_name_from_url(os.environ.get('BUCKET_ENDPOINT_URL', None))

    s3_client.upload_file(
        model_path,
        bucket_name,
        save_path,
    )


def crop_image(image_path, save_path, resolution):
    # Open the image
    image = Image.open(image_path)

    # Crop the image (example cropping to a square in the center)
    width, height = image.size
    min_dimension = min(width, height)
    left = (width - min_dimension) / 2
    top = (height - min_dimension) / 2
    right = (width + min_dimension) / 2
    bottom = (height + min_dimension) / 2
    cropped_image = image.crop((left, top, right, bottom))

    # Resize the cropped image to 512x512 pixels
    new_size = (resolution, resolution)
    resized_image = cropped_image.resize(new_size)

    # Save the cropped and resized image
    resized_image.save(save_path)
    return save_path


def prepare_train_data(
        zipfile_url, 
        instance_prompt,
        class_prompt,
        training_images_repeat,
        training_dir=None, 
        regularization_images_dir="",
        regularization_repeat=1,
        resolution=1024
    ):
    logger.info(f"\nStart Preparing Train Data\nzipfile_url: {zipfile_url}\nunzip_dir: {training_dir}")
    
    if training_dir:
        os.makedirs(training_dir, exist_ok=True)
    else:
        training_dir = TRAIN_DATA_DIR
    
    zip_file = download(zipfile_url, ROOT_DIR)
    new_unzip_dir = extract_and_return_images_parent_dir(zip_file)

    # os.system(
    #     f"mv {new_unzip_dir}/* '{unzip_dir}'"
    # )
    
    # os.system(f"rm -rf '{new_unzip_dir}'")
    # os.remove(zip_file)
    
    for f in os.listdir(new_unzip_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in ALLOWED_EXTENSIONS:
            continue
        image_path = os.path.join(new_unzip_dir,f)
        crop_image(image_path, image_path, resolution)

    from kohya_ss.library.dreambooth_folder_creation_gui import dreambooth_folder_preparation    

    dreambooth_folder_preparation(
        new_unzip_dir,
        training_images_repeat,
        instance_prompt,
        regularization_images_dir,
        regularization_repeat,
        class_prompt,
        training_dir,
    )
    return 


def generate_args(config):
    args = ""
    for k, v in config.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'''--{k}="{v}" '''
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"""--{k}={v} """
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"""--{k}={v} """
        elif isinstance(v, list) and v:
            arg =  f"--{k} "
            for value in v:
                arg += f"{value} "
            args += arg
    return args.strip()


def handler(job):

    job_input = job["input"]
    zipfile_path = job_input["zipfile_path"]
    if job_input.get("bucket_creds", None):
        print("Bucket Creds Provided")
        set_bucket_creds(job_input["bucket_creds"])
    else:
        print("Bucket Creds Not Provided")
    
    prepare_directories()
    
    train_input = job_input["train"]

    prepare_train_data(
        zipfile_path, 
        train_input["token_word"], 
        train_input["class_word"], 
        training_images_repeat=train_input["training_repeats"],
        training_dir=TRAIN_DATA_DIR, 
        resolution=train_input["resolution"]
    )

    train_images_dir = os.path.join(TRAIN_DATA_DIR, "img")
    os.system(f"""ls -la "{train_images_dir}" """)

    # Train
    os.system(f"""
accelerate launch --config_file="accelerate.yaml" --num_cpu_threads_per_process=2 "/kohya_ss/sdxl_train_network.py" \
    --enable_bucket \
    --min_bucket_reso=512 \
    --max_bucket_reso=2048 \
    --pretrained_model_name_or_path="{MODEL_PATH}" \
    --train_data_dir="{train_images_dir}" \
    --resolution="{train_input["resolution"]},{train_input["resolution"]}" \
    --output_dir="{OUTPUT_DIR}" \
    --logging_dir="{LOGGING_DIR}" \
    --network_alpha="{train_input["network_alpha"]}" \
    --save_model_as={train_input["save_model_as"]} \
    --network_module=networks.lora \
    --network_args rank_dropout="0.1" \
    --unet_lr=0.0001 \
    --network_train_unet_only \
    --network_dim={train_input["network_dim"]} \
    --output_name="{train_input["project_name"]}" \
    --lr_scheduler_num_cycles="1" \
    --no_half_vae \
    --learning_rate="{train_input["learning_rate"]}" \
    --lr_scheduler="{train_input["lr_scheduler"]}" \
    --train_batch_size="{train_input["train_batch_size"]}" \
    --max_train_steps="{train_input["max_train_steps"]}" \
    --save_every_n_epochs="3" \
    --mixed_precision="{train_input["mixed_precision"]}" \
    --save_precision="{train_input["save_precision"]}" \
    --cache_latents \
    --cache_latents_to_disk \
    --optimizer_type="{train_input["optimizer_type"]}" \
    --max_data_loader_n_workers="2" \
    --bucket_reso_steps=64 \
    --xformers \
    --bucket_no_upscale \
    --noise_offset=0.0 \
    --log_with="wandb" \
    --wandb_api_key="{os.getenv("WANDB_API_KEY")}"
""")
              
    output_path = job_input.get("output_path", "models")
    save_path = f"{output_path}/{train_input['project_name']}.safetensors"
    model_path = f'{OUTPUT_DIR}/{train_input["project_name"]}.safetensors'
    upload_model(model_path, save_path)

    os.system(f"ls -la '{OUTPUT_DIR}'")
    output = {
        "model_path": save_path,
        "endpoint_url": os.getenv("BUCKET_ENDPOINT_URL"),
        "train": train_input,
    }
    return output


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})