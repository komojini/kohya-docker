import runpod
from runpod.serverless.utils import rp_upload
import json
import requests
import os
import re
import base64
from typing import Optional, Tuple, Literal
import urllib.request
from urllib.parse import urlparse, unquote
from pathlib import Path
import zipfile
import subprocess
import multiprocessing
import logging
import uuid
import shutil
import ast
import time
import boto3
from subprocess import getoutput
from boto3 import session
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


logger = logging.getLogger("runpod upload utility")
FMT = "%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FMT, handlers=[logging.StreamHandler()])

ROOT_DIR = "/"
TRAIN_DATA_DIR = "/fine_tune/train_data"
TRAINING_DIR = os.path.join(ROOT_DIR, "fine_tune")
MODEL_PATH = "/sd-models/sd_xl_base_1.0.safetensors"
REPO_DIR = os.path.join(ROOT_DIR, "kohya_ss")
FINE_TUNE_DIR = os.path.join(REPO_DIR, "fine_tune")
ACCELERATE_CONFIG = "/accelerate.yaml"


def prepare_directories():
    directories = [
        TRAIN_DATA_DIR,
        TRAINING_DIR,
        REPO_DIR,
        FINE_TUNE_DIR,
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

def log_directories():
    print("\nls / -->")
    os.system("ls /")
    print("\nls /kohya_ss -->")
    os.system("ls /kohya_ss")
    print("\nls /kohya_ss/sd_scripts -->")
    os.system("ls /kohya_ss/sd_scripts")


def set_bucket_creds(bucket_creds):
    os.environ["BUCKET_ENDPOINT_URL"] = bucket_creds["endpointUrl"]
    os.environ["BUCKET_ACCESS_KEY_ID"] = bucket_creds["accessId"]
    os.environ["BUCKET_SECRET_ACCESS_KEY"] = bucket_creds["accessSecret"]

def prepare_directory():
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    print(f"Your train data directory: {TRAIN_DATA_DIR}")


def extract_dataset(zip_file, output_path):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_path)
    logger.info(f"\nZipfile extracted: {output_path}\n{os.system(f'ls {output_path}/data/images')}")
    return f"{output_path}/data/images"

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
    downloaded_file = boto_client.download_file(
        bucket_name,
        bucket_path,
        file_path
    )
    return file_path


def download_from_s3(url, output_path):
    pass

def get_filename(url, bearer_token, quiet=True):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    if 'content-disposition' in response.headers:
        content_disposition = response.headers['content-disposition']
        filename = re.findall('filename="?([^"]+)"?', content_disposition)[0]
    else:
        url_path = urlparse(url).path
        filename = unquote(os.path.basename(url_path))

    return filename

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
                    

def download_from_huggingface(url, dst):
    token = os.environ.get("HF_TOKEN", None)
    filename = get_filename(url, token, quiet=False)
    filepath = os.path.join(dst, filename)

    if url.startswith("/workspace"):
        return url
    elif "huggingface.co" in url:
        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
                
        aria2_download(dst, filename, url, token)

    return filepath

def download(zip_path, output_dir):
    if "huggingface" in zip_path:
        downloaded_path = download_from_huggingface(zip_path, output_dir)
    elif "s3" in zip_path:
        downloaded_path = download_from_s3(zip_path, output_dir)
    else:
        output_path = Path(output_dir) / f"{uuid.uuid4()}.zip"
        downloaded_path = download_file_s3(zip_path, output_path)
    return downloaded_path


def prepare_train_data(zipfile_url, unzip_dir=None):
    logger.info(f"Start Preparing Train Data\nzipfile_url: {zipfile_url}\nunzip_dir: {unzip_dir}")
    
    if unzip_dir:
        os.makedirs(unzip_dir, exist_ok=True)
    else:
        unzip_dir = TRAIN_DATA_DIR
    
    zip_file = download(zipfile_url, ROOT_DIR)
    unzip_dir = extract_dataset(zip_file, unzip_dir)
    os.system(
        f"mv {unzip_dir}/* {TRAIN_DATA_DIR}"
    )
    os.system(f"ls {TRAIN_DATA_DIR}")
    os.system(f"rm -rf {unzip_dir}")
    os.remove(zip_file)


def prepare_bucket(
    bucket_resolution=1024,
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16",
    flip_aug=False,
    clean_caption=True,
    recursive=True,
    batch_size=1,
    max_data_loader_n_workers=2,
):
    """
    # Use `clean_caption` option to clean such as duplicate tags, `women` to `girl`, etc
    clean_caption     = True 
    # Use the `recursive` option to process subfolders as well
    recursive         = True
    """

    bucketing_json    = os.path.join(TRAINING_DIR, "meta_lat.json")
    metadata_json     = os.path.join(TRAINING_DIR, "meta_clean.json")

    metadata_config = {
        "_train_data_dir": TRAIN_DATA_DIR,
        "_out_json": metadata_json,
        "recursive": recursive,
        "full_path": recursive,
        "clean_caption": clean_caption
    }

    bucketing_config = {
        "_train_data_dir": TRAIN_DATA_DIR,
        "_in_json": metadata_json,
        "_out_json": bucketing_json,
        "_model_name_or_path": MODEL_PATH,
        "recursive": recursive,
        "full_path": recursive,
        "flip_aug": flip_aug,
        "batch_size": batch_size,
        "max_data_loader_n_workers": max_data_loader_n_workers,
        "max_resolution": f"{bucket_resolution}, {bucket_resolution}",
        "mixed_precision": mixed_precision,
    }

    def generate_args(config):
        args = ""
        for k, v in config.items():
            if k.startswith("_"):
                args += f'"{v}" '
            elif isinstance(v, str):
                args += f'--{k}="{v}" '
            elif isinstance(v, bool) and v:
                args += f"--{k} "
            elif isinstance(v, float) and not isinstance(v, bool):
                args += f"--{k}={v} "
            elif isinstance(v, int) and not isinstance(v, bool):
                args += f"--{k}={v} "
        return args.strip()

    merge_metadata_args = generate_args(metadata_config)
    prepare_buckets_args = generate_args(bucketing_config)

    merge_metadata_command = f"python /kohya_ss/sd_scripts/finetune/merge_all_to_metadata.py {merge_metadata_args}"
    prepare_buckets_command = f"python /kohya_ss/sd_scripts/finetune/prepare_buckets_latents.py {prepare_buckets_args}"

    os.chdir(FINE_TUNE_DIR)
    os.system(merge_metadata_command)
    time.sleep(1)
    os.system(prepare_buckets_command)

    logger.info(f"\n{merge_metadata_args}")
    logger.info(f"\n{prepare_buckets_command}")

def prepare_optimizer_config(
    optimizer_type = "AdaFactor",
    optimizer_args = "[ \"scale_parameter=False\", \"relative_step=False\", \"warmup_init=False\" ]",
    learning_rate=4e-7,
    train_text_encoder=False,
    lr_scheduler="constant_with_warmup",
    lr_warmup_steps=100,
    lr_scheduler_num=0,
):
    if isinstance(optimizer_args, str):
        optimizer_args = optimizer_args.strip()
        if optimizer_args.startswith('[') and optimizer_args.endswith(']'):
            try:
                optimizer_args = ast.literal_eval(optimizer_args)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing optimizer_args: {e}\n")
                optimizer_args = []
        elif len(optimizer_args) > 0:
            print(f"WARNING! '{optimizer_args}' is not a valid list! Put args like this: [\"args=1\", \"args=2\"]\n")
            optimizer_args = []
        else:
            optimizer_args = []
    else:
        optimizer_args = []

    optimizer_config = {
        "optimizer_arguments": {
            "optimizer_type"          : optimizer_type,
            "learning_rate"           : learning_rate,
            "train_text_encoder"      : train_text_encoder,
            "max_grad_norm"           : 1.0,
            "optimizer_args"          : optimizer_args,
            "lr_scheduler"            : lr_scheduler,
            "lr_warmup_steps"         : lr_warmup_steps,
            "lr_scheduler_num_cycles" : lr_scheduler_num if lr_scheduler == "cosine_with_restarts" else None,
            "lr_scheduler_power"      : lr_scheduler_num if lr_scheduler == "polynomial" else None,
            "lr_scheduler_type"       : None,
            "lr_scheduler_args"       : None,
        },
    }
    return optimizer_config

def prepare_advanced_training_config(
    optimizer_state_path="",
    min_snr_gamma=-1,
):
    advanced_training_config = {
        "advanced_training_config" : {
            "resume": optimizer_state_path,
            "min_snr_gamma": min_snr_gamma if not min_snr_gamma == -1 else None,
        }
    }
    return advanced_training_config


def handler(job):
    job_input = job["input"]
    zipfile_path = job_input["zipfile_path"]
    if job.get("bucket_creds", None):
        set_bucket_creds(job["bucket_creds"])
    
    log_directories()
    prepare_directories()
    prepare_train_data(zipfile_path)

    prepare_bucket(**job_input.get("bucketing"))
    optimizer_config = prepare_optimizer_config(**job_input.get("optimizer", {}))
    advanced_training_config = prepare_advanced_training_config(**job_input("advanced_training", {}))

    output = {}
    return output


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})