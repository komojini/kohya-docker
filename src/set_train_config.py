# @title ## **4.4. Training Config**
import toml
import os
from subprocess import getoutput
import random
import glob 


def set_train_config(
        config_path,
        training_dir,
        repo_dir,
        train_data_dir,
        model_path,
        vae_path,
        output_dir,
        optimizer_config,
        lora_config,
        advanced_training_config,
        project_name="sdxl_lora",
        gradient_checkpointing=False,
        no_half_vae=True,
        cache_text_encoder_outputs=False,
        min_timestep=0,
        max_timestep=1000,
        num_repeats=40,
        resolution=1024,
        keep_tokens=0,
        num_epochs=10,
        train_batch_size=2,
        mixed_precision="bf16",
        seed=-1,
        save_precision="fp16",
        save_every_n_epochs=3,
        enable_sample=False,
):

    # @markdown ### **Project Config**
    # project_name                = "sdxl_lora"  # @param {type:"string"}
    # @markdown Get your `wandb_api_key` [here](https://wandb.ai/settings) to logs with wandb.
    wandb_api_key               = "" # @param {type:"string"}
    in_json                     = "/content/LoRA/meta_lat.json"  # @param {type:"string"}
    # @markdown ### **SDXL Config**
    # gradient_checkpointing      = True  # @param {type:"boolean"}
    # no_half_vae                 = True  # @param {type:"boolean"}
    #@markdown Recommended parameter for SDXL training but if you enable it, `shuffle_caption` won't work
    # cache_text_encoder_outputs  = False  # @param {type:"boolean"}
    #@markdown These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.
    # min_timestep                = 0 # @param {type:"number"}
    # max_timestep                = 1000 # @param {type:"number"}
    # @markdown ### **Dataset Config**
    # num_repeats                 = 1  # @param {type:"number"}
    # resolution                  = 1024  # @param {type:"slider", min:512, max:1024, step:128}
    # keep_tokens                 = 0  # @param {type:"number"}
    # @markdown ### **General Config**
    # num_epochs                  = 10  # @param {type:"number"}
    # train_batch_size            = 4  # @param {type:"number"}
    # mixed_precision             = "fp16"  # @param ["no","fp16","bf16"] {allow-input: false}
    # seed                        = -1  # @param {type:"number"}
    optimization                = "scaled dot-product attention" # @param ["xformers", "scaled dot-product attention"]
    # @markdown ### **Save Output Config**
    # save_precision              = "fp16"  # @param ["float", "fp16", "bf16"] {allow-input: false}
    # save_every_n_epochs         = 1  # @param {type:"number"}
    # @markdown ### **Sample Prompt Config**
    # enable_sample               = True  # @param {type:"boolean"}
    sampler                     = "euler_a"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
    positive_prompt             = ""
    negative_prompt             = ""
    quality_prompt              = "NovelAI"  # @param ["None", "Waifu Diffusion 1.5", "NovelAI", "AbyssOrangeMix", "Stable Diffusion XL"] {allow-input: false}
    if quality_prompt          == "NovelAI":
        positive_prompt         = "masterpiece, best quality, "
        negative_prompt         = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, "
    if quality_prompt          == "AbyssOrangeMix":
        positive_prompt         = "masterpiece, best quality, "
        negative_prompt         = "(worst quality, low quality:1.4), "
    if quality_prompt          == "Stable Diffusion XL":
        negative_prompt         = "3d render, smooth, plastic, blurry, grainy, low-resolution, deep-fried, oversaturated"
    custom_prompt               = "face focus, cute, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck" # @param {type:"string"}
    # @markdown Specify `prompt_from_caption` if you want to use caption as prompt instead. Will be chosen randomly.
    prompt_from_caption         = "none"  # @param ["none", ".txt", ".caption"]
    if prompt_from_caption     != "none":
        custom_prompt           = ""
    num_prompt                  = 2  # @param {type:"number"}
    logging_dir                 = os.path.join(training_dir, "logs")
    lowram                      = int(next(line.split()[1] for line in open('/proc/meminfo') if "MemTotal" in line)) / (1024**2) < 15

    os.chdir(repo_dir)

    prompt_config = {
        "prompt": {
            "negative_prompt" : negative_prompt,
            "width"           : resolution,
            "height"          : resolution,
            "scale"           : 12,
            "sample_steps"    : 28,
            "subset"          : [],
        }
    }

    train_config = {
        "sdxl_arguments": {
            "cache_text_encoder_outputs" : cache_text_encoder_outputs,
            "no_half_vae"                : True,
            "min_timestep"               : min_timestep,
            "max_timestep"               : max_timestep,
            "shuffle_caption"            : True if not cache_text_encoder_outputs else False,
            "lowram"                     : lowram
        },
        "model_arguments": {
            "pretrained_model_name_or_path" : model_path,
            "vae"                           : vae_path,
        },
        "dataset_arguments": {
            "debug_dataset"                 : False,
            "in_json"                       : in_json,
            "train_data_dir"                : train_data_dir,
            "dataset_repeats"               : num_repeats,
            "keep_tokens"                   : keep_tokens,
            "resolution"                    : str(resolution) + ',' + str(resolution),
            "color_aug"                     : False,
            "face_crop_aug_range"           : None,
            "token_warmup_min"              : 1,
            "token_warmup_step"             : 0,
        },
        "training_arguments": {
            "output_dir"                    : os.path.join(output_dir, project_name),
            "output_name"                   : project_name if project_name else "last",
            "save_precision"                : save_precision,
            "save_every_n_epochs"           : save_every_n_epochs,
            "save_n_epoch_ratio"            : None,
            "save_last_n_epochs"            : None,
            "resume"                        : None,
            "train_batch_size"              : train_batch_size,
            "max_token_length"              : 225,
            "mem_eff_attn"                  : False,
            "sdpa"                          : True if optimization == "scaled dot-product attention" else False,
            "xformers"                      : True if optimization == "xformers" else False,
            "max_train_epochs"              : num_epochs,
            "max_data_loader_n_workers"     : 8,
            "persistent_data_loader_workers": True,
            "seed"                          : seed if seed > 0 else None,
            "gradient_checkpointing"        : gradient_checkpointing,
            "gradient_accumulation_steps"   : 1,
            "mixed_precision"               : mixed_precision,
        },
        "logging_arguments": {
            "log_with"          : "wandb" if wandb_api_key else "tensorboard",
            "log_tracker_name"  : project_name if wandb_api_key and not project_name == "last" else None,
            "logging_dir"       : logging_dir,
            "log_prefix"        : project_name if not wandb_api_key else None,
        },
        "sample_prompt_arguments": {
            "sample_every_n_steps"    : None,
            "sample_every_n_epochs"   : save_every_n_epochs if enable_sample else None,
            "sample_sampler"          : sampler,
        },
        "saving_arguments": {
            "save_model_as": "safetensors"
        },
    }

    def write_file(filename, contents):
        with open(filename, "w") as f:
            f.write(contents)

    def prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompt):
        if enable_sample:
            search_pattern = os.path.join(train_data_dir, '**/*' + prompt_from_caption)
            caption_files = glob.glob(search_pattern, recursive=True)

            if not caption_files:
                if not custom_prompt:
                    custom_prompt = "masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
                new_prompt_config = prompt_config.copy()
                new_prompt_config['prompt']['subset'] = [
                    {"prompt": positive_prompt + custom_prompt if positive_prompt else custom_prompt}
                ]
            else:
                selected_files = random.sample(caption_files, min(num_prompt, len(caption_files)))

                prompts = []
                for file in selected_files:
                    with open(file, 'r') as f:
                        prompts.append(f.read().strip())

                new_prompt_config = prompt_config.copy()
                new_prompt_config['prompt']['subset'] = []

                for prompt in prompts:
                    new_prompt = {
                        "prompt": positive_prompt + prompt if positive_prompt else prompt,
                    }
                    new_prompt_config['prompt']['subset'].append(new_prompt)

            return new_prompt_config
        else:
            return prompt_config

    def eliminate_none_variable(config):
        for key in config:
            if isinstance(config[key], dict):
                for sub_key in config[key]:
                    if config[key][sub_key] == "":
                        config[key][sub_key] = None
            elif config[key] == "":
                config[key] = None

        return config

    try:
        train_config.update(optimizer_config)
    except NameError:
        raise NameError("'optimizer_config' dictionary is missing. Please run  '4.1. Optimizer Config' cell.")

    try:
        train_config.update(lora_config)
    except NameError:
        raise NameError("'lora_config' dictionary is missing. Please run  '4.1. LoRa: Low-Rank Adaptation Config' cell.")

    advanced_training_warning = False
    try:
        train_config.update(advanced_training_config)
    except NameError:
        advanced_training_warning = True
        pass

    prompt_config = prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompt)

    # config_path         = os.path.join(config_dir, "config_file.toml")
    # prompt_path         = os.path.join(config_dir, "sample_prompt.toml")

    config_str          = toml.dumps(eliminate_none_variable(train_config))
    # prompt_str          = toml.dumps(eliminate_none_variable(prompt_config))

    write_file(config_path, config_str)
    # write_file(prompt_path, prompt_str)

    print(config_str)

    if advanced_training_warning:
        import textwrap
        error_message = "WARNING: This is not an error message, but the [advanced_training_config] dictionary is missing. Please run the '4.2. Advanced Training Config' cell if you intend to use it, or continue to the next step."
        wrapped_message = textwrap.fill(error_message, width=80)
        print('\033[38;2;204;102;102m' + wrapped_message + '\033[0m\n')
        pass

    print(prompt_str)
