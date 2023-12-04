
import toml
import ast


def set_lora_config(
        config_path,
        network_category = "LoRA_LierLa",
        network_dim = 16,
        network_alpha = 8,
        conv_dim = 32,
        conv_alpha = 16,
):
    # @title ## **4.1. LoRa: Low-Rank Adaptation Config**
    # @markdown Kohya's `LoRA` renamed to `LoRA-LierLa` and Kohya's `LoCon` renamed to `LoRA-C3Lier`, read [official announcement](https://github.com/kohya-ss/sd-scripts/blob/849bc24d205a35fbe1b2a4063edd7172533c1c01/README.md#naming-of-lora).
    # network_category = "LoRA_LierLa"  # @param ["LoRA_LierLa", "LoRA_C3Lier", "DyLoRA_LierLa", "DyLoRA_C3Lier", "LoCon", "LoHa", "IA3", "LoKR", "DyLoRA_Lycoris"]

    # @markdown | network_category | network_dim | network_alpha | conv_dim | conv_alpha | unit |
    # @markdown | :---: | :---: | :---: | :---: | :---: | :---: |
    # @markdown | LoRA-LierLa | 32 | 1 | - | - | - |
    # @markdown | LoCon/LoRA-C3Lier | 16 | 8 | 8 | 1 | - |
    # @markdown | LoHa | 8 | 4 | 4 | 1 | - |
    # @markdown | Other Category | ? | ? | ? | ? | - |

    # @markdown Specify `network_args` to add `optional` training args, like for specifying each 25 block weight, read [this](https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E9%9A%8E%E5%B1%A4%E5%88%A5%E5%AD%A6%E7%BF%92%E7%8E%87)
    network_args    = ""  # @param {'type':'string'}

    # @markdown ### **Linear Layer Config**
    # @markdown Used by all `network_category`. When in doubt, set `network_dim = network_alpha`
    # network_dim     = 32  # @param {'type':'number'}
    # network_alpha   = 16  # @param {'type':'number'}

    # @markdown ### **Convolutional Layer Config**
    # @markdown Only required if `network_category` is not `LoRA_LierLa`, as it involves training convolutional layers in addition to linear layers.
    # conv_dim        = 32  # @param {'type':'number'}
    # conv_alpha      = 16  # @param {'type':'number'}

    # @markdown ### **DyLoRA Config**
    # @markdown Only required if `network_category` is `DyLoRA_LierLa` and `DyLoRA_C3Lier`
    unit = 4  # @param {'type':'number'}

    if isinstance(network_args, str):
        network_args = network_args.strip()
        if network_args.startswith('[') and network_args.endswith(']'):
            try:
                network_args = ast.literal_eval(network_args)
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing network_args: {e}\n")
                network_args = []
        elif len(network_args) > 0:
            print(f"WARNING! '{network_args}' is not a valid list! Put args like this: [\"args=1\", \"args=2\"]\n")
            network_args = []
        else:
            network_args = []
    else:
        network_args = []

    network_config = {
        "LoRA_LierLa": {
            "module": "networks.lora",
            "args"  : []
        },
        "LoRA_C3Lier": {
            "module": "networks.lora",
            "args"  : [
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}"
            ]
        },
        "DyLoRA_LierLa": {
            "module": "networks.dylora",
            "args"  : [
                f"unit={unit}"
            ]
        },
        "DyLoRA_C3Lier": {
            "module": "networks.dylora",
            "args"  : [
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
                f"unit={unit}"
            ]
        },
        "LoCon": {
            "module": "lycoris.kohya",
            "args"  : [
                f"algo=locon",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}"
            ]
        },
        "LoHa": {
            "module": "lycoris.kohya",
            "args"  : [
                f"algo=loha",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}"
            ]
        },
        "IA3": {
            "module": "lycoris.kohya",
            "args"  : [
                f"algo=ia3",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}"
            ]
        },
        "LoKR": {
            "module": "lycoris.kohya",
            "args"  : [
                f"algo=lokr",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}"
            ]
        },
        "DyLoRA_Lycoris": {
            "module": "lycoris.kohya",
            "args"  : [
                f"algo=dylora",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}"
            ]
        }
    }

    network_module = network_config[network_category]["module"]
    network_args.extend(network_config[network_category]["args"])

    lora_config = {
        "additional_network_arguments": {
            "no_metadata"                     : False,
            "network_module"                  : network_module,
            "network_dim"                     : network_dim,
            "network_alpha"                   : network_alpha,
            "network_args"                    : network_args,
            "network_train_unet_only"         : True,
            "training_comment"                : None,
        },
    }

    lora_config_string = toml.dumps(lora_config)
    print(lora_config_string)
    with open(config_path, "w") as file:
        file.write(lora_config_string)