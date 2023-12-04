import toml
import ast


def set_optimizer_config(
        optimizer_config_path,
        optimizer_type = "AdaFactor",
        learning_rate = 1e-4,
        lr_scheduler = "constant_with_warmup",
        lr_warmup_steps = 0,
        lr_scheduler_num = 0,
):
    # @title ## **4.2. Optimizer Config**
    # @markdown Use `Adafactor` optimizer. `RMSprop 8bit` or `Adagrad 8bit` may work. `AdamW 8bit` doesn't seem to work.
    # optimizer_type = "AdaFactor"  # @param ["AdamW", "AdamW8bit", "Lion8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation(DAdaptAdamPreprint)", "DAdaptAdaGrad", "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor"]
    # @markdown Specify `optimizer_args` to add `additional` args for optimizer, e.g: `["weight_decay=0.6"]`
    optimizer_args = "[ \"scale_parameter=False\", \"relative_step=False\", \"warmup_init=False\" ]"  # @param {'type':'string'}
    # @markdown ### **Learning Rate Config**
    # @markdown Different `optimizer_type` and `network_category` for some condition requires different learning rate. It's recommended to set `text_encoder_lr = 1/2 * unet_lr`
    # learning_rate = 1e-4  # @param {'type':'number'}
    # @markdown ### **LR Scheduler Config**
    # @markdown `lr_scheduler` provides several methods to adjust the learning rate based on the number of epochs.
    # lr_scheduler = "constant_with_warmup"  # @param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"] {allow-input: false}
    # lr_warmup_steps = 100  # @param {'type':'number'}
    # @markdown Specify `lr_scheduler_num` with `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial`
    # lr_scheduler_num = 0  # @param {'type':'number'}

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
            "max_grad_norm"           : 0,
            "optimizer_args"          : optimizer_args,
            "lr_scheduler"            : lr_scheduler,
            "lr_warmup_steps"         : lr_warmup_steps,
            "lr_scheduler_num_cycles" : lr_scheduler_num if lr_scheduler == "cosine_with_restarts" else None,
            "lr_scheduler_power"      : lr_scheduler_num if lr_scheduler == "polynomial" else None,
            "lr_scheduler_type"       : None,
            "lr_scheduler_args"       : None,
        },
    }

    optimizer_config_string = toml.dumps(optimizer_config)
    print(optimizer_config_string)
    with open(optimizer_config_path, "w") as file:
        file.write(optimizer_config_string)
    
