import toml


def set_dataset_config(
        dataset_config_path,
        train_image_dir,
        instance_token,
        class_token,
        num_repeats,
        resolution,
        batch_size
    ):

    config = {
        "general": {
            "shuffle_caption": True,
            "caption_extension": ".txt",
            "keep_tokens": 1
        },
        "datasets": [
            {
                "resolution": resolution,
                "batch_size": batch_size,
                "keep_tokens": 2,
                "subsets": [
                    {
                        "image_dir": f"{train_image_dir}",
                        "class_tokens": f"{instance_token} {class_token}",
                        "num_repeats": num_repeats
                        # 'keep_tokens' is inherited from the parent 'datasets'
                    },
                    # {
                    #     "image_dir": "C:\\fuga",
                    #     "class_tokens": "fuga boy",
                    #     "keep_tokens": 3
                    # },
                    # {
                    #     "is_reg": True,
                    #     "image_dir": "C:\\reg",
                    #     "class_tokens": "human",
                    #     "keep_tokens": 1
                    # }
                ]
            }
        ]
    }
    dataset_config_string = toml.dumps(config)
    with open(dataset_config_path, "w") as file:
        file.write(dataset_config_string)