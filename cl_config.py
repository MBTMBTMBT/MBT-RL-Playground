def set_up(config: dict,):
    pass


configs = {
    "frozen_lake-custom": {
        "env_type": "frozen_lake",
        "save_path": "./experiments/cl/frozen_lake-custom/frozen_lake-custom",
        "env_names": [
            "target", "env-0", "env-1", "env-2", "env-3", "env-4",
        ],
        "slipperiness": [
            0.66, 0.99,
        ],
        "descs": {
            "target": [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFHFFHHF",
                "FFFFFFFF",
                "FFFFHHFF",
                "FFFFHGFF",
                "FFHFFFFF",
                "FFFFHFFF",
            ],
            "env-0": [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFGFF",
                "FFFFFFFF",
                "FFFFFFFF",
            ],
            "env-1": [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFHHFF",
                "FFFFHGFF",
                "FFFFFFFF",
                "FFFFHFFF",
            ],
            "env-2": [
                "SFFFFFFF",
                "FFFFFFFF",
                "HHHHHHHF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFGFF",
                "FFFFFFFF",
                "FFFFFFFF",
            ],
            "env-3": [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFHFFFFF",
                "FFFFFFFF",
                "FFFFHHFF",
                "FFFFHGFF",
                "FFHFFFFF",
                "FFFFHFFF",
            ],
            "env-4": [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFHFFHHF",
                "FFFFFFFF",
                "FFFFHHFF",
                "FFFFHGFF",
                "FFHFFFFF",
                "FFFFHFFF",
            ],
        }
    }
}
