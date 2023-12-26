conf = {
    "device": "Ascend",
    "device_id": 0,

    "W_surface": 0.1,
    "W_texture": 1,
    "W_structure": 200,
    "W_content": 200,
    "W_tv": 20000,
    "W_motion": 0.1,

    "slic_num": 25,

    "epoch": 30,
    "batch": 16,

    "lr": 1e-4,
    "sn": True,
    "G": True,
    "D": True,

    "continue_training": True,

    "real_train_dataset":
    [
        "./data/train_photo/",
        "./data/test/real/",
    ],
    "real_test_dataset":
    [
        "./data/test/test_photo256/",
    ],
    "cartoon_train_dataset":
    [
        "./data/Hayao/style/",
        "./data/Paprika/style/",
        "./data/Shinkai/style/",
        "./data/SummerWar/style/",
        "./data/Paprika/style/",
    ],
    "cartoon_test_dataset":
    [
        "./data/spirit_away/",
        "./data/test/label_map/",
    ]
}
