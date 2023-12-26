import yaml

from mindyolo.utils.config import load_config


def write_yaml(r):
    with open('yolox-x.yaml', "w", encoding="utf-8") as f:
        yaml.dump(r, f)


cfg, _, _ = load_config('mindyolo/configs/yolox/yolox-x.yaml')
write_yaml(cfg)
