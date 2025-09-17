from omegaconf import OmegaConf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_name: str):
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    return OmegaConf.load(config_path)
