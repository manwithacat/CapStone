"""
Configuration loading and validation.
"""
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: dict) -> None:
    """
    Validate configuration dictionary has required keys.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ['image', 'hog', 'glcm', 'stats', 'xgb', 'runtime']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate image config
    if 'img_size' not in config['image']:
        raise ValueError("Missing 'img_size' in image config")

    # Validate runtime config
    runtime = config['runtime']
    if 'random_state' not in runtime:
        raise ValueError("Missing 'random_state' in runtime config")
