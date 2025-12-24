"""
Configuration Management Module
Module for managing configuration
"""

import yaml
from pathlib import Path


def load_config(config_path: str = 'configs/experiment_config.yaml') -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded config from: {config_path}")
    return config


def get_model_config(config: dict, model_name: str) -> dict:
    """
    Get config for specific model
    
    Args:
        config: Full configuration
        model_name: Model name
    
    Returns:
        dict: Model configuration
    """
    models = config.get('models', {})
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config")
    
    return models[model_name]


def print_config(config: dict, indent: int = 0):
    """
    Display config nicely
    """
    for key, value in config.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}📁 {key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}  • {key}: {value}")


if __name__ == "__main__":
    # Test module
    config = load_config()
    
    print("\n📋 Full Configuration:")
    print("="*50)
    print_config(config)
    
    print("\n🌲 Random Forest Config:")
    print("="*50)
    rf_config = get_model_config(config, 'random_forest')
    print(rf_config)
