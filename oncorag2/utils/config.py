"""
Configuration handling utilities.

This module handles loading and validating configuration from files and
environment variables.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def load_feature_config(config_path: Union[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load feature configuration from a JSON file.

    Args:
        config_path: Path to the feature configuration JSON file

    Returns:
        Dictionary with "features" key containing list of feature dictionaries
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Validate minimum structure
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")

        if "features" not in config:
            raise ValueError('Configuration must contain a "features" key')

        if not isinstance(config["features"], list):
            raise ValueError('"features" must be a list')

        # Log success
        feature_count = len(config["features"])
        logger.info(f"Loaded {feature_count} features from {config_path}")

        return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading feature configuration from {config_path}: {str(e)}")
        # Return empty features list as fallback
        return {"features": []}


def save_feature_config(
        config: Dict[str, List[Dict[str, Any]]],
        config_path: Union[str, Path]
) -> bool:
    """
    Save feature configuration to a JSON file.

    Args:
        config: Feature configuration dictionary
        config_path: Path to save the configuration to

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

        # Write the file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        feature_count = len(config.get("features", []))
        logger.info(f"Saved {feature_count} features to {config_path}")

        return True
    except Exception as e:
        logger.error(f"Error saving feature configuration to {config_path}: {str(e)}")
        return False


def get_env_var(
        name: str,
        default: Optional[str] = None,
        required: bool = False
) -> Optional[str]:
    """
    Get an environment variable with optional default.

    Args:
        name: Name of the environment variable
        default: Default value if environment variable is not set
        required: Whether the environment variable is required

    Returns:
        Value of the environment variable or default

    Raises:
        ValueError: If required is True and the environment variable is not set
    """
    value = os.environ.get(name)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable {name} is not set")
        return default

    return value


def get_api_key_for_platform(platform: str) -> Optional[str]:
    """
    Get the API key for a specific LLM platform.

    Args:
        platform: Platform name (e.g., 'openai', 'anthropic')

    Returns:
        API key for the platform, or None if not found

    Raises:
        ValueError: If the platform is not supported
    """
    platform_env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "cohere": "COHERE_API_KEY",
    }

    if platform.lower() not in platform_env_vars:
        raise ValueError(f"Unsupported platform: {platform}")

    env_var = platform_env_vars[platform.lower()]
    return get_env_var(env_var)