"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a configuration file with optional overrides.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Optional list of CLI-style overrides (e.g., ["model.hidden_size=256"]).

    Returns:
        Merged configuration as a DictConfig.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    config = OmegaConf.load(config_path)

    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_conf)

    return config


def save_config(config: DictConfig | dict[str, Any], path: str | Path) -> None:
    """Save a configuration to a YAML file.

    Args:
        config: Configuration to save.
        path: Path to save the configuration to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, dict):
        config = OmegaConf.create(config)

    OmegaConf.save(config, path)
