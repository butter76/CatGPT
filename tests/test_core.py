"""Tests for core utilities."""

from pathlib import Path

from catgpt.core.configs import load_config, save_config


class TestConfig:
    """Tests for configuration utilities."""

    def test_load_config_from_file(self, tmp_path: Path) -> None:
        """Test loading a config file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("model:\n  hidden_size: 256\n  layers: 4\n")

        config = load_config(config_file)

        assert config.model.hidden_size == 256
        assert config.model.layers == 4

    def test_load_config_with_overrides(self, tmp_path: Path) -> None:
        """Test loading config with CLI overrides."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("model:\n  hidden_size: 256\n")

        config = load_config(config_file, overrides=["model.hidden_size=512"])

        assert config.model.hidden_size == 512

    def test_save_config(self, tmp_path: Path) -> None:
        """Test saving a config file."""
        config = {"model": {"hidden_size": 256}}
        config_file = tmp_path / "output.yaml"

        save_config(config, config_file)

        assert config_file.exists()
        loaded = load_config(config_file)
        assert loaded.model.hidden_size == 256
