"""Tests for core utilities."""

from pathlib import Path

from catgpt.core.configs import load_config, save_config
from catgpt.core.data import Batch, Sample


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


class TestDataTypes:
    """Tests for core data types."""

    def test_sample_creation(self) -> None:
        """Test creating a Sample."""
        sample = Sample(inputs=[1, 2, 3], targets=1)
        assert sample.inputs == [1, 2, 3]
        assert sample.targets == 1
        assert sample.metadata is None

    def test_sample_with_metadata(self) -> None:
        """Test Sample with metadata."""
        sample = Sample(inputs=[1, 2], targets=0, metadata={"id": "abc123"})
        assert sample.metadata == {"id": "abc123"}

    def test_batch_len(self) -> None:
        """Test Batch length."""
        batch = Batch(inputs=[[1], [2], [3]], targets=[0, 1, 0])
        assert len(batch) == 3
