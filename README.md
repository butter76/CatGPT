# CatGPT 🐱

ML research project for chess and beyond.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/butter76/CatGPT.git
cd catgpt

# Install dependencies (creates .venv automatically)
uv sync

# Install with dev dependencies
uv sync --dev

# Install with all optional dependencies
uv sync --all-extras
```

### Development

```bash
# Run linting
uv run ruff check src tests scripts

# Run formatting
uv run ruff format src tests scripts

# Run type checking
uv run pyright src

# Run tests
uv run pytest tests -v

# Run tests with coverage
uv run pytest tests -v --cov=src/catgpt
```

### CLI Usage

```bash
# Show version
uv run catgpt version

# Train a model
uv run catgpt train --config base

# Evaluate a checkpoint
uv run catgpt evaluate checkpoints/model.pt
```

### Training Scripts

```bash
# Run training with Hydra
uv run python scripts/train.py

# Override config values
uv run python scripts/train.py model.hidden_size=512 training.batch_size=128
```

## Project Structure

```
catgpt/
├── src/catgpt/          # Main package (installable)
│   ├── models/          # Model definitions
│   ├── data/            # Data loading & processing
│   ├── training/        # Training loops
│   ├── evaluation/      # Metrics & evaluation
│   ├── chess/           # Chess engine
│   └── utils/           # Shared utilities
├── scripts/             # Standalone scripts
├── configs/             # Hydra/YAML configurations
├── tests/               # Test suite
├── checkpoints/         # Model checkpoints (gitignored)
├── data/                # Datasets (gitignored)
└── outputs/             # Training outputs (gitignored)
```

## Configuration

Configurations are managed with [Hydra](https://hydra.cc/). Base config is in `configs/base.yaml`.

Override any value from CLI:
```bash
uv run python scripts/train.py training.learning_rate=0.001 model.num_layers=12
```

## Contributing

1. Install dev dependencies: `uv sync --dev`
2. Make your changes
3. Run checks: `uv run ruff check . && uv run pyright src && uv run pytest tests`
4. Submit a PR

Bugbot will automatically review your PR for issues.

## License

MIT
