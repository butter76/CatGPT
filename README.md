# CatGPT 🐱

ML research project for chess and beyond. Supports both **PyTorch** and **JAX** frameworks.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/your-org/catgpt.git
cd catgpt

# Install with PyTorch (default)
uv sync --extra torch

# Or install with JAX
uv sync --extra jax

# Or install with both frameworks
uv sync --extra torch --extra jax

# Install with dev dependencies
uv sync --extra torch --extra dev
```

## Framework Support

CatGPT supports multiple ML frameworks with a unified interface:

```python
# PyTorch
from catgpt.torch.models import BaseModel
from catgpt.torch.training import Trainer
from catgpt.torch.optimizers import SPlus

# JAX/Flax
from catgpt.jax.models import BaseModel
from catgpt.jax.training import Trainer

# Shared utilities (framework-agnostic)
from catgpt.core import setup_logging, load_config
from catgpt.core.chess import ChessEngine
```

## Project Structure

```
catgpt/
├── src/catgpt/
│   ├── core/                # Shared, framework-agnostic code
│   │   ├── chess/           # Chess engine (no ML deps)
│   │   ├── configs/         # Configuration management
│   │   ├── data/            # Common data types
│   │   ├── evaluation/      # Shared evaluation logic
│   │   └── utils/           # Logging, etc.
│   ├── torch/               # PyTorch implementations
│   │   ├── models/          # PyTorch models
│   │   ├── optimizers/      # Custom optimizers (SPlus)
│   │   ├── training/        # Training loops
│   │   ├── evaluation/      # PyTorch metrics
│   │   └── data/            # PyTorch datasets
│   └── jax/                 # JAX/Flax implementations
│       ├── models/          # Flax models
│       ├── optimizers/      # Optax extensions
│       ├── training/        # JAX training loops
│       ├── evaluation/      # JAX metrics
│       └── data/            # JAX data loading
├── scripts/                 # Training & evaluation scripts
├── configs/                 # Hydra configurations
├── tests/                   # Test suite
└── ...
```

## Development

```bash
# Run linting
uv run ruff check src tests scripts

# Run type checking
uv run pyright src

# Run tests
uv run pytest tests -v

# Run tests with coverage
uv run pytest tests -v --cov=src/catgpt
```

## CLI Usage

```bash
# Show version
uv run catgpt version

# Train a model (PyTorch)
uv run catgpt train --config base --framework torch

# Train a model (JAX)
uv run catgpt train --config base --framework jax

# Evaluate a checkpoint
uv run catgpt evaluate checkpoints/model.pt --framework torch
```

## Configuration

Configurations are managed with [Hydra](https://hydra.cc/). Base config is in `configs/base.yaml`.

Override any value from CLI:
```bash
uv run python scripts/train.py training.learning_rate=0.001 model.num_layers=12
```

## Optional Dependencies

| Extra | Packages | Install |
|-------|----------|---------|
| `torch` | PyTorch | `uv sync --extra torch` |
| `jax` | JAX, Flax, Optax (CPU) | `uv sync --extra jax` |
| `jax-cuda` | JAX, Flax, Optax (CUDA) | `uv sync --extra jax-cuda` |
| `dev` | pytest, ruff, pyright | `uv sync --extra dev` |
| `notebook` | Jupyter, matplotlib | `uv sync --extra notebook` |
| `all` | Everything | `uv sync --extra all` |

## Contributing

1. Install dev dependencies: `uv sync --extra torch --extra dev`
2. Make your changes
3. Run checks: `uv run ruff check src tests scripts && uv run pyright src && uv run pytest tests`
4. Submit a PR

Bugbot will automatically review your PR for issues.

## License

MIT
