# CatGPT 🐱

ML research project for chess and beyond built with **JAX** and **Flax**.

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

# Install with JAX (CPU)
uv sync --extra jax

# Or install with JAX (CUDA)
uv sync --extra jax-cuda

# Install with dev dependencies
uv sync --extra jax --extra dev
```

## Framework Support

CatGPT is built on JAX/Flax for high-performance machine learning:

```python
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

# Train a model (JAX)
uv run python scripts/train_jax.py

# Evaluate a checkpoint
uv run python scripts/evaluate_jax.py --checkpoint checkpoints/model
```

## Configuration

Configurations are managed with [Hydra](https://hydra.cc/). Base config is in `configs/jax_base.yaml`.

Override any value from CLI:
```bash
uv run python scripts/train_jax.py training.learning_rate=0.001 model.num_layers=12
```

## Optional Dependencies

| Extra | Packages | Install |
|-------|----------|---------|
| `jax` | JAX, Flax, Optax (CPU) | `uv sync --extra jax` |
| `jax-cuda` | JAX, Flax, Optax (CUDA) | `uv sync --extra jax-cuda` |
| `dev` | pytest, ruff, pyright | `uv sync --extra dev` |
| `notebook` | Jupyter, matplotlib | `uv sync --extra notebook` |
| `export` | ONNX, TensorFlow (for model export) | `uv sync --extra export` |
| `all` | Everything | `uv sync --extra all` |

## Contributing

1. Install dev dependencies: `uv sync --extra jax --extra dev`
2. Make your changes
3. Run checks: `uv run ruff check src tests scripts && uv run pyright src && uv run pytest tests`
4. Submit a PR

Bugbot will automatically review your PR for issues.

## License

MIT
