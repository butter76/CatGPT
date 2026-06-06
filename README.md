# CatGPT

A chess GPT trained in **JAX/Flax**, served on the GPU through a custom C++/TensorRT engine, driven by a new search algorithm called **Likelihood Search (LKS)** that targets the failure modes of vanilla MCTS at high node counts.

The point of CatGPT is to break several widely-held assumptions about network design and search stemming from AlphaZero.

The repo holds three first-class pieces:

- **Training stack** (`src/catgpt/`, `scripts/`, `configs/`) — JAX/Flax models, optimizers, Hydra configs, data pipeline, ONNX export.
- **Inference engine** (`cpp/`) — C++23 / GCC 14 / TensorRT 10 UCI engine. See [`cpp/README.md`](cpp/README.md) for build and run instructions.
- **Web viewer** (`web/`) — Next.js UI that streams live search stats from the engine over SSE.

## Repo layout

```
CatGPT/
├── src/catgpt/        # Python package
│   ├── core/          # Framework-agnostic: chess, configs, data, evaluation
│   ├── jax/           # JAX/Flax models, optimizers, training, evaluation
│   ├── cpp/           # Python adapter that drives the UCI engine
│   └── tournament/    # SPRT harness + opening books
├── cpp/               # Native C++ engine (lks_uci, trt_benchmark, ...)
├── web/               # Next.js viewer for live search
├── scripts/           # Train / eval / convert / pack / tournament drivers
├── configs/           # Hydra configs (jax_base, jax_eval, cpp_eval, ...)
├── tests/             # Python test suite
├── checkpoints_jax/   # Local checkpoint dir (gitignored)
├── update.sh          # one-file sudo-less bootstrap (calls scripts/build.sh)
└── pyproject.toml
```

## Quick start

### A. Run the engine on a fresh GPU box (sudo-less bootstrap)

You need only an NVIDIA driver and a CUDA 12.x toolkit installed by an admin; everything else (GCC 14, TensorRT 10.16, submodules) is built/fetched into a working directory of your choice.

```bash
# Put update.sh in an empty dir on the target host, then:
chmod +x update.sh
./update.sh
```

[`update.sh`](update.sh) clones the repo, initializes the `libfork` / `fathom` / `chess-library` submodules, and hands off to [`scripts/build.sh`](scripts/build.sh), which runs an idempotent 8-phase build:

1. Machine scan (logs gcc / cmake / nvidia / cuda state).
2. Locate a CUDA 12.x toolkit (or fail fast).
3. Build GCC 14 from source into `gcc-14/` (skipped if already present).
4. Download and unpack TensorRT 10.16.1.11 into `TensorRT-10.16.1.11/`.
5. Verify `${MODEL}.onnx` is present (either staged manually or fetched via `ONNX_URL`). `MODEL` defaults to `S4`; set `MODEL=S2` to point the same pipeline at `S2.onnx`.
6. Configure cmake and build `lks_uci` + `trt_benchmark`.
7. Pack per-bucket TRT engines into `${MODEL}.network` via [`scripts/trt.sh`](scripts/trt.sh) + [`scripts/pack_network.py`](scripts/pack_network.py).
8. Run `trt_benchmark` against the packed network.

Output artifacts land alongside `update.sh`:

```
./catgpt          # UCI engine binary (copy of cpp/build/bin/lks_uci)
./S4.network      # Packed multi-bucket TRT engine (or ${MODEL}.network)
./trt_benchmark.txt  # Benchmark log
./build.log       # Full build log
```

`./catgpt` speaks UCI; point Cute Chess / Banksia / lichess-bot at it the same way you would Stockfish. See [`cpp/README.md`](cpp/README.md) for tuning env vars (`LKS_*`) and `CATGPT_TRT_ENGINE`.

### B. Hack on Python (training, eval, data)

Prerequisites: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/butter76/CatGPT.git
cd CatGPT

# CPU-only JAX:
uv sync --extra jax

# Or GPU JAX (CUDA 12):
uv sync --extra jax-cuda

# Add dev tooling:
uv sync --extra jax-cuda --extra dev
```

The Python tree is configured via Hydra; the canonical config is [`configs/jax_base.yaml`](configs/jax_base.yaml).

## Training and evaluation

All scripts accept Hydra-style key=value overrides.

```bash
# Train the JAX model
uv run python scripts/train_jax.py
uv run python scripts/train_jax.py model.hidden_size=512 training.batch_size=128
uv run python scripts/train_jax.py +resume_from=checkpoints_jax/epoch_10

# Evaluate a JAX checkpoint
uv run python scripts/evaluate_jax.py

# Export a trained model to ONNX (used by trt.sh + pack_network.py)
uv run python scripts/export_onnx.py

# Run an SPRT match between two C++ engine configurations
uv run python scripts/sprt_tournament.py \
    engine_a.type=mcts engine_a.mcts.num_simulations=400 \
    engine_b.type=mcts engine_b.mcts.num_simulations=800

# Self-play data generation
uv run python scripts/selfplay.py
```

Other scripts of note live under [`scripts/`](scripts/): `pipeline_bagz_to_training.py` and `batch_convert_*.py` for the Leela bag → training-bag data pipeline, `download_syzygy_*.py` for tablebases, `find_unsolved_puzzles.py` / `evaluate_cpp.py` for puzzle evaluation.

## Web viewer

[`web/`](web/) is a Next.js app that spawns the `catgpt_search` binary as a subprocess and streams its JSON-per-line search stats to the browser over Server-Sent Events. It is independent of the training stack.

```bash
cd web
npm install
npm run dev
# open http://localhost:3000
```

The viewer expects the `catgpt_search` binary and a `.network` engine on the host — build them via `update.sh` first.

### Tournament / Playtest environment

The web app also hosts a cutechess-driven tournament environment (`/tournaments`) for running engine-vs-engine matches (e.g. CatGPT `lks_uci` vs Stockfish at 15m+5s). Games stream live, are replayable move-by-move with per-move evals, store full UCI debug logs for later debugging, and each ply deep-links into Quick Analysis.

It requires `cutechess-cli` on the host:

```bash
./scripts/build-cutechess.sh        # builds cutechess-cli from source (Qt5)
```

Then configure the relevant env vars in `web/.env.local` (see [`web/.env.example`](web/.env.example)): `CUTECHESS_CLI_PATH`, `LKS_UCI_PATH`, `LKS_NETWORK_PATH`, `STOCKFISH_PATH`, and `SYZYGY_HOME` (used for tablebase adjudication). Apply the database schema with `npm run db:push` from `web/`.

## C++ engine

The native engine is built around `LksSearch` (Lazy-SMP, log-scale iterative deepening, GPU-batched via TensorRT) plus a bespoke UCI loop. Full details, env vars, model I/O, and the list of secondary binaries (`catgpt_analyze`, legacy MCTS engines, microbenches) live in [`cpp/README.md`](cpp/README.md).

## Development

```bash
# Lint
uv run ruff check src tests scripts

# Type check
uv run pyright src

# Tests
uv run pytest tests -v

# Tests with coverage
uv run pytest tests -v --cov=src/catgpt
```

Bugbot will automatically review PRs for issues.

## Optional dependencies

| Extra | Packages | Install |
|-------|----------|---------|
| `jax` | JAX, Flax, Optax, Orbax (CPU) | `uv sync --extra jax` |
| `jax-cuda` | JAX, Flax, Optax, Orbax (CUDA 12) | `uv sync --extra jax-cuda` |
| `dev` | pytest, ruff, pyright, pre-commit, ipython | `uv sync --extra dev` |
| `notebook` | Jupyter, matplotlib, seaborn, plotly | `uv sync --extra notebook` |
| `export` | ONNX, onnxruntime, jax2onnx, TensorFlow, tf2onnx | `uv sync --extra export` |
| `all` | Everything above | `uv sync --extra all` |

See [`pyproject.toml`](pyproject.toml) for exact pins.

## License

CatGPT is released under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) — see [`LICENSE`](LICENSE).

Free for noncommercial use: personal projects, research, experimentation, education, and use by charitable, educational, public-research, public-safety, environmental, and governmental organizations. **Commercial use is not permitted under this license.** For a commercial license, please open an issue or contact the maintainers.

Third-party components under `cpp/external/` (`libfork`, `fathom`, `chess-library`, …) remain under their original upstream licenses; see the `LICENSE` file inside each dependency directory.
