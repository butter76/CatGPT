# CatGPT C++ Engine

The native side of CatGPT: a C++23 / GCC 14 / TensorRT 10 chess engine built around `LksSearch` — a Lazy-SMP, log-scale iterative-deepening search batched onto the GPU through TensorRT. The production binary is `lks_uci`, a UCI engine; `trt_benchmark` is its companion throughput/latency tool. The legacy `catgpt_mcts` / `catgpt_fractional_mcts` family is still built (when libcoro is present) for SPRT comparison.

## Requirements

- **CMake** 3.22+
- **GCC 14** (system gcc is _not_ sufficient — C++23 `std::print`, coroutines, etc.). The bootstrap builds it from source into `gcc-14/` automatically; otherwise install it yourself and point `-DCMAKE_CXX_COMPILER=g++-14`.
- **CUDA** 12.x toolkit (driver + `libcudart`).
- **TensorRT** 10.16.x (10.16.1.11 is what `scripts/build.sh` pins).
- **Linux x86_64.**
- **Git submodules** under `external/`:
  - [`libfork`](https://github.com/ConorWilliams/libfork) — continuation-stealing fork-join coroutine runtime (used by `lks_uci` and friends).
  - [`fathom`](https://github.com/jdart1/Fathom) — Syzygy tablebase prober.
  - [`chess-library`](https://github.com/Disservin/chess-library) — move generation / board representation.

  Initialize from the repo root:
  ```bash
  git submodule update --init --recursive
  ```

- **Optional:** vcpkg-installed [`libcoro`](https://github.com/jbaldwin/libcoro) at `$HOME/vcpkg`. Only the legacy `catgpt_mcts` / `catgpt_fractional_mcts` / `catgpt_selfplay` / `catgpt_puzzle_eval` targets need it; cmake auto-detects and silently skips them otherwise.

## Building

### Recommended: sudo-less bootstrap

For a fresh GPU box, use the top-level [`scripts/build.sh`](../scripts/build.sh) (normally invoked by [`update.sh`](../update.sh) — see the root [`README.md`](../README.md)). It is idempotent and runs eight phases, sentinel-skipping anything already done:

1. Machine scan (logs uname, glibc, gcc, nvidia-smi, nvcc, candidate CUDA dirs).
2. Locate a CUDA 12.x toolkit (honors `CUDA_ROOT_OVERRIDE` / `CUDA_HOME`).
3. Build **GCC 14** from source into `$WORK_DIR/gcc-14/` (~30–60 min on first run).
4. Download and unpack **TensorRT 10.16.1.11** into `$WORK_DIR/TensorRT-10.16.1.11/`.
5. Verify `$WORK_DIR/main.onnx` exists (or fetch via `MAIN_ONNX_URL`).
6. `cmake` configure and build `lks_uci` + `trt_benchmark` into `cpp/build/bin/`.
7. Build per-bucket TensorRT engines and pack them into `$WORK_DIR/main.network` (via [`scripts/trt.sh`](../scripts/trt.sh) + [`scripts/pack_network.py`](../scripts/pack_network.py)).
8. Run `trt_benchmark` against the packed network.

The result is a self-contained `$WORK_DIR/catgpt` UCI binary plus `$WORK_DIR/main.network`.

### Manual cmake

If you have GCC 14, CUDA, and TensorRT installed in known locations:

```bash
cd cpp
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/path/to/gcc-14 \
    -DCMAKE_CXX_COMPILER=/path/to/g++-14 \
    -DCUDA_ROOT=/usr/local/cuda-12.8 \
    -DTENSORRT_ROOT=/path/to/TensorRT-10.16.1.11 \
    -DGCC14_LIB_DIR=/path/to/gcc-14/lib64

cmake --build build -j$(nproc) --target lks_uci trt_benchmark
```

Add `-DVCPKG_ROOT=$HOME/vcpkg` if you want the legacy libcoro-based targets.

The tiny [`build.sh`](build.sh) in this directory is just a one-shot `cmake .. && cmake --build .` wrapper for local iteration; production builds go through `scripts/build.sh`.

## Headline binaries

### `lks_uci` — production UCI engine

Bespoke UCI loop on top of `LksSearch`. The worker thread emits `info` / `bestmove` lines as it searches; the driver streams them straight to stdout. Tree state is reused across `position` commands when the new line is a strict prefix-extension of the previous one, preserving the shared transposition table.

```bash
# Default engine path: $CATGPT_TRT_ENGINE, else ./main.network
./build/bin/lks_uci

# Or pass it explicitly
./build/bin/lks_uci /path/to/main.network
```

Supported UCI commands: `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, `quit`.

**Tuning environment variables** (defaults tuned for GPU saturation, applied per `go` from [`cpp/src/lks_uci_main.cpp`](src/lks_uci_main.cpp)):

| Variable | Default | Meaning |
|---|---|---|
| `CATGPT_TRT_ENGINE` | `./main.network` | Engine path if not passed on the CLI |
| `LKS_WORKERS_PER_GPU` | `2` | Total workers = this × `#CUDA devices` |
| `LKS_COROS_PER_WORKER` | `256` | Concurrent coroutines per worker |
| `LKS_MAX_BATCH_SIZE` | `112` | Max GPU batch per inference |
| `LKS_LIFETIME_MAX_EVALS` | `1<<27` | Arena lifetime cap (per process) |
| `LKS_DELTA_DEPTH` | `0.2` | Per-iteration log-scale ID step (`LksSearchConfig::delta_depth`) |
| `LKS_C_PUCT` | `1.75` | PUCT exploration constant |
| `LKS_MAX_DEPTH` | `32` | Log-scale ID ceiling per `go` |
| `LKS_SYZYGY_PATH` | `$SYZYGY_HOME` | Directory of `.rtbw`/`.rtbz` files; eligible root positions resolve via DTZ probe instead of NN search |

### `trt_benchmark` — inference throughput / latency

Loads a `.network` (or raw `.trt`) and sweeps a series of batch sizes, reporting average / min / max latency and throughput.

```bash
./build/bin/trt_benchmark                       # uses $CATGPT_TRT_ENGINE or ./main.network
./build/bin/trt_benchmark /path/to/main.network # or pass the path
```

Example output:

```
╔════════════════════════════════════════════════════════════════╗
║         CatGPT Chess Engine - TensorRT Benchmark               ║
╚════════════════════════════════════════════════════════════════╝

┌─ CUDA Device ─────────────────────────────────────────────────┐
│ Device 0: NVIDIA GeForce RTX 5090
│ Compute: 12.0, Memory: 31.4 GB, SMs: 170
└───────────────────────────────────────────────────────────────┘

┌─ Benchmark Results ───────────────────────────────────────────┐
│  Batch │   Avg (ms) │   Throughput │   Min (ms) │   Max (ms) │
│────────┼────────────┼──────────────┼────────────┼────────────│
│      1 │      2.027 │        493/s │      2.005 │      2.187 │
│     64 │      4.386 │      14593/s │      4.232 │      4.543 │
│    256 │     19.326 │      13246/s │     19.273 │     19.659 │
└───────────────────────────────────────────────────────────────┘

★ Optimal batch size: 64 (14960 samples/sec)
```

## Model I/O

The current export is gather-aware: the GPU collapses the full `(64, 73) = 4672` policy tensor down to the legal-move set on-device, so only legal-move logits are copied back to the host.

**Inputs**

| Tensor | dtype | shape | notes |
|---|---|---|---|
| `in_0` (tokens) | `int32` | `(B, 64)` | Tokenized board (one token per square) |
| `in_1` (legal_indices) | `int32` | `(B, MAX_LEGAL_MOVES)` | Flat indices into the 4672-wide policy tensor; `MAX_LEGAL_MOVES = 218` |

**Outputs**

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `wdl_logit` | `float32` | `(B, 3)` | Win / draw / loss logits |
| `bestq_probs` | `float32` | `(B, 81)` | Discretized best-Q distribution (`VALUE_NUM_BINS = 81`) |
| `optimistic_policy_legal_logit` | `float32` | `(B, MAX_LEGAL_MOVES)` | Policy logits gathered at `legal_indices` |

### `.network` bundles

`lks_uci` and `trt_benchmark` accept either a raw single-engine `.trt` file or a packed `.network` produced by [`scripts/pack_network.py`](../scripts/pack_network.py). A `.network` contains multiple TensorRT engines built for different batch-size buckets, plus metadata; the loader picks the right bucket per inference. Produce one with [`scripts/trt.sh`](../scripts/trt.sh):

```bash
TRT_ROOT=/path/to/TensorRT-10.16.1.11 \
ONNX=/path/to/main.onnx \
NETWORK_OUT=/path/to/main.network \
bash scripts/trt.sh
```

## Other binaries

These are built by the same `CMakeLists.txt` and may be useful for development, testing, or comparison.

**Tools**
- `catgpt_analyze` — load a `.network`, run one synchronous inference on a FEN (CLI arg or stdin), pretty-print WDL + BestQ + optimistic policy over legal moves.
- `catgpt_search` — standalone LKS search that emits JSON-per-line search stats; backs the `web/` Next.js viewer over SSE.

**Legacy MCTS family** (built only when libcoro is found via vcpkg)
- `catgpt_mcts` — pure MCTS UCI engine on top of the same TRT batcher.
- `catgpt_fractional_mcts` — fractional MCTS variant.
- `catgpt_selfplay` — batched self-play generator.
- `catgpt_puzzle_eval` — batched puzzle accuracy harness.

**Tests & microbenches** (no NN dependencies unless noted)
- `chess_lib_tests` — sanity checks for the `chess-library` submodule.
- `tt_arena_test` / `tt_arena_bench` / `tt_arena_concurrent_bench` — unit tests + single-threaded and multi-threaded benchmarks for the lock-free TT/arena used by fractional MCTS v2.
- `lks_search_test` — end-to-end LKS lifecycle + real TRT correctness test.
- `lks_throughput_bench` — throughput sweep over real TRT.
- `libfork_smoke_test` — PoC bridging libfork to a fake batched evaluator thread; no TRT, no chess deps.

## Project layout

```
cpp/
├── CMakeLists.txt
├── build.sh                       # local one-shot build wrapper
├── .clang-format
├── cmake/
│   └── GenVersion.cmake           # writes catgpt_version.hpp from git
├── src/
│   ├── main.cpp                   # chess_engine placeholder
│   ├── lks_uci_main.cpp           # production UCI engine entry point
│   ├── trt_benchmark.cpp          # TRT throughput tool
│   ├── analyze_fen_main.cpp       # single-FEN NN dump tool
│   ├── lks_search_main.cpp        # catgpt_search (web backend)
│   ├── mcts_uci_main.cpp          # legacy MCTS UCI
│   ├── fractional_mcts_uci_main.cpp
│   ├── selfplay_main.cpp          # legacy batched selfplay
│   ├── puzzle_eval_main.cpp       # legacy batched puzzle eval
│   ├── libfork_smoke_test.cpp
│   ├── tokenizer.hpp
│   ├── syzygy.hpp
│   ├── engine/                    # search-algo-agnostic engine pieces
│   │   ├── lks/                   # LksSearch (production)
│   │   ├── mcts/                  # legacy MCTS
│   │   ├── fractional_mcts/       # legacy fractional MCTS + v2 TT/arena
│   │   ├── trt_runtime.hpp        # TRT engine + batched evaluator
│   │   ├── network_file.hpp       # .network bundle loader
│   │   ├── nn_constants.hpp       # MAX_LEGAL_MOVES, VALUE_NUM_BINS, ...
│   │   ├── policy.hpp
│   │   ├── search_algo.hpp
│   │   ├── search_limits.hpp
│   │   └── search_result.hpp
│   ├── selfplay/                  # selfplay runner + game-slot coroutines
│   ├── uci/                       # shared UCI handler scaffolding
│   └── lf/                        # libfork helpers (e.g. async_semaphore)
├── external/                      # git submodules
│   ├── libfork/
│   ├── fathom/
│   └── chess-library/
└── tests/
    └── chess_lib_tests.cpp
```

## C++23 features used

- `std::print` / `std::println` and `std::format` everywhere a printf would go.
- `std::span` for non-owning buffer views.
- `std::ranges` for iteration / view composition.
- `std::jthread` + `std::stop_token` for cooperative cancellation in the search worker.
- Designated initializers in configuration aggregates.
- Coroutines (`-fcoroutines`): `libfork`-based fork-join, plus the legacy `libcoro`-based `catgpt_mcts` family.

## CMake cache variables

| Variable | Default | Purpose |
|---|---|---|
| `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` | `gcc-14` / `g++-14` | Toolchain (must be on PATH or absolute) |
| `CUDA_ROOT` | `/usr/local/cuda-12.8` | CUDA toolkit root (must contain `include/cuda.h` and `lib64/libcudart.so`) |
| `TENSORRT_ROOT` | `/home/shadeform/TensorRT-10.16.1.11` | TensorRT root (`include/NvInfer.h`, `lib/libnvinfer.so`) |
| `GCC14_LIB_DIR` | `/usr/local/lib64` | Dir containing `libstdc++.so.6` for GCC 14; baked into BUILD_RPATH/INSTALL_RPATH |
| `VCPKG_ROOT` | `$HOME/vcpkg` | Used to find optional libcoro; legacy targets are silently skipped if missing |
