# CatGPT Chess Engine (C++)

A modern C++ chess engine using C++23 features and TensorRT for GPU-accelerated neural network evaluation.

## Requirements

- CMake 3.22+
- g++-14
- CUDA 12.x
- TensorRT 10.x
- Linux/Unix system

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)
```

Or use the convenience script:

```bash
./build.sh          # Release build
./build.sh Debug    # Debug build
```

## Executables

### `chess_engine`

Basic chess engine binary (placeholder for future development).

```bash
./build/bin/chess_engine
```

### `trt_benchmark`

TensorRT benchmark for neural network evaluation. Tests inference latency and throughput across various batch sizes.

```bash
# Use default engine path (catgpt.trt in project root)
./build/bin/trt_benchmark

# Or specify custom engine path
./build/bin/trt_benchmark /path/to/engine.trt
```

**Example output:**

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

The TensorRT engine expects:

- **Input**: `int32` tensor of shape `(batch, 64)` - tokenized chess position
- **Output**: `float32` tensor of shape `(batch,)` - win probability [0, 1]

## Project Structure

```
cpp/
├── CMakeLists.txt           # Build configuration with CUDA/TensorRT
├── src/
│   ├── main.cpp            # Chess engine entry point
│   └── trt_benchmark.cpp   # TensorRT benchmark binary
├── build.sh                # Convenience build script
├── .clang-format           # Code formatting rules
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## C++23 Features Used

- `std::print` and `std::println` for formatted output
- `std::format` for string formatting
- `std::span` for non-owning array views
- `std::ranges` algorithms
- Designated initializers

## Configuration

TensorRT and CUDA paths can be customized via CMake cache variables:

```bash
cmake .. \
    -DTENSORRT_ROOT=/path/to/TensorRT \
    -DCUDA_ROOT=/path/to/cuda
```
