cmake_minimum_required(VERSION 3.22)

# Set compilers to GCC 14 (must be before project())
set(CMAKE_C_COMPILER gcc-14)
set(CMAKE_CXX_COMPILER g++-14)

project(ChessEngine VERSION 0.1.0 LANGUAGES C CXX)

# Set C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for modern C++ and optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# === vcpkg / libcoro ===
# vcpkg doesn't provide a CMake config for libcoro, so we set paths manually
set(VCPKG_ROOT "$ENV{HOME}/vcpkg" CACHE PATH "vcpkg installation root")
set(VCPKG_INSTALLED_DIR "${VCPKG_ROOT}/installed/x64-linux")

set(LIBCORO_INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/include")
set(LIBCORO_LIB_DIR "${VCPKG_INSTALLED_DIR}/lib")
find_library(LIBCORO_LIB coro PATHS ${LIBCORO_LIB_DIR} REQUIRED)

message(STATUS "vcpkg root: ${VCPKG_ROOT}")
message(STATUS "libcoro include: ${LIBCORO_INCLUDE_DIR}")
message(STATUS "libcoro library: ${LIBCORO_LIB}")

# === Chess Library (header-only) ===
set(CHESS_LIBRARY_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/chess-library/include")
message(STATUS "Chess library include: ${CHESS_LIBRARY_INCLUDE_DIR}")

# === CUDA (runtime only, no compilation) ===
set(CUDA_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA installation root")
set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
set(CUDA_LIB_DIR "${CUDA_ROOT}/lib64")

find_library(CUDART_LIB cudart PATHS ${CUDA_LIB_DIR} REQUIRED)
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDA runtime: ${CUDART_LIB}")

# === TensorRT ===
set(TENSORRT_ROOT "/home/shadeform/TensorRT-10.13.3.9" CACHE PATH "TensorRT installation root")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")

find_library(NVINFER_LIB nvinfer PATHS ${TENSORRT_LIB_DIR} REQUIRED)

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${NVINFER_LIB}")

# === Hello World executable (no dependencies) ===
add_executable(chess_engine
    src/main.cpp
)

# === Chess Library Tests ===
add_executable(chess_lib_tests
    tests/chess_lib_tests.cpp
)

target_include_directories(chess_lib_tests PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
)

# === TensorRT Benchmark executable ===
add_executable(trt_benchmark
    src/trt_benchmark.cpp
)

target_include_directories(trt_benchmark PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
)

target_link_libraries(trt_benchmark PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
)

# GCC 14 libstdc++ path (required for C++23 features)
set(GCC14_LIB_DIR "/usr/local/lib64")

# Set RPATH for finding TensorRT/CUDA/libstdc++ at runtime
set_target_properties(trt_benchmark PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
)

# === CatGPT Fractional MCTS UCI Engine (coroutines + TensorRT) ===
add_executable(catgpt_fractional_mcts
    src/fractional_mcts_uci_main.cpp
)

target_include_directories(catgpt_fractional_mcts PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
)

target_link_libraries(catgpt_fractional_mcts PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
    ${LIBCORO_LIB}
    pthread
)

# libcoro requires -fcoroutines on GCC
target_compile_options(catgpt_fractional_mcts PRIVATE -fcoroutines)

set_target_properties(catgpt_fractional_mcts PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
)

# === CatGPT MCTS UCI Engine (coroutines + TensorRT) ===
add_executable(catgpt_mcts
    src/mcts_uci_main.cpp
)

target_include_directories(catgpt_mcts PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
)

target_link_libraries(catgpt_mcts PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
    ${LIBCORO_LIB}
    pthread
)

# libcoro requires -fcoroutines on GCC
target_compile_options(catgpt_mcts PRIVATE -fcoroutines)

set_target_properties(catgpt_mcts PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
)

# === CatGPT FEN Analyzer (with TensorRT) ===
add_executable(catgpt_analyze
    src/analyze_fen_main.cpp
)

target_include_directories(catgpt_analyze PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
)

target_link_libraries(catgpt_analyze PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
    pthread
)

set_target_properties(catgpt_analyze PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
)

# === CatGPT Search Binary (standalone Fractional MCTS for web UI) ===
# Uses CoroutineSearch (same algorithm as selfplay) with coroutine infrastructure
add_executable(catgpt_search
    src/catgpt_search_main.cpp
)

target_include_directories(catgpt_search PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
)

target_link_libraries(catgpt_search PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
    ${LIBCORO_LIB}
    pthread
)

# libcoro requires -fcoroutines on GCC
target_compile_options(catgpt_search PRIVATE -fcoroutines)

set_target_properties(catgpt_search PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
)

# === Fathom (Syzygy tablebase probing, C library) ===
set(FATHOM_SRC_DIR "${CMAKE_SOURCE_DIR}/external/fathom/src")
add_library(fathom STATIC
    ${FATHOM_SRC_DIR}/tbprobe.c
)
target_include_directories(fathom PUBLIC ${FATHOM_SRC_DIR})
# Fathom is C — suppress C++ warnings, compile as C with optimizations
set_target_properties(fathom PROPERTIES
    C_STANDARD 11
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(fathom PRIVATE -O3 -w)

# === CatGPT Batched Self-Play (coroutines + TensorRT) ===
add_executable(catgpt_selfplay
    src/selfplay_main.cpp
)

target_include_directories(catgpt_selfplay PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
    ${FATHOM_SRC_DIR}
)

target_link_libraries(catgpt_selfplay PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
    ${LIBCORO_LIB}
    fathom
    pthread
)

# libcoro requires -fcoroutines on GCC
target_compile_options(catgpt_selfplay PRIVATE -fcoroutines)

set_target_properties(catgpt_selfplay PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
)

# === CatGPT Batched Puzzle Evaluation (coroutines + TensorRT) ===
add_executable(catgpt_puzzle_eval
    src/puzzle_eval_main.cpp
)

target_include_directories(catgpt_puzzle_eval PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
)

target_link_libraries(catgpt_puzzle_eval PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
    ${LIBCORO_LIB}
    pthread
)

target_compile_options(catgpt_puzzle_eval PRIVATE -fcoroutines)

set_target_properties(catgpt_puzzle_eval PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR};${LIBCORO_LIB_DIR}"
)

# === Fractional MCTS v2 TT/Arena PoC microbenchmark ===
# Standalone — no TRT, no libcoro, no chess-library deps. Just C++23 + pthread.
add_executable(tt_arena_bench
    src/engine/fractional_mcts/v2/tt_arena_bench.cpp
)
target_compile_options(tt_arena_bench PRIVATE -O3 -march=native)
target_link_libraries(tt_arena_bench PRIVATE pthread)
set_target_properties(tt_arena_bench PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR}"
)

# === Lock-free TT/Arena multi-threaded stress + bench ===
# Standalone — no TRT, no libcoro, no chess-library deps.
add_executable(tt_arena_concurrent_bench
    src/engine/fractional_mcts/v2/tt_arena_concurrent_bench.cpp
)
target_compile_options(tt_arena_concurrent_bench PRIVATE -O3 -march=native)
target_link_libraries(tt_arena_concurrent_bench PRIVATE pthread)
set_target_properties(tt_arena_concurrent_bench PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR}"
)

# === LKS search lifecycle + GPU test ===
# Real TensorRT-backed BatchEvaluator per worker. Needs CUDA/TRT/libcoro.
add_executable(lks_search_test
    src/engine/lks/lks_search_test.cpp
)
target_include_directories(lks_search_test PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
)
target_compile_options(lks_search_test PRIVATE -O2 -g -fcoroutines)
target_link_libraries(lks_search_test PRIVATE
    ${LIBCORO_LIB}
    ${NVINFER_LIB}
    ${CUDART_LIB}
    pthread
)
set_target_properties(lks_search_test PROPERTIES
    BUILD_RPATH   "${GCC14_LIB_DIR};${LIBCORO_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${LIBCORO_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
)

# === LKS throughput sweep (real TRT) ===
add_executable(lks_throughput_bench
    src/engine/lks/lks_throughput_bench.cpp
)
target_include_directories(lks_throughput_bench PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
)
target_compile_options(lks_throughput_bench PRIVATE -O3 -march=native -fcoroutines)
target_link_libraries(lks_throughput_bench PRIVATE
    ${LIBCORO_LIB}
    ${NVINFER_LIB}
    ${CUDART_LIB}
    pthread
)
set_target_properties(lks_throughput_bench PROPERTIES
    BUILD_RPATH   "${GCC14_LIB_DIR};${LIBCORO_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${LIBCORO_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
)

# === CatGPT LKS UCI Engine (Lazy-SMP, log-scale ID, TRT-batched) ===
add_executable(lks_uci
    src/lks_uci_main.cpp
)
target_include_directories(lks_uci PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
)
target_link_libraries(lks_uci PRIVATE
    ${NVINFER_LIB}
    ${CUDART_LIB}
    ${LIBCORO_LIB}
    pthread
)
target_compile_options(lks_uci PRIVATE -fcoroutines)
set_target_properties(lks_uci PROPERTIES
    BUILD_RPATH   "${GCC14_LIB_DIR};${LIBCORO_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR};${LIBCORO_LIB_DIR};${TENSORRT_LIB_DIR};${CUDA_LIB_DIR}"
)

# Enable colored output
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(chess_engine PRIVATE -fdiagnostics-color=always)
    target_compile_options(trt_benchmark PRIVATE -fdiagnostics-color=always)
    target_compile_options(chess_lib_tests PRIVATE -fdiagnostics-color=always)
    target_compile_options(catgpt_fractional_mcts PRIVATE -fdiagnostics-color=always)
    target_compile_options(catgpt_mcts PRIVATE -fdiagnostics-color=always)
    target_compile_options(catgpt_analyze PRIVATE -fdiagnostics-color=always)
    target_compile_options(catgpt_selfplay PRIVATE -fdiagnostics-color=always)
    target_compile_options(catgpt_search PRIVATE -fdiagnostics-color=always)
    target_compile_options(catgpt_puzzle_eval PRIVATE -fdiagnostics-color=always)
    target_compile_options(tt_arena_bench PRIVATE -fdiagnostics-color=always)
    target_compile_options(tt_arena_concurrent_bench PRIVATE -fdiagnostics-color=always)
    target_compile_options(lks_search_test PRIVATE -fdiagnostics-color=always)
    target_compile_options(lks_uci PRIVATE -fdiagnostics-color=always)
endif()
