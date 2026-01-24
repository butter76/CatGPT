cmake_minimum_required(VERSION 3.22)
project(ChessEngine VERSION 0.1.0 LANGUAGES CXX)

# Set C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compiler to g++-14
set(CMAKE_CXX_COMPILER g++-14)

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

# === Dummy Search POC executable ===
add_executable(dummy_search
    src/dummy_search/main.cpp
)

target_include_directories(dummy_search PRIVATE
    ${CHESS_LIBRARY_INCLUDE_DIR}
    ${LIBCORO_INCLUDE_DIR}
)

target_link_libraries(dummy_search PRIVATE
    ${LIBCORO_LIB}
    pthread  # Required for libcoro thread_pool
)

set_target_properties(dummy_search PROPERTIES
    BUILD_RPATH "${GCC14_LIB_DIR}"
    INSTALL_RPATH "${GCC14_LIB_DIR}"
)

# Enable colored output
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(chess_engine PRIVATE -fdiagnostics-color=always)
    target_compile_options(trt_benchmark PRIVATE -fdiagnostics-color=always)
    target_compile_options(chess_lib_tests PRIVATE -fdiagnostics-color=always)
    target_compile_options(dummy_search PRIVATE -fdiagnostics-color=always)
endif()
