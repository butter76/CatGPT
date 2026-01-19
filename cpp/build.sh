#!/bin/bash
set -e

# Build script for CatGPT Chess Engine

BUILD_TYPE="${1:-Release}"
BUILD_DIR="build"

echo "Building CatGPT Chess Engine ($BUILD_TYPE)..."

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Configure and build
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build . -j$(nproc)

echo ""
echo "Build complete! Binary location: $BUILD_DIR/bin/chess_engine"
echo ""
echo "To run:"
echo "  ./$BUILD_DIR/bin/chess_engine"
