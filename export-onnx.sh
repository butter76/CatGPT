#!/bin/bash
# Export JAX transformer to ONNX with dynamic batch support
#
# Strategy: Export with a "magic" batch size (477) that won't collide with
# real tensor dimensions, then post-process the ONNX graph to replace all
# occurrences with dynamic dimension (-1).
#
# Usage: ./export-onnx.sh

set -e

uv run python scripts/export_onnx.py \
    --output-path catgpt.onnx \
    --hidden-size 1024 \
    --num-layers 15 \
    --num-heads 32 \
    --seq-length 64 \
    --ff-dim 1536 \
    --dynamic-batch \
    --opset 20 \
    --validate
