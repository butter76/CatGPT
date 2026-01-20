#!/bin/bash
# Export JAX transformer to ONNX with dynamic batch support
#
# Strategy: Export with a "magic" batch size (477) that won't collide with
# real tensor dimensions, then post-process the ONNX graph to replace all
# occurrences with dynamic dimension (-1).
#
# Usage: ./export-onnx.sh [checkpoint_path]

set -e

CHECKPOINT_PATH="${1:-checkpoints_jax/sample}"

uv run python scripts/export_onnx.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output-path catgpt.onnx \
    --dynamic-batch \
    --opset 20 \
    --validate
