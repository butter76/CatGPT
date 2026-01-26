#!/bin/bash
# Export JAX transformer to ONNX with dynamic batch support
#
# Strategy: Export with a "magic" batch size (477) that won't collide with
# real tensor dimensions, then post-process the ONNX graph to replace all
# occurrences with dynamic dimension (-1).
#
# Usage: ./export-onnx.sh [checkpoint_path]

set -e

uv run python scripts/export_onnx.py \
    --checkpoint ./checkpoints_jax/best \
    --output-path sample.onnx \
    --output-keys value hard_policy_logit \
    --batch-size 1 \
    --opset 20 \
    --validate
