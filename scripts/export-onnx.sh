#!/bin/bash
# Export JAX transformer to ONNX with dynamic batch support
#
# Strategy: Export with a "magic" batch size (477) that won't collide with
# real tensor dimensions, then post-process the ONNX graph to replace all
# occurrences with dynamic dimension (-1).
#
# Outputs (in order):
#   wdl_value    — WDL-derived Q value, scalar (batch,)
#   bestq_probs  — BestQ HL-Gauss distribution (batch, 81)
#   wdl_probs    — Win/Draw/Loss probabilities (batch, 3)
#   policy_logit — Move distribution logits (batch, 4672)
#
# Usage: ./export-onnx.sh [checkpoint_path]

set -e

uv run python scripts/export_onnx.py \
    --output-path test.onnx \
    --output-keys wdl_value bestq_probs wdl_probs policy_logit optimistic_policy_logit \
    --dynamic-batch \
    --opset 20 \
    --validate
