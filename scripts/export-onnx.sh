#!/bin/bash
# Export JAX transformer to ONNX with dynamic batch support
#
# Strategy: Export with a "magic" batch size (477) that won't collide with
# real tensor dimensions, then post-process the ONNX graph to replace all
# occurrences with dynamic dimension (-1).
#
# Inputs (in order):
#   tokens                          — int32 (batch, 64) chess position tokens
#   legal_indices                   — int32 (batch, 218) flat policy indices
#                                     of legal moves (padded with 0)
#
# Outputs (in order):
#   wdl_logit                       — raw WDL logits [W, D, L] (batch, 3)
#   bestq_probs                     — BestQ HL-Gauss distribution (batch, 81)
#   optimistic_policy_legal_logit   — Optimistic policy logits gathered at
#                                     legal_indices (batch, 218)
#
# Usage: ./export-onnx.sh [checkpoint_path]

set -e

uv run python scripts/export_onnx.py \
    --checkpoint ./checkpoints_jax/S2/shard_15 \
    --output-path main.onnx \
    --output-keys wdl_logit bestq_probs optimistic_policy_legal_logit \
    --dynamic-batch \
    --opset 20 \
    --validate
