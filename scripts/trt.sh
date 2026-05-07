# Build TensorRT engines from ONNX model: one engine per bucket size, then
# pack into a single .network file consumed by the C++ multi-engine loader.
#
# Bucket sizes: 1, 2, 3, 4, 6, 8, 12, 18, 26, 36, 56, 112
#   - Each bucket gets its own engine with a single optimization profile pinned
#     at min == opt == max so kernel selection is tuned to exactly that batch
#     size. Building separate engines (instead of one engine with 12 profiles)
#     lets TRT choose per-shape-optimal kernels — no cross-profile compromise.
#   - At runtime, BatchEvaluator drains the queue to the largest bucket <= queue
#     size and dispatches on the matching per-bucket engine + context.
#   - The bucket list MUST stay in sync with kBucketSizes in
#     cpp/src/selfplay/batch_evaluator.hpp; the loader validates and throws if
#     any expected bucket is missing from the .network file.
#
# ONNX outputs (auto-detected by TRT):
#   wdl_value    — WDL-derived Q value, scalar (batch,)
#   bestq_probs  — BestQ HL-Gauss distribution (batch, 81)
#   wdl_probs    — Win/Draw/Loss probabilities (batch, 3)
#   policy_logit — Move distribution logits (batch, 4672)

set -euo pipefail

TRT_ROOT=/home/shadeform/TensorRT-10.13.3.9
TRTEXEC=${TRT_ROOT}/bin/trtexec
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${TRT_ROOT}/lib

ONNX=${ONNX:-main.onnx}
NETWORK_OUT=${NETWORK_OUT:-main.network}

BUCKETS=(1 2 3 4 6 8 12 18 26 36 56 112)

TRT_FILES=()
for b in "${BUCKETS[@]}"; do
  out="main.b${b}.trt"
  echo
  echo "=== Building bucket ${b} -> ${out} ==="
  "${TRTEXEC}" \
    --onnx="${ONNX}" \
    --saveEngine="${out}" \
    --fp16 \
    --builderOptimizationLevel=5 \
    --minShapes=in_0:${b}x64 \
    --optShapes=in_0:${b}x64 \
    --maxShapes=in_0:${b}x64
  TRT_FILES+=("${out}")
done

echo
echo "=== Packing ${#TRT_FILES[@]} engines -> ${NETWORK_OUT} ==="
uv run scripts/pack_network.py -o "${NETWORK_OUT}" "${TRT_FILES[@]}"

echo
echo "=== Cleaning up per-bucket .trt files ==="
rm -f "${TRT_FILES[@]}"

echo "Done: ${NETWORK_OUT}"
