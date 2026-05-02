# Build TensorRT engine from ONNX model with one optimization profile per bucket.
#
# Bucket sizes: 1, 2, 3, 4, 6, 8, 12, 18, 26, 36, 56, 112
#   - Each bucket gets a profile pinned at min == opt == max so kernel selection
#     is tuned to exactly that batch size (no shape-range slack).
#   - At runtime, BatchEvaluator drains the queue to the largest bucket <= queue
#     size and dispatches on the matching profile-bound context.
#   - This array MUST stay in sync with kBucketSizes in
#     cpp/src/selfplay/batch_evaluator.hpp; the loader validates and throws if
#     any expected bucket is missing from the engine.
#
# ONNX outputs (auto-detected by TRT):
#   wdl_value    — WDL-derived Q value, scalar (batch,)
#   bestq_probs  — BestQ HL-Gauss distribution (batch, 81)
#   wdl_probs    — Win/Draw/Loss probabilities (batch, 3)
#   policy_logit — Move distribution logits (batch, 4672)

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shadeform/TensorRT-10.13.3.9/lib \
/home/shadeform/TensorRT-10.13.3.9/bin/trtexec \
  --onnx=main.onnx \
  --saveEngine=main.trt \
  --fp16 \
  --builderOptimizationLevel=5 \
  --timingCacheFile=main.trt.cache \
  --profile=0  --minShapes=in_0:1x64   --optShapes=in_0:1x64   --maxShapes=in_0:1x64 \
  --profile=1  --minShapes=in_0:2x64   --optShapes=in_0:2x64   --maxShapes=in_0:2x64 \
  --profile=2  --minShapes=in_0:3x64   --optShapes=in_0:3x64   --maxShapes=in_0:3x64 \
  --profile=3  --minShapes=in_0:4x64   --optShapes=in_0:4x64   --maxShapes=in_0:4x64 \
  --profile=4  --minShapes=in_0:6x64   --optShapes=in_0:6x64   --maxShapes=in_0:6x64 \
  --profile=5  --minShapes=in_0:8x64   --optShapes=in_0:8x64   --maxShapes=in_0:8x64 \
  --profile=6  --minShapes=in_0:12x64  --optShapes=in_0:12x64  --maxShapes=in_0:12x64 \
  --profile=7  --minShapes=in_0:18x64  --optShapes=in_0:18x64  --maxShapes=in_0:18x64 \
  --profile=8  --minShapes=in_0:26x64  --optShapes=in_0:26x64  --maxShapes=in_0:26x64 \
  --profile=9  --minShapes=in_0:36x64  --optShapes=in_0:36x64  --maxShapes=in_0:36x64 \
  --profile=10 --minShapes=in_0:56x64  --optShapes=in_0:56x64  --maxShapes=in_0:56x64 \
  --profile=11 --minShapes=in_0:112x64 --optShapes=in_0:112x64 --maxShapes=in_0:112x64
