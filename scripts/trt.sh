# Build TensorRT engine from ONNX model.
#
# ONNX outputs (auto-detected by TRT):
#   wdl_value    — WDL-derived Q value, scalar (batch,)
#   bestq_probs  — BestQ HL-Gauss distribution (batch, 81)
#   wdl_probs    — Win/Draw/Loss probabilities (batch, 3)
#   policy_logit — Move distribution logits (batch, 4672)

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shadeform/TensorRT-10.13.3.9/lib \
/home/shadeform/TensorRT-10.13.3.9/bin/trtexec \
  --onnx=test.onnx \
  --saveEngine=test.trt \
  --minShapes=in_0:1x64 \
  --optShapes=in_0:112x64 \
  --maxShapes=in_0:112x64 \
  --fp16
