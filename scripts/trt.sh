LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shadeform/TensorRT-10.13.3.9/lib \
/home/shadeform/TensorRT-10.13.3.9/bin/trtexec \
  --onnx=catgpt.onnx \
  --saveEngine=catgpt.trt \
  --minShapes=in_0:1x64 \
  --optShapes=in_0:128x64 \
  --maxShapes=in_0:256x64 \
  --fp16
