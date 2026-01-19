LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shadeform/TensorRT-10.13.3.9/lib \
/home/shadeform/TensorRT-10.13.3.9/bin/trtexec \
  --onnx=catgpt_large.onnx \
  --saveEngine=catgpt_large.trt \
  --fp16
