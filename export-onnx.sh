uv run python scripts/export_onnx.py \
    --output-path catgpt_large.onnx \
    --hidden-size 768 \
    --num-layers 32 \
    --num-heads 48 \
    --seq-length 64 \
    --batch-size 64 \
    --ff-dim 1536
