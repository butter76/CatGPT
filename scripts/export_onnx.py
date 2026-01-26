#!/usr/bin/env python3
"""Export JAX transformer models to ONNX using jax2onnx with dynamic batch support.

Strategy: Export with a "magic" batch size (e.g., 477) that won't collide with real
tensor dimensions, then post-process the ONNX graph to replace all occurrences of
the magic number with a dynamic dimension.

Pipeline: JAX → jax2onnx → ONNX → post-process for dynamic batch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Use highest matmul precision to match ONNX Runtime's float32 behavior
# (JAX defaults to TF32 on Ampere+ GPUs which has lower precision)
jax.config.update("jax_default_matmul_precision", "highest")

# Magic batch size - chosen to be unlikely to collide with real tensor dimensions
# (hidden_size, ff_dim, num_heads, seq_length, vocab_size, etc.)
MAGIC_BATCH_SIZE = 477


def dump_onnx_ir(onnx_path: str | Path, output_path: str | Path | None = None) -> str:
    """Dump ONNX model IR to a human-readable text file.

    Args:
        onnx_path: Path to the ONNX model.
        output_path: Path for the text file. If None, uses onnx_path with .txt extension.

    Returns:
        Path to the output text file.
    """
    import onnx
    from onnx import numpy_helper

    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path.with_suffix(".txt")

    model = onnx.load(str(onnx_path))
    graph = model.graph

    lines = []
    lines.append(f"# ONNX Model IR: {onnx_path.name}")
    lines.append(f"# IR Version: {model.ir_version}")
    opsets = [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import]
    lines.append(f"# Opset: {opsets}")
    lines.append("")

    # Inputs
    lines.append("=" * 80)
    lines.append("INPUTS")
    lines.append("=" * 80)
    for inp in graph.input:
        shape = _format_shape(inp.type.tensor_type.shape)
        dtype = _format_dtype(inp.type.tensor_type.elem_type)
        lines.append(f"  {inp.name}: {dtype}{shape}")
    lines.append("")

    # Outputs
    lines.append("=" * 80)
    lines.append("OUTPUTS")
    lines.append("=" * 80)
    for out in graph.output:
        shape = _format_shape(out.type.tensor_type.shape)
        dtype = _format_dtype(out.type.tensor_type.elem_type)
        lines.append(f"  {out.name}: {dtype}{shape}")
    lines.append("")

    # Initializers (weights) - just shapes
    lines.append("=" * 80)
    lines.append(f"INITIALIZERS ({len(graph.initializer)} tensors)")
    lines.append("=" * 80)
    for init in graph.initializer:
        dtype = _format_dtype(init.data_type)
        shape = list(init.dims)
        size = np.prod(shape) if shape else 1
        lines.append(f"  {init.name}: {dtype}{shape} ({size:,} params)")
    lines.append("")

    # Nodes
    lines.append("=" * 80)
    lines.append(f"NODES ({len(graph.node)} ops)")
    lines.append("=" * 80)
    for i, node in enumerate(graph.node):
        inputs_str = ", ".join(node.input) if node.input else "(none)"
        outputs_str = ", ".join(node.output) if node.output else "(none)"

        lines.append(f"[{i:4d}] {node.op_type}")
        lines.append(f"       name: {node.name or '(unnamed)'}")
        lines.append(f"       inputs: [{inputs_str}]")
        lines.append(f"       outputs: [{outputs_str}]")

        # Attributes
        if node.attribute:
            for attr in node.attribute:
                attr_val = _format_attribute(attr)
                lines.append(f"       {attr.name}: {attr_val}")
        lines.append("")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return str(output_path)


def _format_shape(shape) -> str:
    """Format ONNX TensorShapeProto to string."""
    if not shape or not shape.dim:
        return "[]"
    dims = []
    for d in shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"


def _format_dtype(elem_type: int) -> str:
    """Format ONNX data type to string."""
    import onnx

    dtype_map = {
        onnx.TensorProto.FLOAT: "f32",
        onnx.TensorProto.FLOAT16: "f16",
        onnx.TensorProto.BFLOAT16: "bf16",
        onnx.TensorProto.DOUBLE: "f64",
        onnx.TensorProto.INT32: "i32",
        onnx.TensorProto.INT64: "i64",
        onnx.TensorProto.INT8: "i8",
        onnx.TensorProto.UINT8: "u8",
        onnx.TensorProto.BOOL: "bool",
    }
    return dtype_map.get(elem_type, f"dtype({elem_type})")


def _format_attribute(attr) -> str:
    """Format ONNX AttributeProto to string."""
    import onnx

    if attr.type == onnx.AttributeProto.FLOAT:
        return f"{attr.f}"
    elif attr.type == onnx.AttributeProto.INT:
        return f"{attr.i}"
    elif attr.type == onnx.AttributeProto.STRING:
        return f'"{attr.s.decode()}"'
    elif attr.type == onnx.AttributeProto.FLOATS:
        return f"[{', '.join(str(x) for x in attr.floats)}]"
    elif attr.type == onnx.AttributeProto.INTS:
        return f"[{', '.join(str(x) for x in attr.ints)}]"
    elif attr.type == onnx.AttributeProto.TENSOR:
        from onnx import numpy_helper
        arr = numpy_helper.to_array(attr.t)
        if arr.size <= 8:
            return f"tensor({arr.tolist()})"
        return f"tensor(shape={list(arr.shape)}, dtype={arr.dtype})"
    else:
        return f"<{attr.type}>"


def make_onnx_dynamic_batch(onnx_path: str | Path, output_path: str | Path | None = None) -> str:
    """Post-process ONNX model to replace magic batch size with dynamic dimension.

    Scans all nodes in the ONNX graph and replaces occurrences of MAGIC_BATCH_SIZE
    with -1 (dynamic) in reshape target shapes, and marks input/output batch dims
    as symbolic.

    Args:
        onnx_path: Path to the ONNX model exported with MAGIC_BATCH_SIZE.
        output_path: Path for the modified model. If None, overwrites the original.

    Returns:
        Path to the modified model.
    """
    import onnx
    from onnx import numpy_helper

    onnx_path = Path(onnx_path)
    output_path = Path(output_path) if output_path else onnx_path

    print(f"\n[Post-process] Making batch dimension dynamic...")
    print(f"  Magic batch size: {MAGIC_BATCH_SIZE}")

    model = onnx.load(str(onnx_path))
    graph = model.graph
    modifications = 0

    # 1. Make input dimensions dynamic
    for inp in graph.input:
        shape = inp.type.tensor_type.shape
        if shape.dim and len(shape.dim) > 0:
            dim0 = shape.dim[0]
            if dim0.HasField("dim_value") and dim0.dim_value == MAGIC_BATCH_SIZE:
                dim0.ClearField("dim_value")
                dim0.dim_param = "batch"
                modifications += 1
                print(f"  Input '{inp.name}': dim[0] {MAGIC_BATCH_SIZE} → 'batch'")

    # 2. Make output dimensions dynamic
    for out in graph.output:
        shape = out.type.tensor_type.shape
        if shape.dim and len(shape.dim) > 0:
            dim0 = shape.dim[0]
            if dim0.HasField("dim_value") and dim0.dim_value == MAGIC_BATCH_SIZE:
                dim0.ClearField("dim_value")
                dim0.dim_param = "batch"
                modifications += 1
                print(f"  Output '{out.name}': dim[0] {MAGIC_BATCH_SIZE} → 'batch'")

    # 3. Fix Reshape nodes - replace magic batch size in shape tensors
    for node in graph.node:
        if node.op_type == "Reshape":
            # The shape input is usually the second input
            if len(node.input) >= 2:
                shape_input_name = node.input[1]

                # Find the initializer for this shape
                for init in graph.initializer:
                    if init.name == shape_input_name:
                        shape_array = numpy_helper.to_array(init)
                        if MAGIC_BATCH_SIZE in shape_array:
                            # Replace magic batch size with -1 (dynamic)
                            new_shape = np.where(
                                shape_array == MAGIC_BATCH_SIZE, -1, shape_array
                            ).astype(np.int64)

                            # Update the initializer
                            new_init = numpy_helper.from_array(new_shape, init.name)
                            init.CopyFrom(new_init)
                            modifications += 1
                            print(
                                f"  Reshape '{node.name}': {shape_array.tolist()} → {new_shape.tolist()}"
                            )
                        break

    # 4. Fix Constant nodes that might contain the magic batch size
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t is not None:
                    tensor = attr.t
                    if tensor.data_type == onnx.TensorProto.INT64:
                        arr = numpy_helper.to_array(tensor)
                        if MAGIC_BATCH_SIZE in arr:
                            new_arr = np.where(arr == MAGIC_BATCH_SIZE, -1, arr).astype(
                                np.int64
                            )
                            new_tensor = numpy_helper.from_array(new_arr)
                            tensor.CopyFrom(new_tensor)
                            modifications += 1
                            print(
                                f"  Constant '{node.name}': {arr.tolist()} → {new_arr.tolist()}"
                            )

    # 5. Clear value_info - we'll re-infer shapes with symbolic batch dimension
    # This removes stale shape metadata that still contains the magic batch size
    num_value_info = len(graph.value_info)
    if num_value_info > 0:
        graph.ClearField("value_info")
        print(f"  Cleared {num_value_info} value_info entries (will be re-inferred)")

    print(f"  Total modifications: {modifications}")

    # 6. Run shape inference to propagate symbolic batch dimension through the graph
    # This regenerates value_info with the correct dynamic shapes
    print("  Running shape inference to propagate dynamic batch...")
    model = onnx.shape_inference.infer_shapes(model)

    # Save the modified model
    onnx.save(model, str(output_path))
    print(f"  Saved to: {output_path}")

    return str(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a JAX transformer to ONNX with dynamic batch support."
    )
    parser.add_argument("--output-path", type=Path, required=True, help="Destination .onnx path.")
    parser.add_argument("--opset", type=int, default=20, help="ONNX opset version (>=20 for Gelu).")

    # Checkpoint loading (preferred method)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint directory. If provided, loads model config and params from checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/jax_base.yaml"),
        help="Path to config YAML file. Used when --checkpoint is not provided (default: configs/jax_base.yaml).",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use EMA params from checkpoint if available (default: True).",
    )
    parser.add_argument(
        "--no-use-ema",
        dest="use_ema",
        action="store_false",
        help="Use regular training params instead of EMA params.",
    )

    # Output keys to export
    parser.add_argument(
        "--output-keys",
        type=str,
        nargs="+",
        default=["value", "policy_logit"],
        help="Model output keys to export (default: value policy_logit).",
    )

    # Export settings
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        default=True,
        help="Export with dynamic batch dimension (default: True).",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        dest="dynamic_batch",
        action="store_false",
        help="Export with fixed batch dimension.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for fixed export or validation.",
    )

    # Validation settings
    parser.add_argument("--validate", action="store_true", help="Validate ONNX output with ORT.")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument(
        "--logit-atol",
        type=float,
        default=0.05,
        help="Absolute tolerance for logit outputs (default: 0.05, ~5%% prob diff).",
    )

    # Debug/inspection
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Dump ONNX IR to a human-readable text file (.txt).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing files
    data_path = output_path.with_suffix(".onnx.data")
    for p in [output_path, data_path]:
        if p.exists():
            p.unlink()

    # Import heavy dependencies after arg parsing
    from jax2onnx import to_onnx
    from omegaconf import OmegaConf

    from catgpt.jax.configs import jax_config_from_dict
    from catgpt.jax.evaluation.checkpoint import load_checkpoint
    from catgpt.jax.models.transformer import BidirectionalTransformer

    print("=" * 60)
    print("JAX → ONNX Export (via jax2onnx)")
    print("=" * 60)

    if args.checkpoint is not None:
        # Load checkpoint
        print(f"\n[Checkpoint] Loading from: {args.checkpoint}")
        print(f"  Use EMA params: {args.use_ema}")

        ckpt = load_checkpoint(args.checkpoint, evaluation=args.use_ema)
        model = ckpt.model
        params = ckpt.params
        model_config = ckpt.model_config
    else:
        # Create fresh model from config
        print(f"\n[Config] Creating new model from: {args.config}")

        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

        config_dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        experiment_config = jax_config_from_dict(config_dict)  # type: ignore[arg-type]
        model_config = experiment_config.model

        # Override seq_length from tokenizer config if specified
        model_config.seq_length = experiment_config.tokenizer.sequence_length

        # Create and initialize model with random parameters
        rng = jax.random.PRNGKey(experiment_config.seed)
        model, params = BidirectionalTransformer.create_and_init(
            model_config, rng, compute_dtype=jnp.float32
        )
        print("  Created model with random initialization")

    print(f"\nModel config:")
    print(f"  hidden_size: {model_config.hidden_size}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  num_heads: {model_config.num_heads}")
    print(f"  ff_dim: {model_config.ff_dim or 4 * model_config.hidden_size}")
    print(f"  seq_length: {model_config.seq_length}")
    print(f"  output_keys: {args.output_keys}")
    print(f"  dynamic_batch: {args.dynamic_batch}")

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nModel parameters: {param_count:,}")

    # Define the forward function returning multiple outputs
    output_keys = args.output_keys

    def forward(x: jax.Array) -> tuple[jax.Array, ...]:
        outputs = model.apply(params, x, train=False, compute_dtype=jnp.float32)
        result = []
        for key in output_keys:
            if key not in outputs:
                available = ", ".join(sorted(outputs.keys()))
                raise KeyError(f"Output '{key}' not found. Available: {available}")
            result.append(outputs[key])
        return tuple(result)

    # Choose batch size for export
    if args.dynamic_batch:
        export_batch_size = MAGIC_BATCH_SIZE
        print(f"\n[Export] Using magic batch size {MAGIC_BATCH_SIZE} for dynamic export")
    else:
        export_batch_size = args.batch_size
        print(f"\n[Export] Using fixed batch size {export_batch_size}")

    # Input spec
    seq_length = model_config.seq_length
    input_specs = [jax.ShapeDtypeStruct((export_batch_size, seq_length), jnp.int32)]

    # Export to ONNX
    print(f"[Export] Converting to ONNX (opset {args.opset})...")

    model_path = to_onnx(
        forward,
        inputs=input_specs,
        model_name=output_path.stem,
        opset=args.opset,
        return_mode="file",
        output_path=str(output_path),
    )

    print(f"  ONNX model saved to: {model_path}")

    # Post-process for dynamic batch
    if args.dynamic_batch:
        make_onnx_dynamic_batch(output_path)

    # Dump IR for inspection
    if args.dump_ir:
        ir_path = dump_onnx_ir(output_path)
        print(f"\n[IR Dump] Written to: {ir_path}")

    # Validate with ONNX Runtime
    if args.validate:
        print("\n[Validation] Testing ONNX model with ONNX Runtime...")
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output_path))
        input_name = sess.get_inputs()[0].name
        ort_output_names = [o.name for o in sess.get_outputs()]

        print(f"  ONNX outputs: {ort_output_names}")

        # Test multiple batch sizes if dynamic
        test_batches = [1, 4, args.batch_size, 128] if args.dynamic_batch else [args.batch_size]

        # Threshold for "relevant" logits: only compare logits within this range of the max
        LOGIT_RELEVANCE_THRESHOLD = 3.0

        for batch_size in test_batches:
            test_input = np.random.randint(
                0, model_config.vocab_size, size=(batch_size, seq_length), dtype=np.int32
            )
            jax_input = jnp.array(test_input)

            # JAX reference outputs
            jax_outputs = forward(jax_input)

            # ONNX Runtime outputs
            ort_outputs = sess.run(None, {input_name: test_input})

            # Compare each output
            all_close = True
            max_diffs = []
            for jax_out, ort_out, key in zip(jax_outputs, ort_outputs, output_keys):
                jax_out_np = np.array(jax_out)

                if "logit" in key:
                    # For logits: only check relevant logits (within threshold of max)
                    # and verify argmax matches
                    jax_argmax = np.argmax(jax_out_np, axis=-1)
                    ort_argmax = np.argmax(ort_out, axis=-1)
                    argmax_match = np.all(jax_argmax == ort_argmax)

                    # Create mask for relevant logits (within threshold of max per sample)
                    jax_max = np.max(jax_out_np, axis=-1, keepdims=True)
                    relevant_mask = jax_out_np >= (jax_max - LOGIT_RELEVANCE_THRESHOLD)

                    # Compare only relevant logits
                    relevant_diffs = np.abs(jax_out_np - ort_out) * relevant_mask
                    max_diff = np.max(relevant_diffs)

                    is_close = argmax_match and max_diff < args.logit_atol
                    max_diffs.append(f"{key}={max_diff:.2e}")
                    if not argmax_match:
                        max_diffs[-1] += "(ARGMAX MISMATCH!)"
                else:
                    # For other outputs (e.g., value): standard comparison
                    is_close = np.allclose(jax_out_np, ort_out, rtol=args.rtol, atol=args.atol)
                    max_diff = np.max(np.abs(jax_out_np - ort_out))
                    max_diffs.append(f"{key}={max_diff:.2e}")

                if not is_close:
                    all_close = False

            status = "✓" if all_close else "✗"
            diff_str = ", ".join(max_diffs)
            print(f"  Batch {batch_size:3d}: {status} (max diff: {diff_str})")

            if not all_close:
                raise RuntimeError(f"ONNX validation failed for batch size {batch_size}")

        print("  All validations PASSED")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
