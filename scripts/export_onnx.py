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


# Magic batch size - chosen to be unlikely to collide with real tensor dimensions
# (hidden_size, ff_dim, num_heads, seq_length, vocab_size, etc.)
MAGIC_BATCH_SIZE = 477


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

    # 5. Check value_info for intermediate tensors
    for vi in graph.value_info:
        if vi.type.HasField("tensor_type"):
            shape = vi.type.tensor_type.shape
            if shape.dim and len(shape.dim) > 0:
                dim0 = shape.dim[0]
                if dim0.HasField("dim_value") and dim0.dim_value == MAGIC_BATCH_SIZE:
                    dim0.ClearField("dim_value")
                    dim0.dim_param = "batch"
                    # Don't count these as they're just metadata

    print(f"  Total modifications: {modifications}")

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

    # Model architecture
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=28)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--activation", type=str, default="gelu")

    # Output heads
    parser.add_argument("--self-head", action="store_true")
    parser.add_argument("--no-self-head", dest="self_head", action="store_false")
    parser.set_defaults(self_head=False)

    parser.add_argument("--value-head", action="store_true")
    parser.add_argument("--no-value-head", dest="value_head", action="store_false")
    parser.set_defaults(value_head=True)

    parser.add_argument("--policy-head", action="store_true")
    parser.add_argument("--no-policy-head", dest="policy_head", action="store_false")
    parser.set_defaults(policy_head=False)

    parser.add_argument("--value-num-bins", type=int, default=81)

    parser.add_argument(
        "--output-key",
        type=str,
        default="value",
        help="Model output key to export.",
    )

    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--validate", action="store_true", help="Validate ONNX output with ORT.")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)

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

    from catgpt.jax.configs import JaxModelConfig, JaxOutputHeadConfig
    from catgpt.jax.models.transformer import BidirectionalTransformer

    print("=" * 60)
    print("JAX → ONNX Export (via jax2onnx)")
    print("=" * 60)

    # Build model config
    output_heads = JaxOutputHeadConfig(
        self_head=args.self_head,
        value_head=args.value_head,
        policy_head=args.policy_head,
        soft_policy_head=False,
        value_num_bins=args.value_num_bins,
    )
    model_config = JaxModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        activation=args.activation,
        output_heads=output_heads,
    )

    print(f"\nModel config:")
    print(f"  hidden_size: {model_config.hidden_size}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  num_heads: {model_config.num_heads}")
    print(f"  ff_dim: {model_config.ff_dim or 4 * model_config.hidden_size}")
    print(f"  seq_length: {model_config.seq_length}")
    print(f"  output_key: {args.output_key}")
    print(f"  dynamic_batch: {args.dynamic_batch}")

    # Create and initialize model
    rng = jax.random.key(args.seed)
    model, params = BidirectionalTransformer.create_and_init(
        model_config, rng, compute_dtype=jnp.float32
    )

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\nModel parameters: {param_count:,}")

    # Define the forward function
    output_key = args.output_key

    def forward(x: jax.Array) -> jax.Array:
        outputs = model.apply(params, x, train=False, compute_dtype=jnp.float32)
        if output_key not in outputs:
            available = ", ".join(sorted(outputs.keys()))
            raise KeyError(f"Output '{output_key}' not found. Available: {available}")
        return outputs[output_key]

    # Choose batch size for export
    if args.dynamic_batch:
        export_batch_size = MAGIC_BATCH_SIZE
        print(f"\n[Export] Using magic batch size {MAGIC_BATCH_SIZE} for dynamic export")
    else:
        export_batch_size = args.batch_size
        print(f"\n[Export] Using fixed batch size {export_batch_size}")

    # Input spec
    input_specs = [jax.ShapeDtypeStruct((export_batch_size, args.seq_length), jnp.int32)]

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

    # Validate with ONNX Runtime
    if args.validate:
        print("\n[Validation] Testing ONNX model with ONNX Runtime...")
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output_path))
        input_name = sess.get_inputs()[0].name

        # Test multiple batch sizes if dynamic
        test_batches = [1, 4, args.batch_size, 128] if args.dynamic_batch else [args.batch_size]

        for batch_size in test_batches:
            test_input = np.random.randint(
                0, args.vocab_size, size=(batch_size, args.seq_length), dtype=np.int32
            )
            jax_input = jnp.array(test_input)

            # JAX reference output
            jax_output = forward(jax_input)
            jax_output_np = np.array(jax_output)

            # ONNX Runtime output
            ort_output = sess.run(None, {input_name: test_input})[0]

            # Compare
            is_close = np.allclose(jax_output_np, ort_output, rtol=args.rtol, atol=args.atol)
            max_diff = np.max(np.abs(jax_output_np - ort_output))

            status = "✓" if is_close else "✗"
            print(f"  Batch {batch_size:3d}: {status} (max diff: {max_diff:.2e})")

            if not is_close:
                raise RuntimeError(f"ONNX validation failed for batch size {batch_size}")

        print("  All validations PASSED")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
