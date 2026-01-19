#!/usr/bin/env python3
"""Export JAX transformer models to ONNX for architecture benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax2onnx import allclose, to_onnx

from catgpt.jax.configs import JaxModelConfig, JaxOutputHeadConfig
from catgpt.jax.models.transformer import BidirectionalTransformer


def _parse_output_keys(raw: Iterable[str]) -> list[str]:
    keys = [key.strip() for key in raw if key.strip()]
    if not keys:
        return ["value"]
    return keys


def _select_outputs(outputs: dict[str, jax.Array], keys: list[str]) -> jax.Array | tuple[jax.Array, ...]:
    missing = [key for key in keys if key not in outputs]
    if missing:
        available = ", ".join(sorted(outputs.keys()))
        msg = f"Requested outputs not found: {missing}. Available outputs: {available}"
        raise KeyError(msg)
    if len(keys) == 1:
        return outputs[keys[0]]
    return tuple(outputs[key] for key in keys)


def _build_model_config(args: argparse.Namespace) -> JaxModelConfig:
    output_heads = JaxOutputHeadConfig(
        self_head=args.self_head,
        value_head=args.value_head,
        policy_head=args.policy_head,
        soft_policy_head=args.soft_policy_head,
        value_num_bins=args.value_num_bins,
    )
    return JaxModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        activation=args.activation,
        output_heads=output_heads,
    )


def _create_validation_input(batch_size: int, seq_length: int, vocab_size: int) -> np.ndarray:
    return np.random.randint(0, vocab_size, size=(batch_size, seq_length), dtype=np.int32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a JAX transformer to ONNX.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination .onnx path.")
    parser.add_argument("--model-name", type=str, default=None, help="Name to embed in ONNX model.")
    parser.add_argument("--opset", type=int, default=21, help="ONNX opset version.")

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=28)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--activation", type=str, default="gelu")

    parser.add_argument("--self-head", action="store_true", help="Enable self reconstruction head.")
    parser.add_argument("--no-self-head", dest="self_head", action="store_false")
    parser.set_defaults(self_head=True)

    parser.add_argument("--value-head", action="store_true", help="Enable value head.")
    parser.add_argument("--no-value-head", dest="value_head", action="store_false")
    parser.set_defaults(value_head=True)

    parser.add_argument("--policy-head", action="store_true", help="Enable policy head.")
    parser.add_argument("--no-policy-head", dest="policy_head", action="store_false")
    parser.set_defaults(policy_head=False)

    parser.add_argument("--soft-policy-head", action="store_true", help="Enable soft policy head.")
    parser.add_argument("--no-soft-policy-head", dest="soft_policy_head", action="store_false")
    parser.set_defaults(soft_policy_head=False)

    parser.add_argument("--value-num-bins", type=int, default=81)
    parser.add_argument(
        "--output-key",
        action="append",
        default=["value"],
        help="Model output key to export (repeat for multiple outputs).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with a dynamic batch dimension (dtype inferred).",
    )
    parser.add_argument("--validate", action="store_true", help="Validate ONNX output with ORT.")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_keys = _parse_output_keys(args.output_key)
    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = _build_model_config(args)

    rng = jax.random.key(args.seed)
    model, params = BidirectionalTransformer.create_and_init(
        model_config, rng, compute_dtype=jnp.float32
    )

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")

    def forward(x: jax.Array) -> jax.Array | tuple[jax.Array, ...]:
        outputs = model.apply(params, x, train=False, compute_dtype=jnp.float32)
        return _select_outputs(outputs, output_keys)

    model_name = args.model_name or output_path.stem
    if args.dynamic_batch:
        input_specs = [("B", args.seq_length)]
    else:
        input_specs = [jax.ShapeDtypeStruct((args.batch_size, args.seq_length), jnp.int32)]

    model_path = to_onnx(
        forward,
        inputs=input_specs,
        model_name=model_name,
        opset=args.opset,
        return_mode="file",
        output_path=str(output_path),
    )

    print(f"Exported ONNX model to: {model_path}")

    if args.validate:
        validation_input = _create_validation_input(
            args.batch_size, args.seq_length, args.vocab_size
        )
        is_match, msg = allclose(
            forward,
            model_path,
            inputs=[validation_input],
            rtol=args.rtol,
            atol=args.atol,
        )
        if not is_match:
            raise RuntimeError(f"ONNX validation failed: {msg}")
        print("ONNX validation passed.")


if __name__ == "__main__":
    main()
