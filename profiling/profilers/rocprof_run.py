"""Launch driver run *under* rocprofv3 to capture one registered kernel.

``measure_registered_via_rocprofv3`` spawns this as
``rocprofv3 ... -- python -m profiling.profilers.rocprof_run --kind K --backend B
--spec '{...}' --warmup W --rep R``. rocprofv3 traces every GPU dispatch of this
whole process; this driver rebuilds the exact kernel from ``(kind, backend,
spec)`` (same fixed seed as the runner) and launches it ``warmup + rep`` times,
so the rocpd database holds those launches for the parser to attribute. It does
no timing itself -- the tracer and the rocpd parser own that.

Only ROCm backends that expose a kernel builder are driveable here; the builder
registry below maps a ``(kind, backend)`` to the module-level ``build`` that
returns the callable and its fixed inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any


def _rms_norm_torch_rocm_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.norm.rms_norm_torch_rocm import build_rms_norm_kernel

    return build_rms_norm_kernel(spec["m"], spec["hidden"], spec["dtype"])


def _kda_recurrent_decode_torch_rocm_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.attention.kda_recurrent_decode_torch_rocm import (
        build_kda_recurrent_decode_kernel,
    )

    return build_kda_recurrent_decode_kernel(
        spec["batch_size"], spec["num_heads"], spec["head_dim"], spec["dtype"]
    )


def _kda_chunk_prefill_torch_rocm_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.attention.kda_chunk_prefill_torch_rocm import (
        build_kda_chunk_prefill_kernel,
    )

    return build_kda_chunk_prefill_kernel(
        spec["num_tokens"],
        spec["max_sequence_length"],
        spec["num_decode_sequences"],
        spec["num_heads"],
        spec["head_dim"],
        spec["dtype"],
    )


def _gdn_causal_conv_decode_torch_rocm_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.attention.gdn_causal_conv_decode_torch_rocm import (
        build_gdn_causal_conv_decode_kernel,
    )

    return build_gdn_causal_conv_decode_kernel(
        spec["batch_size"],
        spec["channels"],
        spec["kernel_size"],
        spec["dtype"],
        spec["state_dtype"],
    )


def _gdn_causal_conv_prefill_torch_rocm_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.attention.gdn_causal_conv_prefill_torch_rocm import (
        build_gdn_causal_conv_prefill_kernel,
    )

    return build_gdn_causal_conv_prefill_kernel(
        spec["batch_size"],
        spec["sequence_length"],
        spec["channels"],
        spec["kernel_size"],
        spec["dtype"],
        spec["state_dtype"],
    )


def _q_kv_rms_norm_torch_rocm_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.attention.q_kv_rms_norm_torch_rocm import (
        build_q_kv_rms_norm_kernel,
    )

    return build_q_kv_rms_norm_kernel(
        spec["num_tokens"],
        spec["q_dim"],
        spec["kv_dim"],
        spec["rms_eps"],
        spec["dtype"],
    )


def _rocm_aiter_fp8_block_fused_moe_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.moe.rocm_aiter_fp8_block_fused_moe import (
        build_rocm_aiter_fp8_block_fused_moe_kernel,
    )

    return build_rocm_aiter_fp8_block_fused_moe_kernel(
        spec["num_tokens"],
        spec["hidden_size"],
        spec["intermediate_size"],
        spec["num_experts"],
        spec["num_local_experts"],
        spec["top_k"],
        spec["input_dtype"],
        spec["weight_format"],
        spec["group_size"],
        spec["routing_method"],
        spec["n_group"],
        spec["topk_group"],
        spec["routed_scaling_numerator"],
        spec["routed_scaling_denominator"],
        tuple(spec["per_expert_batches"]),
    )


def _dsa_sparse_mla_attention_rocm_triton_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.attention.dsa_sparse_mla_attention_rocm_triton import (
        build_dsa_sparse_mla_attention_rocm_triton_kernel,
    )

    return build_dsa_sparse_mla_attention_rocm_triton_kernel(**spec)


def _dsa_sparse_mla_prefill_rocm_triton_builder(spec: dict[str, Any]) -> dict[str, Any]:
    from profiling.runners.attention.dsa_sparse_mla_prefill_rocm_triton import (
        build_dsa_sparse_mla_prefill_rocm_triton_kernel,
    )

    return build_dsa_sparse_mla_prefill_rocm_triton_kernel(**spec)


# (kind, backend) -> builder returning at least {"torch", "kernel"}.
_BUILDERS: dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]] = {
    ("rms_norm", "torch_rocm"): _rms_norm_torch_rocm_builder,
    ("kda_recurrent_decode", "torch_rocm"): _kda_recurrent_decode_torch_rocm_builder,
    ("kda_chunk_prefill", "torch_rocm"): _kda_chunk_prefill_torch_rocm_builder,
    ("gdn_causal_conv_decode", "torch_rocm"): _gdn_causal_conv_decode_torch_rocm_builder,
    ("gdn_causal_conv_prefill", "torch_rocm"): _gdn_causal_conv_prefill_torch_rocm_builder,
    ("q_kv_rms_norm", "torch_rocm"): _q_kv_rms_norm_torch_rocm_builder,
    ("nvfp4_fused_moe", "rocm_aiter_fp8_block"): _rocm_aiter_fp8_block_fused_moe_builder,
    (
        "dsa_sparse_mla_attention",
        "rocm_triton_mla_sparse",
    ): _dsa_sparse_mla_attention_rocm_triton_builder,
    (
        "dsa_sparse_mla_prefill",
        "rocm_triton_mla_sparse",
    ): _dsa_sparse_mla_prefill_rocm_triton_builder,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--spec", required=True, help="JSON kernel spec")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args(argv)

    builder = _BUILDERS.get((args.kind, args.backend))
    if builder is None:
        raise SystemExit(
            f"no rocprof launch builder for {args.kind}:{args.backend}; "
            f"known: {sorted(_BUILDERS)}"
        )
    built = builder(json.loads(args.spec))
    torch = built["torch"]
    kernel = built["kernel"]

    for _ in range(args.warmup):
        kernel()
    torch.cuda.synchronize()
    for _ in range(args.rep):
        kernel()
    torch.cuda.synchronize()
    # Stdout is captured by the caller only on failure; a success line helps when
    # inspecting the rocprofv3 wrapper log directly.
    print(f"rocprof_run ok: {args.kind}:{args.backend} warmup={args.warmup} rep={args.rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
