"""Names of the persisted FlashInfer tactic files the L1 runners tune into.

Each runner that tunes wraps its call in ``autotune_cached(autotune, label)``
(``profiling/runners/autotune_cache.py``); the label names the file. The labels
live here, apart from the runners, because the parent process needs them too:
under ``VIBESIM_AUTOTUNE=forbid`` it checks that every spec's file exists before
a worker starts (``profiling/exec/local.py``), and registry rows may not import
a runner module, which is free to need the worker's environment.

Each function takes the spec's schema fields as keywords, like a runner.
"""

from __future__ import annotations

from typing import Any


def _moe_shape(args: dict[str, Any]) -> str:
    return (
        f".e{args['num_experts']}.l{args['num_local_experts']}"
        f".h{args['hidden_size']}.i{args['intermediate_size']}"
    )


def nvfp4_fused_moe(args: dict[str, Any], stack: str, precomputed_routing: bool) -> str:
    """The file for one NVFP4 TRT-LLM MoE runner configuration.

    FlashInfer's file key for ``trtllm_fp4_block_scale_moe`` holds only the
    custom-op name, the runner class and the bucketed input shapes (output,
    logits, top-k ids/weights, hidden states and scales); ``MoERunner`` adds no
    ``get_cache_key_extras``. The local expert count, the intermediate size and,
    for the routed call, the global expert count shape the GEMMs but are absent
    from those shapes, so EP4 and TP8 rows of one model would share a key and
    the second would replay the first one's tactic. vLLM never sees that: it
    tunes in-process, where the key also hashes the runner. One file per
    configuration restores that separation. The routing method selects a
    different routing kernel in the monolithic call, so it names the file too.
    """

    label = f"nvfp4_fused_moe.{stack}" + (
        ".routed" if precomputed_routing else f".{args['routing_method']}"
    )
    return label + _moe_shape(args)


def nvfp4_fused_moe_vllm(**args: Any) -> str:
    return nvfp4_fused_moe(args, "vllm", precomputed_routing=False)


def nvfp4_fused_moe_vllm_routed(**args: Any) -> str:
    return nvfp4_fused_moe(args, "vllm", precomputed_routing=True)


def mxfp4_bf16_fused_moe(**args: Any) -> str:
    """See ``nvfp4_fused_moe``: FlashInfer's key omits the expert counts and sizes."""

    return f"nvfp4_fused_moe.mxfp4_bf16_vllm.{args['routing_method']}" + _moe_shape(args)


def mxfp4_fused_moe(**_args: Any) -> str:
    return "nvfp4_fused_moe.mxfp4_vllm"


def fp8_block_fused_moe(**_args: Any) -> str:
    return "nvfp4_fused_moe.fp8_block_vllm"


def flashinfer_fp8_tensor(**_args: Any) -> str:
    return "single_gemm.flashinfer_fp8_tensor"


def flashinfer_mxfp8(**_args: Any) -> str:
    return "single_gemm.flashinfer_mxfp8"
