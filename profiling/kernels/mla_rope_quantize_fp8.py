"""FlashInfer fused MLA RoPE, FP8 quantization, and query-concat kernel kind."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "mla_rope_quantize_fp8"


@dataclass(frozen=True)
class MlaRopeQuantizeFp8Args(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Query and key token rows to transform.")
    num_heads: int = arg(unit="heads", doc="Query heads on this GPU.")
    kv_lora_rank: int = arg(
        unit="elements", doc="Width of the non-rotary latent key and query part."
    )
    rope_dim: int = arg(unit="elements", doc="Width of the rotary query and key part.")
    max_position: int = arg(unit="positions", doc="Rows in the cosine and sine lookup table.")
    is_neox_style: bool = arg(doc="Whether the rotary dimensions use NeoX pairing.")
    input_dtype: DType = arg(doc="Element type of the input query and key parts.")
    quant_dtype: DType = arg(doc="Element type of the transformed outputs.")


DOC = KernelDoc(
    title="MLA RoPE and FP8 quantization",
    summary="Rotate MLA queries and keys, quantize them to FP8, and assemble each query.",
    description=(
        "On SGLang's MLA path, one FlashInfer call applies RoPE to "
        "the rope_dim columns of each query head and of the shared key, and "
        "quantizes both rotary and non-rotary parts to FP8. The query's "
        "non-rotary part is already absorbed into the latent space, so it is "
        "kv_lora_rank wide. Query parts go into slices of one FP8 tensor; key "
        "parts go to separate outputs. Positions are random rows of the "
        "max_position-row lookup table."
    ),
    category="Attention",
    subcategory="MLA",
    formula=(
        "query output = FP8([q_nope, RoPE(q_rope)]); key outputs = FP8(k_nope), FP8(RoPE(k_rope))",
        "TFLOPS = num_tokens · (num_heads + 1) · rope_dim · 3 / time",
        "GB/s = num_tokens · ((num_heads + 1) · (kv_lora_rank + rope_dim) · "
        "(input bytes + output bytes) + rope_dim · 4 + 8) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "One call and a synchronization run first, then five warm-up calls; "
        "every launch of the FlashInfer call is counted."
    ),
    caveats=(
        "The cosine and sine table holds random values, not real RoPE angles; "
        "the access pattern is the same.",
        "TFLOPS counts three operations per rotary element as a nominal rate.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer",
        # No capability rule: FlashInfer JIT-builds rope.cu for the current device;
        # its PDL instructions are guarded by __CUDA_ARCH__ >= 900 (pos_enc.cuh).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mla_rope_quantize_fp8",
            function_name="profile_mla_rope_quantize_fp8_flashinfer",
        ),
        table_name=KIND,
        args_schema=MlaRopeQuantizeFp8Args,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer mla_rope_quantize_fp8 with pre-split MLA inputs and FP8 E4M3 outputs."
            ),
        ),
    )
)

__all__ = ["KIND", "MlaRopeQuantizeFp8Args"]
