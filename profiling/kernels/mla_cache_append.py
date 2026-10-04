"""Plain MLA paged-cache append kernel kind.

The first backend measures the two-write Torch semantic composite for
GLM-5.2's 512-wide latent plus 64-wide RoPE cache entry. The production-aligned
backend measures vLLM's fused one-launch CUDA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "mla_cache_append"


@dataclass(frozen=True)
class MlaCacheAppendArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows appended to the cache.")
    kv_lora_rank: int = arg(unit="elements", doc="Width of each compressed KV latent.")
    rope_dim: int = arg(unit="elements", doc="Width of each rotary key part; 0 when there is none.")
    block_size: int = arg(unit="tokens", doc="Token positions in each cache page.")
    input_dtype: DType = arg(doc="Element type of the source rows.")
    kv_dtype: DType = arg(doc="Element type of the cache entries.")
    cache_format: str = arg(doc="Layout of the MLA cache entries.")


DOC = KernelDoc(
    title="MLA cache append",
    summary="Write compressed KV latents and rotary keys into paged MLA cache entries.",
    description=(
        "After the MLA projections and RoPE, each token writes a "
        "kv_lora_rank-wide latent and a rope_dim-wide rotary key into one paged"
        " cache entry. The measurement scatters the rows to distinct random "
        "slots. The plain cache format takes any kv_lora_rank, rope_dim and "
        "page size; vllm_cuda also accepts rope_dim = 0, a latent without a "
        "rotary key, which it writes alone."
    ),
    category="Attention",
    subcategory="MLA",
    formula=(
        "cache[position] = [latent, rotary key]",
        "GB/s = num_tokens · ((kv_lora_rank + rope_dim) · (input bytes + cache bytes) + 8) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        "torch is timed with CUDA events: five warm-up calls, then a loop of "
        "back-to-back calls, taking the median of three runs, so the gap "
        "between its two launches is included. vllm_cuda and sglang_cuda use "
        "CUPTI kernel time with the L2 cache flushed before each launch: "
        "vllm_cuda counts only concat_and_cache_mla_kernel, after checking that"
        " every slot holds its row, and sglang_cuda counts every launch of its "
        "call after one setup call."
    ),
    caveats=(
        "torch rows are warm-cache elapsed time and CUPTI rows are cold-cache "
        "kernel time, so the backends are not directly comparable.",
        "sglang_cuda scatters bytes that are already FP8; vllm_cuda can convert"
        " BF16 input to an FP8 cache as it writes.",
        "GB/s counts logical source bytes, cache bytes and one 8-byte slot "
        "index per token; cache initialization is excluded.",
    ),
    reference="profiling.runners.attention.mla_cache_append_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mla_cache_append",
            function_name="profile_mla_cache_append_torch",
        ),
        table_name=KIND,
        args_schema=MlaCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(summary="PyTorch indexed assignment in two separate cache writes."),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16, DType.FP8_E4M3}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mla_cache_append",
            function_name="profile_mla_cache_append_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MlaCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM concat_and_cache_mla, a fused CUDA append that can "
                "convert BF16 rows to FP8 cache entries."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mla_cache_append",
            function_name="profile_mla_cache_append_sglang_cuda",
        ),
        table_name=KIND,
        args_schema=MlaCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang set_mla_kv_buffer_triton, a Triton kernel that scatters "
                "pre-quantized FP8 rows."
            ),
        ),
    )
)
