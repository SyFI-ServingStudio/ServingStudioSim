"""KV compression: save partial states, then compress, normalize, rotate and store keys."""

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

KIND = "kv_compress_store"


@dataclass(frozen=True)
class KvCompressStoreArgs(KernelArgs):
    """Exact token topology plus the production model/cache identity."""

    row_positions: tuple[int, ...] = arg(unit="tokens", doc="Absolute position of each input row.")
    row_request_ids: tuple[int, ...] = arg(
        unit="requests", doc="Request index associated with each input row."
    )
    state_block_table_width: int = arg(
        unit="blocks", doc="Partial-state block-table entries allocated per request."
    )
    compress_ratio: int = arg(
        unit="tokens", doc="Source tokens represented by one compressed cache position."
    )
    num_kv_heads: int = arg(unit="heads", doc="KV heads in the compressed cache.")
    head_dim: int = arg(unit="elements", doc="Elements in each compressed key.")
    rope_head_dim: int = arg(unit="elements", doc="Compressed-key elements rotated by RoPE.")
    logical_block_size: int = arg(unit="tokens", doc="Source tokens in a logical cache block.")
    rms_eps: float = arg(unit="unitless", doc="Stabilizer added to the RMS variance.")
    state_dtype: DType = arg(doc="Element type of saved partial states.")
    norm_dtype: DType = arg(doc="Element type of the normalization weight.")
    cache_dtype: str = arg(doc="Packed compressed-cache format.")
    cache_layout: str = arg(doc="Ordering of cache data and scales within a block.")
    scale_format: str = arg(doc="Encoding of the compressed-key scales.")


DOC = KernelDoc(
    title="KV compression and store",
    summary="Save partial states, then compress and store keys at compression boundaries.",
    description=(
        "A compressed KV cache is built incrementally: every token saves a "
        "partial state, and at every compress_ratio-th position a window of saved "
        "states is combined with softmax weights, RMS-normalized, rotated with "
        "RoPE, quantized and stored as one compressed key. row_positions and "
        "row_request_ids give each row's position and request; a row stores a "
        "key when position + 1 is divisible by compress_ratio. The 512-element "
        "attention cache uses a CuTe DSL kernel; the 128-element indexer cache "
        "uses Triton."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=(
        "compressed = Σwindow state_key · softmax(state_score); key = RoPE(RMSNorm(compressed))",
        "N = len(row_positions); A = number of boundary rows; "
        "W = compress_ratio·(2 if compress_ratio = 4 else 1)",
        "TFLOPS = [A·(W·head_dim·5 + head_dim·8) + N·state_width] / time",
        "GB/s = [A·W·head_dim·2·4 + A·cache_row_bytes + N·(4 + 8 + 8 + 8) "
        "+ N·(state_width·5·4 + 16)] / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Every launch is counted: saving the partial states, then compressing "
        "and storing the boundary rows. The 512-element C128 path takes three "
        "launches; the C4 paths take two."
    ),
    caveats=(
        "state_width is 1024 for C4 with CuTe DSL and 512 otherwise; "
        "cache_row_bytes is 584 for CuTe DSL and 132 for Triton.",
        "Partial states and the RoPE table are generated values, not model activations.",
    ),
    # The PyTorch correctness calculation is local to a measured runner.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cutedsl",
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.kv_compress_store_cutedsl"),
            function_name="profile_kv_compress_store_cutedsl",
        ),
        table_name=KIND,
        args_schema=KvCompressStoreArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # FP8 e4m3 conversion needs SM89+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            kv=frozenset({DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's save_partial_states, then the CuTe DSL "
                "compress_norm_rope_store_cutedsl, writing the 512-element FP8 "
                "attention cache."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.kv_compress_store_triton"),
            function_name="profile_kv_compress_store_triton",
        ),
        table_name=KIND,
        args_schema=KvCompressStoreArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # compress_norm_rope_store_triton stores tl.float8e4nv, which Triton lowers
        # only on SM89+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            kv=frozenset({DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's save_partial_states, then the Triton "
                "compress_norm_rope_store_triton, writing the 128-element FP8 indexer "
                "cache."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py",
        ),
    )
)

__all__ = ["KvCompressStoreArgs", "KIND"]
