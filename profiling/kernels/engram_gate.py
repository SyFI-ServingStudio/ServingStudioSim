"""DeepSeek-V4.1 Engram gate: add the gated n-gram value to every mHC stream, one fused launch.

Source (SGLang v0.5.23): ``srt/layers/engram.py`` ``Engram.forward`` (line 1332) -> ``engram_gate``
(line 1253) -> ``kernels/ops/embeddings/engram_gate.py`` ``fused_engram_gate`` (line 49): one Triton
``_engram_gate_kernel`` program per (token, hc copy). It follows the ``wkv`` projection of the looked-up
rows (a block-FP8 linear, priced by ``single_gemm``). One call per Engram layer (V4.1-Flash: layers 1
and 14) per iteration, on every TP rank over all scheduled tokens.
"""

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

KIND = "engram_gate"


@dataclass(frozen=True)
class EngramGateArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    hidden_dtype: DType = arg(doc="Element type of the streams, the wkv output and the result.")


DOC = KernelDoc(
    title="Engram gate",
    summary=(
        "Gate the Engram value with a normalized query-key dot product per mHC stream and add it to "
        "the stream."
    ),
    description=(
        "kv = wkv(rows) holds one key per hc copy and a shared value. For each (token, copy): "
        "dot = Σ(x·q_w·k_w·key)·rsqrt(mean(x²)+ε)·rsqrt(mean(key²)+ε)/√d, gate = "
        "σ(sign(dot)·√max(|dot|, c)), out = x + gate·value. Inputs are seeded random bf16 tensors."
    ),
    category="Other",
    formula=(
        "out[t, i] = x[t, i] + σ(±√max(|dot[t, i]|, c)) · value[t]",
        "GB/s = 2 · num_tokens · hidden_size · (3·hc_mult + 1) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=f"{CUPTI_METHOD} Compared with SGLang's PyTorch engram_gate before timing.",
    caveats=("Image-token masking (input_ids) is off, as for text-only serving.",),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.embedding.engram_sglang",
            function_name="profile_engram_gate_sglang_triton",
        ),
        table_name=KIND,
        args_schema=EngramGateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary="SGLang fused_engram_gate (kernels/ops/embeddings/engram_gate.py:49).",
            url="https://github.com/sgl-project/sglang",
        ),
    )
)

__all__ = ["EngramGateArgs", "KIND"]
