"""Single-GEMM kernel kind.

All Python-side per-kernel knowledge for ``single_gemm`` lives here: the wire
string ``KIND``, the ``SingleGemmArgs`` schema, and the ``register(...)`` call
that wires this kernel into ``profiling.db.registry``.

Wire string: ``"single_gemm"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/single_gemm.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_single_gemm_times`` /
``count_missing_single_gemm``.

Seven backends share this kind/table/schema: ``torch`` (contiguous-RHS
``torch.mm``), ``torch_linear`` (model-weight-layout ``F.linear`` in the main
environment), ``torch_linear_vllm`` (the same expression in vLLM's pinned
environment), ``sglang_bf16_auto`` and ``sglang_fused_a_auto`` (SGLang's
production BF16 dispatches, SM100 only), ``deepgemm`` (FP8 dense GEMM,
``dtype = fp8_e4m3``), and ``flashinfer_mxfp8`` (MXFP8 dense linear:
activation quant + FlashInfer CuTe-DSL block-scaled GEMM, ``dtype = mxfp8_e4m3``).
BF16/FP16 model defaults offer both generic Torch variants and the timing cache
selects the faster one per shape.

Importing this module has a side effect: it appends ``KernelProfilerSpec`` rows
to the registry. The runner modules under ``profiling.runners.gemm``
are referenced lazily via ``RunnerRef`` so the main process never eager-imports
torch/cuda.
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

KIND: str = "single_gemm"


@dataclass(frozen=True)
class SingleGemmArgs(KernelArgs):
    m: int = arg(unit="tokens", doc="Rows of the activation: tokens in the batch.")
    n: int = arg(unit="elements", doc="Output features of the weight.")
    k: int = arg(unit="elements", doc="Input features, the reduction dimension.")
    dtype: DType = arg(doc="Element type of A and B. fp8_e4m3 writes bf16 output.")


DOC = KernelDoc(
    title="Dense GEMM",
    summary="One matrix multiply: an activation block of m tokens times a weight matrix.",
    description=(
        "Every linear layer in a transformer is a dense GEMM: QKV and output "
        "projections, dense MLPs, the LM head. m is the number of tokens in the "
        "batch; n and k come from the weight shape. Backends that share this table "
        "differ in how the weight is laid out and which library dispatches the "
        "multiply, and the simulator picks the fastest measured backend per shape."
    ),
    category="GEMM",
    formula=(
        "C[m, n] = A[m, k] · B[k, n]",
        "TFLOPS = 2·m·n·k / time",
        "GB/s = (m·k + k·n + m·n) · bytes per element / time",
        "deepgemm (fp8 in, bf16 out): GB/s = (m·k + k·n + 2·m·n) / time",
    ),
    default_metric="tflops",
    caveats=(
        "Inputs are random normal tensors, so data-dependent effects such as "
        "sparsity are not measured.",
        "torch and torch_linear compute the same product. They are separate rows "
        "because the weight layout changes which cuBLAS kernel runs.",
    ),
    method=(
        f"{CUPTI_METHOD} The torch and SGLang backends count overlapping launches "
        "once, by the time the GPU is busy, because Blackwell can split one logical "
        "GEMM into several; deepgemm sums its launches."
    ),
    # torch.mm is its own reference: the torch backend measures exactly it.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.torch",
            function_name="profile_single_gemm",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="torch.mm on a contiguous right-hand side.",
            url="https://pytorch.org/docs/stable/generated/torch.mm.html",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_linear_vllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.torch",
            function_name="profile_single_gemm_linear",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="The same F.linear call, run in vLLM's pinned environment.",
            url="https://github.com/vllm-project/vllm",
        ),
        subprocess_env="vllm_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_linear",
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.torch",
            function_name="profile_single_gemm_linear",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="F.linear with the (n, k) weight layout used by vLLM and Transformers.",
            url="https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_bf16_auto",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.sglang",
            function_name="profile_single_gemm_sglang_bf16",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "SGLang's BF16 dispatch with bf16_gemm_backend='auto': cutedsl_bf16_gemm "
                "when use_cutedsl_bf16_gemm(m, n, k) accepts the shape, otherwise "
                "F.linear. SM100 only."
            ),
            url="https://github.com/sgl-project/sglang",
        ),
        subprocess_env="sglang_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_fused_a_auto",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.sglang",
            function_name="profile_single_gemm_sglang_fused_a",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "SGLang's full fused-A dispatch: dsv3_fused_a_gemm when 1 ≤ m ≤ 16, "
                "n % 16 == 0 and k % 256 == 0; otherwise cutedsl_bf16_gemm when "
                "use_cutedsl_bf16_gemm(m, n, k) accepts the shape; otherwise F.linear. "
                "SM100 only."
            ),
            url="https://github.com/sgl-project/sglang",
        ),
        subprocess_env="sglang_env",
    )
)

# DeepGEMM FP8 dense kernel — same wire schema / table, FP8 compute
# (dtype = fp8_e4m3, fp8 in / bf16 out). subprocess_env=None (default env;
# deep_gemm is a pinned project dep, see CLAUDE.md / `just sync`).
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm",
        supports=BackendSupport(compute=frozenset({DType.FP8_E4M3})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.deepgemm",
            function_name="profile_single_gemm",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "DeepGEMM fp8_gemm_nt: fp8 A with one scale per token per 128 "
                "k-elements, fp8 B with one scale per 128×128 block, bf16 output. "
                "On Blackwell the scales are UE8M0, packed into DeepGEMM's layout "
                "before timing, as vLLM passes them."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)

# vLLM-fork MXFP8 dense linear (DeepSeek-V4.1): one slot = the swizzled MXFP8
# activation quant + FlashInfer CuTe-DSL block-scaled GEMM that
# FlashInferCutedslMxfp8LinearKernel.apply_weights issues, bf16 out.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_mxfp8",
        supports=BackendSupport(
            compute=frozenset({DType.MXFP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.flashinfer_mxfp8",
            function_name="profile_single_gemm_flashinfer_mxfp8",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_upstream_fork_env",
    )
)
