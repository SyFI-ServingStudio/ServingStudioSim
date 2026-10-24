"""Single-GEMM kernel kind.

All Python-side per-kernel knowledge for ``single_gemm`` lives here: the wire
string ``KIND``, the ``SingleGemmArgs`` schema, and the ``register(...)`` call
that wires this kernel into ``profiling.db.registry``.

Wire string: ``"single_gemm"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/single_gemm.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_single_gemm_times`` /
``count_missing_single_gemm``.

Fourteen backends share this kind/table/schema: ``torch`` (contiguous-RHS
``torch.mm``), ``torch_linear`` (model-weight-layout ``F.linear`` in the main
environment), ``torch_linear_vllm`` (the same expression in vLLM's pinned
environment), ``sglang_bf16_auto`` and ``sglang_fused_a_auto`` (SGLang's
production BF16 dispatches, SM100 only), ``deepgemm`` (FP8 dense GEMM,
``dtype = fp8_e4m3``), ``flashinfer_fp8_tensor`` (vLLM's static per-tensor FP8
linear on SM10x, ``dtype = fp8_e4m3``), ``flashinfer_cutedsl`` (vLLM's default NVFP4 W4A4
linear on SM10x, ``dtype = nvfp4_e2m1``), and ``flashinfer_mxfp8`` (MXFP8 dense
linear: activation quant + FlashInfer CuTe-DSL block-scaled GEMM,
``dtype = mxfp8_e4m3``), and TensorFold's checkpoint-math ``tensorfold_lane``
(decode/verify) and ``tensorfold_ck`` (prefill) GEMMs (``fp8_e4m3`` /
``nvfp4_e2m1``, SM12x), and TensorFold's affine 4-bit lane matmul
``tensorfold_qmm`` (``w4a16_affine_g64``, SM12x), and TensorFold's verify-path
bf16 GDN gate pair ``tensorfold_b16`` and its prefill form ``tensorfold_b16_prompt``
(two n/2 halves in one launch).
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
    dtype: DType = arg(
        doc=(
            "Element type of A and B. fp8_e4m3, nvfp4_e2m1 and mxfp8_e4m3 write bf16 output; "
            "w4a16_affine_g64 is bf16 A times affine 4-bit B (bf16 scale and bias per 64 inputs)."
        )
    )


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
        "deepgemm, flashinfer_fp8_tensor (fp8 in, bf16 out): GB/s = (m·k + k·n + 2·m·n) / time",
        "flashinfer_cutedsl (nvfp4 in, bf16 out): GB/s = ((m·k + k·n)·(1/2 + 1/16) + 2·m·n) / time",
        "tensorfold_lane, tensorfold_ck: the fp8 or nvfp4 formula above, by dtype",
        "tensorfold_b16, tensorfold_b16_prompt (bf16, two n/2-wide weights in one launch): "
        "the bf16 formula, n the pair's total",
        "tensorfold_qmm (w4a16_affine_g64): GB/s = (2·m·k + 4·m·k/64 + k·n/2 + 4·n·k/64 + 2·m·n) / time",
    ),
    default_metric="tflops",
    caveats=(
        "Inputs are random normal tensors, so data-dependent effects such as "
        "sparsity are not measured.",
        "deepgemm scales fp8_e4m3 operands per 128-element block; "
        "flashinfer_fp8_tensor scales each operand by one FP32 scalar.",
        "torch and torch_linear compute the same product. They are separate rows "
        "because the weight layout changes which cuBLAS kernel runs.",
        "flashinfer_mxfp8 takes bf16 activations and includes their MXFP8 "
        "quantization in the time; its GB/s adds the bf16 read and the "
        "quantized write of the activation.",
    ),
    method=(
        f"{CUPTI_METHOD} The torch and SGLang backends count overlapping launches "
        "once, by the time the GPU is busy, because Blackwell can split one logical "
        "GEMM into several; deepgemm, flashinfer_fp8_tensor, flashinfer_cutedsl and "
        "flashinfer_mxfp8 sum their launches. flashinfer_cutedsl times only the GEMM: "
        "the activation quantization before it is the nvfp4_quant kind. flashinfer_mxfp8 includes "
        "its activation quantization and autotunes the call once before timing. "
        "flashinfer_fp8_tensor times only the GEMM on already-quantized fp8 input, "
        "after autotuning FlashInfer's hybrid token buckets as vLLM's warmup does. "
        "tensorfold_lane and tensorfold_ck time only TensorFold's GEMM launch on rows "
        "already quantized under the static input scale (quant4 / quant8 is its own kind). "
        "tensorfold_qmm times only the GEMM launch, with the rows' fp32 group sums already "
        "computed (TensorFold's small _group_sums launch, when the caller does not pass them, "
        "is not included). tensorfold_b16 times the one b16_kernel launch of matmul_pair "
        "on bf16 rows, tensorfold_b16_prompt the one b16_prompt_kernel launch of prompt_pair."
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
        # SGLang's initialize_bf16_gemm_config resolves `auto` to cutedsl when
        # is_sm100_supported(): compute capability major 10.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100f"}),
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
        # SGLang's initialize_bf16_gemm_config resolves `auto` to cutedsl when
        # is_sm100_supported(): compute capability major 10.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100f"}),
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

# vLLM's static per-tensor FP8 linear (ModelOpt FP8) on SM10x:
# FlashInferFP8ScaledMMLinearKernel heads _POSSIBLE_FP8_KERNELS and its
# is_supported requires compute capability 100+; flashinfer.bmm_fp8's common
# check (_check_bmm_fp8_problem_size) admits 100/103/107, the sm_100f family.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_fp8_tensor",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.flashinfer_fp8_tensor",
            function_name="profile_single_gemm_flashinfer_fp8_tensor",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's FlashInferFP8ScaledMMLinearKernel after its activation "
                "quantization: flashinfer bmm_fp8(backend='auto') on fp8 A and the "
                "column-major fp8 weight, one FP32 scale each, bf16 output, tactic "
                "autotuned per hybrid token bucket. SM10x only."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py",
        ),
    )
)

# vLLM's default NVFP4 (ModelOpt W4A4, group 16) dense linear on SM10x:
# FlashInferCuteDslNvFp4LinearKernel heads _POSSIBLE_NVFP4_KERNELS and its
# is_supported requires is_device_capability_family(100); FlashInfer's
# _cute_dsl_gemm_fp4_requirement accepts SM100/103/107, all in the sm_100f family.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_cutedsl",
        supports=BackendSupport(
            compute=frozenset({DType.NVFP4_E2M1}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.flashinfer_fp4",
            function_name="profile_single_gemm",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "vLLM's FlashInferCuteDslNvFp4LinearKernel after its activation "
                "quantization: flashinfer mm_fp4(backend='cute-dsl') on packed FP4 A "
                "and B with swizzled E4M3 scales per 16 k-elements, bf16 output, "
                "heuristic tactic (vLLM skips fp4_gemm autotuning). SM10x only."
            ),
            url="https://github.com/flashinfer-ai/flashinfer",
        ),
        subprocess_env="vllm_env",
    )
)

# MXFP8 dense linear: one slot = the swizzled MXFP8 activation quant +
# FlashInfer CuTe-DSL block-scaled GEMM that
# FlashInferCutedslMxfp8LinearKernel.apply_weights issues, bf16 out.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_mxfp8",
        # vLLM's FlashInferCutedslMxfp8LinearKernel.is_supported():
        # is_device_capability_family(100), the SM10x family.
        supports=BackendSupport(
            compute=frozenset({DType.MXFP8_E4M3}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.flashinfer_mxfp8",
            function_name="profile_single_gemm_flashinfer_mxfp8",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's FlashInferCutedslMxfp8LinearKernel: FlashInfer MXFP8 "
                "quantization of the bf16 activation (UE8M0 per 32, swizzled) and "
                "the CuTe-DSL block-scaled GEMM mm_mxfp8, bf16 output."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py",
        ),
    )
)

# TensorFold's checkpoint-math linears (tensorfold.cuda.nvfp4.checkpoint), the
# GEMM after the row quantization (quant4 / quant8 is its own kind). The one
# extension builds both modes; its NVFP4 mode needs the SM12x block-scaled FP4
# mma (mma4.cuh guards on __CUDA_ARCH_FEAT_SM120_ALL / SM121_ALL;
# checkpoint.available() and precision.own_math: NVFP4 own math only at major
# 12). The FP8 mode builds from 8.9, but a row declares one capability rule, so
# both rows take the SM12x family the NVFP4 mode requires.
_TENSORFOLD_SUPPORT = BackendSupport(
    compute=frozenset({DType.FP8_E4M3, DType.NVFP4_E2M1}),
    sm_targets=frozenset({"sm_120f"}),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_lane",
        supports=_TENSORFOLD_SUPPORT,
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.tensorfold",
            function_name="profile_single_gemm_tensorfold_lane",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's decode/verify linear in checkpoint math after its row "
                "quantization: lane_kernel (lane4.cu) on NVFP4 rows x NVFP4 weights or "
                "FP8 rows x per-tensor FP8 weights, bf16 output, K split by weight shape "
                "with the slices summed in a thread-block cluster. SM12x."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_ck",
        supports=_TENSORFOLD_SUPPORT,
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.tensorfold",
            function_name="profile_single_gemm_tensorfold_ck",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's prefill linear in checkpoint math after its row "
                "quantization: gemm_kernel (gemm_ck.cu, plain epilogue) on NVFP4 or FP8 "
                "rows, one fp32 chain over K a row, bf16 output. SM12x."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

# TensorFold's bf16 GDN gate pair (families/qwen3_5/cuda/b16.py matmul_pair ->
# b16.cu b16_kernel, grid z = 2): a verify forward's in_proj_b and in_proj_a in
# one launch, priced as one GEMM of their total width n (two n/2 halves). The
# extension builds through tensorfold.cuda.build.load, which refuses a GPU below
# compute capability 8.9 (MIN_CAPABILITY); the kernel itself is plain CUDA-core.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_b16",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.tensorfold_b16",
            function_name="profile_single_gemm_tensorfold_b16",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's verify-path bf16 linear pair (b16.matmul_pair, the GDN gates b and "
                "a): one b16_kernel launch, grid z picking one of two n/2-wide weights, a warp an "
                "output column over 1-16 rows with fp32 CUDA-core FMAs, bf16 output. n is the "
                "pair's total width and must be even."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

# The same pair in prefill (b16.py prompt_pair -> b16.cu b16_prompt_kernel,
# bf16 mma.sync, grid z = 2). Same build gate as tensorfold_b16.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_b16_prompt",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.tensorfold_b16",
            function_name="profile_single_gemm_tensorfold_b16_prompt",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's prefill bf16 linear pair (b16.prompt_pair, the GDN gates b and a): "
                "one b16_prompt_kernel launch of bf16 mma.sync BM x 64 tiles, grid z picking one "
                "of two n/2-wide weights, bf16 output. n is the pair's total width and must be "
                "even; k a multiple of 64. The two halves are scheduled one after the other (z "
                "is the slowest grid index), so each re-reads the rows: past L2 (about 4096 rows "
                "of k = 5120 on a GB10) the pair runs slower than one n-wide GEMM, and the row "
                "includes that."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

# TensorFold's affine 4-bit lane matmul (tensorfold.cuda.kernels.qmm), which runs
# the DFlash2 drafter's projections after its load-time re-quantization
# (dflash2.quantize4 + qmm_fast.tile). qmm.grouped() sends groups of 64 through
# qmm_group.cu's group_kernel only at compute capability major 12; other GPUs
# take qmm.cu's kernel, a different implementation this row does not measure.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_qmm",
        supports=BackendSupport(
            compute=frozenset({DType.W4A16_AFFINE_G64}),
            sm_targets=frozenset({"sm_120f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.tensorfold_qmm",
            function_name="profile_single_gemm_tensorfold_qmm",
        ),
        table_name=KIND,
        args_schema=SingleGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's affine 4-bit lane matmul (qmm_fast.matmul, the DFlash2 drafter's "
                "projections): bf16 rows times MLX-style 4-bit codes with a bf16 scale and bias "
                "per 64 inputs, group_kernel (qmm_group.cu) on bf16 mma, K split by weight shape "
                "and summed in a thread-block cluster, bf16 output. Rows' group sums precomputed. SM12x."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)
