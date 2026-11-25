"""Batched-GEMM kernel kind for MLA's per-head query absorption and value
expansion, and a grouped attention output projection.

Each backend freezes one production storage layout (vLLM's packed Q-absorption
or V-up views, or the padded group-slot attention output a DeepGEMM MXFP8
einsum reads) instead of presenting its constants as generic batched GEMM
behavior. Any per-rank head count launches, except that the padded V-up layout
holds at most 64 heads.

``deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out`` is DeepInfer's grouped output
projection: the same einsum on a contiguous attention output, writing e4m3 with
UE8M0 scales per 32 (DeepGEMM's FP8-casting epilogue) instead of bf16.
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

KIND: str = "batched_gemm"


@dataclass(frozen=True)
class BatchedGemmArgs(KernelArgs):
    num_batches: int = arg(
        unit="heads",
        doc=(
            "Attention heads (or, for the grouped output projection, head groups) "
            "multiplied independently on this GPU."
        ),
    )
    m: int = arg(unit="tokens", doc="Token rows per attention head.")
    n: int = arg(unit="elements", doc="Output features per attention head.")
    k: int = arg(unit="elements", doc="Input features per attention head.")
    dtype: DType = arg(
        doc=(
            "Element type of both inputs: bf16 for the MLA backends, with bf16 "
            "output; mxfp8_e4m3 for the grouped output projection, with bf16 output."
        )
    )


DOC = KernelDoc(
    title="Batched GEMM",
    summary=(
        "Multiply one activation and weight matrix pair per attention head (MLA) "
        "or per head group (grouped output projection)."
    ),
    description=(
        "vLLM's MLA runs two per-head batched multiplies. Query absorption "
        "multiplies each head's query by W_UK, mapping it into the compressed KV "
        "space; value expansion multiplies each head's attention output by W_UV, "
        "mapping it back to the value head. num_batches is the heads on this GPU "
        "and m the tokens. Each backend builds the same strided views of the "
        "packed weight, activation and output that vLLM passes to torch.bmm. "
        "A grouped low-rank attention output projection is the same shape of "
        "work: each group of 8 heads' 4,096-wide attention output is multiplied "
        "by that group's weight, as one DeepGEMM einsum on MXFP8 operands. "
        "Two or more same-shape linears whose inputs are adjacent column slices "
        "of one packed activation can also run as one torch.bmm "
        "(torch_bmm_packed_lhs), in place of one linear call each."
    ),
    category="GEMM",
    formula=(
        "C[b, m, n] = A[b, m, k] · B[b, k, n]",
        "TFLOPS = 2·num_batches·m·n·k / time",
        "GB/s = num_batches·(m·k + k·n + m·n)·bytes per element / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} Each MLA backend times only its torch.bmm call; operand "
        "construction and any head-padding copy are outside the timed call. The "
        "grouped output projection times its one DeepGEMM launch after three "
        "warm-up calls."
    ),
    caveats=(
        "GB/s counts the logical operand elements, not the gaps in the packed "
        "storage the strided views skip.",
        "Each MLA backend fixes k and n to its layout: k = 192 or 256, n = 512 for "
        "query absorption and k = 512, n = 256 for value expansion. "
        "torch_bmm_packed_lhs takes any k and n.",
        "torch_mla_v_up reads an attention output padded to 64 heads, so it takes "
        "at most 64 heads.",
        "The grouped output projection's activation is the FP8 attention output "
        "with UE8M0 scales, stored in 8 padded group slots of which the call reads "
        "the first num_batches. Its GB/s counts one byte per input element plus one "
        "scale byte per 32, and two bytes per output element.",
        "deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out reads a contiguous "
        "[m, num_batches, k] activation and writes an e4m3 output with a UE8M0 "
        "scale per 32 (DeepInfer's FP8-output einsum); its GB/s counts one byte "
        "plus 1/32 scale byte per input and output element.",
    ),
    reference="profiling.runners.gemm.batched_gemm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_q_absorb",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_q_absorb",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="torch.bmm for query absorption, on vLLM's packed W_UK view.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_v_up",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_v_up",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "torch.bmm for value expansion, on vLLM's packed W_UV view and strided output."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py",
        ),
    )
)


# MLA layouts with an unpadded attention output: qk_nope 256 with no RoPE part,
# v_head 256 (the first pair), and qk_nope 128 + RoPE 64, v_head 128 (the second
# pair, Kimi-K3.1); kv_lora 512 for both. Same torch.bmm call as the two
# backends above.
_MLA_URL = "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py"
for _backend, _function, _summary in (
    (
        "torch_mla_q_absorb_no_rope",
        "profile_mla_q_absorb_no_rope",
        "torch.bmm for query absorption when the query head has no RoPE part, "
        "on vLLM's packed W_UK view.",
    ),
    (
        "torch_mla_v_up_unpadded",
        "profile_mla_v_up_unpadded",
        "torch.bmm for value expansion from an attention output with no head "
        "padding, on vLLM's packed W_UV view.",
    ),
    (
        "torch_mla_q_absorb_nope128_rope64",
        "profile_mla_q_absorb_nope128_rope64",
        "torch.bmm for query absorption of the 128-wide nope part of a 192-wide "
        "head (Kimi-K3.1 MLA), on vLLM's packed W_UK view.",
    ),
    (
        "torch_mla_v_up_v128_unpadded",
        "profile_mla_v_up_v128_unpadded",
        "torch.bmm for value expansion to 128-wide heads from an attention output "
        "with no head padding (Kimi-K3.1 MLA), on vLLM's packed W_UV view.",
    ),
):
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=BackendSupport(
                compute=frozenset({DType.BF16}),
            ),
            runner_ref=RunnerRef(
                module_name="profiling.runners.gemm.batched_gemm",
                function_name=_function,
            ),
            table_name=KIND,
            args_schema=BatchedGemmArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="vllm_env",
            doc=BackendDoc(summary=_summary, url=_MLA_URL),
        )
    )
# Grouped output projection (wo_a): one DeepGEMM fp8_einsum "bhr,hdr->bhd" over
# the local groups. Both operands are MXFP8 (e4m3 + ue8m0 per 32 K); the
# activation is the fused attention output in its padded 8-slot layout; bf16 out.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_mxfp8_einsum_grouped_o_proj",
        # vLLM's DeepGemmMxfp8BmmLinearKernel.is_supported():
        # is_device_capability_family(100), the SM10x family.
        supports=BackendSupport(
            compute=frozenset({DType.MXFP8_E4M3}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.deepgemm_mxfp8_einsum",
            function_name="profile_batched_gemm_deepgemm_mxfp8_einsum_grouped_o_proj",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                'DeepGEMM fp8_einsum("bhr,hdr->bhd") with a (1, 1, 32) MXFP8 recipe, '
                "as vLLM's DeepGemmMxfp8BmmLinearKernel applies a grouped output "
                "projection to the fused attention's FP8 output."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)

# Two or more same-shape low-rank linears whose inputs are adjacent column
# slices of one packed activation (for example the up halves of two low-rank
# gates read from one fused down projection), merged into one torch.bmm over
# stacked (n, k) weights. Same Torch stack as single_gemm's torch_linear_vllm,
# which prices the linears run one by one.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_bmm_packed_lhs",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_bmm_packed_lhs",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "torch.bmm over num_batches linears: each LHS is a k-wide column "
                "slice of one packed (m, num_batches*k) activation, the weights "
                "are stacked (n, k) linear weights, the output (num_batches, m, n)."
            ),
            url="https://pytorch.org/docs/stable/generated/torch.bmm.html",
        ),
    )
)

# DeepInfer's grouped output projection (wv / wo_a): the same DeepGEMM fp8_einsum
# "bhr,hdr->bhd" with recipe (1, 1, 32), on a contiguous [m, h, k] MXFP8 attention
# output, writing an FP8 (e4m3 + UE8M0 per 32) output tuple: DeepGEMM's
# CastToFP8WithClamp epilogue, a different kernel instance from the bf16 one above.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out",
        # DeepGEMM's fp8_einsum asserts arch_major == 10 for an FP8 output pair
        # and runs per-32 scales only on its SM100 path (csrc/apis/einsum.hpp).
        supports=BackendSupport(
            compute=frozenset({DType.MXFP8_E4M3}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.deepgemm_mxfp8_einsum_fp8_out",
            function_name="profile_batched_gemm_deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                'DeepInfer\'s V4 attention wv projection: DeepGEMM fp8_einsum("bhr,hdr->bhd") '
                "with a (1, 1, 32) MXFP8 recipe and an FP8 output tuple (e4m3 + UE8M0 "
                "per 32) for the next linear. One sm100_fp8_fp4_gemm_1d1d launch."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
