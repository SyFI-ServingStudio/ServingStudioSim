"""CPU contract tests for the FlashInfer TRT-LLM fused TP boundary."""

from __future__ import annotations

import subprocess
import sys

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.all_reduce_residual_rms_norm import (
    KIND,
    AllReduceResidualRmsNormArgs,
)


def test_args_field_contract_matches_runner_kwargs():
    assert list(AllReduceResidualRmsNormArgs.__dataclass_fields__) == [
        "num_gpus",
        "num_tokens",
        "hidden_dim",
        "dtype",
        "fabric",
        "strategy",
        "launch_with_pdl",
        "trigger_completion_at_end",
        "fp32_acc",
    ]


def test_registry_contract_and_capability_gate():
    spec = find_kernel_profiler_spec(KIND, "flashinfer_trtllm")
    assert spec.table_name == KIND
    assert spec.args_schema is AllReduceResidualRmsNormArgs
    assert spec.subprocess_env == "vllm_env"
    assert spec.list_native is True
    assert spec.gpu_count_fn is not None
    assert spec.gpu_count_fn({"num_gpus": 4}) == 4
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA B200")


def test_kernel_import_is_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels.all_reduce_residual_rms_norm; "
                "print('profiling.runners.comm.flashinfer_trtllm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"
