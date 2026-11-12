"""CPU contract tests for the FlashInfer fused TP boundary (TRT-LLM and MNNVL)."""

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
    assert spec.subprocess_env == "flashinfer_pip_env"
    assert spec.list_native is True
    assert spec.gpu_count_fn is not None
    assert spec.gpu_count_fn({"num_gpus": 4}) == 4
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA B200")


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


def test_mnnvl_registry_row_runs_in_production_flashinfer_env():
    # vLLM's B200 default fused backend; measured in the vLLM fork env and only
    # under the SM100 budget its runner carries.
    spec = find_kernel_profiler_spec(KIND, "flashinfer_mnnvl")
    assert spec.args_schema is AllReduceResidualRmsNormArgs
    assert spec.subprocess_env == "vllm_env"
    assert spec.gpu_count_fn({"num_gpus": 4}) == 4
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA B300")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA H200")


def test_mnnvl_fused_batch_rejects_non_production_specs(monkeypatch):
    # vLLM never forces the MNNVL strategy, and never fuses above the budget.
    from profiling.runners.comm import flashinfer_mnnvl
    from profiling.runners.metrics import CommMetrics, RunnerResult

    def spec(num_tokens: int, strategy: str = "auto") -> dict:
        return {
            "num_gpus": 4,
            "num_tokens": num_tokens,
            "hidden_dim": 4096,
            "dtype": "bf16",
            "fabric": "nvlink",
            "strategy": strategy,
            "launch_with_pdl": True,
            "trigger_completion_at_end": num_tokens > 16,
            "fp32_acc": True,
        }

    forwarded: list[tuple] = []

    def fake_run_comm_batch(launcher, per_rank_fn, kwargs_list):
        forwarded.append((per_rank_fn, kwargs_list))
        return [
            RunnerResult(
                metrics=CommMetrics(time_ms=float(s["num_tokens"]), algbw_gbps=0.0, busbw_gbps=0.0)
            )
            for s in kwargs_list
        ]

    monkeypatch.setattr(flashinfer_mnnvl, "run_comm_batch", fake_run_comm_batch)
    results = flashinfer_mnnvl.profile_all_reduce_residual_rms_norm_batch(
        [spec(32), spec(32, strategy="oneshot"), spec(4097), spec(2048)]
    )

    per_rank_fn, kwargs_list = forwarded[0]
    assert per_rank_fn is flashinfer_mnnvl._fused_per_rank_batch
    assert [s["num_tokens"] for s in kwargs_list] == [32, 2048]
    assert results[0].metrics.time_ms == 32.0
    assert "strategy auto" in results[1].error
    assert "budget" in results[2].error
    assert results[3].metrics.time_ms == 2048.0
