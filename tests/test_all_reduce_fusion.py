"""CPU contract tests for FlashInfer's shape-aware standalone all-reduce."""

from __future__ import annotations

import subprocess
import sys

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.all_reduce_fusion import KIND, AllReduceFusionArgs


def test_args_field_contract_matches_runner_kwargs():
    assert list(AllReduceFusionArgs.__dataclass_fields__) == [
        "num_gpus",
        "num_tokens",
        "hidden_dim",
        "dtype",
        "fabric",
    ]


def test_registry_contract_and_capability_gate():
    spec = find_kernel_profiler_spec(KIND, "flashinfer_trtllm")
    assert spec.table_name == KIND
    assert spec.args_schema is AllReduceFusionArgs
    assert spec.subprocess_env == "flashinfer_pip_env"
    assert spec.list_native is True
    assert spec.gpu_count_fn is not None
    assert spec.gpu_count_fn({"num_gpus": 4}) == 4
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA H200")


def test_kernel_import_is_lazy():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels.all_reduce_fusion; "
                "print('profiling.runners.comm.flashinfer_trtllm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_mnnvl_registry_row_runs_in_production_flashinfer_env():
    # Catches the row silently falling back to the project venv's older FlashInfer.
    spec = find_kernel_profiler_spec(KIND, "flashinfer_mnnvl")
    assert spec.args_schema is AllReduceFusionArgs
    assert spec.subprocess_env == "vllm_fork_env"
    assert spec.list_native is True
    assert spec.gpu_count_fn({"num_gpus": 4}) == 4
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA B200")


def test_mnnvl_oneshot_rule_matches_flashinfer_threshold():
    # GLM-5.3-Flash decode (T=32) is one-shot; one more token flips to two-shot.
    from profiling.runners.comm.flashinfer_mnnvl import uses_oneshot

    assert uses_oneshot(4, 32, 4096, DType.BF16)
    assert not uses_oneshot(4, 33, 4096, DType.BF16)
    assert uses_oneshot(8, 16, 4096, DType.BF16)
    assert not uses_oneshot(8, 17, 4096, DType.BF16)


def test_mnnvl_batch_rejects_non_production_specs_and_forwards_the_rest(monkeypatch):
    # Catches measuring shapes vLLM never routes to FlashInfer (over the SM100
    # TP4 32 MiB budget, or off NVLink) and misaligned per-spec results.
    from profiling.runners.comm import flashinfer_mnnvl
    from profiling.runners.metrics import CommMetrics, RunnerResult

    def spec(num_tokens: int, fabric: str = "nvlink") -> dict:
        return {
            "num_gpus": 4,
            "num_tokens": num_tokens,
            "hidden_dim": 4096,
            "dtype": "bf16",
            "fabric": fabric,
        }

    forwarded: list[list[dict]] = []

    def fake_run_comm_batch(launcher, per_rank_fn, kwargs_list):
        forwarded.append(kwargs_list)
        return [
            RunnerResult(
                metrics=CommMetrics(time_ms=float(s["num_tokens"]), algbw_gbps=0.0, busbw_gbps=0.0)
            )
            for s in kwargs_list
        ]

    monkeypatch.setattr(flashinfer_mnnvl, "run_comm_batch", fake_run_comm_batch)
    results = flashinfer_mnnvl.profile_all_reduce_fusion_batch(
        [spec(32), spec(4097), spec(4096), spec(32, fabric="pcie")]
    )

    assert [s["num_tokens"] for s in forwarded[0]] == [32, 4096]
    assert results[0].metrics.time_ms == 32.0
    assert "budget" in results[1].error
    assert results[2].metrics.time_ms == 4096.0
    assert "NVLink" in results[3].error
