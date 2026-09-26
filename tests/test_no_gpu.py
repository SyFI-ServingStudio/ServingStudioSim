"""SERVINGSTUDIO_NO_GPU / `--no-gpu`: every GPU entry point refuses, the rest runs."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from profiling.gpu_policy import NO_GPU_ENV, GpuDisabledError, gpu_disabled


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setenv(NO_GPU_ENV, "1")


def _untouchable(*args, **kwargs):
    pytest.fail("a GPU path ran under SERVINGSTUDIO_NO_GPU")


@pytest.mark.parametrize(
    ("value", "disabled"),
    [("", False), ("0", False), ("false", False), ("Off", False), ("1", True), ("yes", True)],
)
def test_the_variable_reads_off_values_as_unset(monkeypatch, value, disabled):
    monkeypatch.setenv(NO_GPU_ENV, value)

    assert gpu_disabled() is disabled


def test_profiling_a_batch_is_refused(no_gpu):
    from profiling.db.batch import execute_profile_batch

    pool = SimpleNamespace(acquire_chunks=_untouchable)
    spec = {"m": 1, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"}

    with pytest.raises(GpuDisabledError, match="profiling 1 single_gemm spec"):
        execute_profile_batch("single_gemm", [spec], pool=pool, db_path=None, gpu_name="H200")


def test_a_cache_fill_is_refused_only_when_something_is_missing(no_gpu, monkeypatch):
    import profiling.exec.local as local
    import profiling.perf_api as perf_api
    from profiling.plan import active_collector

    monkeypatch.setattr(local, "find_idle_gpus", _untouchable)
    perf_api.begin_collect()
    perf_api.issue_collected(0)  # a warm cache still passes

    perf_api.begin_collect()
    active_collector().record("single_gemm", "torch_linear", [{"m": 1, "n": 8, "k": 8}])
    with pytest.raises(GpuDisabledError, match="1 missing profile.db spec"):
        perf_api.issue_collected(1)


def test_a_forced_gpu_set_and_nvidia_smi_are_refused(no_gpu, monkeypatch):
    import profiling.exec.local as local

    monkeypatch.setattr(local.subprocess, "run", _untouchable)

    with pytest.raises(GpuDisabledError):
        next(local.LocalGpuPool(gpus=[0]).acquire_chunks(1))
    with pytest.raises(GpuDisabledError):
        local.find_idle_gpus()


def test_a_gpu_name_is_never_detected_from_cuda(no_gpu):
    from profiling.facade import _resolve_gpu_name

    assert _resolve_gpu_name("NVIDIA H200") == "NVIDIA H200"
    with pytest.raises(ValueError, match=NO_GPU_ENV):
        _resolve_gpu_name(None)


def test_an_alignment_capture_is_refused_before_any_setup(no_gpu):
    from alignment.runner import run_profile

    config = SimpleNamespace(profile_kind="nsys", engine="vllm")

    with pytest.raises(GpuDisabledError, match="alignment nsys capture"):
        run_profile(config)


@pytest.mark.parametrize("flag", [True, False])
def test_the_launcher_hides_every_device_for_its_children(monkeypatch, flag):
    from launcher import __main__ as launcher_main
    from launcher import timing_predict

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    if flag:
        monkeypatch.delenv(NO_GPU_ENV, raising=False)
    else:
        monkeypatch.setenv(NO_GPU_ENV, "1")
    seen = []
    monkeypatch.setattr(timing_predict, "main", lambda argv: seen.append(argv) or 0)

    argv = ["timing-predict", "predict.json"] + (["--no-gpu"] if flag else [])
    assert launcher_main.main(argv) == 0

    assert seen == [["predict.json"]]
    assert os.environ[NO_GPU_ENV] == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_a_cache_prebuild_with_missing_rows_fails_without_starting_a_builder(
    no_gpu, monkeypatch, tmp_path, capsys
):
    from launcher import cache_build

    async def probe(*args, **kwargs):
        return 7

    monkeypatch.setattr(cache_build, "_unique_by_cache_key", lambda params, registry: params)
    monkeypatch.setattr(cache_build, "_prebuild_log_dir", lambda base, config: tmp_path)
    monkeypatch.setattr(cache_build, "build_cli_command", lambda *args, **kwargs: ["build"])
    monkeypatch.setattr(cache_build, "_probe_missing", probe)
    monkeypatch.setattr(cache_build._PROCESS_SUPERVISOR, "run", _untouchable)

    ok = asyncio.run(cache_build.prebuild_caches([{}], None, base_dir=tmp_path))

    assert ok is False
    assert "filling 7 missing profile.db spec(s) needs a GPU" in capsys.readouterr().err
