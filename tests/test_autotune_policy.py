"""The autotune policy (no GPU): who decides VIBESIM_AUTOTUNE, the two preflights,
the in-process guard, and how a forbidden tune ends a run.

Defect these guard against: a measurement that tunes kernel tactics inside
itself (a TP4 vLLM capture spent 8.5 min in FlashInfer's NVFP4 MoE search, and
an L1 refresh re-derived the tactics its row was measured under).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from alignment import warm_caches
from alignment.profiler.config import PythonRuntimeConfig, ServerConfig
from profiling.db import KernelKind, Table, run_profile_batch
from profiling.db.batch import coerce_args
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec import ChunkResult, GpuChunk, GpuPool, local
from profiling.exec.local import LocalGpuChunk, autotune_preflight
from profiling.runners import tactic_labels
from profiling.runners.autotune_cache import tactic_file_stem
from profiling.runners.metrics import ComputeMetrics
from reference_engines import autotune_guard as guard
from reference_engines.autotune_guard import AutotuneForbidden, AutotuneForbiddenError

# --- policy table -------------------------------------------------------------


def test_l1_policy_allows_jit_fill_and_forbids_refresh_unless_allowed() -> None:
    assert local.autotune_policy() == "allow"  # jit-fill: tuning a new shape fills it
    with local.refreshing_rows():
        assert local.autotune_policy() == "forbid"  # --force / --fresh
        local.set_allow_autotune(True)
        try:
            assert local.autotune_policy() == "allow"  # ... --allow-autotune
        finally:
            local.set_allow_autotune(False)
    assert local.autotune_policy() == "allow"


@pytest.mark.parametrize(
    ("allow_autotune", "warm", "policy"),
    [(False, False, "forbid"), (True, False, "allow"), (False, True, "allow")],
)
def test_capture_policy(allow_autotune: bool, warm: bool, policy: str) -> None:
    assert warm_caches.capture_policy(allow_autotune=allow_autotune, warm_caches=warm) == policy


# --- in-process guard -----------------------------------------------------------


def _hooked(monkeypatch) -> type:
    """A fresh tuner class with both kinds of hooked method (plain, classmethod)."""

    class AutoTuner:
        def _profile_single_kernel(self, runner, inputs, tactic):
            return f"measured {tactic}"

        @classmethod
        def benchmark_choices(cls, choices):
            return len(choices)

    monkeypatch.setattr(guard, "_TUNED", {})
    assert guard._wrap(AutoTuner, "_profile_single_kernel", lambda *a, **k: {"tactic": a[3]}, "fi")
    assert guard._wrap(
        AutoTuner, "benchmark_choices", lambda _c, choices: {"n": len(choices)}, "ind"
    )
    return AutoTuner


@pytest.mark.parametrize("policy", [None, "forbid"])
def test_a_tactic_measurement_raises_unless_allowed(monkeypatch, capsys, policy) -> None:
    tuner = _hooked(monkeypatch)
    if policy is None:
        monkeypatch.delenv(guard.POLICY_ENV, raising=False)  # unset is forbid
    else:
        monkeypatch.setenv(guard.POLICY_ENV, policy)
    monkeypatch.delenv(guard.PREFLIGHT_ENV, raising=False)

    with pytest.raises(AutotuneForbidden) as raised:
        tuner()._profile_single_kernel(object(), [], 7)
    with pytest.raises(AutotuneForbidden):
        tuner.benchmark_choices([1, 2])

    # A BaseException: FlashInfer's per-tactic `except Exception` cannot eat it.
    assert not isinstance(raised.value, Exception)
    assert guard.forbidden_line(capsys.readouterr().err) is not None
    assert "warm-caches" in str(raised.value)


def test_allowed_tuning_runs_and_is_tallied_per_op_not_per_tactic(monkeypatch) -> None:
    tuner = _hooked(monkeypatch)
    monkeypatch.setenv(guard.POLICY_ENV, "allow")

    assert tuner()._profile_single_kernel(object(), [], 3) == "measured 3"
    assert tuner()._profile_single_kernel(object(), [], 4) == "measured 4"

    (row,) = guard.tuned()
    assert (row["source"], row["measurements"]) == ("fi", 2)
    assert guard.tuned() == []  # read once


def test_a_tune_after_a_preflight_hit_is_reported_as_a_key_mismatch(monkeypatch) -> None:
    tuner = _hooked(monkeypatch)
    monkeypatch.setenv(guard.POLICY_ENV, "forbid")
    monkeypatch.setenv(guard.PREFLIGHT_ENV, "warm-caches manifest m.json")

    with pytest.raises(AutotuneForbidden, match="manifest/actual mismatch"):
        tuner()._profile_single_kernel(object(), [], 1)


def test_the_finder_hooks_a_module_when_it_is_first_imported(tmp_path, monkeypatch) -> None:
    (tmp_path / "fake_fi_autotuner.py").write_text(
        "class AutoTuner:\n"
        "    def _profile_single_kernel(self, runner, inputs, tactic):\n"
        "        return tactic\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(guard._PATCHES, "fake_fi_autotuner", guard._flashinfer)
    monkeypatch.setattr(sys, "meta_path", [guard._Finder(), *sys.meta_path])
    monkeypatch.delitem(sys.modules, "fake_fi_autotuner", raising=False)
    monkeypatch.setenv(guard.POLICY_ENV, "forbid")

    import fake_fi_autotuner  # noqa: PLC0415 - the import is the thing under test

    with pytest.raises(AutotuneForbidden):
        fake_fi_autotuner.AutoTuner()._profile_single_kernel(object(), [], 0)


# --- L1 preflight ---------------------------------------------------------------

_MOE = {
    "num_tokens": 64,
    "hidden_size": 4096,
    "intermediate_size": 2048,
    "num_experts": 256,
    "num_local_experts": 32,
    "routing_method": "deepseek_v3",
    "backend": "flashinfer_trtllm_sm100",
}


def test_l1_preflight_finds_each_specs_tactic_file(tmp_path) -> None:
    spec = find_kernel_profiler_spec("nvfp4_fused_moe", "flashinfer_trtllm_sm100")
    label = tactic_labels.nvfp4_fused_moe_vllm(**_MOE)
    (tmp_path / f"{tactic_file_stem(label, 'NVIDIA B200')}.fi0-7-3.json").write_text("{}")

    found = autotune_preflight(
        "nvfp4_fused_moe", spec, [_MOE], tactic_dir=tmp_path, device="NVIDIA B200"
    )

    assert found is not None and "fi0-7-3" in found


def test_l1_preflight_lists_every_spec_without_tactics(tmp_path) -> None:
    spec = find_kernel_profiler_spec("nvfp4_fused_moe", "flashinfer_trtllm_sm100")
    other = {**_MOE, "num_local_experts": 64}

    with pytest.raises(AutotuneForbiddenError, match="2 of 2 spec") as raised:
        autotune_preflight(
            "nvfp4_fused_moe", spec, [_MOE, other], tactic_dir=tmp_path, device="NVIDIA B200"
        )
    assert "-l32-" in str(raised.value) and "-l64-" in str(raised.value)


def test_l1_preflight_refuses_a_row_that_retunes_in_every_worker(tmp_path) -> None:
    spec = find_kernel_profiler_spec("kda_chunk_prefill", "vllm_triton")
    assert spec.tunes_every_worker

    with pytest.raises(AutotuneForbiddenError, match="AutotunePin"):
        autotune_preflight("kda_chunk_prefill", spec, [{}], tactic_dir=tmp_path, device="x")


def test_l1_preflight_passes_a_row_that_keeps_no_tactic_file(tmp_path) -> None:
    spec = find_kernel_profiler_spec("single_gemm", "torch")
    assert autotune_preflight("single_gemm", spec, [{}], tactic_dir=tmp_path, device="x") is None


# --- a forbidden tune ends the L1 run, and keeps nothing --------------------------


def test_a_worker_stopped_by_the_guard_fails_the_run(monkeypatch) -> None:
    monkeypatch.setenv("VIBESIM_PROFILE_RUNTIME", "host")
    marker = f'{guard.MARKER} {{"source": "triton.autotune"}} -- hint'

    def worker(cmd, **kwargs):
        assert kwargs["env"][guard.POLICY_ENV] == "allow"  # jit-fill
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=marker)

    monkeypatch.setattr("profiling.exec.local.subprocess.run", worker)
    with pytest.raises(AutotuneForbiddenError, match="triton.autotune"):
        LocalGpuChunk([0]).run(
            "single_gemm", [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"}]
        )


class _GoodChunk(GpuChunk):
    def run(self, kernel_kind: KernelKind, specs: list[dict]) -> list[ChunkResult]:
        metrics = ComputeMetrics(time_ms=1.0, tflops=0.1, memory_bandwidth_gbps=0.2, energy_j=0.0)
        return [ChunkResult(metrics=metrics, observed_gpu_name="NVIDIA H200") for _ in specs]


class _ForbiddenChunk(GpuChunk):
    def run(self, kernel_kind: KernelKind, specs: list[dict]) -> list[ChunkResult]:
        raise AutotuneForbiddenError("worker tuned")


class _HalfForbiddenPool(GpuPool):
    def acquire_chunks(self, k: int, max_concurrent: int):
        yield _GoodChunk()
        yield _ForbiddenChunk()


def test_a_forbidden_tune_persists_no_row_not_even_its_neighbours(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("profiling.db.batch.require_gpu", lambda _why: None)
    db = tmp_path / "profile.db"
    specs = [{"m": m, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"} for m in (8, 16)]

    with pytest.raises(AutotuneForbiddenError):
        run_profile_batch(
            "single_gemm", specs, pool=_HalfForbiddenPool(), db_path=db, gpu_name="NVIDIA H200"
        )

    profiler_spec = find_kernel_profiler_spec("single_gemm", "torch")
    args = [
        coerce_args(profiler_spec.args_schema, {k: v for k, v in s.items() if k != "backend"})
        for s in specs
    ]
    stored = Table(profiler_spec, db).rows_for(args, backend="torch", gpu_name="NVIDIA H200")
    assert stored == [None, None]


# --- capture preflight ------------------------------------------------------------


@dataclass
class _Cfg:
    engine: str
    server: ServerConfig
    cuda_visible_devices: str = "0,1,2,3"
    python_runtime: PythonRuntimeConfig | None = None
    name: str = "smoke"
    log_dir: str = "/tmp/x"


def _cfg(tmp_path: Path, **server) -> _Cfg:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    return _Cfg(
        "vllm",
        ServerConfig(model_path=str(model), tp_size=4, **server),
        python_runtime=PythonRuntimeConfig(packages=[], environment={"A": "1"}),
    )


def test_capture_preflight_admits_the_warmed_deployment_only(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "home").mkdir()
    before = warm_caches.snapshot(cache)
    (cache / "home" / "tactics.json").write_text("{}")
    warmed = _cfg(tmp_path)
    written = warm_caches.write_manifest(
        warmed,
        tmp_path / "p.yaml",
        cache,
        "vllm-abc-sm100",
        image="img",
        image_id="sha256:1",
        before=before,
        seconds=1.0,
    )
    record = json.loads(written.read_text())
    assert record["files_produced"] == ["home/tactics.json"]

    # The same deployment on another port is admitted ...
    assert (
        warm_caches.preflight(
            _cfg(tmp_path, port=9000), tmp_path / "p.yaml", cache, "vllm-abc-sm100"
        )
        == written
    )
    # ... another chunk size, or another GPU architecture, is not.
    with pytest.raises(AutotuneForbiddenError, match="server.chunk_size") as raised:
        warm_caches.preflight(
            _cfg(tmp_path, chunk_size=4096), tmp_path / "p.yaml", cache, "vllm-abc-sm100"
        )
    assert "just ref warm-caches vllm" in str(raised.value)
    with pytest.raises(AutotuneForbiddenError):
        warm_caches.preflight(warmed, tmp_path / "p.yaml", cache, "vllm-abc-sm103")


def test_capture_preflight_without_any_manifest_names_warm_caches(tmp_path) -> None:
    with pytest.raises(AutotuneForbiddenError, match="warm-caches"):
        warm_caches.preflight(_cfg(tmp_path), tmp_path / "p.yaml", tmp_path, "k")


# --- the persistent-cache key --------------------------------------------------------


def test_an_engine_image_is_keyed_by_base_and_gpu_arch(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        if cmd[0] == "docker":
            return subprocess.CompletedProcess(cmd, 0, stdout="vllm-a3a232b8e67f\n", stderr="")
        assert cmd[0] == "nvidia-smi" and "--query-gpu=compute_cap" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="10.0\n10.0\n", stderr="")

    monkeypatch.setattr("profiling.exec.local.subprocess.run", fake_run)
    monkeypatch.setattr("profiling.exec.local.require_gpu", lambda _why: None)

    key = local.image_cache_key("vibesim-profiler-vllm:1234", ["GPU-a", "GPU-b"])

    assert key == "vllm-a3a232b8e67f-sm100"


def test_an_unlabelled_image_keeps_one_cache_per_tag(monkeypatch) -> None:
    monkeypatch.setattr(
        "profiling.exec.local.subprocess.run",
        lambda cmd, **_: subprocess.CompletedProcess(cmd, 0, stdout="<no value>\n", stderr=""),
    )
    assert (
        local.image_cache_key("vibesim-profiler-default:abcd", ["GPU-a"])
        == "vibesim-profiler-default_abcd"
    )


def test_engine_images_stamp_the_base_half_of_the_key() -> None:
    from profiling.container.images import IMAGES
    from reference_engines import host_platform, read_base

    for engine in ("vllm", "sglang"):
        digest = read_base(engine).digests[host_platform()].removeprefix("sha256:")
        assert IMAGES[engine].build_args(local._PROJECT_ROOT)["VIBESIM_CACHE_KEY"] == (
            f"{engine}-{digest[:12]}"
        )
        assert "reference_engines/autotune_guard.py" in IMAGES[engine].inputs
