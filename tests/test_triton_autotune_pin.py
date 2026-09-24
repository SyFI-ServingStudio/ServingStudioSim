"""Autotune determinism for Triton-autotuned runners (no GPU, no Triton launch).

Background: vLLM sets TRITON_CACHE_AUTOTUNING=1 and the FLA/KDA autotune keys
omit the token count, so whichever row tuned first in a TRITON_CACHE_DIR fixed
the configs of every later row and process (b2-stab: 331 vs 364 vs 434 us at
one shape). These tests guard the pieces of the fix that run on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from profiling.db.registry import (
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    _validate_registry,
    find_kernel_profiler_spec,
)
from profiling.db.outlier import BatchOutlierPolicy
from profiling.exec import local as local_exec
from profiling.exec import local_worker
from profiling.exec.env import ContainerProfileEnv, ProfileEnv
from profiling.kernels.kda_chunk_prefill import KdaChunkPrefillArgs
from profiling.runners import triton_autotune_pin as pin_module
from profiling.runners.metrics import ComputeMetrics, RunnerResult


class _FakeTuner:
    def __init__(self, cache_results: bool = False) -> None:
        self.cache: dict = {}
        self.cache_results = cache_results


def _pin_with(monkeypatch: pytest.MonkeyPatch, tuners: list[tuple[str, _FakeTuner]]):
    monkeypatch.setattr(pin_module, "iter_autotuners", lambda _prefixes: tuners)
    return pin_module.AutotunePin(("fake",))


def test_anchor_tunes_once_and_later_rows_reuse_its_selection(monkeypatch) -> None:
    # Defect: a later row re-tuning (or tuning first) and so picking the configs.
    tuner = _FakeTuner()
    tuner.cache[("stale",)] = "left over from an earlier shape"
    pin = _pin_with(monkeypatch, [("k", tuner)])
    calls = []

    def tune() -> None:
        calls.append(1)
        tuner.cache[(16,)] = "anchor-config"

    note = pin.ensure((16, 128, "bf16"), "T2048/L2019/D29/H16", tune)
    assert pin.ensure((16, 128, "bf16"), "T2048/L2019/D29/H16", tune) == note
    assert calls == [1]
    assert tuner.cache == {(16,): "anchor-config"}  # the stale selection is gone
    assert note.startswith("anchor=T2048/L2019/D29/H16 configs=1:")
    assert pin.note((16, 128, "bf16")) == note and pin.note((8, 128, "bf16")) is None


def test_note_flags_a_selection_made_after_the_anchor(monkeypatch) -> None:
    # Defect: a row reaching a tuning key the anchor did not, tuning it at the
    # row's own shape, and still being stamped as anchor-tuned.
    tuner = _FakeTuner()
    pin = _pin_with(monkeypatch, [("k", tuner)])
    note = pin.ensure("h16", "a16", lambda: tuner.cache.update({(16,): "anchor"}))
    assert pin.note("h16") == note
    tuner.cache[(16, "other")] = "row-tuned"
    assert pin.note("h16").startswith(note + " retuned=2:")


def test_alternating_state_keys_restore_rather_than_retune(monkeypatch) -> None:
    # Defect: an H-less kernel keeping H=16's winner when an H=8 row follows, or
    # an H=16 row after H=8 running under H=8's selections.
    tuner = _FakeTuner()
    pin = _pin_with(monkeypatch, [("k", tuner)])
    tuned = []

    def tune_for(label: str):
        def tune() -> None:
            tuned.append(label)
            assert tuner.cache == {}
            tuner.cache[("shared",)] = label

        return tune

    pin.ensure("h16", "a16", tune_for("h16"))
    pin.ensure("h8", "a8", tune_for("h8"))
    assert tuner.cache == {("shared",): "h8"}
    pin.ensure("h16", "a16", tune_for("h16-again"))
    assert tuned == ["h16", "h8"]
    assert tuner.cache == {("shared",): "h16"}


def test_refuses_to_pin_while_autotune_results_persist(monkeypatch) -> None:
    # Defect: the worker_env not reaching the worker, so the anchor silently
    # loads a selection some earlier process wrote to TRITON_CACHE_DIR.
    pin = _pin_with(monkeypatch, [("k", _FakeTuner(cache_results=True))])
    with pytest.raises(RuntimeError, match="TRITON_CACHE_AUTOTUNING=0"):
        pin.ensure("h16", "a", lambda: None)


def test_digest_depends_on_selection_not_insertion_order() -> None:
    first, second = _FakeTuner(), _FakeTuner()
    first.cache.update({(1,): "a", (2,): "b"})
    second.cache.update({(2,): "b", (1,): "a"})
    assert pin_module.selection_digest([("k", first)]) == pin_module.selection_digest(
        [("k", second)]
    )
    second.cache[(2,)] = "c"
    assert pin_module.selection_digest([("k", first)]) != pin_module.selection_digest(
        [("k", second)]
    )


@pytest.mark.parametrize("kind", ["kda_chunk_prefill", "kda_recurrent_decode"])
def test_kda_rows_disable_autotune_persistence_and_record_provenance(kind: str) -> None:
    spec = find_kernel_profiler_spec(kind, "vllm_triton")
    assert dict(spec.worker_env)["TRITON_CACHE_AUTOTUNING"] == "0"
    assert spec.row_provenance_ref is not None
    assert spec.row_provenance_ref.module_name == spec.runner_ref.module_name


def test_host_worker_env_overrides_the_inherited_autotune_setting(monkeypatch) -> None:
    # Defect: the parent's (or vLLM's) TRITON_CACHE_AUTOTUNING=1 winning.
    monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "1")
    env_spec = ProfileEnv("plain", Path("/usr/bin/python3"))
    _cmd, env = local_exec._host_worker_command(
        env_spec,
        [3],
        Path("/tmp/in.json"),
        Path("/tmp/out.json"),
        worker_env={"TRITON_CACHE_AUTOTUNING": "0"},
    )
    assert env["TRITON_CACHE_AUTOTUNING"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"


def test_container_worker_receives_the_row_env(monkeypatch, tmp_path) -> None:
    # Defect: a container worker (fresh environment) silently tuning with
    # persistence on while host workers beside it do not.
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(local_exec, "gpu_uuids_for_indices", lambda gpus: ["GPU-x"])
    cmd, _env = local_exec._container_worker_command(
        ContainerProfileEnv("c", "image:tag"),
        [0],
        tmp_path,
        worker_env={"TRITON_CACHE_AUTOTUNING": "0"},
    )
    index = cmd.index("TRITON_CACHE_AUTOTUNING=0")
    assert cmd[index - 1] == "--env" and index < cmd.index("image:tag")


def test_registry_rejects_worker_env_that_overrides_backend_owned_vars() -> None:
    spec = KernelProfilerSpec(
        kernel_kind="kda_chunk_prefill",
        backend="bogus",
        runner_ref=RunnerRef("m", "f"),
        table_name="kda_chunk_prefill",
        args_schema=KdaChunkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        worker_env=(("CUDA_VISIBLE_DEVICES", "0"),),
    )
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        _validate_registry([spec])


def test_worker_appends_row_provenance_to_backend_version() -> None:
    # Defect: rows measured under different autotune states being indistinguishable.
    ok = RunnerResult(metrics=ComputeMetrics(1.0, 1.0, 1.0))
    failed = RunnerResult(error="boom")
    versions = {"cuda_version": "13.0", "backend_version": "0.1.dev1"}

    def note(**kwargs):
        return f"anchor=A configs=7:{kwargs['num_heads']}"

    stamped = local_worker._with_row_provenance(versions, note, {"num_heads": 16}, ok)
    assert stamped["backend_version"] == "0.1.dev1; anchor=A configs=7:16"
    assert local_worker._with_row_provenance(versions, note, {"num_heads": 16}, failed) is versions
    assert local_worker._with_row_provenance(versions, None, {}, ok) is versions
