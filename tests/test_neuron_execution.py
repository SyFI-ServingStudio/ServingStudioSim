"""Protect hardware routing and resource ownership without a Neuron SDK."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from profiling.db.args import DType
from profiling.db.registry import BackendSupport
from profiling.exec import neuron
from profiling.gpu_policy import GpuDisabledError


def test_backend_family_prevents_cuda_neuron_cross_selection():
    cuda = BackendSupport(compute=frozenset({DType.BF16}))
    nki = BackendSupport(
        compute=frozenset({DType.BF16}),
        device_family="neuron",
        architectures=frozenset({"Trainium2"}),
    )
    assert not cuda.allows(DType.BF16, gpu=neuron.TRAINIUM2_LNC2)
    assert nki.allows(DType.BF16, gpu=neuron.TRAINIUM2_LNC2)
    assert not nki.allows(DType.BF16, gpu="NVIDIA H200")
    assert not nki.allows(DType.BF16, gpu="unrecognized accelerator")
    assert not nki.allows(DType.FP16, gpu=neuron.TRAINIUM2_LNC2)


def test_no_gpu_blocks_neuron_discovery_before_subprocess(monkeypatch):
    monkeypatch.setenv("SERVINGSTUDIO_NO_GPU", "1")
    monkeypatch.setattr(neuron.subprocess, "run", lambda *a, **k: pytest.fail("device access"))
    with pytest.raises(GpuDisabledError):
        neuron.neuron_devices()


def test_device_identity_comes_from_neuron_ls_not_requested_label(monkeypatch):
    monkeypatch.delenv("SERVINGSTUDIO_NO_GPU", raising=False)
    row = {
        "instance_type": "trn2.3xlarge",
        "neuron_device": 0,
        "neuroncore_ids": [0, 1, 2, 3],
        "logical_neuroncore_config": 2,
        "memory_size": 96 * 1024**3,
        "neuron_processes": [],
    }
    monkeypatch.setattr(
        neuron.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps([row]))
    )
    assert neuron.current_neuron_name() == neuron.TRAINIUM2_LNC2
    row["instance_type"] = "trn3.3xlarge"
    with pytest.raises(RuntimeError, match="unsupported Neuron target"):
        neuron.current_neuron_name()


def test_busy_chip_is_never_reserved(monkeypatch):
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    monkeypatch.setattr(
        neuron,
        "neuron_devices",
        lambda: [neuron.NeuronDevice(0, (0, 1, 2, 3), 2, 96 * 1024**3, True, "trn2.3xlarge")],
    )
    with pytest.raises(RuntimeError, match="no idle"):
        list(neuron.LocalNeuronPool().acquire_chunks(1))


@pytest.mark.parametrize("visibility", ["NEURON_RT_VISIBLE_CORES", "NEURON_VISIBLE_DEVICES"])
def test_pool_cannot_override_inherited_core_visibility(monkeypatch, visibility):
    monkeypatch.setenv(visibility, "2")
    monkeypatch.setattr(neuron, "neuron_devices", lambda: [])
    with pytest.raises(RuntimeError, match="owns allocation"):
        neuron.LocalNeuronPool().idle_devices()


def test_chip_reservation_is_shared_across_workspace_tmpdirs(monkeypatch, tmp_path):
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    device = neuron.NeuronDevice(0, (0, 1, 2, 3), 2, 96 * 1024**3, False, "trn2.3xlarge")
    monkeypatch.setattr(neuron, "neuron_devices", lambda: [device])
    monkeypatch.setattr(neuron, "NEURON_LOCK_DIR", tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "workspace-a"))
    first = list(neuron.LocalNeuronPool().acquire_chunks(1))[0]
    try:
        monkeypatch.setenv("TMPDIR", str(tmp_path / "workspace-b"))
        with pytest.raises(RuntimeError, match="no idle"):
            list(neuron.LocalNeuronPool().acquire_chunks(1))
    finally:
        first.lock.close()
    second = list(neuron.LocalNeuronPool().acquire_chunks(1))[0]
    second.lock.close()


def test_rejected_chunk_releases_chip_reservation(monkeypatch, tmp_path):
    monkeypatch.delenv("SERVINGSTUDIO_NO_GPU", raising=False)
    from profiling.profilers import energy

    monkeypatch.setattr(energy, "energy_enabled", lambda: False)
    device = neuron.NeuronDevice(0, (0, 1, 2, 3), 2, 96 * 1024**3, False, "trn2.3xlarge")
    lock = (tmp_path / "reservation").open("a")
    chunk = neuron.LocalNeuronChunk(device, lock)
    with pytest.raises(ValueError, match="not a Neuron backend"):
        chunk.run("single_gemm", [{"backend": "torch"}])
    assert lock.closed
