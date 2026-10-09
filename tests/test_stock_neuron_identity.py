"""Numerical bytes, warm loader scope and legacy HTTP provenance regressions."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from profiling.runners.neuron import vllm_identity as identity


def write(root, name, value):
    (root / name).write_text(json.dumps(value))


def loader_log(keys, *, source="backend.py", ranks=range(4)):
    return "\n".join(
        f"(Worker_TP{rank} pid=12) INFO - {source}:241 - "
        f"Compilation cache key: {key}\n"
        f"(Worker_TP{rank} pid=12) INFO - {source}:241 - Local cache hit for key: {key}"
        for key in keys for rank in ranks
    )


@pytest.fixture
def numeric_run(tmp_path):
    root = tmp_path / "run"
    cache = tmp_path / "cache"
    root.mkdir()
    cases, rows = [], []
    for number, (phase, bucket) in enumerate((("prefill", 512), ("decode", 1), ("decode", 16))):
        key = f"{number + 1:032x}"
        folder = cache / key
        folder.mkdir(parents=True)
        (folder / f"graph_{key}.neff").write_bytes(f"original-{key}".encode())
        slot = ("Shape: (6782, 2, 32, 128)\n Dtype: bfloat16" if phase == "prefill"
                else f"Shape: ({bucket}, 16)\n Dtype: int32")
        metadata = (f"Input 0:\n Shape: ({bucket},)\nInput 5:\n {slot}\nInput 6:\n"
                    " Shape: (6782, 2, 32, 128)\n Dtype: bfloat16")
        (folder / "example_inputs.txt").write_text(metadata)
        if phase == "decode":
            case = {"id": f"b{bucket}", "shape_id": f"b{bucket}-s504", "bucket": bucket,
                    "prompts": [[1, 2]]}
            cases.append(case)
            row = {"case_id": f"b{bucket}-slot0", "shape_id": case["shape_id"],
                   "prompt_ids": [1, 2], "token_ids": [3] * 8}
            rows.append(row)
            (root / f"native-{row['case_id']}.npy").write_bytes(b"saved-vector-fixture")
    snapshot = identity.compile_cache_snapshot(cache)
    for name in ("before-load", "before", "after"):
        write(root, f"accuracy-binaries-{name}.json", snapshot)
    source = {"loader": identity.LOADER_SOURCE_HASHES, "producer": {"engine": "same-source"}}
    for name in ("before", "after"):
        write(root, f"accuracy-source-{name}.json", source)
    write(root, "plan.json", {"context": 512, "identity": {"model_sha256": identity.MODEL_HASHES}})
    write(root, "cases.json", cases)
    write(root, "accuracy-outputs.json", rows)
    write(root, "accuracy-config.json", {"model": "immutable-checkpoint"})
    write(root, "precision.json", {
        "passed": True, "by_shape": {r["shape_id"]: {"passed": True} for r in rows},
        "criterion": "vendor",
    })
    (root / "accuracy.log").write_text(loader_log(snapshot))
    return root, cache, snapshot


def test_same_key_neff_replacement_rejected(numeric_run):
    root, cache, _ = numeric_run
    identity.seal_accuracy_binaries(root)
    proof = identity.load_accuracy_binaries(root)
    assert proof["available"]
    identity.verify_cache_binaries(cache, proof["graphs"])
    key = next(iter(proof["graphs"]))
    (cache / key / f"graph_{key}.neff").write_bytes(b"different compiler binary, same key")
    with pytest.raises(ValueError, match="NEFF differs"):
        identity.verify_cache_binaries(cache, proof["graphs"])
    # The numerical acquisition remains immutable even after the live cache changes.
    assert identity.load_accuracy_binaries(root)["available"]


@pytest.mark.parametrize("field,value", [("phase", "decode"), ("token_bucket", 128)])
def test_receipt_geometry_is_recomputed(numeric_run, field, value):
    root, _, _ = numeric_run
    receipt = identity.seal_accuracy_binaries(root)
    next(iter(receipt["graphs"].values()))[field] = value
    write(root, "binary-provenance.json", receipt)
    with pytest.raises(ValueError, match="differs from acquisition"):
        identity.load_accuracy_binaries(root)


@pytest.mark.parametrize(
    "source,ranks", [("capture_backend.py", range(4)), ("backend.py", range(3))],
)
def test_capture_only_or_missing_rank_is_not_loaded_binary_proof(numeric_run, source, ranks):
    root, _, snapshot = numeric_run
    (root / "accuracy.log").write_text(loader_log(snapshot, source=source, ranks=ranks))
    identity.seal_accuracy_binaries(root)
    assert identity.load_accuracy_binaries(root)["available"] is False


def test_cold_load_is_explicitly_unproven_without_breaking_profile(numeric_run):
    root, _, _ = numeric_run
    write(root, "accuracy-binaries-before-load.json", {})
    identity.seal_accuracy_binaries(root)
    assert identity.load_accuracy_binaries(root)["scope"] == "cold_load_binary_binding_unproven"


@pytest.mark.parametrize("name", ["accuracy-outputs.json", "native-b1-slot0.npy", "accuracy.log"])
def test_frozen_acquisition_mutation_rejected(numeric_run, name):
    root, _, _ = numeric_run
    identity.seal_accuracy_binaries(root)
    with (root / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="file changed"):
        identity.load_accuracy_binaries(root)


def test_numerical_initialization_binary_mutation_rejected(numeric_run):
    root, _, snapshot = numeric_run
    key = next(iter(snapshot))
    snapshot[key]["neff_sha256"] = "changed"
    write(root, "accuracy-binaries-after.json", snapshot)
    with pytest.raises(ValueError, match="compiled binary changed"):
        identity.seal_accuracy_binaries(root)


def test_checkpoint_weight_mutation_without_config_change_rejected(tmp_path, monkeypatch):
    values = {"config.json": b"unchanged-config", "weights.safetensors": b"original-weights"}
    for name, data in values.items():
        (tmp_path / name).write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in values.items()}
    monkeypatch.setattr(identity, "MODEL_HASHES", hashes)
    assert identity.verify_checkpoint(tmp_path) == hashes
    (tmp_path / "weights.safetensors").write_bytes(b"different-weights")
    with pytest.raises(ValueError, match="checkpoint/config/tokenizer"):
        identity.verify_checkpoint(tmp_path)


def test_fresh_accuracy_only_http_authority_and_legacy_refusal(numeric_run, tmp_path):
    from alignment.neuron import vllm_runner as runner
    root, cache, snapshot = numeric_run
    identity.seal_accuracy_binaries(root)
    evidence = runner.accepted_forward_evidence(root)
    cfg = SimpleNamespace(server=SimpleNamespace(cache_path=str(cache.parent)))
    # Configured cache root is <root>/cache/neuron/compile_cache; use the real helper separately.
    assert set(evidence["graphs"]) == {"prefill:512", "decode:1", "decode:16"}
    assert evidence["binary_provenance"]["available"]
    with pytest.raises(ValueError, match="requires producer-recorded"):
        runner.capture_binary_snapshot(cfg, {"binary_provenance": {"available": False}})
    capture = tmp_path / "capture"
    capture.mkdir()
    assert runner.captured_binary_provenance(capture, evidence)["available"] is False
    for name in ("before", "after"):
        write(capture, f"capture-binaries-{name}.json", snapshot)
    write(capture, "checkpoint-provenance.json", {"model_sha256": identity.MODEL_HASHES})
    (capture / "server.log").write_text(loader_log(snapshot))
    assert runner.captured_binary_provenance(capture, evidence)["available"]
    (capture / "server.log").write_text(loader_log(snapshot, ranks=[0, 1, 2]))
    with pytest.raises(ValueError, match="four rank"):
        runner.captured_binary_provenance(capture, evidence)


def test_legacy_receipt_does_not_gain_binary_proof(tmp_path):
    assert identity.load_accuracy_binaries(tmp_path) == {
        "available": False, "scope": "legacy_graph_keys_only",
    }
