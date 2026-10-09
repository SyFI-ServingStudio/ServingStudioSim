"""Lightweight stock checkpoint and numerical-run binary identity; no SDK imports."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path

MODEL_HASHES = {
    "model-00001-of-00004.safetensors": (
        "f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b"
    ),
    "model-00002-of-00004.safetensors": (
        "c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0"
    ),
    "model-00003-of-00004.safetensors": (
        "d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383"
    ),
    "model-00004-of-00004.safetensors": (
        "e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d"
    ),
    "config.json": ("54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd"),
    "tokenizer.json": ("76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa"),
    "tokenizer_config.json": ("8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a"),
    "model.safetensors.index.json": (
        "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b"
    ),
}

# Pinned lite loader: get_local -> _build_cache_artifact -> build_executable
# passes cache/<key>/graph_<key>.neff directly to torch.classes.neuron.Executor.
LOADER_SOURCE_HASHES = {
    "libtorch_neuronx_lite.envs":
        "b9921ee562755006084b88eca7ef5e1983c64665b5fb8412e602423eb13d3d97",
    "libtorch_neuronx_lite.compile.backend":
        "de5d7816c35d72ab6a4bd1ca06e01136908a7f40a17a1cb077f10b206ded63ea",
    "libtorch_neuronx_lite.compile.cache":
        "466337a4bd516b05ca499a0412222847acf8f5127133c736f3ccc3e7ada6dd46",
}


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint(model: Path) -> dict[str, str]:
    observed = {name: sha256(model / name) for name in MODEL_HASHES}
    if observed != MODEL_HASHES:
        raise ValueError("checkpoint/config/tokenizer differs from pinned Llama3.1-8B")
    return observed


def stock_compile_cache_dir() -> Path:
    """Use the same public environment resolver as the executable loader."""
    from libtorch_neuronx_lite.envs import get_neuron_compile_cache_dir

    return Path(get_neuron_compile_cache_dir()).resolve()


def accuracy_source_identity() -> dict:
    """Called only inside the stock SDK worker, never by the host adapter."""
    loader = {
        name: sha256(Path(importlib.import_module(name).__file__))
        for name in LOADER_SOURCE_HASHES
    }
    if loader != LOADER_SOURCE_HASHES:
        raise ValueError("stock cache/Executor loader differs from pinned source")
    source = Path(__file__).parent
    return {
        "loader": loader,
        "compile_cache_dir": str(stock_compile_cache_dir()),
        "producer": {name: sha256(source / name) for name in (
            "vllm_identity.py", "vllm_forward_engine.py", "vllm_forward_reference.py",
        )},
    }


def warm_loader_coverage(log: str, keys) -> dict:
    """Require executable-loader hits, not capture-only log messages or substrings."""
    hits = re.findall(
        r"\(Worker_TP([0-3]) pid=\d+\)[^\n]*(?<![\w])backend\.py:\d+[^\n]*"
        r"Local cache hit for key: ([0-9a-f]{32})(?:\s|$)", log,
    )
    observed = {key: sorted({int(rank) for rank, item in hits if item == key}) for key in keys}
    return {"complete": all(ranks == [0, 1, 2, 3] for ranks in observed.values()),
            "ranks_by_graph": observed}


def compile_cache_snapshot(cache: Path) -> dict:
    """Hash the canonical files passed by the lite backend to its Executor.

    A cache key describes compilation inputs, not the NEFF bytes. Record the
    bytes separately. Runtime model-name suffixes are opaque and not decoded.
    """
    result = {}
    for neff in sorted(cache.glob("*/graph_*.neff")):
        key = neff.parent.name
        if neff.name != f"graph_{key}.neff":
            raise ValueError("unexpected stock cache NEFF filename")
        metadata = neff.parent / "example_inputs.txt"
        result[key] = {
            "neff_sha256": sha256(neff), "neff_bytes": neff.stat().st_size,
            "input_sha256": sha256(metadata), "input_metadata": metadata.read_text(),
        }
    return result


def verify_cache_binaries(cache: Path, graphs: dict) -> dict:
    observed = compile_cache_snapshot(cache)
    for key, expected in graphs.items():
        if observed.get(key) != expected["binary"]:
            raise ValueError(f"compiled NEFF differs from numerical validation under key {key}")
    return {key: observed[key] for key in graphs}


def _accuracy_graphs(root: Path) -> tuple[dict, dict]:
    """Reconstruct the receipt from acquisition files, including geometry and load scope."""
    from profiling.runners.neuron.vllm_forward_trace import model_geometry

    plan = json.loads((root / "plan.json").read_text())
    if plan["identity"]["model_sha256"] != MODEL_HASHES:
        raise ValueError("numerical run has different checkpoint values")
    before_load = json.loads((root / "accuracy-binaries-before-load.json").read_text())
    before = json.loads((root / "accuracy-binaries-before.json").read_text())
    after = json.loads((root / "accuracy-binaries-after.json").read_text())
    log = (root / "accuracy.log").read_text()
    expected = {("prefill", plan["context"])} | {
        ("decode", case["bucket"])
        for case in json.loads((root / "cases.json").read_text())
    }
    graphs = {}
    for key, binary in before.items():
        # Capture logs alone are not evidence that Executor loaded the file.
        if not re.fullmatch(r"[0-9a-f]{32}", key):
            raise ValueError("invalid compiled graph key")
        if f"Compilation cache key: {key}" not in log:
            continue
        try:
            geometry = model_geometry(binary["input_metadata"], plan["context"])
        except ValueError:
            continue  # Other context entries can share the persistent cache.
        if geometry not in expected:
            continue
        if hashlib.sha256(binary["input_metadata"].encode()).hexdigest() != binary["input_sha256"]:
            raise ValueError("numerical graph input metadata changed")
        if after.get(key) != binary or (key in before_load and before_load[key] != binary):
            raise ValueError("compiled binary changed across public initialization/numerical calls")
        if any((row["phase"], row["token_bucket"]) == geometry for row in graphs.values()):
            raise ValueError("ambiguous numerical graph geometry")
        graphs[key] = {"phase": geometry[0], "token_bucket": geometry[1], "binary": binary}
    if {(g["phase"], g["token_bucket"]) for g in graphs.values()} != expected:
        raise ValueError("numerical run lacks complete before/after compiled binary evidence")
    source_before = json.loads((root / "accuracy-source-before.json").read_text())
    source_after = json.loads((root / "accuracy-source-after.json").read_text())
    if source_before != source_after or source_before.get("loader") != LOADER_SOURCE_HASHES:
        raise ValueError("numerical producer/loader source changed")
    coverage = warm_loader_coverage(log, graphs)
    warm = coverage["complete"] and all(before_load.get(key) == row["binary"]
                                        for key, row in graphs.items())
    return graphs, {"available": warm, "loader_coverage": coverage,
                    "source": source_before}


def seal_accuracy_binaries(root: Path) -> dict:
    """Bind a fresh warm numerical run; cold runs explicitly require revalidation."""
    graphs, load = _accuracy_graphs(root)
    rows = json.loads((root / "accuracy-outputs.json").read_text())
    _check_accuracy_rows(root, rows)
    precision = json.loads((root / "precision.json").read_text())
    names = [
        "plan.json", "cases.json", "accuracy.log", "accuracy-config.json",
        "accuracy-outputs.json", "precision.json", "accuracy-binaries-before-load.json",
        "accuracy-binaries-before.json", "accuracy-binaries-after.json",
        "accuracy-source-before.json", "accuracy-source-after.json",
        *[f"native-{row['case_id']}.npy" for row in rows],
    ]
    receipt = {
        "schema_version": 1,
        "scope": "warm_cache_neff_bytes_stable_across_public_accuracy_calls",
        "binary_binding_available": load["available"],
        "load": load,
        "checkpoint_sha256": MODEL_HASHES, "graphs": graphs,
        "precision_passed": precision["passed"],
        "files": {name: sha256(root / name) for name in names},
        "runtime_model_name_suffix_decoded": False,
    }
    (root / "binary-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _check_accuracy_rows(root: Path, rows: list) -> None:
    cases = json.loads((root / "cases.json").read_text())
    expected = {
        f"{case['id']}-slot{slot}": (case["shape_id"], prompt)
        for case in cases for slot, prompt in enumerate(case["prompts"])
    }
    if not rows or len(rows) != len(expected) or {r["case_id"] for r in rows} != set(expected):
        raise ValueError("numerical request coverage changed")
    for row in rows:
        if ((row["shape_id"], row["prompt_ids"]) != expected[row["case_id"]]
                or len(row["token_ids"]) != 8):
            raise ValueError("numerical request history changed")


def load_accuracy_binaries(root: Path) -> dict:
    path = root / "binary-provenance.json"
    if not path.exists():
        return {"available": False, "scope": "legacy_graph_keys_only"}
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema_version") != 1
        or receipt.get("scope") != "warm_cache_neff_bytes_stable_across_public_accuracy_calls"
        or receipt.get("checkpoint_sha256") != MODEL_HASHES
        or receipt.get("precision_passed") is not True
    ):
        raise ValueError("invalid numerical binary provenance")
    rows = json.loads((root / "accuracy-outputs.json").read_text())
    _check_accuracy_rows(root, rows)
    required = {
        "plan.json", "cases.json", "accuracy.log", "accuracy-config.json",
        "accuracy-outputs.json", "precision.json", "accuracy-binaries-before-load.json",
        "accuracy-binaries-before.json", "accuracy-binaries-after.json",
        "accuracy-source-before.json", "accuracy-source-after.json",
        *[f"native-{row['case_id']}.npy" for row in rows],
    }
    if not rows or set(receipt["files"]) != required or not receipt["graphs"]:
        raise ValueError("incomplete numerical binary provenance")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or sha256(root / name) != expected:
            raise ValueError("numerical binary provenance file changed")
    graphs, load = _accuracy_graphs(root)
    if (graphs != receipt["graphs"] or load != receipt["load"]
            or load["available"] != receipt.get("binary_binding_available")
            or json.loads((root / "precision.json").read_text()).get("passed") is not True):
        raise ValueError("numerical binary provenance differs from acquisition")
    if not load["available"]:
        return {"available": False, "scope": "cold_load_binary_binding_unproven",
                "path": str(path), "sha256": sha256(path),
                "reason": "warm the cache then repeat canonical accuracy/reference"}
    return {"available": True, "path": str(path), "sha256": sha256(path), **receipt}
