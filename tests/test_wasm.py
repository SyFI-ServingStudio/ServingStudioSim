"""The wasm32 simulator against the native one.

Native runs on the Python bridge (profile.db). The wasm module reads what a
browser gets from the public API instead: the config document
(`GET /kernels/{kind}/configs/{hash}`) of every kernel config the run builds,
made here by the same `KernelLibrary.config` the API serves, and the files the
arch block names (a token corpus's manifest, never its payload), passed in
memory. Results must match bit for bit: the browser has no other oracle.

Skips unless the release binary, the wasm module and `node` are all present and
profile.db holds every row the config reads.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from launcher.corpus import resolve_hf_references
from launcher.exec import _build_subprocess_env
from profiling.db.kernel_config import content_hash
from public_api.kernel.library import KernelLibrary
from public_api.kernel.sources import KernelSources

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "target" / "wasm-pkg"
RUNNER = REPO_ROOT / "tests" / "wasm" / "run_predict.mjs"

GLM52_B200_EP4 = {
    "iter": {
        "type": "glm52_vllm_nvfp4_dsa_moe",
        "model_config": "model/config/glm52_nvfp4.json",
        "fp8": False,
        "ep_size": 4,
        "nvl_num_gpu": 4,
        "max_model_len": 8192,
        "routing": "popularity",
        "mtp_mode": "off",
        "expert_popularity_file": "presets/alignment/glm52_nvfp4_b200/expert_popularity.json",
    }
}
LLAMA3_8B_TP1 = {
    "iter": {
        "type": "llama3_dense_tp",
        "model_config": "model/config/llama3_8b.json",
        "fp8": False,
        "tp_size": 1,
    }
}

MIXED_CASES = [
    {"groups": [{"decode_kv_lens": [4096] * 300}]},
    {"groups": [{"prefill_chunk_pairs": [[0, 512]] * 20, "decode_kv_lens": [2048] * 400}]},
    {"groups": [{"prefill_chunk_pairs": [[1024, 3000]]}]},
]

# The AFD layer-wise pair, as registered (presets/predict_afd_*.json). The only
# registered ffn data set routes uniformly: synthetic, but every cell measured.
QWEN3_ATTN_TP4 = json.loads((REPO_ROOT / "presets" / "predict_afd_attn.json").read_text())
QWEN3_FFN_EP8 = json.loads((REPO_ROOT / "presets" / "predict_afd_ffn.json").read_text())


def _preset_cases(preset: dict) -> list:
    return json.loads((REPO_ROOT / "presets" / preset["cases_file"]).read_text())


# GLM-5.2 NVFP4 drafting MTP-5 on B200 EP4, routed by the recorded token corpus
# of that deployment: a corpus binds by its manifest alone in wasm.
SPEC5_CORPUS_PRESET = "presets/glm52_nvfp4_b200_ep4_speculative.yaml"
SPEC5_CASES = [
    {"groups": [{"decode_requests": [[4096, 6]] * 64}]},
    {"groups": [{"prefill_chunk_pairs": [[0, 512]] * 4, "decode_requests": [[2048, 6]] * 100}]},
    {"groups": [{"prefill_chunk_pairs": [[1024, 3000]]}]},
]


def _spec5_corpus_arch() -> dict:
    """The preset's arch block with its corpus reference fetched (hub cache)."""

    preset = yaml.safe_load((REPO_ROOT / SPEC5_CORPUS_PRESET).read_text())
    [group] = preset["pools"]["main"]["groups"]
    try:
        return {"speculative_iter": resolve_hf_references(group["arch"])}
    except Exception as error:  # no hub access and nothing cached
        pytest.skip(f"token corpus unavailable: {error}")


# name -> (arch, gpu, cases); an arch that must be resolved first is a callable.
CONFIGS = {
    "glm52_nvfp4_b200_ep4_popularity": (GLM52_B200_EP4, "NVIDIA B200", MIXED_CASES),
    "llama3_8b_h200_tp1": (LLAMA3_8B_TP1, "NVIDIA H200", MIXED_CASES),
    "qwen3_235b_h200_attn_tp4": (
        QWEN3_ATTN_TP4["arch"],
        QWEN3_ATTN_TP4["gpu"],
        _preset_cases(QWEN3_ATTN_TP4),
    ),
    "qwen3_235b_h200_ffn_ep8_uniform": (
        QWEN3_FFN_EP8["arch"],
        QWEN3_FFN_EP8["gpu"],
        _preset_cases(QWEN3_FFN_EP8),
    ),
    "glm52_nvfp4_b200_ep4_spec5_corpus": (_spec5_corpus_arch, "NVIDIA B200", SPEC5_CASES),
}
# Arch fields that name a file the simulator reads.
FILE_FIELDS = ("model_config", "expert_popularity_file", "token_corpus_file")

pytestmark = [pytest.mark.needs_binary, pytest.mark.needs_db]


def _arch_files(arch: dict) -> dict[str, str]:
    """The repo files an arch block names, as the browser receives them."""

    [block] = arch.values()
    return {block[key]: str(REPO_ROOT / block[key]) for key in FILE_FIELDS if key in block}


def _simulator(sim_bin: Path, *args: str) -> subprocess.CompletedProcess:
    env = _build_subprocess_env() | {"SERVINGSTUDIO_NO_GPU": "1", "RUST_LOG": "warn"}
    return subprocess.run(
        [str(sim_bin), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def _kernel_data(records: Path, out: Path) -> None:
    """The config document of every kernel config in `records` (what
    `--kernel-configs-out` writes), as the kernel API serves each."""

    library = KernelLibrary(KernelSources(db_path=REPO_ROOT / "profiling" / "profile.db"))
    documents = {}
    for record in json.loads(records.read_text())["configs"]:
        key = (record["kind"], content_hash(record["identity"]), record["gpu_name"])
        if key not in documents:
            documents[key] = library.config(*key)
    out.write_text(json.dumps({"configs": list(documents.values())}))


@pytest.fixture(scope="module")
def node() -> str:
    path = shutil.which("node")
    if path is None:
        pytest.skip("node not on PATH")
    if not (PKG_DIR / "simulator_wasm_bg.wasm").is_file():
        pytest.skip("wasm module not built (just setup-wasm, then just build-wasm)")
    return path


@pytest.mark.parametrize("name", CONFIGS)
def test_wasm_matches_native_timing_predict(name, sim_bin, node, tmp_path):
    arch, gpu, cases = CONFIGS[name]
    arch = arch() if callable(arch) else arch
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    config = tmp_path / "predict.json"
    config.write_text(
        json.dumps(
            {
                "arch": arch,
                "gpu": gpu,
                "log_dir": str(tmp_path / "logs"),
                "cases_file": "cases.json",
            }
        )
    )

    records = tmp_path / "records.json"
    dry = _simulator(
        sim_bin, "timing-predict", "--dry-run", "--kernel-configs-out", str(records), str(config)
    )
    assert dry.returncode == 0, dry.stderr
    missing = re.search(r"total: (\d+) / \d+ specs missing", dry.stdout)
    if missing is None or int(missing.group(1)):
        pytest.skip(f"profile.db lacks rows for {name}")
    kernel_data = tmp_path / "kernel_data.json"
    _kernel_data(records, kernel_data)

    native = _simulator(sim_bin, "timing-predict", str(config))
    assert native.returncode == 0, native.stderr
    raw = tmp_path / "logs" / "raw"
    rows = pq.read_table(raw / "cost_log" / "worker_predict_0.parquet").to_pylist()
    doc = json.loads((raw / "cost_manifest" / "worker_predict_0.json").read_text())

    request = tmp_path / "wasm_input.json"
    request.write_text(
        json.dumps(
            {
                "config": {"arch": arch, "gpu": gpu},
                "kernel_data_file": str(kernel_data),
                "files": _arch_files(arch),
                "cases": cases,
            }
        )
    )
    run = subprocess.run(
        [node, str(RUNNER), str(PKG_DIR), str(request)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)

    assert out["version"]["kernel_data_format"] == 1
    [(selector, block)] = arch.items()
    assert out["info"]["selector"] == selector
    assert out["info"]["max_model_len"] == block.get("max_model_len")
    assert out["info"]["draft_tokens"] == block.get("draft_tokens")
    assert out["manifest"]["sections"] == [
        {
            "section": section["section"],
            "slots": [slot["name"] for slot in section["slots"]],
            "nodes": section["nodes"],
            "node_labels": section["node_labels"],
        }
        for section in doc["sections"]
    ]

    # Every case's sections, in order, are native's cost_log rows.
    assert len(out["cases"]) == len(cases)
    predicted = [section for case in out["cases"] for section in case["sections"]]
    assert len(predicted) == len(rows)
    for got, row in zip(predicted, rows, strict=True):
        assert (got["section"], got["layer"]) == (row["section"], row["layer"])
        assert got["total_time_ms"] == row["total_time_ms"]
        # f32 on both sides; wasm prints the shortest decimal that reads back as it.
        np.testing.assert_array_equal(
            np.float32(got["slot_time_ms"]), np.float32(row["slot_time_ms"])
        )
        assert np.float32(got["node_time_ms"][0]) == np.float32(row["total_time_ms"])
