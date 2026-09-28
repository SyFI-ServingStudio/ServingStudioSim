"""The wasm32 simulator against the native one.

Native runs on the Python bridge (profile.db). The wasm module reads what a
browser gets from the public API instead: the config document
(`GET /kernels/{kind}/configs/{hash}`) of every kernel config the run builds,
made here by the same `KernelLibrary.config` the API serves, and the files the
arch block names, passed in memory. Results must match bit for bit: the browser
has no other oracle.

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

CONFIGS = {
    "glm52_nvfp4_b200_ep4_popularity": (GLM52_B200_EP4, "NVIDIA B200"),
    "llama3_8b_h200_tp1": (LLAMA3_8B_TP1, "NVIDIA H200"),
}

pytestmark = [pytest.mark.needs_binary, pytest.mark.needs_db]


def _arch_files(arch: dict) -> dict[str, str]:
    """The repo files an arch block names, as the browser receives them."""

    [block] = arch.values()
    return {
        block[key]: str(REPO_ROOT / block[key])
        for key in ("model_config", "expert_popularity_file")
        if key in block
    }


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
        pytest.skip("wasm module not built (just build-wasm)")
    return path


@pytest.mark.parametrize("name", CONFIGS)
def test_wasm_matches_native_timing_predict(name, sim_bin, node, tmp_path):
    arch, gpu = CONFIGS[name]
    (tmp_path / "cases.json").write_text(json.dumps(MIXED_CASES))
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
    table = pq.read_table(raw / "cost_log" / "worker_predict_0.parquet")
    [section] = json.loads((raw / "cost_manifest" / "worker_predict_0.json").read_text())[
        "sections"
    ]

    request = tmp_path / "wasm_input.json"
    request.write_text(
        json.dumps(
            {
                "config": {"arch": arch, "gpu": gpu},
                "kernel_data_file": str(kernel_data),
                "files": _arch_files(arch),
                "cases": MIXED_CASES,
            }
        )
    )
    run = subprocess.run(
        [node, str(RUNNER), str(PKG_DIR), str(request)], capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)

    assert out["version"]["kernel_data_format"] == 1
    [block] = arch.values()
    assert out["info"] == {
        "selector": "iter",
        "num_attn_dp_groups": out["info"]["num_attn_dp_groups"],
        "gpus_per_replica": out["info"]["gpus_per_replica"],
        "max_model_len": block.get("max_model_len"),
        "draft_tokens": None,
    }
    manifest = out["manifest"]
    assert manifest["slots"] == [slot["name"] for slot in section["slots"]]
    assert manifest["nodes"] == section["nodes"]
    assert manifest["node_labels"] == section["node_labels"]

    totals = table["total_time_ms"].to_pylist()
    slots = table["slot_time_ms"].to_pylist()
    assert len(out["cases"]) == len(totals) == len(MIXED_CASES)
    for case, total, native_slots in zip(out["cases"], totals, slots, strict=True):
        assert case["total_time_ms"] == total
        # f32 on both sides; wasm prints the shortest decimal that reads back as it.
        np.testing.assert_array_equal(np.float32(case["slot_time_ms"]), np.float32(native_slots))
        assert np.float32(case["node_time_ms"][0]) == np.float32(total)
        assert len(case["node_time_ms"]) == len(section["nodes"])
