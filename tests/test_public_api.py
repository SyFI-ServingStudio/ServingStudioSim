"""The public API serves the kernel library read-only.

The simulator's introspection, the kind docs and the model catalog are
fixtures; the profiling registry and a small profile.db are real.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from profiling.db import kernel_config
from profiling.db.args import DType
from profiling.db.doc import CUPTI_METHOD, KernelDoc, arg_docs
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.db.table import ProfileRow, Table
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.metrics import ComputeMetrics
from public_api.app import PREFIX, create_app
from public_api.kernel import library
from public_api.kernel.library import PROVENANCE, KernelLibrary
from public_api.kernel.sources import KernelSources

GEMM_COLUMNS = (
    "gpu_name TEXT, backend TEXT, m INTEGER, n INTEGER, k INTEGER, dtype TEXT, "
    "time_ms REAL, tflops REAL, memory_bandwidth_gbps REAL, energy_j REAL, "
    "profiler_git_hash TEXT, profiler_run_at TEXT, cuda_version TEXT, "
    "driver_version TEXT, backend_version TEXT"
)
PROV = ("abc123", "2026-09-01T00:00:00+00:00", "12.9", "580", None)
GEMM_ROWS = [
    ("NVIDIA H200", "torch", 1, 6144, 4096, "bf16", 0.01, 5.0, 100.0, None, *PROV),
    ("NVIDIA H200", "torch", 8, 6144, 4096, "bf16", 0.02, 20.0, 200.0, None, *PROV),
    ("NVIDIA H200", "torch", 8, 4096, 4096, "bf16", 0.02, 13.0, 150.0, None, *PROV),
    ("NVIDIA B200", "torch_linear", 8, 6144, 4096, "fp16", 0.01, 40.0, 300.0, None, *PROV),
]
QKV_CONFIG = {"n": {"value": 6144, "expression": "qkv", "bindings": {"tp": 1}}, "k": 4096}
BUILD = {
    "arch": "llama3_dense_tp",
    "gpu": "NVIDIA H200",
    "params": {"model_config": "llama3_8b", "tp_size": 1},
    "gpus_per_replica": 1,
    "error": None,
    "cost_manifest": {
        "sections": [
            {
                "section": "iter",
                "slots": [
                    {"name": "attn.qkv_proj", "kind": "single_gemm", "kernel_config": QKV_CONFIG},
                    # A rank copy of the same leaf is listed once.
                    {"name": "attn.qkv_proj", "kind": "single_gemm", "kernel_config": QKV_CONFIG},
                ],
            }
        ]
    },
}


GEMM_DOC = KernelDoc(
    title="Dense GEMM",
    summary="One matrix multiply.",
    description="C = A · B.",
    category="GEMM",
    formula=("C[m, n] = A[m, k] · B[k, n]",),
    default_metric="tflops",
    method=CUPTI_METHOD,
)


class FixtureSources(KernelSources):
    """Real profile.db access, canned simulator introspection."""

    def supported_builds(self) -> list[dict]:
        return [BUILD]

    def kernel_list(self) -> list[dict]:
        return [{"kind": "single_gemm", "compute_dtype": "dtype", "kv_dtype": None}]

    def rows_report(self, kind: str, config: dict) -> dict:
        assert (kind, config) == ("single_gemm", QKV_CONFIG)
        return {
            "fixed": {"backend": "torch", "n": 6144, "k": 4096, "dtype": "bf16"},
            "swept": ["m"],
        }


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "profile.db"
    conn = sqlite3.connect(path)
    conn.execute(f"create table single_gemm ({GEMM_COLUMNS})")
    conn.executemany(f"insert into single_gemm values ({', '.join('?' * 15)})", GEMM_ROWS)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def client(db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("llama3_8b: {name: Llama 3 8B, family: Llama}\n")
    monkeypatch.setattr(library, "MODEL_CATALOG", catalog)
    monkeypatch.setattr(
        library, "kernel_doc", lambda kind: GEMM_DOC if kind == "single_gemm" else None
    )
    return TestClient(create_app(KernelLibrary(FixtureSources(db_path=db))))


def test_catalog_lists_every_kind_with_coverage_and_models(client: TestClient) -> None:
    catalog = client.get(f"{PREFIX}/kernels").json()
    kernels = {k["kind"]: k for k in catalog["kernels"]}
    gemm = kernels["single_gemm"]
    assert gemm["documented"] and gemm["category"] == "GEMM"
    assert gemm["rows"] == 4
    assert gemm["precisions"] == ["bf16", "fp16"]
    assert {"gpu": "NVIDIA B200", "backend": "torch_linear", "precision": "fp16", "rows": 1} in (
        gemm["coverage"]
    )
    assert gemm["used_by"] == ["Llama 3 8B"]
    # A kind with no table and no supported deployment is still listed.
    assert kernels["rms_norm"]["rows"] == 0 and kernels["rms_norm"]["used_by"] == []
    assert [g["name"] for g in catalog["gpus"]] == ["NVIDIA H200", "NVIDIA B200"]
    peaks = catalog["gpus"][0]["peaks"]
    assert peaks["memory_bandwidth_gbps"]["value"] > 0
    assert peaks["busbw_gbps"]["note"].startswith("NVLink")
    # Throughput ceilings are per compute dtype, for the dtypes the rows record.
    assert set(peaks["tflops"]["by_dtype"]) == {"bf16", "fp16"}
    assert catalog["snapshot"]["last_measured_at"] == PROV[1]
    assert catalog["models"][0]["model_config"] == "llama3_8b"


def test_kernel_detail_joins_docs_roles_and_shapes(client: TestClient) -> None:
    kernel = client.get(f"{PREFIX}/kernels/single_gemm").json()
    roles = {a["name"]: a["role"] for a in kernel["args"]}
    assert roles == {"m": "sweep", "n": "config", "k": "config", "dtype": "config"}
    assert [a["name"] for a in kernel["args"] if a["precision"]] == ["dtype"]
    types = {a["name"]: a["type"] for a in kernel["args"]}
    assert types == {"m": "number", "n": "number", "k": "number", "dtype": "dtype"}
    # Units and meanings are the kind module's own arg(...) docs.
    documented = arg_docs(SingleGemmArgs)
    assert [(a["unit"], a["doc"]) for a in kernel["args"]] == [
        (d["unit"], d["doc"]) for d in documented.values()
    ]
    assert kernel["metrics"][1] == {"name": "tflops", "label": "Throughput", "unit": "TFLOPS"}
    assert set(kernel["backends"]["torch"]) >= {"summary", "url", "supports", "env"}
    assert kernel["method"].startswith(CUPTI_METHOD)
    (deployment,) = kernel["used_by"]
    assert deployment["model"]["name"] == "Llama 3 8B"
    assert deployment["parallel"] == {"tp_size": 1}
    (shape,) = deployment["shapes"]
    assert shape["db"] == {"n": 6144, "k": 4096, "dtype": "bf16"}
    assert shape["why"] == {"n": {"expression": "qkv", "bindings": {"tp": 1}}}


def test_kernel_without_a_supported_deployment_has_unknown_roles(client: TestClient) -> None:
    kernel = client.get(f"{PREFIX}/kernels/rms_norm").json()
    assert kernel["used_by"] == []
    assert {a["role"] for a in kernel["args"]} == {None}


def test_rows_filter_by_typed_columns(client: TestClient) -> None:
    rows = client.get(
        f"{PREFIX}/kernels/single_gemm/rows", params={"gpu": "NVIDIA H200", "n": "6144"}
    )
    document = rows.json()
    assert document["columns"][:6] == ["gpu", "backend", "m", "n", "k", "dtype"]
    assert [row[2] for row in document["rows"]] == [1, 8]
    assert document["provenance"] == [dict(zip(PROVENANCE, PROV))]


def test_rows_as_csv(client: TestClient) -> None:
    response = client.get(
        f"{PREFIX}/kernels/single_gemm/rows", params={"backend": "torch_linear", "format": "csv"}
    )
    assert response.headers["content-type"].startswith("text/csv")
    header, row = response.text.strip().splitlines()
    assert header.startswith("gpu,backend,m,n,k,dtype,time_ms") and header.endswith(
        "backend_version"
    )
    assert row.startswith("NVIDIA B200,torch_linear,8,6144,4096,fp16")


@pytest.mark.parametrize(
    ("path", "params", "status"),
    [
        ("/kernels/no_such_kind", {}, 404),
        ("/kernels/no_such_kind/rows", {}, 404),
        ("/kernels/single_gemm/rows", {"hidden": "1"}, 400),
        ("/kernels/single_gemm/rows", {"n": "big"}, 400),
        ("/kernels/single_gemm/rows", {"format": "xml"}, 400),
    ],
)
def test_bad_requests(client: TestClient, path: str, params: dict, status: int) -> None:
    assert client.get(f"{PREFIX}{path}", params=params).status_code == status


def test_the_database_is_never_written(client: TestClient, db: Path) -> None:
    before = (hashlib.sha256(db.read_bytes()).hexdigest(), db.stat().st_mtime_ns)
    for path in (
        "/kernels",
        "/kernels/single_gemm",
        "/kernels/single_gemm/rows",
        "/kernels/single_gemm/configs",
    ):
        assert client.get(f"{PREFIX}{path}").status_code == 200
    assert (hashlib.sha256(db.read_bytes()).hexdigest(), db.stat().st_mtime_ns) == before
    with FixtureSources(db_path=db).connect() as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("delete from single_gemm")


def _registered_db(path: Path) -> str:
    """A profile.db with torch rows at m=1 and m=2 of one n=6144 config, and that
    config registered on a grid of m = 1, 2, 16 whose last cell cannot run.
    Returns the config's hash."""

    table = Table(next(iter_kernel_profiler_specs("single_gemm")), path)
    table.insert(
        [
            ProfileRow(
                args=SingleGemmArgs(m=m, n=6144, k=4096, dtype=DType.BF16),
                metrics=ComputeMetrics(
                    time_ms=0.01 * m, tflops=float(m), memory_bandwidth_gbps=1.0
                ),
                gpu_name="NVIDIA H200",
                backend="torch",
                profiler_git_hash="abc",
                profiler_run_at="2026-09-26T00:00:00+00:00",
            )
            for m in (1, 2)
        ]
    )
    identity = {"n": 6144, "k": 4096, "dtype": "bf16"}
    record = {
        "kind": "single_gemm",
        "profile_kind": "single_gemm",
        "gpu_name": "NVIDIA H200",
        "identity": identity,
        "grid": {
            "cache_coords": ["m"],
            "axes": [[1.0, 2.0, 16.0]],
            "cells": [{"m": m, **identity} for m in (1, 2, 16)],
            "infeasible": [2],
        },
        "uses": [{"pool": "main", "role": "unified.qkv"}],
    }
    source = {"timing_predict": "presets/predict_x.json", "gpu": "NVIDIA H200", "arch": {}}
    kernel_config.register_kernel_configs(
        path,
        {"schema_version": kernel_config.RECORDS_SCHEMA_VERSION, "configs": [record]},
        {"main": source},
    )
    return kernel_config.content_hash(identity)


@pytest.fixture
def registered(tmp_path: Path) -> tuple[TestClient, str, Path]:
    path = tmp_path / "registered.db"
    config_hash = _registered_db(path)
    client = TestClient(create_app(KernelLibrary(FixtureSources(db_path=path))))
    return client, config_hash, path


def test_configs_list_what_each_backend_measured(registered) -> None:
    client, config_hash, _ = registered
    [config] = client.get(f"{PREFIX}/kernels/single_gemm/configs").json()["configs"]
    assert config["config_hash"] == config_hash
    assert config["gpu"] == "NVIDIA H200"
    assert config["identity"] == {"n": 6144, "k": 4096, "dtype": "bf16"}
    assert (config["cache_coords"], config["shape"], config["cells"]) == (["m"], [3], 3)
    assert (config["infeasible"], config["measured"]) == (1, {"torch": 2})
    assert config["uses"] == [
        {
            "source": {
                "timing_predict": "presets/predict_x.json",
                "gpu": "NVIDIA H200",
                "arch": {},
            },
            "pool": "main",
            "role": "unified.qkv",
        }
    ]


def test_config_joins_grid_cells_to_rows(registered) -> None:
    client, config_hash, path = registered
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)

    config = client.get(f"{PREFIX}/kernels/single_gemm/configs/{config_hash}").json()

    assert config["axes"] == [[1.0, 2.0, 16.0]]
    points = config["points"]
    assert [p["coords"] for p in points] == [[1.0], [2.0], [16.0]]
    assert [p["feasible"] for p in points] == [True, True, False]
    assert points[1]["args"] == {"m": 2, "n": 6144, "k": 4096, "dtype": "bf16"}
    assert points[1]["measured"]["torch"]["tflops"] == 2.0
    assert points[1]["measured"]["torch"]["outlier"] is False
    assert points[2]["measured"] == {}
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before


def test_config_lookup_errors(registered) -> None:
    client, config_hash, _ = registered
    base = f"{PREFIX}/kernels/single_gemm/configs"
    assert client.get(f"{base}/no_such_hash").status_code == 404
    assert client.get(f"{base}/{config_hash}", params={"gpu": "NVIDIA B200"}).status_code == 404
    assert client.get(f"{PREFIX}/kernels/no_such_kind/configs").status_code == 404
