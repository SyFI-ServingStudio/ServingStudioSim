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


# `list-params`, cut down: an arch with `#[supported]` rows (its label names
# their params) and one without (its label falls back to the scalar params that
# change its kernel configs; a file path stays out).
SCHEMA = {
    "arch_common": [
        {"name": "model_config", "type": "string", "required": True, "affects_cache": True},
        {"name": "fp8", "type": "bool", "default": False, "affects_cache": True},
    ],
    "providers": {
        "arch": {
            "iter_wise": {
                "llama3_dense_tp": {
                    "params": [
                        {"name": "tp_size", "type": "int", "default": 2, "affects_cache": True}
                    ],
                    "supported": [
                        {"gpu": ["NVIDIA H200"], "model_config": ["llama3_8b"], "tp_size": [1, 4]}
                    ],
                },
                # The same tag under another contract: a build's `contract` picks.
                "attn_x": {
                    "params": [
                        {"name": "ep_size", "type": "int", "default": 1, "affects_cache": True}
                    ],
                    "supported": [{"gpu": ["NVIDIA H200"], "ep_size": [1]}],
                },
                "moe_x": {
                    "params": [
                        {"name": "ep_size", "type": "int", "default": 4, "affects_cache": True},
                        {"name": "routing", "type": "string", "choices": ["uniform", "popularity"]},
                        {
                            "name": "mtp_mode",
                            "type": "string",
                            "default": "off",
                            "affects_cache": True,
                            "choices": ["off", "on"],
                        },
                        {"name": "popularity_file", "type": "string", "affects_cache": True},
                    ]
                },
            },
            "layer_wise_attn": {
                "attn_x": {
                    "params": [
                        {"name": "tp_size", "type": "int", "default": 1, "affects_cache": True}
                    ],
                    "supported": [
                        {"gpu": ["NVIDIA H200"], "model_config": ["llama3_8b"], "tp_size": [2]}
                    ],
                }
            },
        }
    },
}


class FixtureSources(KernelSources):
    """Real profile.db access, canned simulator introspection."""

    def supported_builds(self) -> list[dict]:
        return [BUILD]

    def deployment_schema(self) -> dict:
        return SCHEMA

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


@pytest.fixture(autouse=True)
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text("llama3_8b: {name: Llama 3 8B, family: Llama}\n")
    monkeypatch.setattr(library, "MODEL_CATALOG", path)
    monkeypatch.setattr(
        library, "kernel_doc", lambda kind: GEMM_DOC if kind == "single_gemm" else None
    )


@pytest.fixture
def client(db: Path) -> TestClient:
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
    assert gemm["used_by"] == ["llama3_8b"]
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
    assert deployment["params"] == {"tp_size": 1}
    assert deployment["label"] == "llama3_dense_tp, tp_size 1"
    assert (deployment["sources"], deployment["gpus_per_replica"]) == (["supported"], 1)
    (shape,) = deployment["shapes"]
    assert shape["db"] == {"n": 6144, "k": 4096, "dtype": "bf16"}
    assert shape["why"] == {"n": {"expression": "qkv", "bindings": {"tp": 1}}}
    assert shape["config_hash"] is None


class LayerWiseSources(FixtureSources):
    """A layer-wise supported build: its cost manifest has several sections."""

    def supported_builds(self) -> list[dict]:
        slot = {"kind": "single_gemm", "kernel_config": QKV_CONFIG}
        return [
            {
                "arch": "attn_x",
                "contract": "layer_wise_attn",
                "gpu": "NVIDIA H200",
                "params": {"model_config": "llama3_8b", "tp_size": 2},
                "gpus_per_replica": 2,
                "error": None,
                "cost_manifest": {
                    "sections": [
                        {"section": "attn", "slots": [{"name": "attn.qkv_proj", **slot}]},
                        {"section": "post_attn", "slots": [{"name": "attn.o_proj", **slot}]},
                    ]
                },
            }
        ]


def test_layer_wise_builds_read_every_section_under_their_contract(db: Path) -> None:
    client = TestClient(create_app(KernelLibrary(LayerWiseSources(db_path=db))))
    (deployment,) = client.get(f"{PREFIX}/kernels/single_gemm").json()["used_by"]
    # The layer-wise attn_x's supported rows name tp_size; the iter-wise tag of
    # the same name would have named ep_size.
    assert deployment["label"] == "attn_x, tp_size 2"
    assert [(s["section"], s["layer"]) for s in deployment["shapes"]] == [
        ("attn", "attn.qkv_proj"),
        ("post_attn", "attn.o_proj"),
    ]


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


QKV = {"n": 6144, "k": 4096, "dtype": "bf16"}
GATE = {"n": 512, "k": 4096, "dtype": "bf16"}
PREDICT = {
    "timing_predict": "presets/predict_x.json",
    "gpu": "NVIDIA H200",
    "arch": {
        "iter": {
            "type": "llama3_dense_tp",
            "model_config": "model/config/llama3_8b.json",
            "tp_size": 4,
        }
    },
}
MOE_ARCH = {
    "type": "moe_x",
    "model_config": "model/config/moe_x.json",
    "ep_size": 8,
    "routing": "popularity",
    "popularity_file": "presets/popularity.json",
}
RUN = {
    "preset": "presets/moe_x.yaml",
    "deployment": "unified",
    "pool": "main",
    "groups": [{"gpu": "NVIDIA H200", "arch": MOE_ARCH}],
}
# The deployment-level pool of the same run, which names no arch.
RUN_DEPLOYMENT = {"preset": "presets/moe_x.yaml", "deployment": "unified", "pool": None}


def _row(m: int, n: int, backend: str) -> ProfileRow:
    return ProfileRow(
        args=SingleGemmArgs(m=m, n=n, k=4096, dtype=DType.BF16),
        metrics=ComputeMetrics(time_ms=0.01 * m, tflops=float(m), memory_bandwidth_gbps=1.0),
        gpu_name="NVIDIA H200",
        backend=backend,
        profiler_git_hash="abc",
        profiler_run_at="2026-09-26T00:00:00+00:00",
    )


def _record(identity: dict, uses: list[dict]) -> dict:
    return {
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
        "uses": uses,
    }


def _registered_db(path: Path) -> tuple[str, str]:
    """A profile.db with two configs registered on a grid of m = 1, 2, 16 whose
    last cell cannot run: the qkv config (torch rows at m = 1 and 2, torch_linear
    at m = 2), built by a Llama prediction, and the gate config (one torch row),
    built by a MoE preset run. Returns both hashes."""

    table = Table(next(iter_kernel_profiler_specs("single_gemm")), path)
    table.insert([_row(1, 6144, "torch"), _row(2, 6144, "torch"), _row(2, 6144, "torch_linear")])
    table.insert([_row(1, 512, "torch")])
    kernel_config.register_kernel_configs(
        path,
        {
            "schema_version": kernel_config.RECORDS_SCHEMA_VERSION,
            "configs": [_record(QKV, [{"pool": "main", "role": "unified.qkv"}])],
        },
        {"main": PREDICT},
    )
    kernel_config.register_kernel_configs(
        path,
        {
            "schema_version": kernel_config.RECORDS_SCHEMA_VERSION,
            "configs": [
                _record(
                    GATE,
                    [
                        {"pool": "main", "role": "unified.moe.gate"},
                        {"pool": "", "role": "unified.transfer"},
                    ],
                )
            ],
        },
        {"main": RUN, "": RUN_DEPLOYMENT},
    )
    return kernel_config.content_hash(QKV), kernel_config.content_hash(GATE)


@pytest.fixture
def registered(tmp_path: Path) -> tuple[TestClient, tuple[str, str], Path]:
    path = tmp_path / "registered.db"
    hashes = _registered_db(path)
    client = TestClient(create_app(KernelLibrary(FixtureSources(db_path=path))))
    return client, hashes, path


def test_configs_list_is_slim_and_labels_each_use(registered) -> None:
    client, (qkv, gate), _ = registered
    document = client.get(f"{PREFIX}/kernels/single_gemm/configs").json()
    configs = {c["config_hash"]: c for c in document["configs"]}
    config = configs[qkv]
    assert config["gpu"] == "NVIDIA H200"
    assert "identity" not in config
    assert (config["fixed"], config["swept"]) == (QKV, ["m"])
    assert (config["config_args"], config["config_args_omitted"]) == (QKV, [])
    assert (config["cache_coords"], config["axes"], config["shape"]) == (["m"], [[1, 2, 16]], [3])
    assert (config["cells"], config["infeasible"]) == (3, 1)
    assert config["measured"] == {"torch": 2, "torch_linear": 1}
    assert configs[gate]["measured"] == {"torch": 1}

    sources = {s["id"]: s for s in document["sources"]}
    deployments = {d["id"]: d for d in document["deployments"]}
    [use] = config["uses"]
    assert (use["pool"], use["role"]) == ("main", "unified.qkv")
    assert sources[use["source"]] == {
        "id": use["source"],
        "type": "timing_predict",
        "ref": "presets/predict_x.json",
    }
    [llama] = [deployments[i] for i in use["deployments"]]
    # The model config path is named by its stem, as #[supported] rows and the
    # catalog spell it; the label carries the params the supported rows name.
    assert (llama["model_config"], llama["model"]["name"]) == ("llama3_8b", "Llama 3 8B")
    assert (llama["params"], llama["label"]) == ({"tp_size": 4}, "llama3_dense_tp, tp_size 4")

    by_role = {u["role"]: u for u in configs[gate]["uses"]}
    [moe] = [deployments[i] for i in by_role["unified.moe.gate"]["deployments"]]
    # No supported rows: the scalar params that change its kernel configs,
    # defaults filled in; the popularity file path stays out of the label.
    assert (moe["model_config"], moe["model"]) == ("moe_x", None)
    assert moe["params"] == {"ep_size": 8, "mtp_mode": "off"}
    assert moe["label"] == "moe_x, ep_size 8, mtp_mode off"
    assert sources[by_role["unified.moe.gate"]["source"]]["type"] == "preset"
    # The deployment-level pool names no arch, so no deployment.
    assert by_role["unified.transfer"]["deployments"] == []


def test_config_joins_grid_cells_to_rows(registered) -> None:
    client, (qkv, _), path = registered
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)

    config = client.get(f"{PREFIX}/kernels/single_gemm/configs/{qkv}").json()

    assert config["identity"] == QKV
    assert config["axes"] == [[1.0, 2.0, 16.0]]
    points = config["points"]
    assert [p["coords"] for p in points] == [[1.0], [2.0], [16.0]]
    assert [p["feasible"] for p in points] == [True, True, False]
    assert points[1]["args"] == {"m": 2, "n": 6144, "k": 4096, "dtype": "bf16"}
    assert points[1]["measured"]["torch"]["tflops"] == 2.0
    assert points[1]["measured"]["torch"]["outlier"] is False
    assert set(points[1]["measured"]) == {"torch", "torch_linear"}
    assert set(points[0]["measured"]) == {"torch"}
    assert points[2]["measured"] == {}
    [use] = config["uses"]
    assert use["source"] == PREDICT
    assert use["via"]["type"] == "timing_predict"
    assert [d["label"] for d in use["deployments"]] == ["llama3_dense_tp, tp_size 4"]
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before


def test_registered_configs_add_models_to_catalog_and_detail(registered) -> None:
    client, (_, gate), _ = registered
    catalog = client.get(f"{PREFIX}/kernels").json()
    gemm = next(k for k in catalog["kernels"] if k["kind"] == "single_gemm")
    # The supported Llama deployment, then the registry's MoE model the catalog
    # does not name.
    assert gemm["used_by"] == ["llama3_8b", "moe_x"]
    assert catalog["models"][-1] == {
        "model_config": "moe_x",
        "name": None,
        "family": None,
        "checkpoint": None,
    }

    kernel = client.get(f"{PREFIX}/kernels/single_gemm").json()
    assert {a["name"]: a["role"] for a in kernel["args"]}["m"] == "sweep"
    labels = [d["label"] for d in kernel["used_by"]]
    # Catalog order, then the uncatalogued model.
    assert labels == [
        "llama3_dense_tp, tp_size 1",
        "llama3_dense_tp, tp_size 4",
        "moe_x, ep_size 8, mtp_mode off",
    ]
    moe = kernel["used_by"][-1]
    assert moe["sources"] == ["preset"]
    [shape] = moe["shapes"]
    assert (shape["layer"], shape["section"], shape["config_hash"]) == (
        "unified.moe.gate",
        "main",
        gate,
    )
    assert shape["db"] == GATE


def test_config_lookup_errors(registered) -> None:
    client, (qkv, _), _ = registered
    base = f"{PREFIX}/kernels/single_gemm/configs"
    assert client.get(f"{base}/no_such_hash").status_code == 404
    assert client.get(f"{base}/{qkv}", params={"gpu": "NVIDIA B200"}).status_code == 404
    assert client.get(f"{PREFIX}/kernels/no_such_kind/configs").status_code == 404
