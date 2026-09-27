"""The public API serves the kernel library read-only.

The simulator's introspection, the kind docs and the model catalog are
fixtures; the profiling registry and a small profile.db are real.
"""

from __future__ import annotations

import copy
import hashlib
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from profiling.db import kernel_config
from profiling.db.args import DType
from profiling.db.doc import CUPTI_METHOD, KernelDoc, arg_docs, kernel_doc
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.db.table import ProfileRow, Table
from profiling.kernels.nvfp4_fused_moe import Nvfp4FusedMoeArgs
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.metrics import ComputeMetrics
from public_api.app import PREFIX, create_app
from public_api.arch import library as arch_library
from public_api.arch.library import ArchLibrary, config_identity
from public_api.kernel import library
from public_api.kernel.library import PROVENANCE, KernelLibrary
from public_api.kernel.sources import REPO_ROOT, KernelSources

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
                "moe_x": {
                    "params": [
                        {"name": "ep_size", "type": "int", "default": 4, "affects_cache": True},
                        {
                            "name": "routing",
                            "type": "string",
                            "default": "uniform",
                            "set_when_predicting": True,
                            "choices": ["uniform", "popularity", "corpus"],
                        },
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
        }
    },
}


def _dim(value: int, expression: str) -> dict:
    """A rich ``Dim`` as a slot's kernel config serializes it."""

    return {"value": value, "expression": expression, "bindings": {"hidden": 4096}}


def _gemm_config(n: dict | int) -> dict:
    return {
        "gpu_name": "NVIDIA H200",
        "backends": ["torch", "torch_linear"],
        "n": n,
        "k": _dim(4096, "hidden"),
        "dtype": "bf16",
    }


def _llama_build(tp_size: int) -> dict:
    """``supported-cost-trees`` for one llama3_dense_tp expansion, cut down:
    Sum(embedding, Scale{32}(Sum(Max(qkv rank 0, qkv rank 1), mlp))). Both
    qkv ranks carry the registered qkv config (``QKV``)."""

    qkv = _gemm_config(_dim(6144, "(heads+2*kv_heads)*head_dim"))
    slots = [
        {
            "name": "unified.embedding",
            "kind": "elementwise",
            "kernel_config": {
                "gpu_name": "NVIDIA H200",
                "backends": ["triton"],
                "input_bytes_per_token": 8192,
                "output_bytes_per_token": 8192,
            },
        },
        {"name": "unified.layer.attn.qkv", "kind": "single_gemm", "kernel_config": qkv},
        {"name": "unified.layer.attn.qkv", "kind": "single_gemm", "kernel_config": qkv},
        {"name": "unified.layer.mlp", "kind": "single_gemm", "kernel_config": _gemm_config(14336)},
    ]
    nodes = [
        {"Sum": {"children": {"start": 1, "end": 3}}},
        {"Leaf": 0},
        {"Scale": {"n": 32, "children": {"start": 3, "end": 4}}},
        {"Sum": {"children": {"start": 4, "end": 6}}},
        {"Max": {"overlap": 1.0, "children": {"start": 6, "end": 8}}},
        {"Leaf": 3},
        {"Leaf": 1},
        {"Leaf": 2},
    ]
    labels = [f"unified [dense TP (tp={tp_size})]", None, "layer", None, "attn [Max]", None, None]
    return {
        "contract": "iter_wise",
        "arch": "llama3_dense_tp",
        "gpu": "NVIDIA H200",
        "params": {"model_config": "llama3_8b", "tp_size": tp_size},
        "gpus_per_replica": tp_size,
        "cost_manifest": {
            "sections": [
                {"section": "iter", "slots": slots, "nodes": nodes, "node_labels": [*labels, None]}
            ]
        },
        "error": None,
    }


class FixtureSources(KernelSources):
    """Real profile.db access, canned simulator introspection."""

    def deployment_schema(self) -> dict:
        return SCHEMA

    def kernel_list(self) -> list[dict]:
        return [{"kind": "single_gemm", "compute_dtype": "dtype", "kv_dtype": None}]

    def supported_cost_trees(self) -> list[dict]:
        return [_llama_build(1), _llama_build(4)]

    def arch_cost_trees(self, blocks: list[dict]) -> list[dict]:
        """``supported-cost-trees --archs``: a Llama block builds as its row's
        combination does."""

        return [_llama_build(block["arch"]["tp_size"]) for block in blocks]


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
    archs = tmp_path / "arch_catalog.yaml"
    archs.write_text("llama3_dense_tp: {name: 'Llama 3, TP', summary: Megatron TP.}\n")
    monkeypatch.setattr(arch_library, "ARCH_CATALOG", archs)
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
    # No registered config reads its rows yet.
    assert gemm["used_by"] == []
    # A kind with no table and no registered config is still listed.
    assert kernels["rms_norm"]["rows"] == 0 and kernels["rms_norm"]["used_by"] == []
    assert [g["name"] for g in catalog["gpus"]] == ["NVIDIA H200", "NVIDIA B200"]
    peaks = catalog["gpus"][0]["peaks"]
    assert peaks["memory_bandwidth_gbps"]["value"] > 0
    assert peaks["busbw_gbps"]["note"].startswith("NVLink")
    # Throughput ceilings are per compute dtype, for the dtypes the rows record.
    assert set(peaks["tflops"]["by_dtype"]) == {"bf16", "fp16"}
    assert catalog["snapshot"]["last_measured_at"] == PROV[1]
    assert catalog["models"][0]["model_config"] == "llama3_8b"


def test_a_catalog_edit_shows_without_a_restart(registered) -> None:
    client, _, _ = registered

    def model(document: dict) -> dict:
        return document["used_by"][0]["model"]

    assert model(client.get(f"{PREFIX}/kernels/single_gemm").json())["name"] == "Llama 3 8B"
    library.MODEL_CATALOG.write_text("llama3_8b: {name: Llama 3 8B Base, family: Llama}\n")
    assert model(client.get(f"{PREFIX}/kernels/single_gemm").json())["name"] == "Llama 3 8B Base"
    models = client.get(f"{PREFIX}/kernels").json()["models"]
    assert models[0]["name"] == "Llama 3 8B Base"


def test_kernel_detail_joins_docs_roles_and_shapes(registered) -> None:
    client, (qkv, _), _ = registered
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
    deployment = kernel["used_by"][0]
    assert deployment["model"]["name"] == "Llama 3 8B"
    # The supported row lists tp_size [1, 4]; only tp_size 4 is registered.
    assert (deployment["params"], deployment["label"]) == (
        {"tp_size": [4]},
        "llama3_dense_tp, tp_size 4",
    )
    assert deployment["members"] == [{"params": {"tp_size": 4}}]
    assert "validated" not in deployment and "sources" not in deployment
    (shape,) = deployment["shapes"]
    assert shape == {
        "layer": "unified.qkv",
        "pool": "main",
        "db": QKV,
        "config_hash": qkv,
        "members": [0],
    }


def _nvfp4_row(backend: str) -> ProfileRow:
    return ProfileRow(
        args=Nvfp4FusedMoeArgs(
            num_tokens=16,
            hidden_size=6144,
            intermediate_size=2048,
            num_experts=4,
            num_local_experts=4,
            top_k=2,
            input_dtype=DType.BF16,
            weight_format=DType.NVFP4_E2M1,
            group_size=16,
            routing_method="minimax2",
            n_group=1,
            topk_group=1,
            routed_scaling_numerator=5,
            routed_scaling_denominator=2,
            per_expert_batches=(8, 8, 8, 8),
        ),
        metrics=ComputeMetrics(time_ms=0.05, tflops=100.0, memory_bandwidth_gbps=1.0),
        gpu_name="NVIDIA B200",
        backend=backend,
        profiler_git_hash="abc",
        profiler_run_at="2026-09-26T00:00:00+00:00",
    )


class Nvfp4Sources(FixtureSources):
    """The kind's compute dtype is the field Rust tags `#[compute_dtype]`
    (`Nvfp4FusedMoeKernelConfig::COMPUTE_DTYPE_FIELD`, pinned in Rust and
    against the binary below)."""

    def kernel_list(self) -> list[dict]:
        return [{"kind": "nvfp4_fused_moe", "compute_dtype": "weight_format", "kv_dtype": None}]


def test_nvfp4_fused_moe_precision_is_its_nvfp4_compute_dtype(tmp_path: Path) -> None:
    path = tmp_path / "nvfp4.db"
    for spec in iter_kernel_profiler_specs("nvfp4_fused_moe"):
        Table(spec, path).insert([_nvfp4_row(spec.backend)])
    client = TestClient(create_app(KernelLibrary(Nvfp4Sources(db_path=path))))

    catalog = client.get(f"{PREFIX}/kernels").json()
    kernel = next(k for k in catalog["kernels"] if k["kind"] == "nvfp4_fused_moe")
    # Not bf16: that is input_dtype, the activation before in-kernel quantization.
    assert kernel["precisions"] == ["nvfp4_e2m1"]
    assert catalog["precisions"] == ["nvfp4_e2m1"]
    (b200,) = catalog["gpus"]
    assert b200["peaks"]["tflops"] == {"by_dtype": {"nvfp4_e2m1": 9000.0}, "note": "dense"}

    detail = client.get(f"{PREFIX}/kernels/nvfp4_fused_moe").json()
    assert [a["name"] for a in detail["args"] if a["precision"]] == ["weight_format"]
    types = {a["name"]: a["type"] for a in detail["args"]}
    assert types["weight_format"] == types["input_dtype"] == "dtype"
    assert {b["supports"]["compute"][0] for b in detail["backends"].values()} == {"nvfp4_e2m1"}


@pytest.mark.needs_binary
def test_the_binary_tags_nvfp4_weight_format_as_the_compute_dtype(tmp_path: Path) -> None:
    entries = KernelSources(db_path=tmp_path / "unused.db").kernel_list()
    (entry,) = [e for e in entries if e["kind"] == "nvfp4_fused_moe"]
    assert entry["compute_dtype"] == "weight_format"


def test_kernel_without_a_registered_config_has_unknown_roles(client: TestClient) -> None:
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
        "/archs",
        "/archs/llama3_dense_tp",
        "/archs/llama3_dense_tp/cost-tree?gpu=NVIDIA H200&model=llama3_8b&tp_size=4",
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


def test_a_new_registration_shows_after_the_detail_was_cached(registered) -> None:
    client, _, path = registered
    before = client.get(f"{PREFIX}/kernels/single_gemm").json()["used_by"]
    _register(path, {"n": 256, "k": 4096, "dtype": "bf16"}, _supported(4))
    after = client.get(f"{PREFIX}/kernels/single_gemm").json()["used_by"]
    assert after != before


def test_warming_builds_every_document_without_raising(registered) -> None:
    _, _, path = registered
    KernelLibrary(FixtureSources(db_path=path)).warm()


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

    # What registered a config (a preset or predict path) is not published.
    assert "sources" not in document
    deployments = {d["id"]: d for d in document["deployments"]}
    [use] = config["uses"]
    assert (use["pool"], use["role"]) == ("main", "unified.qkv")
    assert use["deployments"] == [{"id": 0, "members": [0]}]
    llama = deployments[0]
    # The model config path is named by its stem, as #[supported] rows and the
    # catalog spell it; the label carries the params the supported rows name.
    assert (llama["model_config"], llama["model"]["name"]) == ("llama3_8b", "Llama 3 8B")
    assert (llama["params"], llama["label"]) == ({"tp_size": [4]}, "llama3_dense_tp, tp_size 4")
    assert llama["varies"] == ["tp_size"]

    by_role = {u["role"]: u for u in configs[gate]["uses"]}
    [asked] = by_role["unified.moe.gate"]["deployments"]
    moe = deployments[asked["id"]]
    # No supported rows: the scalar params that change its kernel configs,
    # defaults filled in; the popularity file path stays out of the label.
    assert (moe["model_config"], moe["model"]) == ("moe_x", None)
    assert (moe["params"], moe["varies"]) == ({"ep_size": 8, "mtp_mode": "off"}, [])
    assert moe["label"] == "moe_x, ep_size 8, mtp_mode off"
    assert moe["members"] == [{"params": {}}]
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
    assert set(use) == {"pool", "role", "deployments"}
    [asked] = use["deployments"]
    assert [d["label"] for d in config["deployments"] if d["id"] == asked["id"]] == [
        "llama3_dense_tp, tp_size 4"
    ]
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before


def test_registered_configs_add_models_to_catalog_and_detail(registered) -> None:
    client, (_, gate), _ = registered
    catalog = client.get(f"{PREFIX}/kernels").json()
    gemm = next(k for k in catalog["kernels"] if k["kind"] == "single_gemm")
    # The catalog's Llama, then the MoE model the catalog does not name.
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
    assert labels == ["llama3_dense_tp, tp_size 4", "moe_x, ep_size 8, mtp_mode off"]
    moe = kernel["used_by"][-1]
    [shape] = moe["shapes"]
    assert (shape["layer"], shape["pool"], shape["config_hash"]) == (
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


def _supported(tp_size: int) -> dict:
    """The source ``--register-supported-kernel-configs`` writes for one
    expansion of the llama3_dense_tp row (``tp_size`` in [1, 4])."""

    return {
        "supported": {
            "arch": "llama3_dense_tp",
            "gpu": "NVIDIA H200",
            "params": {"model_config": "llama3_8b", "tp_size": tp_size},
        }
    }


def _alignment(pack: str, variant: str, tp_size: int) -> dict:
    return {
        "alignment": {"pack": pack, "variant": variant, "cases": ["01_micro"]},
        "deployment": "unified",
        "pool": "main",
        "preset": None,
        "groups": [
            {
                "gpu": "NVIDIA H200",
                "arch": {
                    "type": "llama3_dense_tp",
                    "model_config": "model/config/llama3_8b.json",
                    "tp_size": tp_size,
                },
            }
        ],
    }


def _register(path: Path, identity: dict, source: dict) -> None:
    kernel_config.register_kernel_configs(
        path,
        {
            "schema_version": kernel_config.RECORDS_SCHEMA_VERSION,
            "configs": [_record(identity, [{"pool": "main", "role": "unified.qkv"}])],
        },
        {"main": source},
    )


@pytest.fixture
def supported(tmp_path: Path) -> tuple[TestClient, tuple[str, str]]:
    """Both expansions of one supported row read the qkv config; only tp_size 4
    reads the gate config. An alignment run of tp_size 4 names that deployment
    again, and stays one member of it."""

    path = tmp_path / "supported.db"
    table = Table(next(iter_kernel_profiler_specs("single_gemm")), path)
    table.insert([_row(1, 6144, "torch"), _row(1, 512, "torch")])
    _register(path, QKV, _supported(1))
    _register(path, QKV, _supported(4))
    _register(path, GATE, _supported(4))
    pack = "presets/alignment/glm52_nvfp4_b200_sglang_tp4"
    _register(path, GATE, _alignment(pack, "tp4_dp1_sglang", 4))
    client = TestClient(create_app(KernelLibrary(FixtureSources(db_path=path))))
    return client, (kernel_config.content_hash(QKV), kernel_config.content_hash(GATE))


def test_a_supported_row_is_one_deployment_with_its_value_list(supported) -> None:
    client, (qkv, gate) = supported
    document = client.get(f"{PREFIX}/kernels/single_gemm/configs").json()
    [deployment] = document["deployments"]
    assert deployment["label"] == "llama3_dense_tp, tp_size 1 · 4"
    assert (deployment["params"], deployment["varies"]) == ({"tp_size": [1, 4]}, ["tp_size"])
    assert [m["params"] for m in deployment["members"]] == [{"tp_size": 1}, {"tp_size": 4}]
    # Each config names the members that read it: gate is tp_size 4 only.
    asked = {c["config_hash"]: c["uses"][0]["deployments"] for c in document["configs"]}
    assert asked == {qkv: [{"id": 0, "members": [0, 1]}], gate: [{"id": 0, "members": [1]}]}

    [used] = client.get(f"{PREFIX}/kernels/single_gemm").json()["used_by"]
    assert {s["config_hash"]: s["members"] for s in used["shapes"]} == {qkv: [0, 1], gate: [1]}


def test_local_paths_in_a_config_identity_are_cut_to_file_names() -> None:
    identity = {"expert_demand": {"corpus": {"data_file": "/raid/hf/hub/blobs/dfb7"}}, "n": 1}
    assert library._without_local_paths(identity) == {
        "expert_demand": {"corpus": {"data_file": "dfb7"}},
        "n": 1,
    }


def test_public_responses_carry_no_source_paths(supported) -> None:
    client, (qkv, gate) = supported
    base = f"{PREFIX}/kernels/single_gemm"
    for path in ("", "/configs", f"/configs/{qkv}", f"/configs/{gate}"):
        text = client.get(f"{base}{path}").text
        assert "presets/" not in text and "model/config/" not in text, path
        assert '"source' not in text and '"via"' not in text, path


# The routed demand of five nvfp4_fused_moe configs and the arch blocks that
# built each: the name comes from the blocks, the fingerprint from the demand.
TRACKED = "presets/alignment/glm52_nvfp4_b200/expert_popularity.json"
HUB_REFERENCE = "hf://o/corpora@abc/run1/manifest.json"


def _popularity(share: int) -> dict:
    return {"popularity": {"layerwise_global_ppm": [[share, 1_000_000 - share, 0, 0]]}}


CORPUS = {
    "corpus": {
        "schema_version": 1,
        "data_file": "/hub/models--o--corpora/blobs/f00d",
        "num_tokens": 64,
        "num_layers": 2,
        "num_experts": 4,
        "top_k": 2,
        "checksum_fnv1a64": 0xABC,
        "group_size": 6,
        "layer_start": 0,
        "layer_end": 1,
        "seed": 1,
        "sampling_candidates": 16,
    }
}


def _moe_run(**routing) -> dict:
    return {**RUN, "groups": [{"gpu": "NVIDIA B200", "arch": {**MOE_ARCH, **routing}}]}


def _nvfp4_record(demand: dict, position: int) -> dict:
    row = _nvfp4_row("flashinfer_trtllm_sm100").args
    cell = {**asdict(row), "input_dtype": "bf16", "weight_format": "nvfp4_e2m1"}
    cell["per_expert_batches"] = list(cell["per_expert_batches"])
    identity = {
        **{k: v for k, v in cell.items() if k not in ("num_tokens", "per_expert_batches")},
        "expert_demand": demand,
        "folded_rank_position": position,
    }
    return {
        "kind": "nvfp4_fused_moe",
        "profile_kind": "nvfp4_fused_moe",
        "gpu_name": "NVIDIA B200",
        "identity": identity,
        "grid": {
            "cache_coords": ["num_tokens"],
            "axes": [[16.0]],
            "cells": [cell],
            "infeasible": [],
        },
        "uses": [{"pool": "main", "role": "unified.moe.fused_moe"}],
    }


@pytest.fixture
def routings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, dict]:
    monkeypatch.setattr(library, "kernel_doc", kernel_doc)
    path = tmp_path / "routings.db"
    for spec in iter_kernel_profiler_specs("nvfp4_fused_moe"):
        Table(spec, path).insert([_nvfp4_row(spec.backend)])
    built = {
        "uniform": (_popularity(250_000), _moe_run(routing="uniform")),
        "tracked": (_popularity(400_000), _moe_run(expert_popularity_file=TRACKED)),
        "absolute": (
            _popularity(500_000),
            _moe_run(expert_popularity_file=str(REPO_ROOT / TRACKED)),
        ),
        "outside": (
            _popularity(600_000),
            _moe_run(expert_popularity_file="/elsewhere/run/expert_popularity.json"),
        ),
        "corpus": (CORPUS, _moe_run(routing="corpus", token_corpus_file=HUB_REFERENCE)),
    }
    hashes = {}
    for name, (demand, source) in built.items():
        records = [_nvfp4_record(demand, position) for position in (0, 1)]
        kernel_config.register_kernel_configs(
            path,
            {"schema_version": kernel_config.RECORDS_SCHEMA_VERSION, "configs": records},
            {"main": source},
        )
        hashes[name] = kernel_config.content_hash(records[0]["identity"])
    client = TestClient(create_app(KernelLibrary(Nvfp4Sources(db_path=path))))
    return client, hashes


def test_a_kind_doc_declares_its_view(routings) -> None:
    routed, _ = routings
    view = routed.get(f"{PREFIX}/kernels/nvfp4_fused_moe").json()["view"]
    assert view == asdict(kernel_doc("nvfp4_fused_moe").view)
    assert (view["series"]["field"], view["workload"]["field"]) == (
        "folded_rank_position",
        "expert_demand",
    )
    assert routed.get(f"{PREFIX}/kernels/moe_finalize_routing").json()["view"] is None


def test_each_config_names_its_routing(routings) -> None:
    client, hashes = routings
    document = client.get(f"{PREFIX}/kernels/nvfp4_fused_moe/configs").json()
    names = {
        c["config_hash"]: c["config_labels"]["expert_demand"]
        for c in document["configs"]
        if c["config_args"]["folded_rank_position"] == 0
    }
    label = {name: names[h]["label"] for name, h in hashes.items()}
    assert label["uniform"] == "uniform"
    # A file this checkout tracks, named relative or absolute: its repo path.
    assert label["tracked"] == label["absolute"] == TRACKED
    assert names[hashes["tracked"]]["reference"] == TRACKED
    # A hub artifact: the reference the preset wrote, as the launcher records it.
    assert label["corpus"] == names[hashes["corpus"]]["reference"] == HUB_REFERENCE
    # Any other file: its name and the demand's fingerprint, no path.
    outside = names[hashes["outside"]]
    assert outside["reference"] is None
    assert outside["label"] == f"expert_popularity.json · {outside['fingerprint']}"
    assert outside["fingerprint"].startswith("sha256:")
    assert names[hashes["corpus"]]["fingerprint"] == "fnv1a64:0000000000000abc"
    assert names[hashes["corpus"]]["binding"] == {
        "group_size": 6,
        "layer_start": 0,
        "layer_end": 1,
    }
    assert names[hashes["uniform"]]["binding"] == {"layers": 1}
    # A measured routing is preferred, a corpus first; uniform never leads.
    order = sorted(names.values(), key=lambda n: n["preference"])
    assert (order[0]["routing"], order[-1]["routing"]) == ("corpus", "uniform")
    assert "/elsewhere" not in client.get(f"{PREFIX}/kernels/nvfp4_fused_moe/configs").text

    detail = client.get(
        f"{PREFIX}/kernels/nvfp4_fused_moe/configs/{hashes['tracked']}",
        params={"gpu": "NVIDIA B200"},
    ).json()
    assert detail["config_labels"]["expert_demand"]["label"] == TRACKED


# -- archs -----------------------------------------------------------------------

LLAMA_TP4 = {"gpu": "NVIDIA H200", "model": "llama3_8b", "tp_size": "4"}


def test_arch_catalog_names_each_arch_and_counts_its_parameter_sets(client: TestClient) -> None:
    catalog = client.get(f"{PREFIX}/archs").json()
    archs = {a["arch"]: a for a in catalog["archs"]}
    llama = archs["llama3_dense_tp"]
    assert (llama["name"], llama["summary"]) == ("Llama 3, TP", "Megatron TP.")
    assert (llama["contract"], llama["models"], llama["families"]) == (
        "iter_wise",
        ["llama3_8b"],
        ["Llama"],
    )
    assert (llama["gpus"], llama["params"]) == (["NVIDIA H200"], ["tp_size"])
    # One row, so one parameter set; it lists tp_size 1 and 4.
    assert (llama["param_sets"], llama["combinations"]) == (1, 2)
    # An arch without rows or catalog entry is listed, last, with nothing to pick.
    assert catalog["archs"][-1]["arch"] == "moe_x"
    assert (archs["moe_x"]["name"], archs["moe_x"]["combinations"]) == (None, 0)
    assert catalog["models"] == [
        {"model_config": "llama3_8b", "name": "Llama 3 8B", "family": "Llama"}
    ]


def test_arch_detail_lists_params_and_the_supported_sets(registered) -> None:
    client, _, _ = registered
    arch = client.get(f"{PREFIX}/archs/llama3_dense_tp").json()
    params = {p["name"]: p for p in arch["params"]}
    assert params["tp_size"]["default"] == 2 and params["tp_size"]["values"] == [1, 4]
    assert params["model_config"]["values"] == ["llama3_8b"]
    # fp8 is an arch param the rows do not choose: the trees take its default.
    assert params["fp8"]["values"] is None
    assert arch["query"] == ["gpu", "model", "tp_size"]
    [param_set] = arch["param_sets"]
    # The kernel library's deployment-entry shape.
    assert param_set["label"] == "llama3_dense_tp, tp_size 1 · 4"
    assert (param_set["params"], param_set["varies"]) == ({"tp_size": [1, 4]}, ["tp_size"])
    one, four = param_set["members"]
    assert four["query"] == {"gpu": "NVIDIA H200", "model": "llama3_8b", "tp_size": 4}
    assert (four["label"], four["gpus_per_replica"], four["error"]) == (
        "llama3_dense_tp, tp_size 4",
        4,
        None,
    )
    # Three distinct configs: qkv is registered and measured, the other two are not.
    assert four["counts"] == {"leaves": 4, "configs": 3, "registered": 1, "measured": 1}


def test_cost_tree_nests_the_manifest_in_the_analyzer_shape(registered) -> None:
    client, (qkv, _), _ = registered
    tree = client.get(f"{PREFIX}/archs/llama3_dense_tp/cost-tree", params=LLAMA_TP4).json()
    assert (tree["label"], tree["params"], tree["gpus_per_replica"]) == (
        "llama3_dense_tp, tp_size 4",
        {"tp_size": 4},
        4,
    )
    # Params no row chooses take their schema defaults; the document says which.
    assert (tree["defaults"], tree["set_when_predicting"]) == ({"fp8": False}, [])
    [section] = tree["sections"]
    root = section["root"]
    assert (section["section"], root["kind"], root["id"], root["path"]) == (
        "iter",
        "sum",
        0,
        "unified",
    )
    assert root["label"] == "unified [dense TP (tp=4)]"
    embedding, layers = root["children"]
    assert embedding == {
        "id": 1,
        "kind": "leaf",
        "slot": {
            "index": 0,
            "name": "unified.embedding",
            "kind": "elementwise",
            "backends": ["triton"],
            "config_hash": kernel_config.content_hash(
                {"input_bytes_per_token": 8192, "output_bytes_per_token": 8192}
            ),
        },
    }
    assert (layers["kind"], layers["n"], layers["label"]) == ("scale", 32, "layer")
    [layer] = layers["children"]
    assert layer["path"] == "unified.layer"
    ranks, mlp = layer["children"]
    assert (ranks["kind"], ranks["overlap"], ranks["path"]) == (
        "max",
        1.0,
        "unified.layer.attn.qkv",
    )
    # Both ranks are one config: the Dims reduce to the registered qkv identity.
    assert [r["slot"]["config_hash"] for r in ranks["children"]] == [qkv, qkv]
    assert [r["slot"]["index"] for r in ranks["children"]] == [1, 2]
    # Structure only: no time anywhere in the tree.
    assert "timing" not in str(tree["sections"]) and "ms" not in tree
    config = tree["configs"][qkv]
    assert config == {
        "kind": "single_gemm",
        "args": QKV,
        "args_omitted": [],
        "registry": {"cells": 3, "infeasible": 1, "measured": {"torch": 2, "torch_linear": 1}},
    }
    assert tree["configs"][mlp["slot"]["config_hash"]]["registry"] is None
    assert tree["kernels"]["single_gemm"] == {
        "documented": True,
        "title": "Dense GEMM",
        "category": "GEMM",
    }
    assert tree["kernels"]["elementwise"]["documented"] is False


@pytest.mark.parametrize(
    ("params", "status", "message"),
    [
        ({"gpu": "NVIDIA H200", "model": "llama3_8b"}, 400, "needs tp_size"),
        ({**LLAMA_TP4, "routing": "corpus"}, 400, "does not choose routing"),
        ({**LLAMA_TP4, "tp_size": "big"}, 400, "must be an integer"),
        ({**LLAMA_TP4, "tp_size": "2"}, 404, "no #[supported] row"),
        ({**LLAMA_TP4, "gpu": "NVIDIA B200"}, 404, "no #[supported] row"),
    ],
)
def test_a_cost_tree_query_must_name_a_supported_set(
    client: TestClient, params: dict, status: int, message: str
) -> None:
    response = client.get(f"{PREFIX}/archs/llama3_dense_tp/cost-tree", params=params)
    assert response.status_code == status
    detail = response.json()["detail"]
    assert message in detail["message"]
    # Never a guess: the valid queries instead.
    assert [c["tp_size"] for c in detail["choices"]] == [1, 4]


class RoutedSources(FixtureSources):
    """``moe_x`` with a ``#[supported]`` row (and no build), so it has a tree."""

    def deployment_schema(self) -> dict:
        schema = copy.deepcopy(SCHEMA)
        moe = schema["providers"]["arch"]["iter_wise"]["moe_x"]
        moe["supported"] = [{"gpu": ["NVIDIA B200"], "model_config": ["moe_m"], "ep_size": [4]}]
        return schema


def test_a_traffic_param_is_set_when_predicting_not_defaulted(db: Path) -> None:
    client = TestClient(create_app(KernelLibrary(RoutedSources(db_path=db))))
    query = {"gpu": "NVIDIA B200", "model": "moe_m", "ep_size": "4"}
    tree = client.get(f"{PREFIX}/archs/moe_x/cost-tree", params=query).json()
    # The schema marks routing as traffic: its `uniform` default is no choice,
    # so the tree names it instead of listing it among the defaults.
    assert tree["defaults"] == {"fp8": False, "mtp_mode": "off"}
    assert tree["set_when_predicting"] == ["routing"]
    params = {p["name"]: p for p in client.get(f"{PREFIX}/archs/moe_x").json()["params"]}
    assert params["routing"]["set_when_predicting"] is True
    assert "set_when_predicting" not in params["mtp_mode"]
    # A tree query cannot pick it either, and the refusal does not call it a default.
    response = client.get(f"{PREFIX}/archs/moe_x/cost-tree", params={**query, "routing": "corpus"})
    assert response.status_code == 400
    assert response.json()["detail"]["message"].endswith(
        "every other param takes its default, except routing, set when predicting"
    )


def test_an_unknown_arch_is_404(client: TestClient) -> None:
    for path in ("/archs/no_such_arch", "/archs/no_such_arch/cost-tree"):
        assert client.get(f"{PREFIX}{path}").status_code == 404


def test_warming_builds_every_arch_document_without_raising(registered) -> None:
    _, _, path = registered
    kernels = KernelLibrary(FixtureSources(db_path=path))
    ArchLibrary(kernels).warm()


@pytest.mark.needs_binary
def test_a_leaf_config_hash_is_the_registry_key_of_its_config(tmp_path: Path) -> None:
    """Every leaf the build records a config for hashes, from its own kernel
    config, to that record's registry key."""

    sources = KernelSources(db_path=tmp_path / "unused.db")
    builds = sources._simulator(["supported-cost-trees", "--kernel-configs"])
    checked = 0
    for build in builds:
        records: dict[str, set[str]] = {}
        for record in build["kernel_configs"]["configs"]:
            for use in record["uses"]:
                records.setdefault(use["role"], set()).add(
                    kernel_config.content_hash(record["identity"])
                )
        for section in build["cost_manifest"]["sections"]:
            for slot in section["slots"]:
                if slot["name"] in records:
                    identity = config_identity(slot["kernel_config"])
                    assert kernel_config.content_hash(identity) in records[slot["name"]]
                    checked += 1
    assert checked > 1000


@pytest.mark.needs_binary
def test_the_arch_catalog_names_every_arch_tag(tmp_path: Path) -> None:
    import yaml

    schema = KernelSources(db_path=tmp_path / "unused.db").deployment_schema()
    tags = {tag for providers in schema["providers"]["arch"].values() for tag in providers}
    named = yaml.safe_load((REPO_ROOT / "model" / "arch_catalog.yaml").read_text())
    assert set(named) == tags
    assert all(entry["name"] and entry["summary"] for entry in named.values())


# -- trees built as registered runs ------------------------------------------------

GATE_ON = {"n": 256, "k": 4096, "dtype": "bf16"}
MOE_QUERY = {"gpu": "NVIDIA H200", "model": "moe_m", "ep_size": "4"}
MOE_TREE = f"{PREFIX}/archs/moe_x/cost-tree"
POPULARITY = "presets/popularity.json"
PACK = "presets/alignment/moe_x"


def _moe_block(**params) -> dict:
    return {"type": "moe_x", "model_config": "model/config/moe_m.json", "ep_size": 4, **params}


def _moe_source(block: dict, *, preset: str | None = None, cases: list[str] | None = None) -> dict:
    source = {
        "deployment": "unified",
        "pool": "main",
        "preset": preset,
        "groups": [{"gpu": "NVIDIA H200", "arch": block}],
    }
    if cases is not None:
        source["alignment"] = {"pack": PACK, "variant": "ep4", "cases": cases}
    return source


def _moe_build(block: dict) -> dict:
    """A moe_x tree: qkv (always measured), the gate (measured only with
    ``mtp_mode`` off) and the experts, whose config carries the routed demand
    the block's routing gives (``uniform`` where the block names none, as the
    binary defaults it)."""

    gate = GATE if block.get("mtp_mode", "off") == "off" else GATE_ON
    share = 250_000 if block.get("routing", "uniform") == "uniform" else 400_000
    config = {"gpu_name": "NVIDIA H200", "backends": ["torch"]}
    slots = [
        {"name": "unified.qkv", "kind": "single_gemm", "kernel_config": {**config, **QKV}},
        {"name": "unified.moe.gate", "kind": "single_gemm", "kernel_config": {**config, **gate}},
        {
            "name": "unified.moe.experts",
            "kind": "moe_experts",
            "kernel_config": {**config, "expert_demand": _popularity(share)},
        },
    ]
    nodes = [{"Sum": {"children": {"start": 1, "end": 4}}}, {"Leaf": 0}, {"Leaf": 1}, {"Leaf": 2}]
    return {
        "contract": "iter_wise",
        "arch": "moe_x",
        "gpu": "NVIDIA H200",
        "params": {k: v for k, v in block.items() if k != "type"},
        "gpus_per_replica": 4,
        "cost_manifest": {
            "sections": [
                {"section": "iter", "slots": slots, "nodes": nodes, "node_labels": [None] * 4}
            ]
        },
        "error": None,
    }


class RunSources(FixtureSources):
    """``moe_x`` with one ``#[supported]`` row on the H200 and the routing
    files a run names; the binary builds a block with :func:`_moe_build`,
    and git tracks the preset, the pack and the popularity file."""

    built: list[dict]

    def deployment_schema(self) -> dict:
        schema = copy.deepcopy(SCHEMA)
        moe = schema["providers"]["arch"]["iter_wise"]["moe_x"]
        moe["supported"] = [{"gpu": ["NVIDIA H200"], "model_config": ["moe_m"], "ep_size": [4]}]
        moe["params"] = [p for p in moe["params"] if p["name"] != "popularity_file"] + [
            {"name": "expert_popularity_file", "type": "string", "affects_cache": True},
            {"name": "token_corpus_file", "type": "string", "affects_cache": True},
        ]
        return schema

    def supported_cost_trees(self) -> list[dict]:
        # The defaults tree: every param at its default, routing uniform.
        build = _moe_build(_moe_block())
        build["params"] = {"model_config": "moe_m", "ep_size": 4}
        return [build]

    def arch_cost_trees(self, blocks: list[dict]) -> list[dict]:
        self.built = [*getattr(self, "built", []), *blocks]
        return [_moe_build(block["arch"]) for block in blocks]

    def tracked(self, paths: list[str]) -> set[str]:
        return {p for p in paths if p in ("presets/moe_x.yaml", POPULARITY, PACK)}


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Registered runs of the moe_x set, in registration order:

    - an alignment case, popularity routing, ``mtp_mode`` off: 2 of 3 measured;
    - a preset with the same params: the same run, recorded twice;
    - an alignment case with ``mtp_mode`` on: 1 of 3;
    - a preset with ``popularity2.json``, ``mtp_mode`` off: 2 of 3, as the first;
    - a run that did not record its routing (built uniform), and one naming a
      hub corpus this machine does not hold: both skipped;
    - the ``supported`` source, which registers the defaults tree: never a run.
    """

    root = tmp_path / "repo"
    (root / "model" / "config").mkdir(parents=True)
    (root / "model" / "config" / "moe_m.json").write_text("{}")
    (root / "presets").mkdir()
    (root / POPULARITY).write_text("{}")
    (root / "presets" / "popularity2.json").write_text("{}")
    monkeypatch.setattr(arch_library, "REPO_ROOT", root)

    from launcher import corpus

    def no_download(*args):
        raise AssertionError("the public API never downloads")

    monkeypatch.setattr(corpus, "_download", no_download)
    monkeypatch.setattr(corpus, "_cached", lambda *a: (_ for _ in ()).throw(FileNotFoundError()))

    path = tmp_path / "runs.db"
    table = Table(next(iter_kernel_profiler_specs("single_gemm")), path)
    table.insert([_row(1, 6144, "torch"), _row(1, 512, "torch")])
    popular = {"routing": "popularity", "expert_popularity_file": POPULARITY, "mtp_mode": "off"}
    _register(path, QKV, _moe_source(_moe_block(**popular), cases=["01_micro"]))
    _register(path, GATE, _moe_source(_moe_block(**popular), preset="presets/moe_x.yaml"))
    _register(path, GATE_ON, _moe_source(_moe_block(**{**popular, "mtp_mode": "on"}), cases=["02"]))
    second = {**popular, "expert_popularity_file": "presets/popularity2.json"}
    _register(path, QKV, _moe_source(_moe_block(**second), preset="presets/moe_y.yaml"))
    _register(path, QKV, _moe_source(_moe_block(mtp_mode="off"), preset="presets/old.yaml"))
    corpus_run = {"routing": "corpus", "token_corpus_file": "hf://o/c@abc1234/r/manifest.json"}
    _register(path, QKV, _moe_source(_moe_block(**corpus_run), preset="presets/moe_x.yaml"))
    supported = {"arch": "moe_x", "gpu": "NVIDIA H200", "params": {"model_config": "moe_m"}}
    _register(
        path, QKV, {"supported": {**supported, "params": {**supported["params"], "ep_size": 4}}}
    )
    sources = RunSources(db_path=path)
    return TestClient(create_app(KernelLibrary(sources))), sources


def test_a_tree_is_built_as_its_best_measured_registered_run(runs) -> None:
    client, sources = runs
    tree = client.get(MOE_TREE, params=MOE_QUERY).json()
    run = tree["run"]
    assert run["basis"] == "registry"
    assert tree["counts"] == {"leaves": 3, "configs": 3, "registered": 2, "measured": 2}
    assert run["params"] == {
        "fp8": False,
        "routing": "popularity",
        "mtp_mode": "off",
        "expert_popularity_file": POPULARITY,
    }
    # Two sources recorded that run: the preset leads the alignment case, and
    # it beats the equally measured popularity2 run, a preset registered later.
    assert [(s["kind"], s["name"]) for s in run["sources"]] == [
        ("preset", "presets/moe_x.yaml"),
        ("alignment", PACK),
    ]
    assert run["sources"][1]["cases"] == ["01_micro"]
    assert run["routing"]["routing"] == "popularity"
    assert run["routing"]["label"] == POPULARITY
    assert run["query"] == {**{k: v for k, v in MOE_QUERY.items()}, "ep_size": 4, **run["params"]}
    # Built with the run's own params; no default stands in for routing.
    assert (tree["defaults"], tree["set_when_predicting"]) == ({"fp8": False}, [])
    # The binary got local paths, and no build of a run that names no routing.
    assert all(b["arch"].get("routing") == "popularity" for b in sources.built)
    assert all(Path(b["arch"]["model_config"]).is_file() for b in sources.built)
    assert {Path(b["arch"]["expert_popularity_file"]).name for b in sources.built} == {
        "popularity.json",
        "popularity2.json",
    }
    # The arch's set lists the same counts and run.
    [member] = client.get(f"{PREFIX}/archs/moe_x").json()["param_sets"][0]["members"]
    assert member["counts"] == tree["counts"]
    assert member["run"]["sources"] == ["presets/moe_x.yaml", PACK]


def test_the_pickers_offer_only_values_a_registered_run_used(runs) -> None:
    client, _ = runs
    run = client.get(MOE_TREE, params=MOE_QUERY).json()["run"]
    pickers = {p["name"]: p for p in run["pickers"]}
    assert list(pickers) == ["fp8", "routing", "mtp_mode"]
    assert pickers["fp8"]["fixed"] and not pickers["mtp_mode"]["fixed"]
    assert pickers["routing"]["keys"] == ["routing", "expert_popularity_file"]
    mtp = {o["value"]["mtp_mode"]: o for o in pickers["mtp_mode"]["options"]}
    assert (mtp["off"]["selected"], mtp["on"]["selected"]) == (True, False)
    assert mtp["on"]["compatible"] and mtp["on"]["counts"]["measured"] == 1
    # A tracked file by its repo path; another by its name and demand fingerprint.
    files = [o["value"]["expert_popularity_file"] for o in pickers["routing"]["options"]]
    assert files[1] == POPULARITY and files[0].startswith("popularity2.json · sha256:")
    # Picking another value builds that run's tree.
    other = client.get(MOE_TREE, params={**MOE_QUERY, "mtp_mode": "on"}).json()
    assert other["run"]["params"]["mtp_mode"] == "on"
    assert other["counts"]["measured"] == 1
    assert [s["kind"] for s in other["run"]["sources"]] == ["alignment"]
    on = {p["name"]: p for p in other["run"]["pickers"]}["routing"]["options"]
    # Only the first popularity file was run with mtp_mode on.
    assert [o["compatible"] for o in on] == [
        o["value"]["expert_popularity_file"] == POPULARITY for o in on
    ]


def test_a_run_param_no_registered_run_used_is_refused(runs) -> None:
    client, _ = runs
    response = client.get(MOE_TREE, params={**MOE_QUERY, "mtp_mode": "sometimes"})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "no registered run" in detail["message"]
    assert sorted(c["mtp_mode"] for c in detail["choices"]) == ["off", "off", "on"]
    # A routing no run recorded: uniform was never chosen, so it is no value.
    assert client.get(MOE_TREE, params={**MOE_QUERY, "routing": "uniform"}).status_code == 404
    response = client.get(MOE_TREE, params={**MOE_QUERY, "draft_tokens": "3"})
    assert response.status_code == 400
    assert "registered runs of this set also choose" in response.json()["detail"]["message"]


def test_runs_that_did_not_record_their_routing_or_lack_a_file_are_skipped(runs) -> None:
    client, sources = runs
    run = client.get(MOE_TREE, params=MOE_QUERY).json()["run"]
    skipped = {s["sources"][0]["name"]: s for s in run["skipped"]}
    assert set(skipped) == {"old.yaml", "presets/moe_x.yaml"}
    assert skipped["old.yaml"]["error"] == "the run did not record its routing"
    assert "routing" not in skipped["old.yaml"]["params"]
    assert "not in this machine's hub cache" in skipped["presets/moe_x.yaml"]["error"]
    assert not any(b["arch"].get("routing", "uniform") == "uniform" for b in sources.built)
    assert run["combinations"] == 3


def test_a_set_no_registered_run_matches_is_built_at_its_defaults(runs, db: Path) -> None:
    client = TestClient(create_app(KernelLibrary(RunSources(db_path=db))))
    tree = client.get(MOE_TREE, params=MOE_QUERY).json()
    assert tree["run"]["basis"] == "defaults"
    assert (tree["run"]["pickers"], tree["run"]["sources"]) == ([], [])
    assert tree["set_when_predicting"] == ["routing"]
    assert "routing" not in tree["defaults"]
    response = client.get(MOE_TREE, params={**MOE_QUERY, "mtp_mode": "on"})
    assert response.status_code == 400
    assert response.json()["detail"]["message"].endswith(
        "every other param takes its default, except routing, set when predicting"
    )
