"""The public API serves the kernel library read-only.

The simulator's introspection, the kind docs and the model catalog are
fixtures; the profiling registry and a small profile.db are real.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from profiling.db import storage
from profiling.db.args import DType
from profiling.db.doc import CUPTI_METHOD, KernelDoc, arg_docs, kernel_doc
from profiling.db.registry import BackendSupport, iter_kernel_profiler_specs
from profiling.db.table import ProfileRow, Table
from profiling.kernels.nvfp4_fused_moe import Nvfp4FusedMoeArgs
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.metrics import ComputeMetrics
from public_api import kernels as library
from public_api import predict
from public_api.app import PREFIX, create_app
from public_api.deployments import Config, DeploymentIndex, Member, Preset, config_id
from public_api.kernels import PROVENANCE, REPO_ROOT, KernelLibrary
from public_api.sources import Sources

GEMM_ARGS = {"m": "INTEGER", "n": "INTEGER", "k": "INTEGER", "dtype": "TEXT"}
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


class FixtureSources(Sources):
    """Real profile.db access, canned simulator introspection."""

    def kernel_list(self) -> list[dict]:
        return [{"kind": "single_gemm", "compute_dtype": "dtype", "kv_dtype": None}]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "profile.db"
    conn = sqlite3.connect(path)
    conn.execute(storage.RUN_SCHEMA)
    conn.execute(
        storage.kind_table_schema(
            "single_gemm",
            [f"{c} {t} NOT NULL" for c, t in GEMM_ARGS.items()],
            ["time_ms REAL", "tflops REAL", "memory_bandwidth_gbps REAL", "energy_j REAL"],
        )
    )
    for gpu, backend, *rest in GEMM_ROWS:
        args = dict(zip(GEMM_ARGS, rest[:4], strict=True))
        metrics, (git_hash, run_at, *versions) = rest[4:8], rest[8:]
        conn.execute(
            "insert into single_gemm (gpu_name, backend, m, n, k, dtype, args_hash, run_key, "
            "profiler_run_at, time_ms, tflops, memory_bandwidth_gbps, energy_j) "
            f"values ({', '.join('?' * 13)})",
            (
                gpu,
                backend,
                *args.values(),
                storage.args_hash(args, GEMM_ARGS),
                storage.run_key(conn, (git_hash, *versions)),
                storage.epoch(run_at),
                *metrics,
            ),
        )
    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "meta-llama/Meta-Llama-3-8B: {name: Llama 3 8B, family: Llama, config: llama3_8b}\n"
    )
    monkeypatch.setattr(library, "MODEL_CATALOG", path)
    monkeypatch.setattr(
        library, "kernel_doc", lambda kind: GEMM_DOC if kind == "single_gemm" else None
    )


CHECKPOINT = "meta-llama/Meta-Llama-3-8B"
PRESET = "Meta-Llama-3-8B/dense"
GEMM_IDENTITY = {"n": 6144, "k": 4096, "dtype": "bf16"}
GEMM_CELLS = [{"m": m, **GEMM_IDENTITY} for m in (1, 8, 16)]
GEMM_CONFIG = config_id("single_gemm", "NVIDIA H200", GEMM_IDENTITY)


def _member(tp_size: int) -> Member:
    return Member(
        preset=PRESET,
        params={"tp_size": tp_size},
        gpu="NVIDIA H200",
        arch={"type": "dense", "tp_size": tp_size},
        block={},
        predict={"cases": "fields"},
        sections=[
            {
                "section": "layer",
                "nodes": [{"kind": "leaf", "slot": 0}],
                "slots": [
                    {
                        "name": "qkv",
                        "kernel": "single_gemm",
                        "config": GEMM_CONFIG,
                        "backends": ["torch"],
                    }
                ],
            }
        ],
    )


def _index() -> DeploymentIndex:
    """One preset of two members over one gemm config, three cells of which
    the fixture database measured two on the H200; the tp 2 member lacks one."""
    index = DeploymentIndex(None, {"prefill": ["input_len"]}, {CHECKPOINT: {"name": "Llama 3 8B"}})
    members = [_member(1), _member(2)]
    index.presets[PRESET] = Preset(
        id=PRESET,
        checkpoint=CHECKPOINT,
        arch="dense",
        gpu="NVIDIA H200",
        axes=[{"name": "tp_size", "values": [1, 2]}],
        members=members,
    )
    index.configs[GEMM_CONFIG] = Config(
        id=GEMM_CONFIG,
        kind="single_gemm",
        profile_kind="single_gemm",
        gpu="NVIDIA H200",
        identity=GEMM_IDENTITY,
        grid={
            "cache_coords": ["m"],
            "axes": [[1.0, 8.0, 16.0]],
            "cells": GEMM_CELLS,
            "infeasible": [2],
        },
        uses={(PRESET, 0): ["layer.qkv"], (PRESET, 1): ["layer.qkv"]},
    )
    index.check(lambda member: {"layer.qkv": 1} if member.params["tp_size"] == 2 else {})
    return index


@pytest.fixture
def client(db: Path, tmp_path: Path) -> TestClient:
    return TestClient(create_app(KernelLibrary(FixtureSources(db_path=db), _index()), tmp_path))


def test_catalog_lists_every_kind_with_coverage_and_the_models(client: TestClient) -> None:
    catalog = client.get(f"{PREFIX}/kernels").json()
    kernels = {k["kind"]: k for k in catalog["kernels"]}
    gemm = kernels["single_gemm"]
    assert gemm["documented"] and gemm["category"] == "GEMM"
    assert gemm["rows"] == 4
    assert gemm["used_by"] == [PRESET]
    assert kernels["rms_norm"]["used_by"] == []
    assert gemm["precisions"] == ["bf16", "fp16"]
    assert {"gpu": "NVIDIA B200", "backend": "torch_linear", "precision": "fp16", "rows": 1} in (
        gemm["coverage"]
    )
    # A kind with no table is still listed.
    assert kernels["rms_norm"]["rows"] == 0
    assert [g["name"] for g in catalog["gpus"]] == ["NVIDIA H200", "NVIDIA B200"]
    peaks = catalog["gpus"][0]["peaks"]
    assert peaks["memory_bandwidth_gbps"]["value"] > 0
    assert peaks["busbw_gbps"]["note"].startswith("NVLink")
    # Throughput ceilings are per compute dtype, for the dtypes the rows record.
    assert set(peaks["tflops"]["by_dtype"]) == {"bf16", "fp16"}
    assert catalog["snapshot"]["last_measured_at"] == PROV[1]
    assert catalog["models"] == [
        {
            "checkpoint": "meta-llama/Meta-Llama-3-8B",
            "name": "Llama 3 8B",
            "family": "Llama",
            "config": "llama3_8b",
        }
    ]


def test_a_catalog_edit_shows_without_a_restart(client: TestClient) -> None:
    def name() -> str:
        return client.get(f"{PREFIX}/kernels").json()["models"][0]["name"]

    assert name() == "Llama 3 8B"
    library.MODEL_CATALOG.write_text(
        "meta-llama/Meta-Llama-3-8B: {name: Llama 3 8B Base, family: Llama, config: llama3_8b}\n"
    )
    assert name() == "Llama 3 8B Base"


def test_kernel_detail_joins_docs_args_and_backends(client: TestClient) -> None:
    kernel = client.get(f"{PREFIX}/kernels/single_gemm").json()
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
    # torch has no device requirement: any CUDA GPU.
    assert kernel["backends"]["torch"]["supports"]["gpus"] is None
    assert kernel["method"].startswith(CUPTI_METHOD)
    assert kernel["view"] is None


def test_a_backends_gpus_follow_its_compute_capability() -> None:
    sm10x = library._supports(BackendSupport(compute=None, sm_targets=frozenset({"sm_100f"})))
    assert sm10x["sm_targets"] == ["sm_100f"]
    assert "H200-SXM-141GB" not in sm10x["gpus"]
    assert {"B200-SXM-180GB", "B300-SXM-288GB"} <= set(sm10x["gpus"])
    fp8 = library._supports(BackendSupport(compute=None, min_compute_capability=(8, 9)))
    assert fp8["min_compute_capability"] == "8.9"
    assert "L40S" in fp8["gpus"] and not any(gpu.startswith("A100") for gpu in fp8["gpus"])


# The FP8 block-scale backend of this kind takes FP8 weights in 128-wide blocks.
_FP8_BLOCK_BACKEND = "flashinfer_trtllm_fp8_block_sm100"


def _nvfp4_row(backend: str) -> ProfileRow:
    fp8 = backend == _FP8_BLOCK_BACKEND
    return ProfileRow(
        args=Nvfp4FusedMoeArgs(
            num_tokens=16,
            hidden_size=6144,
            intermediate_size=2048,
            num_experts=4,
            num_local_experts=4,
            top_k=2,
            input_dtype=DType.BF16,
            weight_format=DType.FP8_E4M3 if fp8 else DType.NVFP4_E2M1,
            group_size=128 if fp8 else 16,
            routing_method="deepseek_v3" if fp8 else "minimax2",
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


def test_nvfp4_fused_moe_precision_is_each_backends_weight_format(tmp_path: Path) -> None:
    path = tmp_path / "nvfp4.db"
    for spec in iter_kernel_profiler_specs("nvfp4_fused_moe"):
        Table(spec, path).insert([_nvfp4_row(spec.backend)])
    client = TestClient(create_app(KernelLibrary(Nvfp4Sources(db_path=path), _index())))

    catalog = client.get(f"{PREFIX}/kernels").json()
    kernel = next(k for k in catalog["kernels"] if k["kind"] == "nvfp4_fused_moe")
    # Not bf16: that is input_dtype, the activation before in-kernel quantization.
    assert kernel["precisions"] == ["fp8_e4m3", "nvfp4_e2m1"]
    assert catalog["precisions"] == ["fp8_e4m3", "nvfp4_e2m1"]
    (b200,) = catalog["gpus"]
    assert b200["peaks"]["tflops"] == {
        "by_dtype": {"fp8_e4m3": 4500.0, "nvfp4_e2m1": 9000.0},
        "note": "dense",
    }

    detail = client.get(f"{PREFIX}/kernels/nvfp4_fused_moe").json()
    assert [a["name"] for a in detail["args"] if a["precision"]] == ["weight_format"]
    types = {a["name"]: a["type"] for a in detail["args"]}
    assert types["weight_format"] == types["input_dtype"] == "dtype"
    assert {
        name: backend["supports"]["compute"] for name, backend in detail["backends"].items()
    } == {
        name: ["fp8_e4m3"] if name == _FP8_BLOCK_BACKEND else ["nvfp4_e2m1"]
        for name in detail["backends"]
    }


@pytest.mark.needs_binary
def test_the_binary_tags_nvfp4_weight_format_as_the_compute_dtype(tmp_path: Path) -> None:
    entries = Sources(db_path=tmp_path / "unused.db").kernel_list()
    (entry,) = [e for e in entries if e["kind"] == "nvfp4_fused_moe"]
    assert entry["compute_dtype"] == "weight_format"


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
        "/kernels/single_gemm/rows?format=csv",
    ):
        assert client.get(f"{PREFIX}{path}").status_code == 200
    assert (hashlib.sha256(db.read_bytes()).hexdigest(), db.stat().st_mtime_ns) == before
    with FixtureSources(db_path=db).connect() as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("delete from single_gemm")


def test_a_kind_doc_declares_its_view(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "kernel_doc", kernel_doc)
    client = TestClient(create_app(KernelLibrary(FixtureSources(db_path=db), _index())))
    view = client.get(f"{PREFIX}/kernels/nvfp4_fused_moe").json()["view"]
    assert view == asdict(kernel_doc("nvfp4_fused_moe").view)
    assert (view["series"]["field"], view["workload"]["field"]) == (
        "folded_rank_position",
        "expert_demand",
    )
    assert client.get(f"{PREFIX}/kernels/moe_finalize_routing").json()["view"] is None


def test_every_checkpoint_in_the_catalog_names_a_config_file() -> None:
    catalog = yaml.safe_load((REPO_ROOT / "model" / "catalog.yaml").read_text())
    for checkpoint, entry in catalog.items():
        assert "/" in checkpoint, f"{checkpoint}: key a checkpoint by its Hugging Face repo"
        assert {"name", "family", "config"} <= set(entry), checkpoint
        assert (REPO_ROOT / "model" / "config" / f"{entry['config']}.json").is_file(), checkpoint


def test_a_configs_grid_joins_the_rows_measured_on_its_gpu(client: TestClient) -> None:
    [listed] = client.get(f"{PREFIX}/kernels/single_gemm/configs").json()["configs"]
    assert listed["id"] == GEMM_CONFIG
    assert listed["measured"] == {"torch": 2}
    assert (listed["fixed"], listed["swept"]) == (GEMM_IDENTITY, ["m"])
    assert listed["uses"] == [
        {"preset": PRESET, "params": {"tp_size": 1}, "roles": ["layer.qkv"]},
        {"preset": PRESET, "params": {"tp_size": 2}, "roles": ["layer.qkv"]},
    ]

    detail = client.get(f"{PREFIX}/kernels/single_gemm/configs/{GEMM_CONFIG}").json()
    assert detail["identity"] == GEMM_IDENTITY
    assert [p["coords"] for p in detail["points"]] == [[1.0], [8.0], [16.0]]
    assert [p["feasible"] for p in detail["points"]] == [True, True, False]
    # The B200 row of the same args belongs to another GPU's config.
    assert [p["measured"] for p in detail["points"]] == [
        {
            "torch": {
                "time_ms": 0.01,
                "tflops": 5.0,
                "memory_bandwidth_gbps": 100.0,
                "energy_j": None,
                "outlier": False,
            }
        },
        {
            "torch": {
                "time_ms": 0.02,
                "tflops": 20.0,
                "memory_bandwidth_gbps": 200.0,
                "energy_j": None,
                "outlier": False,
            }
        },
        {},
    ]
    assert client.get(f"{PREFIX}/kernels/single_gemm/configs/0000000000000000").status_code == 404
    assert client.get(f"{PREFIX}/kernels/nvfp4_fused_moe/configs/{GEMM_CONFIG}").status_code == 404


def test_models_list_presets_and_a_tree_is_named_by_its_axes(client: TestClient) -> None:
    [checkpoint] = client.get(f"{PREFIX}/models").json()["checkpoints"]
    [preset] = checkpoint["presets"]
    assert preset["id"] == PRESET
    assert [m["missing"] for m in preset["members"]] == [{}, {"single_gemm": 1}]

    tree = client.get(f"{PREFIX}/models/{PRESET}/tree", params={"tp_size": 1}).json()
    assert tree["configs"][GEMM_CONFIG]["identity"] == GEMM_IDENTITY
    assert tree["configs"][GEMM_CONFIG]["missing"] == 0
    lacking = client.get(f"{PREFIX}/models/{PRESET}/tree", params={"tp_size": 2}).json()
    assert lacking["configs"][GEMM_CONFIG]["missing"] == 1
    bad = client.get(f"{PREFIX}/models/{PRESET}/tree", params={"tp_size": 4})
    assert bad.status_code == 400
    assert bad.json()["detail"]["choices"] == [{"tp_size": 1}, {"tp_size": 2}]
    assert client.get(f"{PREFIX}/models/Nope/dense/tree").status_code == 404


def test_predict_refuses_what_it_cannot_cost(client: TestClient) -> None:
    def post(tp_size: int, cases: list) -> object:
        return client.post(
            f"{PREFIX}/predict",
            json={"preset": PRESET, "params": {"tp_size": tp_size}, "cases": cases},
        )

    lacking = post(2, [{"prefill": []}])
    assert lacking.status_code == 409
    assert "lacks profile.db rows: single_gemm 1" in lacking.json()["detail"]
    assert post(1, []).status_code == 400
    assert client.post(f"{PREFIX}/predict", json={"preset": PRESET}).status_code == 422


def test_only_a_predictions_routes_are_forwarded(client: TestClient) -> None:
    # This service runs no Analyzer; a route outside a prediction is no route.
    assert client.get(f"{PREFIX}/analyzer/predictions/abc/descriptor").status_code == 503
    assert client.get(f"{PREFIX}/analyzer/runs").status_code == 404


def test_a_forwarded_report_names_its_prediction_not_the_host_path(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from public_api import app as app_module

    log_dir = tmp_path.resolve() / "72dd8769"

    def analyzer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"log_dir": str(log_dir)})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        app_module.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=httpx.MockTransport(analyzer), **kwargs),
    )
    kernels = KernelLibrary(FixtureSources(db_path=db), _index())
    client = TestClient(create_app(kernels, tmp_path, "http://analyzer"))
    answer = client.get(f"{PREFIX}/analyzer/predictions/p_1/subjects/scoped-optimality/report")
    assert answer.json() == {"log_dir": "72dd8769"}


def test_a_failed_run_names_no_host_path(tmp_path: Path) -> None:
    scratch, log_dir = tmp_path / "scratch", tmp_path / "runs" / "abc"
    stdout = f"Error: parsing JSON cases file {scratch}/cases.json: unknown field\n"
    assert (
        predict._cause(stdout, scratch, log_dir)
        == "parsing JSON cases file cases.json: unknown field"
    )
