"""The public API serves the kernel library read-only.

The simulator's introspection, the kind docs and the model catalog are
fixtures; the profiling registry and a small profile.db are real.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from profiling.db import storage
from profiling.db.args import DType
from profiling.db.doc import CUPTI_METHOD, KernelDoc, arg_docs, kernel_doc
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.db.table import ProfileRow, Table
from profiling.kernels.nvfp4_fused_moe import Nvfp4FusedMoeArgs
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.metrics import ComputeMetrics
from public_api.app import PREFIX, create_app
from public_api.kernel import library
from public_api.kernel.library import PROVENANCE, KernelLibrary
from public_api.kernel.sources import KernelSources

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


class FixtureSources(KernelSources):
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
    path.write_text("llama3_8b: {name: Llama 3 8B, family: Llama}\n")
    monkeypatch.setattr(library, "MODEL_CATALOG", path)
    monkeypatch.setattr(
        library, "kernel_doc", lambda kind: GEMM_DOC if kind == "single_gemm" else None
    )


@pytest.fixture
def client(db: Path) -> TestClient:
    return TestClient(create_app(KernelLibrary(FixtureSources(db_path=db))))


def test_catalog_lists_every_kind_with_coverage_and_the_models(client: TestClient) -> None:
    catalog = client.get(f"{PREFIX}/kernels").json()
    kernels = {k["kind"]: k for k in catalog["kernels"]}
    gemm = kernels["single_gemm"]
    assert gemm["documented"] and gemm["category"] == "GEMM"
    assert gemm["rows"] == 4
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
        {"model_config": "llama3_8b", "name": "Llama 3 8B", "family": "Llama"}
    ]


def test_a_catalog_edit_shows_without_a_restart(client: TestClient) -> None:
    def name() -> str:
        return client.get(f"{PREFIX}/kernels").json()["models"][0]["name"]

    assert name() == "Llama 3 8B"
    library.MODEL_CATALOG.write_text("llama3_8b: {name: Llama 3 8B Base, family: Llama}\n")
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
    assert kernel["method"].startswith(CUPTI_METHOD)
    assert kernel["view"] is None


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
    client = TestClient(create_app(KernelLibrary(Nvfp4Sources(db_path=path))))

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
    entries = KernelSources(db_path=tmp_path / "unused.db").kernel_list()
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
    client = TestClient(create_app(KernelLibrary(FixtureSources(db_path=db))))
    view = client.get(f"{PREFIX}/kernels/nvfp4_fused_moe").json()["view"]
    assert view == asdict(kernel_doc("nvfp4_fused_moe").view)
    assert (view["series"]["field"], view["workload"]["field"]) == (
        "folded_rank_position",
        "expert_demand",
    )
    assert client.get(f"{PREFIX}/kernels/moe_finalize_routing").json()["view"] is None
