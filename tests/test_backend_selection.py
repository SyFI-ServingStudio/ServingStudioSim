"""Unit tests for the per-kernel backend-selection launcher module.

CPU tier (unmarked): these drive `launcher.backends` with SYNTHETIC
`emit-backends` records (the JSON the Rust enumerator would emit — config,
dtype, and GPU are typed fields), so no binary / GPU is needed.
They assert shape parsing, dedup/count, skeleton render, and — against the real
`profiling.db.registry` capability table — the validation gate, including the
two-axis `fa2` bf16-query/fp8-KV case (the current prod config).
"""

from __future__ import annotations

import pytest

from launcher.backends import (
    BackendEnumError,
    Role,
    _merge_role_variants,
    _parse_shape,
    _structure_key,
    dedup_roles,
    render_skeleton,
    validate_backend_map,
    validate_backends_for_candidates,
)
from profiling.db.args import DType


def _rec(pool, name, kind, config, backends, *, compute=None, kv=None, gpu=None):
    """A synthetic `emit-backends` record. dtype/gpu are TYPED fields — as Rust
    emits them, not scraped from `config`: `compute`/`kv` take a `DType` (serialized
    to its wire form), `gpu` the nvidia-smi name. Only `_parse_shape` reads `config`."""
    return {
        "pool": pool,
        "name": name,
        "kind": kind,
        "gpu": gpu,
        "compute_dtype": compute.value if compute is not None else None,
        "kv_dtype": kv.value if kv is not None else None,
        "config": config,
        "backends": backends,
    }


def _dim(value, expression=None, bindings=None):
    return {"value": value, "expression": expression, "bindings": bindings or {}}


# ── shape parsing (geometry tokens, dtype/backends/gpu/routing stripped) ──────


def test_parse_shape_keeps_geometry_drops_noise():
    # attention: heads/dim kept; backends, gpu_name, all dtype columns dropped.
    cfg = {
        "backends": ["fa2", "fa3"],
        "gpu_name": "NVIDIA H200",
        "num_qo_heads": _dim(16),
        "num_kv_heads": _dim(1),
        "head_dim": _dim(128),
        "q_dtype": "bf16",
        "kv_dtype": "bf16",
        "o_dtype": "bf16",
    }
    assert _parse_shape(cfg) == "num_qo_heads=16 num_kv_heads=1 head_dim=128"


def test_parse_shape_gemm_and_drops_routing_load():
    # grouped GEMM: n/k kept; the 16-int per-expert local_ppm routing load dropped.
    cfg = {
        "backends": ["deepgemm"],
        "gpu_name": "NVIDIA H200",
        "n": _dim(6144),
        "k": _dim(4096),
        "dtype": "fp8_e4m3",
        "local_ppm": [7812, 7812, 7812, 7812],
    }
    assert _parse_shape(cfg) == "n=6144 k=4096"


def test_parse_shape_comm_keeps_fabric():
    assert (
        _parse_shape(
            {
                "backends": ["nccl"],
                "gpu_name": "NVIDIA H200",
                "num_gpus": 4,
                "fabric": "nvlink",
            }
        )
        == "num_gpus=4 fabric=nvlink"
    )


# ── dedup + occurrence count ─────────────────────────────────────────────────


def test_dedup_counts_reused_roles():
    recs = [
        _rec(
            "ffn",
            "afd.moe_expert_compute.gate_up",
            "grouped_gemm",
            {"k": _dim(1), "dtype": "fp8_e4m3"},
            ["deepgemm"],
        ),
        _rec(
            "ffn",
            "afd.moe_expert_compute.gate_up",
            "grouped_gemm",
            {"k": _dim(1), "dtype": "fp8_e4m3"},
            ["deepgemm"],
        ),
        _rec(
            "ffn",
            "afd.lm_head",
            "single_gemm",
            {"n": _dim(1), "k": _dim(1), "dtype": "bf16"},
            ["torch"],
        ),
    ]
    roles = dedup_roles(recs)
    assert [r.key for r in roles] == ["ffn/afd.moe_expert_compute.gate_up", "ffn/afd.lm_head"]
    assert roles[0].count == 2  # two build sites → one knob, xN=2
    assert roles[1].count == 1


# ── skeleton render ──────────────────────────────────────────────────────────


def _afd_roles():
    return dedup_roles(
        [
            _rec(
                "attn",
                "afd.attn.prefill",
                "flashinfer_attn_prefill",
                {"q_dtype": "bf16", "kv_dtype": "bf16", "o_dtype": "bf16"},
                ["fa2", "fa3"],
                compute=DType.BF16,
                kv=DType.BF16,
            ),
            _rec(
                "ffn",
                "afd.moe_expert_compute.gate_up",
                "grouped_gemm",
                {"dtype": "fp8_e4m3"},
                ["deepgemm"],
                compute=DType.FP8_E4M3,
            ),
            _rec(
                "ffn",
                "afd.moe_expert_compute.gate_up",
                "grouped_gemm",
                {"dtype": "fp8_e4m3"},
                ["deepgemm"],
                compute=DType.FP8_E4M3,
            ),
            _rec(
                "", "afd_qkv_transfer", "p2p_inter", {"fabric": "infiniband"}, ["nccl", "nvshmem"]
            ),
        ]
    )


def test_render_skeleton_shape():
    text = render_skeleton(_afd_roles(), {"attn": "qwen3_attn_tp", "ffn": "qwen3_ffn_moe"})
    assert "# ── pool: attn   (arch qwen3_attn_tp) ──" in text
    assert "# ── pool: ffn   (arch qwen3_ffn_moe) ──" in text
    # attention: bf16, options exclude trt (fp8-only) and include cudnn.
    assert "attn/afd.attn.prefill:" in text
    assert "options: fa2 fa3 cudnn" in text
    # reused expert GEMM collapses to one knob with an xN marker.
    assert "ffn/afd.moe_expert_compute.gate_up:" in text
    assert "| x2 |" in text
    assert "dtype=fp8_e4m3" in text
    # the deployment-level transfer kernel is NOT an editable entry, but is noted.
    assert "\n  /afd_qkv_transfer" not in text
    assert "deployment-level kernels (not per-pool overridable): afd_qkv_transfer" in text


def test_render_skeleton_annotates_shape():
    roles = dedup_roles(
        [
            _rec(
                "attn",
                "afd.attn.prefill",
                "flashinfer_attn_prefill",
                {
                    "backends": ["fa2"],
                    "gpu_name": "H200",
                    "num_qo_heads": _dim(16),
                    "num_kv_heads": _dim(1),
                    "head_dim": _dim(128),
                    "q_dtype": "bf16",
                    "kv_dtype": "bf16",
                    "o_dtype": "bf16",
                },
                ["fa2", "fa3"],
                compute=DType.BF16,
                kv=DType.BF16,
            ),
            _rec(
                "ffn",
                "afd.lm_head",
                "single_gemm",
                {"backends": ["deepgemm"], "n": _dim(152064), "k": _dim(4096), "dtype": "fp8_e4m3"},
                ["deepgemm"],
                compute=DType.FP8_E4M3,
            ),
            # a comm kernel with no geometry beyond fabric, at the deployment level.
            _rec(
                "",
                "afd_qkv_transfer",
                "p2p_inter",
                {"backends": ["nccl"], "fabric": "infiniband"},
                ["nccl", "nvshmem"],
            ),
        ]
    )
    text = render_skeleton(roles)
    # geometry surfaced in-line; dtype columns and the candidate set are NOT.
    assert "shape: num_qo_heads=16 num_kv_heads=1 head_dim=128" in text
    assert "shape: n=152064 k=4096" in text
    assert "kv_dtype" not in text  # dtype columns never leak into the shape note
    # deployment-level kernel shows its fabric in the trailing note.
    assert "afd_qkv_transfer (p2p_inter, fabric=infiniband)" in text


# ── multi-variant reconcile (emit across a sweep) ────────────────────────────


def _r(pool, name, shape, kind="single_gemm", default=("deepgemm",)):
    return Role(
        pool=pool, name=name, kind=kind, compute=None, kv=None, default=list(default), shape=shape
    )


def test_merge_variants_same_roles_marks_varying_shape():
    # tp=2 and tp=4: identical role names, but attn shape moves and hidden does not.
    tp2 = [_r("attn", "prefill", "num_qo_heads=32"), _r("ffn", "norm", "hidden=4096")]
    tp4 = [_r("attn", "prefill", "num_qo_heads=16"), _r("ffn", "norm", "hidden=4096")]
    roles = _merge_role_variants([("tp2", tp2), ("tp4", tp4)])
    got = {r.name: r.shape for r in roles}
    assert got["prefill"] == "num_qo_heads=32 (varies)"  # run-0's value + moved flag
    assert got["norm"] == "hidden=4096"  # constant → no mark


def test_merge_variants_rejects_divergent_role_set():
    # tp=1 drops the all_reduce role → one file can't cover both.
    tp1 = [_r("attn", "prefill", "num_qo_heads=32")]
    tp2 = [_r("attn", "prefill", "num_qo_heads=16"), _r("ffn", "tp_allreduce", "num_gpus=2")]
    with pytest.raises(BackendEnumError) as exc:
        _merge_role_variants([("tp2", tp2), ("tp1", tp1)])
    msg = str(exc.value)
    assert "distinct kernel role sets" in msg
    assert "missing ffn/tp_allreduce" in msg  # names the divergent role + variant
    assert "tp1:" in msg


def test_merge_variants_single_structure_is_passthrough():
    only = [_r("attn", "prefill", "num_qo_heads=16")]
    roles = _merge_role_variants([("tp4", only)])
    assert roles[0].shape == "num_qo_heads=16"  # no sweep → no (varies)


# ── validation gate ──────────────────────────────────────────────────────────


def _full_map(roles):
    """A strict-coverage map pinning every pooled role to its default."""
    out: dict = {}
    for r in roles:
        if r.pool:
            out.setdefault(r.pool, {})[r.name] = list(r.default)
    return out


def test_validate_ok_full_coverage():
    roles = _afd_roles()
    assert validate_backend_map(_full_map(roles), roles) == []


def test_validate_unknown_role():
    roles = _afd_roles()
    m = _full_map(roles)
    m["ffn"]["afd.does_not_exist"] = ["triton"]
    errs = validate_backend_map(m, roles)
    assert any("unknown backend role ffn/afd.does_not_exist" in e for e in errs)


def test_validate_incompatible_backend_kind():
    # `fa2` is not a single_gemm backend at all.
    roles = dedup_roles(
        [
            _rec(
                "main",
                "m.qkv",
                "single_gemm",
                {"n": _dim(1), "k": _dim(1), "dtype": "bf16"},
                ["torch"],
                compute=DType.BF16,
            )
        ]
    )
    errs = validate_backend_map({"main": {"m.qkv": ["fa2"]}}, roles)
    assert any("unknown backend 'fa2' for single_gemm" in e for e in errs)


def test_validate_incompatible_backend_dtype():
    # torch has no fp8 rows — torch@fp8 is a registered-but-incompatible combo.
    roles = dedup_roles(
        [
            _rec(
                "ffn",
                "g",
                "single_gemm",
                {"n": _dim(1), "k": _dim(1), "dtype": "fp8_e4m3"},
                ["deepgemm"],
                compute=DType.FP8_E4M3,
            )
        ]
    )
    errs = validate_backend_map({"ffn": {"g": ["torch"]}}, roles)
    assert any("unsupported for single_gemm at dtype=fp8_e4m3" in e for e in errs)


def test_validate_missing_coverage():
    roles = _afd_roles()
    m = _full_map(roles)
    del m["ffn"]["afd.moe_expert_compute.gate_up"]
    errs = validate_backend_map(m, roles)
    assert any("strict coverage" in e and "moe_expert_compute.gate_up" in e for e in errs)


def test_validate_deployment_level_exempt_from_coverage():
    # the pool="" transfer kernel must NOT be required for strict coverage.
    roles = _afd_roles()
    errs = validate_backend_map(_full_map(roles), roles)
    assert not any("afd_qkv_transfer" in e for e in errs)


def test_emit_filters_and_rejects_backends_by_gpu():
    # An fp8 attention role carries its run GPU (parsed from `gpu_name=`); the
    # options column and the validator both honour the capability GPU axis.
    def prefill_role(gpu, backends):
        return dedup_roles(
            [
                _rec(
                    "attn",
                    "afd.attn.prefill",
                    "flashinfer_attn_prefill",
                    {"gpu_name": gpu, "q_dtype": "fp8_e4m3", "kv_dtype": "fp8_e4m3"},
                    backends,
                    compute=DType.FP8_E4M3,
                    kv=DType.FP8_E4M3,
                    gpu=gpu,
                )
            ]
        )[0]

    h200 = prefill_role("NVIDIA H200", ["fa3"])
    assert h200.gpu == "NVIDIA H200"
    assert "trt" not in h200.options and "fa3" in h200.options  # trt filtered on H200
    errs = validate_backend_map({"attn": {"afd.attn.prefill": ["trt"]}}, [h200])
    assert any("gpu=NVIDIA H200" in e for e in errs)  # pinning trt on H200 rejected

    b200 = prefill_role("NVIDIA B200", ["fa3", "trt"])
    assert "trt" in b200.options  # trt is a valid option on B200
    assert validate_backend_map({"attn": {"afd.attn.prefill": ["trt"]}}, [b200]) == []


def test_fa2_bf16_query_fp8_kv_is_valid():
    # THE two-axis case: fa2 with bf16 query + fp8 KV cache — the current prod
    # config. Must validate (fp8 *compute* would not, but fp8 *KV* alone does).
    roles = dedup_roles(
        [
            _rec(
                "attn",
                "afd.attn.prefill",
                "flashinfer_attn_prefill",
                {"q_dtype": "bf16", "kv_dtype": "fp8_e4m3", "o_dtype": "bf16"},
                ["fa2", "fa3"],
                compute=DType.BF16,
                kv=DType.FP8_E4M3,
            ),
        ]
    )
    assert validate_backend_map({"attn": {"afd.attn.prefill": ["fa2"]}}, roles) == []
    # cudnn is bf16-only on the KV axis → rejected for fp8 KV.
    errs = validate_backend_map({"attn": {"afd.attn.prefill": ["cudnn"]}}, roles)
    assert any("unsupported for flashinfer_attn_prefill" in e for e in errs)


def _candidate(log_dir: str, *, threshold: int, trace: str) -> dict:
    return {
        "deployment": "unified",
        "workload": {"trace_files": [trace]},
        "io": {"log_dir": log_dir},
        "pools": {"main": {"migration_threshold": threshold}},
        "backends": {"main": {"unified.attn": ["fa3"]}},
        "_env": {"trace": trace},
    }


def test_workload_io_and_backend_sweeps_share_one_structure():
    base = _candidate("logs/a", threshold=8, trace="a.jsonl")
    same = _candidate("logs/b", threshold=8, trace="b.jsonl")
    same["backends"] = {"main": {"unified.attn": ["fa2"]}}
    other_pool = _candidate("logs/c", threshold=16, trace="a.jsonl")

    assert _structure_key(base) == _structure_key(same)
    assert _structure_key(base) != _structure_key(other_pool)


def test_validation_enumerates_each_structure_once(monkeypatch):
    candidates = [
        _candidate(f"logs/{threshold}_{trace}", threshold=threshold, trace=trace)
        for threshold in (8, 16)
        for trace in ("a.jsonl", "b.jsonl", "c.jsonl")
    ]
    enumerated: list[int] = []

    def enumerate_kernels(config, _build_type):
        threshold = config["pools"]["main"]["migration_threshold"]
        enumerated.append(threshold)
        return [_rec("main", f"role_{threshold}", "rms_norm", {}, ["fa3"])]

    checked: dict[str, list[str]] = {}

    def validate(_backends, roles, *_args, **_kwargs):
        return [role.name for role in roles]

    monkeypatch.setattr("launcher.backends.enumerate_kernels", enumerate_kernels)
    monkeypatch.setattr("launcher.backends.validate_backend_map", validate)

    for error in validate_backends_for_candidates(candidates):
        log_dir, role = error.split(": ")
        checked.setdefault(log_dir, []).append(role)

    assert sorted(enumerated) == [8, 16]
    assert checked == {
        f"logs/{threshold}_{trace}": [f"role_{threshold}"]
        for threshold in (8, 16)
        for trace in ("a.jsonl", "b.jsonl", "c.jsonl")
    }


@pytest.mark.needs_binary
def test_emit_backends_reads_the_real_binary(tmp_path):
    """The binary logs its cost tree to stderr while stdout carries the JSON; the
    launcher must parse stdout alone."""
    from launcher.__main__ import main as launcher_main
    from launcher.exec import REPO_ROOT

    skeleton = tmp_path / "backends.yaml"
    preset = REPO_ROOT / "presets" / "unified_smoke.yaml"
    assert launcher_main(["--emit-backends", str(skeleton), str(preset)]) == 0
    assert "main/unified.pre_attn.qkv_proj" in skeleton.read_text()
