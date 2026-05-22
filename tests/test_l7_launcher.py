"""L7-α launcher unit tests (design §1 verification).

Most tests run against an inline schema fixture (deterministic, no build
dependency); one integration test loads the real Rust-generated schema if the
binary has been built. Covers: validators V1–V6, the five-step sweep expansion
(list / dict / sweep_groups / derived / constraints), argv construction, and
cache-key de-duplication.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from launcher import metadata
from launcher.cache_build import cache_key
from launcher.exec import SimulationRunner, _build_subprocess_env, binary_path
from launcher.schema import (
    _format_log_dir,
    build_cli_command,
    expand_sweep_params,
    normalize_params,
    validate_params,
    validate_unique_log_dirs,
)
from launcher.schema.loader import SchemaNotFound, load_schema, schema_from_dict
from launcher.sweep import (
    COMPLETE_MARKER,
    _aggregate,
    _experiment_root,
    _group_map,
    _is_complete,
    _launch_one,
    _mark_complete,
    _run_single_async,
    _run_sweep_async,
    _sweep_axes,
    run_single,
    run_sweep,
)

# Inline schema mirroring the unified deployment's relevant fields.
_FIXTURE = {
    "deployment_schemas": {
        "unified": [
            {"name": "model_config", "type": "string", "required": True, "description": ""},
            {"name": "tp_size", "type": "int", "default": 4, "description": ""},
            {"name": "ep_size", "type": "int", "default": 8, "description": ""},
            {"name": "head_parallel", "type": "int", "default": 1, "description": ""},
            {
                "name": "cp_plan",
                "type": "string",
                "default": "no-cp",
                "choices": ["no-cp", "ring", "replicated-q", "optimal-prefill"],
                "description": "",
            },
            {"name": "fp8", "type": "bool", "default": False, "description": ""},
            {"name": "trace_files", "type": "path_list", "required": False, "description": ""},
            {"name": "run_to_end", "type": "bool", "default": False, "description": ""},
            {"name": "max_batch_tokens", "type": "int", "required": False, "description": ""},
            {"name": "request_rate", "type": "float", "default": 10.0, "description": ""},
            {"name": "log_dir", "type": "path", "default": "logs", "description": ""},
            {
                "name": "log_level",
                "type": "string",
                "default": "info",
                "choices": ["trace", "debug", "info", "warn", "error"],
                "description": "",
            },
        ]
    },
    # Display-only (drives `list-params --human` grouping). max_batch_tokens is
    # intentionally left out of every fragment → it is a deployment-own param.
    "pool_fragments": {
        "ModelCommon": ["model_config", "fp8"],
        "ParallelismCommon": ["tp_size", "ep_size", "head_parallel", "cp_plan"],
        "WorkloadCommon": ["trace_files", "run_to_end", "request_rate"],
        "IoCommon": ["log_dir", "log_level"],
    },
}


@pytest.fixture
def schema():
    return schema_from_dict(_FIXTURE)


def _base():
    return {"deployment": "unified", "model_config": "m.json"}


# ── validation ──────────────────────────────────────────────────────────────


def test_validate_ok(schema):
    assert validate_params({**_base(), "tp_size": 4}, schema) == []


def test_validate_unknown_key(schema):
    errs = validate_params({**_base(), "bogus_param": 1}, schema)
    assert any("unknown param 'bogus_param'" in e for e in errs)


def test_validate_missing_required(schema):
    errs = validate_params({"deployment": "unified"}, schema)
    assert any("model_config" in e for e in errs)


def test_validate_unknown_deployment(schema):
    errs = validate_params({"deployment": "afd", "model_config": "m"}, schema)
    assert any("unknown deployment" in e for e in errs)


def test_v1_derived_lhs_not_schema_param(schema):
    errs = validate_params({**_base(), "derived": {"aux_name": "tp_size"}}, schema)
    assert any("aux_name" in e and "not a schema param" in e for e in errs)


def test_v2_derived_also_sweep(schema):
    preset = {**_base(), "tp_size": [1, 2], "derived": {"tp_size": "ep_size"}}
    errs = validate_params(preset, schema)
    assert any("also appears as a sweep dim" in e for e in errs)


def test_double_definition_sweep_and_group(schema):
    # §1.2.1.1 step 2: tp_size is both an independent list_sweep and a
    # sweep_groups field → double-definition error.
    preset = {
        **_base(),
        "tp_size": [1, 2],
        "sweep_groups": {"par": [{"tp_size": 4, "head_parallel": 4}]},
    }
    errs = validate_params(preset, schema)
    assert any("double definition" in e and "tp_size" in e for e in errs)


def test_v3_derived_rhs_undefined(schema):
    errs = validate_params({**_base(), "derived": {"tp_size": "nonexistent + 1"}}, schema)
    assert any("undefined names" in e and "nonexistent" in e for e in errs)


def test_v4_required_after_derived(schema):
    # model_config supplied via derived satisfies the required check.
    errs = validate_params(
        {"deployment": "unified", "derived": {"model_config": "'m.json'"}}, schema
    )
    assert errs == []


def test_v5_constraint_undefined_var(schema):
    errs = validate_params({**_base(), "constraints": ["ghost > 0"]}, schema)
    assert any("ghost" in e for e in errs)


def test_v6_disallowed_call(schema):
    errs = validate_params({**_base(), "constraints": ["evil(tp_size)"]}, schema)
    assert any("call to 'evil' not allowed" in e for e in errs)


def test_allowlisted_calls_ok(schema):
    errs = validate_params(
        {**_base(), "derived": {"head_parallel": "max(min(tp_size, 4), 1)"}}, schema
    )
    assert errs == []


def test_ternary_validates_and_evaluates(schema):
    preset = {**_base(), "tp_size": [1, 2], "derived": {"head_parallel": "2 if tp_size > 1 else 1"}}
    assert validate_params(preset, schema) == []
    cands = expand_sweep_params(preset, schema)
    assert {(c["tp_size"], c["head_parallel"]) for c in cands} == {(1, 1), (2, 2)}


def test_unsupported_grammar_rejected(schema):
    # subscript / attribute / comprehension are not part of the expression grammar.
    for expr in ["tp_size[0] > 0", "tp_size.bit_length() > 0", "[x for x in tp_size]"]:
        errs = validate_params({**_base(), "constraints": [expr]}, schema)
        assert errs, f"expected rejection for {expr!r}"


def test_eval_failure_is_loud_not_silent_none(schema):
    # division by zero passes the grammar but must raise, not yield None.
    import pytest as _pytest

    with _pytest.raises(ValueError):
        expand_sweep_params({**_base(), "derived": {"head_parallel": "tp_size // 0"}}, schema)


def test_param_choices_reject_invalid_scalar(schema):
    errs = validate_params({**_base(), "cp_plan": "not-a-plan"}, schema)
    assert any("not one of" in e for e in errs)


def test_param_choices_cover_list_and_dict_sweeps(schema):
    list_errs = validate_params({**_base(), "cp_plan": ["ring", "bad-plan"]}, schema)
    dict_errs = validate_params(
        {**_base(), "log_level": {"quiet": "info", "bad": "verbose"}}, schema
    )
    group_errs = validate_params(
        {**_base(), "sweep_groups": {"io": [{"log_level": "debug"}, {"log_level": "loud"}]}},
        schema,
    )
    assert any("cp_plan='bad-plan'" in e for e in list_errs)
    assert any("log_level='verbose'" in e for e in dict_errs)
    assert any("log_level='loud'" in e for e in group_errs)


# ── sweep expansion ───────────────────────────────────────────────────────


def test_expand_single_no_sweep(schema):
    cands = expand_sweep_params({**_base(), "tp_size": 4}, schema)
    assert len(cands) == 1
    assert cands[0]["tp_size"] == 4


def test_expand_list_sweep_count(schema):
    cands = expand_sweep_params({**_base(), "tp_size": [1, 2, 4, 8]}, schema)
    assert sorted(c["tp_size"] for c in cands) == [1, 2, 4, 8]


def test_expand_list_value_not_swept(schema):
    # path_list value is the value, never a sweep dim.
    cands = expand_sweep_params({**_base(), "trace_files": ["a.csv", "b.csv"]}, schema)
    assert len(cands) == 1
    assert cands[0]["trace_files"] == ["a.csv", "b.csv"]


def test_expand_dict_sweep_labels(schema):
    cands = expand_sweep_params({**_base(), "tp_size": {"lo": 1, "hi": 8}}, schema)
    labels = {c["_sweep_labels"]["tp_size"]: c["tp_size"] for c in cands}
    assert labels == {"lo": 1, "hi": 8}


def test_expand_sweep_groups_zip(schema):
    preset = {
        **_base(),
        "sweep_groups": {
            "parallelism": [
                {"tp_size": 1, "head_parallel": 1},
                {"tp_size": 2, "head_parallel": 2},
            ]
        },
    }
    cands = expand_sweep_params(preset, schema)
    assert {(c["tp_size"], c["head_parallel"]) for c in cands} == {(1, 1), (2, 2)}


def test_expand_cartesian_product(schema):
    preset = {**_base(), "tp_size": [1, 2], "ep_size": [4, 8]}
    cands = expand_sweep_params(preset, schema)
    assert len(cands) == 4


def test_expand_derived_and_constraint(schema):
    preset = {
        **_base(),
        "tp_size": [1, 2, 4, 8],
        "derived": {"head_parallel": "tp_size"},
        "constraints": ["tp_size * ep_size <= 32"],
    }
    cands = expand_sweep_params(preset, schema)
    # ep_size defaults to 8 → tp ∈ {1,2,4} pass (8*8=64 dropped).
    assert sorted(c["tp_size"] for c in cands) == [1, 2, 4]
    assert all(c["head_parallel"] == c["tp_size"] for c in cands)


def test_expand_cartesian_count(schema):
    cands = expand_sweep_params({**_base(), "tp_size": [1, 2], "ep_size": [4, 8]}, schema)
    assert len(cands) == 4


def test_expand_groups_times_independent(schema):
    preset = {
        **_base(),
        "sweep_groups": {
            "par": [{"tp_size": 1, "head_parallel": 1}, {"tp_size": 2, "head_parallel": 2}]
        },
        "ep_size": [4, 8],
    }
    cands = expand_sweep_params(preset, schema)
    seen = {(c["tp_size"], c["head_parallel"], c["ep_size"]) for c in cands}
    assert seen == {(1, 1, 4), (1, 1, 8), (2, 2, 4), (2, 2, 8)}


def test_expand_derived_chain(schema):
    # head_parallel ← tp_size, then a second derived reads the first.
    preset = {**_base(), "tp_size": [2, 4], "derived": {"head_parallel": "tp_size * 1"}}
    cands = expand_sweep_params(preset, schema)
    assert all(c["head_parallel"] == c["tp_size"] for c in cands)


def test_expand_two_dict_sweeps_merge_labels(schema):
    cands = expand_sweep_params(
        {**_base(), "tp_size": {"a": 1, "b": 2}, "ep_size": {"lo": 4, "hi": 8}}, schema
    )
    assert len(cands) == 4
    for c in cands:
        assert set(c["_sweep_labels"]) == {"tp_size", "ep_size"}


# ── normalize + argv ────────────────────────────────────────────────────────


def test_normalize_defaults_and_coerce(schema):
    out = normalize_params({**_base(), "tp_size": "2"}, schema)
    assert out["tp_size"] == 2  # coerced str → int
    assert out["ep_size"] == 8  # default filled
    assert out["fp8"] is False


def test_build_cli_command_basic(schema):
    out = normalize_params({**_base(), "tp_size": 2}, schema)
    argv = build_cli_command(out, "/bin/sim")
    assert argv[:3] == ["/bin/sim", "run", "unified"]
    assert "--tp-size" in argv and argv[argv.index("--tp-size") + 1] == "2"
    # default-false bool → flag omitted.
    assert "--fp8" not in argv and "--run-to-end" not in argv


def test_build_cli_command_bool_and_list(schema):
    out = normalize_params(
        {**_base(), "fp8": True, "trace_files": ["a.csv", "b.csv"]}, schema
    )
    argv = build_cli_command(out, "/bin/sim")
    assert "--fp8" in argv
    assert argv.count("--trace-files") == 2


def test_build_cache_only_subcommand(schema):
    out = normalize_params(_base(), schema)
    argv = build_cli_command(out, "/bin/sim", subcommand="build-cache-only")
    assert argv[1] == "build-cache-only"


# ── cache key ────────────────────────────────────────────────────────────────


# The cache-key field set is Rust-authoritative (affects_cache); the fixture
# tags model_config + tp_size as kernel-determining for these unit tests.
_CACHE_FIELDS = ("model_config", "tp_size")


def test_cache_key_ignores_rate_and_logdir():
    a = {"model_config": "m", "tp_size": 4, "request_rate": 1.0, "log_dir": "x"}
    b = {"model_config": "m", "tp_size": 4, "request_rate": 99.0, "log_dir": "y"}
    assert cache_key(a, _CACHE_FIELDS) == cache_key(b, _CACHE_FIELDS)


def test_cache_key_distinguishes_tp():
    a = {"model_config": "m", "tp_size": 4}
    b = {"model_config": "m", "tp_size": 8}
    assert cache_key(a, _CACHE_FIELDS) != cache_key(b, _CACHE_FIELDS)


def test_cache_key_fields_from_real_schema():
    try:
        real = load_schema("debug")
    except SchemaNotFound:
        pytest.skip("simulator not built")
    fields = set(real.deployment_schemas["unified"].cache_key_fields)
    # kernel-shaping params are tagged; pure workload/IO params are not.
    assert {"model_config", "tp_size", "ep_size", "cp_plan"} <= fields
    assert "request_rate" not in fields and "log_dir" not in fields


# ── log_dir templating ───────────────────────────────────────────────────────


def test_format_log_dir_aliases():
    out = _format_log_dir(
        {
            "deployment": "unified",
            "model_config": "a/llama3_8b.json",
            "tp_size": 4,
            "request_rate": 10.0,
            "log_dir": "logs/{model}_tp{tp}_r{rate}",
        }
    )
    assert out["log_dir"] == "logs/llama3_8b_tp4_r10.0"


def test_format_log_dir_uses_sweep_label():
    out = _format_log_dir(
        {
            "deployment": "unified",
            "tp_size": 2,
            "log_dir": "logs/{tp_size}",
            "_sweep_labels": {"tp_size": "small"},
        }
    )
    assert out["log_dir"] == "logs/small"


def test_format_log_dir_noop_without_template():
    out = _format_log_dir({"deployment": "unified", "log_dir": "logs/plain"})
    assert out["log_dir"] == "logs/plain"


def test_format_log_dir_warns_unknown_placeholder(capsys):
    out = _format_log_dir({"deployment": "unified", "log_dir": "logs/{unknown_sweep}"})
    captured = capsys.readouterr()
    assert out["log_dir"] == "logs/{unknown_sweep}"
    assert "unknown log_dir placeholder {unknown_sweep}" in captured.err


# ── metadata persistence ─────────────────────────────────────────────────────


def test_write_shared_and_run_metadata(tmp_path):
    root_dir = tmp_path / "sweep"
    log_dir = tmp_path / "run"
    preset = {"deployment": "unified", "tp_size": [1, 4]}
    params = {"deployment": "unified", "tp_size": 4, "_sweep_labels": {"x": "y"}}
    metadata.write_shared_metadata(root_dir, preset)
    metadata.write_run_metadata(log_dir, params, ["/bin/sim", "run", "unified"])

    shared_names = {p.relative_to(root_dir).as_posix() for p in root_dir.rglob("*") if p.is_file()}
    names = {p.relative_to(log_dir).as_posix() for p in log_dir.rglob("*") if p.is_file()}
    dirs = {p.relative_to(log_dir).as_posix() for p in log_dir.rglob("*") if p.is_dir()}
    assert {"preset.json", "git_snapshot/commit.txt"} <= shared_names
    assert {"raw/params.json", "raw/command.txt", "manifest.json"} <= names
    assert {"raw", "plots", "reports", "payloads", "traces"} <= dirs
    assert "preset.json" not in names
    assert "git_snapshot/commit.txt" not in names

    params = json.loads((log_dir / "raw" / "params.json").read_text())
    assert "_sweep_labels" not in params  # launcher-internal stripped
    assert "start_ts" in params
    manifest = json.loads((log_dir / "manifest.json").read_text())
    assert manifest["pending_l7_beta"] is True  # no sidecar yet (L7-β not built)


# ── experiment-root (sweep base_dir) ─────────────────────────────────────────


def _root(*paths):
    return str(_experiment_root([{"log_dir": p} for p in paths]))


def test_experiment_root_siblings():
    assert _root("/x/logs/a", "/x/logs/b").endswith("/x/logs")


def test_experiment_root_nested_runs():
    assert _root("/x/logs/a/run", "/x/logs/b/run").endswith("/x/logs")


def test_experiment_root_disjoint_reaches_root():
    assert _root("/x/a/r", "/y/b/r") == "/"


# ── log_dir collision validation ─────────────────────────────────────────────


def test_validate_unique_log_dirs_rejects_duplicates(capsys):
    params = [
        {"deployment": "unified", "model_config": "m", "log_dir": "logs/same"},
        {"deployment": "unified", "model_config": "m", "log_dir": "./logs/same"},
    ]
    assert validate_unique_log_dirs(params) is False
    assert "runs share log_dir" in capsys.readouterr().err


def test_run_sweep_aborts_on_log_dir_collision(schema, capsys):
    params = [
        {"deployment": "unified", "model_config": "m", "log_dir": "logs/same"},
        {"deployment": "unified", "model_config": "m", "log_dir": "logs/same"},
    ]
    assert asyncio.run(_run_sweep_async(params, _base(), schema, "debug", parallelism=1)) == 2
    assert "aborting sweep" in capsys.readouterr().err


def test_main_validates_log_dirs_before_dry_run(tmp_path, schema, monkeypatch, capsys):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset_path = tmp_path / "preset.json"
    preset_path.write_text(
        json.dumps({**_base(), "tp_size": [1, 2], "log_dir": "logs/{model}"})
    )
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)

    assert main_module.main([str(preset_path), "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert "[plan] 2 run(s) across 1 preset(s)" in captured.out
    assert "aborting sweep" in captured.err


# ── aggregator contract (sweep axes + group hint) ───────────────────────────


def test_sweep_axes_are_mechanism_agnostic(schema):
    # tp/ep vary; model_config constant; derived-style varying param surfaces too.
    runs = [
        {"deployment": "unified", "model_config": "m", "tp_size": t, "ep_size": e, "nvl": n}
        for t, e, n in [(1, 4, 1), (2, 4, 2), (1, 8, 1)]
    ]
    axes = _sweep_axes(runs)
    assert set(axes) == {"tp_size", "ep_size", "nvl"}  # varying params
    assert "model_config" not in axes  # constant


def test_sweep_axes_excludes_log_dir(schema):
    runs = [
        {"deployment": "unified", "tp_size": 1, "log_dir": "a"},
        {"deployment": "unified", "tp_size": 2, "log_dir": "b"},
    ]
    assert _sweep_axes(runs) == ["tp_size"]  # log_dir is output location, not an axis


def test_group_map_extracts_zip_fields():
    preset = {
        "sweep_groups": {
            "par": [{"tp_size": 1, "head_parallel": 1}, {"tp_size": 2, "head_parallel": 2}]
        }
    }
    assert _group_map(preset) == {"par": ["head_parallel", "tp_size"]}


def test_aggregate_passes_groups_kwarg(monkeypatch, tmp_path):
    calls = []

    def aggregate_sweep(run_infos, base_dir, *, groups):
        calls.append((run_infos, base_dir, groups))

    package = types.ModuleType("analyze_aggregator")
    package.__path__ = []
    module = types.ModuleType("analyze_aggregator.aggregator")
    module.aggregate_sweep = aggregate_sweep
    monkeypatch.setitem(sys.modules, "analyze_aggregator", package)
    monkeypatch.setitem(sys.modules, "analyze_aggregator.aggregator", module)

    param_sets = [
        {
            "deployment": "unified",
            "tp_size": 1,
            "head_parallel": 1,
            "request_rate": 1.0,
            "log_dir": str(tmp_path / "par_0" / "rr_lo"),
            "_sweep_labels": {"par": "0", "request_rate": "lo"},
        },
        {
            "deployment": "unified",
            "tp_size": 2,
            "head_parallel": 2,
            "request_rate": 100.0,
            "log_dir": str(tmp_path / "par_1" / "rr_hi"),
            "_sweep_labels": {"par": "1", "request_rate": "hi"},
        },
    ]
    preset = {
        "sweep_groups": {
            "par": [{"tp_size": 1, "head_parallel": 1}, {"tp_size": 2, "head_parallel": 2}]
        }
    }

    _aggregate(param_sets, tmp_path, preset)

    run_infos, base_dir, groups = calls[0]
    assert base_dir == tmp_path
    assert groups == {"par": ["head_parallel", "tp_size"]}
    assert run_infos[0]["log_dir"] == str((tmp_path / "par_0" / "rr_lo").resolve())
    assert run_infos[0]["sweep"] == {
        "head_parallel": 1,
        "request_rate": 1.0,
        "tp_size": 1,
    }
    assert run_infos[0]["labels"] == {"par": "0", "request_rate": "lo"}


# ── real subprocess plumbing (uses the built stub binary) ───────────────────


def test_simulation_runner_captures_stdout(tmp_path):
    binary = binary_path("debug")
    if not binary.is_file():
        pytest.skip("simulator not built")
    # `run unified` with no trace fails fast (the L7-β driver / L4 build run, but
    # there's no profile.db here); we just verify the subprocess wrapper spawns,
    # captures stdout to stdout.log, and reports the non-zero exit.
    argv = [str(binary), "run", "unified", "--model-config", "model/config/llama3_8b.json"]
    runner = SimulationRunner(argv=argv, log_dir=tmp_path, env=_build_subprocess_env())
    ok = asyncio.run(runner.run())
    assert ok is False
    assert (tmp_path / "stdout.log").read_text().strip()  # captured the error output


# ── resume / --refresh (.complete marker, INV-5) ────────────────────────────


def test_complete_marker_roundtrip(tmp_path):
    assert not _is_complete(tmp_path)
    _mark_complete(tmp_path)
    assert _is_complete(tmp_path)
    assert (tmp_path / COMPLETE_MARKER).is_file()


def test_resume_skips_completed_run(tmp_path, schema):
    # A run whose log_dir already carries .complete is skipped (returns True)
    # before any prebuild/spawn — so this path needs no built binary.
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    _mark_complete(log_dir)
    params = {"deployment": "unified", "model_config": "m", "log_dir": str(log_dir)}
    assert asyncio.run(_run_single_async(params, None, schema, "debug")) is True


def test_public_run_entrypoints_require_explicit_schema():
    # The CLI owns cargo_build + load_schema. Direct callers must pass a loaded
    # Schema explicitly; None must not silently trigger a schema load here.
    with pytest.raises(TypeError, match="requires a loaded Schema"):
        run_single(_base(), None, None)
    with pytest.raises(TypeError, match="requires a loaded Schema"):
        run_sweep([_base()], _base(), None)


def test_failed_run_leaves_no_marker(tmp_path):
    # Marker is written only on zero exit; the stub binary exits non-zero, so a
    # failed run must NOT leave a `.complete` (otherwise resume would skip it).
    binary = binary_path("debug")
    if not binary.is_file():
        pytest.skip("simulator not built")
    log_dir = tmp_path / "run"
    params = {
        "deployment": "unified",
        "model_config": "model/config/llama3_8b.json",
        "log_dir": str(log_dir),
    }
    assert asyncio.run(_launch_one(params, "debug")) is False
    assert not _is_complete(log_dir)


# ── --override parsing ───────────────────────────────────────────────────────


def test_apply_overrides_json_typing():
    from launcher.__main__ import _apply_overrides

    out = _apply_overrides(
        {"deployment": "unified"},
        ["tp_size=4", "request_rate=2.5", "fp8=true", "cp_plan=ring", "model_config=a/b.json"],
    )
    assert out["tp_size"] == 4 and isinstance(out["tp_size"], int)
    assert out["request_rate"] == 2.5
    assert out["fp8"] is True  # lowercase JSON bool
    assert out["cp_plan"] == "ring"  # bare string fallback
    assert out["model_config"] == "a/b.json"  # path-like bare string


# ── list-params --human grouping (display-only pool_fragments) ──────────────


def test_human_table_groups_by_pool_fragment(schema, capsys):
    from launcher.__main__ import _print_params_table

    _print_params_table(schema, human=True)
    out = capsys.readouterr().out

    # Fragment headings appear in pool_fragments declaration order, own last.
    order = [
        out.index("[ModelCommon]"),
        out.index("[ParallelismCommon]"),
        out.index("[WorkloadCommon]"),
        out.index("[IoCommon]"),
        out.index("[unified-own]"),
    ]
    assert order == sorted(order)
    # max_batch_tokens is in no fragment → shown under the deployment-own group.
    own_section = out.split("[unified-own]", 1)[1]
    assert "max_batch_tokens" in own_section


def test_human_table_no_fragments_falls_back_to_own(capsys):
    # With no pool_fragments, every param lands under the own group (no crash).
    bare = schema_from_dict({"deployment_schemas": _FIXTURE["deployment_schemas"]})
    from launcher.__main__ import _print_params_table

    _print_params_table(bare, human=True)
    out = capsys.readouterr().out
    assert "[unified-own]" in out and "tp_size" in out
    assert "[ModelCommon]" not in out


# ── integration with the real Rust-generated schema ─────────────────────────


def test_real_schema_unified_present():
    try:
        real = load_schema("debug")
    except SchemaNotFound:
        pytest.skip("simulator not built; run `uv run cargo build` + list-params first")
    assert "unified" in real.deployment_schemas
    dep_schema = real.deployment_schemas["unified"]
    assert dep_schema.params["model_config"].get("required") is True
    assert dep_schema.params["max_batch_tokens"].get("required") is False
    assert dep_schema.params["tp_size"]["default"] == 4
    if "choices" not in dep_schema.params["cp_plan"]:
        pytest.skip("real deployment_schema.json was generated before ParamDef choices")
    assert "ring" in dep_schema.params["cp_plan"]["choices"]
