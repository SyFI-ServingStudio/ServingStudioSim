"""L7 launcher unit tests (structured config interface, new-interface-design §11).

Most tests run against an inline `Registry` fixture (deterministic, no build
dependency); a few integration tests load the real Rust-generated schema /
binary if present. Covers: the tree-walk validator (unknown-key guard, provider
tags, choices, required), `${name}` / `sweep` expansion + `derived` /
`constraints`, schema-walked normalization (default fill + coercion),
`io.log_dir` `{name}` templating, cache-key de-duplication, config-file argv,
metadata, and the sweep aggregator contract.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest
import yaml

from launcher import metadata
from launcher.cache_build import cache_key
from launcher.exec import (
    _build_subprocess_env,
    _cargo_build_env,
    _run_capture,
    binary_path,
    run_analysis,
    run_logged_process,
)
from launcher.process import ProcessResult
from launcher.process.artifacts import ArtifactValidation
from launcher.schema import (
    _format_log_dir,
    build_cli_command,
    expand_sweep_params,
    normalize_params,
    strip_internal,
    validate_distinct_configs,
    validate_expanded,
    validate_params,
    validate_unique_log_dirs,
)
from launcher.schema.loader import SchemaNotFound, load_schema, schema_from_dict
from launcher.sweep import (
    COMPLETE_MARKER,
    _aggregate,
    _experiment_root,
    _is_complete,
    _launch_one,
    _mark_complete,
    _run_single_async,
    _run_sweep_async,
    _sweep_axes,
    _write_sweep_manifest,
    run_single,
    run_sweep,
)

# Inline list-params document mirroring the real shape (unified / iter_wise).
_FIXTURE = {
    "deployments": {
        "unified": {"pools": {"main": "iter_wise"}},
        "pd": {"pools": {"prefill": "iter_wise", "decode": "iter_wise"}},
    },
    "providers": {
        "arch": {
            "iter_wise": {
                "llama3_dense": {"params": []},
                "llama3_dense_tp": {
                    "params": [
                        {
                            "name": "tp_size",
                            "type": "int",
                            "default": 2,
                            "affects_cache": True,
                            "description": "",
                        },
                    ]
                },
            }
        },
        "worker": {
            "iter_wise": {
                "barebone": {
                    "params": [
                        {
                            "name": "attn_gpu_memory_gb",
                            "type": "float",
                            "default": 80.0,
                            "description": "",
                        },
                    ]
                },
                "chunked_prefill": {
                    "params": [
                        {
                            "name": "attn_gpu_memory_gb",
                            "type": "float",
                            "default": 80.0,
                            "description": "",
                        },
                        {
                            "name": "max_batch_tokens",
                            "type": "int",
                            "required": True,
                            "description": "",
                        },
                        {
                            "name": "batch_policy",
                            "type": "string",
                            "default": "mix",
                            "choices": ["mix", "separate-prefill-priority"],
                            "description": "",
                        },
                    ]
                },
            }
        },
    },
    "arch_common": [
        {
            "name": "model_config",
            "type": "string",
            "required": True,
            "affects_cache": True,
            "description": "",
        },
        {"name": "num_layers", "type": "int", "required": False, "description": ""},
        {"name": "sim_num_layers", "type": "int", "required": False, "description": ""},
        {"name": "fp8", "type": "bool", "default": False, "affects_cache": True, "description": ""},
    ],
    "group_common": [
        {"name": "gpu", "type": "string", "required": True, "description": ""},
        {"name": "replicas", "type": "int", "default": 1, "description": ""},
    ],
    "pool_common": [
        {
            "name": "placement",
            "type": "string",
            "default": "least-queued",
            "choices": ["least-queued", "round-robin"],
            "description": "",
        },
    ],
    "common": {
        "workload": [
            {"name": "trace_files", "type": "path_list", "required": True, "description": ""},
            {
                "name": "input_file_format",
                "type": "string",
                "required": True,
                "choices": [
                    "text-generation-independent",
                    "text-generation-session-execution-v2",
                ],
                "description": "",
            },
            {
                "name": "input_file_tags",
                "type": "string_list",
                "required": False,
                "choices": ["session", "slo", "priority", "speculative"],
                "description": "",
            },
            {"name": "duration_ms", "type": "float", "default": 5000.0, "description": ""},
            {"name": "run_to_end", "type": "bool", "default": False, "description": ""},
            {"name": "request_rate", "type": "float", "default": 10.0, "description": ""},
            {
                "name": "arrival_mode",
                "type": "string",
                "required": True,
                "choices": ["trace_timed", "saturated"],
                "description": "",
            },
            {
                "name": "session_dependency",
                "type": "string",
                "required": True,
                "choices": ["independent", "chained"],
                "description": "",
            },
        ],
        "io": [
            {"name": "log_dir", "type": "path", "default": "logs", "description": ""},
            {
                "name": "log_level",
                "type": "string",
                "default": "info",
                "choices": ["trace", "debug", "info", "warn", "error"],
                "description": "",
            },
            {"name": "quiet", "type": "bool", "default": False, "description": ""},
            {"name": "force_cache_build", "type": "bool", "default": False, "description": ""},
        ],
    },
}


@pytest.fixture
def schema():
    return schema_from_dict(_FIXTURE)


def _base(*, arch=None, worker=None, log_dir="logs/x", **extra):
    """A minimal valid `unified` config tree. `arch` / `worker` overlay the
    default llama3_dense + barebone providers; `extra` merges into the root."""
    arch_node = {"type": "llama3_dense", "model_config": "m.json"}
    arch_node.update(arch or {})
    worker_node = {"type": "barebone"}
    worker_node.update(worker or {})
    preset = {
        "deployment": "unified",
        "workload": {
            "trace_files": ["t.csv"],
            "input_file_format": "text-generation-independent",
            "arrival_mode": "trace_timed",
            "session_dependency": "independent",
        },
        "io": {"log_dir": log_dir},
        "pools": {
            "main": {
                "groups": [{"gpu": "H200", "replicas": 1, "arch": arch_node, "worker": worker_node}]
            }
        },
    }
    preset.update(extra)
    return preset


def _arch(cand):
    return cand["pools"]["main"]["groups"][0]["arch"]


def test_run_analysis_delegates_both_optimality_modes_to_analyzer(monkeypatch, tmp_path):
    analyzer_path = tmp_path / "analyze"
    analyzer_path.write_text("")
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "stdout.log").write_text("")
    captured_commands: list[list[str]] = []

    async def supervise(spec):
        command = [str(argument) for argument in spec.argv]
        captured_commands.append(command)
        return ProcessResult(
            argv=tuple(command),
            pid=1,
            process_group_id=1,
            exit_code=0,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr("launcher.exec.analyzer_binary_path", lambda _build_type: analyzer_path)
    monkeypatch.setattr("launcher.exec._PROCESS_SUPERVISOR.run", supervise)
    monkeypatch.setattr(
        "launcher.exec.validate_json_outputs", lambda _path: ArtifactValidation(())
    )
    monkeypatch.setattr(
        "launcher.exec.validate_render_artifacts", lambda _path: ArtifactValidation(())
    )
    monkeypatch.setattr(
        "launcher.exec.validate_trace_artifacts", lambda _path: ArtifactValidation(())
    )

    asyncio.run(run_analysis(log_dir, subjects=["optimality"]))

    analyzer_run_commands = [
        command for command in captured_commands if len(command) > 1 and command[1] == "run"
    ]
    assert analyzer_run_commands == [[str(analyzer_path), "run", str(log_dir), "optimality"]]


def test_run_capture_avoids_asyncio_subprocess_transport(monkeypatch):
    async def reject_asyncio_subprocess(*_arguments, **_keyword_arguments):
        raise AssertionError("analysis capture must not use the asyncio subprocess transport")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject_asyncio_subprocess)

    return_code, output = asyncio.run(
        _run_capture(
            [
                sys.executable,
                "-c",
                "import sys; print('captured stdout'); print('captured stderr', file=sys.stderr)",
            ]
        )
    )

    assert return_code == 0
    assert "captured stdout" in output
    assert "captured stderr" in output


# ── validation ──────────────────────────────────────────────────────────────


def test_validate_ok(schema):
    assert validate_params(_base(), schema) == []


def test_validate_unknown_root_key(schema):
    errs = validate_params(_base(bogus=1), schema)
    assert any("'bogus'" in e for e in errs)


def test_validate_rejects_legacy_replay_mode(schema):
    preset = _base()
    preset["workload"]["replay_mode"] = preset["workload"].pop("arrival_mode")
    errs = validate_params(preset, schema)
    assert any("workload.replay_mode" in error and "unknown" in error for error in errs)


@pytest.mark.parametrize("axis", ["arrival_mode", "session_dependency"])
def test_validate_requires_both_replay_axes(schema, axis):
    preset = _base()
    del preset["workload"][axis]
    errs = validate_params(preset, schema)
    assert any(axis in error and "required" in error for error in errs)


def test_validate_unknown_arch_payload_key(schema):
    # G1: a tagged-enum payload typo is caught by the launcher walk (Rust's
    # deny_unknown_fields is silently ignored on a tagged variant).
    errs = validate_params(_base(arch={"tp_zize": 4}), schema)
    assert any("arch.tp_zize" in e for e in errs)


def test_validate_missing_required_gpu(schema):
    preset = _base()
    del preset["pools"]["main"]["groups"][0]["gpu"]
    errs = validate_params(preset, schema)
    assert any("gpu" in e and "required" in e for e in errs)


def test_validate_missing_required_model_config(schema):
    errs = validate_params(_base(arch={"model_config": None}), schema)
    assert any("model_config" in e for e in errs)


def test_validate_unknown_deployment(schema):
    errs = validate_params({"deployment": "nope"}, schema)
    assert any("unknown deployment" in e for e in errs)


def test_validate_bad_arch_tag(schema):
    errs = validate_params(_base(arch={"type": "no_such_arch"}), schema)
    assert any("arch.type='no_such_arch'" in e for e in errs)


def test_validate_bad_worker_tag(schema):
    errs = validate_params(_base(worker={"type": "no_such_worker"}), schema)
    assert any("worker.type='no_such_worker'" in e for e in errs)


# ── params-only: structure may not be a placeholder (finding 3) ──────────────


def test_validate_rejects_placeholder_arch_tag(schema):
    # A provider `type` tag selects the schema — it is structure, not a value, so a
    # `${...}` is rejected at the raw phase (not deferred). params-only principle.
    errs = validate_params(
        _base(arch={"type": "${arch_kind}"}, sweep={"arch_kind": ["llama3_dense"]}),
        schema,
    )
    assert any("arch.type='${arch_kind}'" in e and "structure" in e for e in errs)


def test_validate_rejects_placeholder_worker_tag(schema):
    errs = validate_params(_base(worker={"type": "${wk}"}, sweep={"wk": ["barebone"]}), schema)
    assert any("worker.type='${wk}'" in e and "structure" in e for e in errs)


def test_validate_rejects_placeholder_whole_arch_block(schema):
    # A whole arch/worker block cannot be injected via ${...} — structure is literal.
    preset = _base(sweep={"a": ["x"]})
    preset["pools"]["main"]["groups"][0]["arch"] = "${a}"
    errs = validate_params(preset, schema)
    assert any("arch is the placeholder" in e and "structure" in e for e in errs)


def test_validate_rejects_placeholder_deployment(schema):
    errs = validate_params({"deployment": "${d}", "sweep": {"d": ["unified"]}}, schema)
    assert any("deployment '${d}' is a placeholder" in e for e in errs)


def test_validate_placement_choices(schema):
    preset = _base()
    preset["pools"]["main"]["placement"] = "bogus-policy"
    errs = validate_params(preset, schema)
    assert any("placement" in e and "not one of" in e for e in errs)


def test_validate_log_level_choices(schema):
    preset = _base()
    preset["io"]["log_level"] = "verbose"
    errs = validate_params(preset, schema)
    assert any("log_level" in e and "not one of" in e for e in errs)


def test_validate_batch_policy_choices(schema):
    errs = validate_params(
        _base(
            worker={"type": "chunked_prefill", "max_batch_tokens": 16384, "batch_policy": "loud"}
        ),
        schema,
    )
    assert any("batch_policy" in e and "not one of" in e for e in errs)


def test_validate_list_choices_apply_to_each_element(schema):
    preset = _base()
    preset["workload"]["input_file_tags"] = ["session", "slo"]
    assert validate_params(preset, schema) == []

    preset["workload"]["input_file_tags"] = ["session", "unknown"]
    errs = validate_params(preset, schema)
    assert any(
        "input_file_tags[1]='unknown'" in error and "not one of" in error
        for error in errs
    )


def test_v1_v2_derived_collides_with_sweep(schema):
    preset = _base(sweep={"ptp": [1, 2]}, derived={"ptp": "1"})
    errs = validate_params(preset, schema)
    assert any("also appears as a sweep dim" in e for e in errs)


def test_v3_derived_rhs_undefined(schema):
    preset = _base(sweep={"ptp": [1, 2]}, derived={"hp": "nonexistent + 1"})
    errs = validate_params(preset, schema)
    assert any("undefined names" in e and "nonexistent" in e for e in errs)


def test_v5_constraint_undefined_var(schema):
    preset = _base(sweep={"ptp": [1, 2]}, constraints=["ghost > 0"])
    errs = validate_params(preset, schema)
    assert any("ghost" in e for e in errs)


def test_v6_disallowed_call(schema):
    preset = _base(sweep={"ptp": [1, 2]}, constraints=["evil(ptp)"])
    errs = validate_params(preset, schema)
    assert any("call to 'evil' not allowed" in e for e in errs)


def test_allowlisted_calls_ok(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2]},
        derived={"hp": "max(min(ptp, 4), 1)"},
        log_dir="logs/h{hp}",
    )
    assert validate_params(preset, schema) == []


def test_ternary_validates_and_evaluates(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2]},
        derived={"hp": "2 if ptp > 1 else 1"},
        log_dir="logs/h{hp}",
    )
    assert validate_params(preset, schema) == []
    cands = expand_sweep_params(preset, schema)
    assert {(c["_env"]["ptp"], c["_env"]["hp"]) for c in cands} == {(1, 1), (2, 2)}


# ── suspicious-use rejection (R1–R6) ────────────────────────────────────────


def test_r1_partial_placeholder_rejected(schema):
    # `${ptp}` embedded in a larger string is silently dropped by substitution.
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "v${ptp}"},
        sweep={"ptp": [1, 2]},
    )
    assert any("not the whole value" in e for e in validate_params(preset, schema))


def test_r2_dead_sweep_rejected(schema):
    # ptp is declared but referenced nowhere → N identical runs.
    errs = validate_params(_base(sweep={"ptp": [1, 2]}), schema)
    assert any("sweep dim 'ptp' never reaches a run config value" in e for e in errs)


def test_r2_constraint_only_sweep_rejected(schema):
    # A sweep dim used only in a constraint is a phantom axis — it filters the plan
    # but never lands in any run's config (the single-survivor bypass R5 can't see).
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2], "cap": [2]},
        constraints=["ptp <= cap"],
    )
    errs = validate_params(preset, schema)
    assert any("'cap' never reaches a run config value" in e for e in errs)


def test_r2_log_dir_only_sweep_rejected(schema):
    # A dim referenced only in io.log_dir changes the path, not the config.
    errs = validate_params(_base(sweep={"ptp": [1, 2]}, log_dir="logs/tp{ptp}"), schema)
    assert any("'ptp' never reaches a run config value" in e for e in errs)


def test_r2_derived_correlation_is_effective(schema):
    # The correlated-columns pattern (sweep one, derive the other; both land in
    # config) must stay valid — derived reachability makes the swept dim effective.
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ep}"},
        sweep={"tp": [4, 8]},
        derived={"ep": "tp * 2"},
    )
    preset["pools"]["main"]["groups"][0]["replicas"] = "${tp}"
    assert validate_params(preset, schema) == []


def test_control_block_bad_shape_rejected(schema):
    assert any("`sweep` must be a mapping" in e for e in validate_params(_base(sweep=[]), schema))
    assert any(
        "`derived` must be a mapping" in e for e in validate_params(_base(derived=[]), schema)
    )
    assert any(
        "`constraints` must be a list" in e for e in validate_params(_base(constraints="1"), schema)
    )
    assert any(
        "constraints[0] must be a string" in e
        for e in validate_params(_base(sweep={"ptp": [1]}, constraints=[1]), schema)
    )


@pytest.mark.parametrize(
    "template",
    [
        "logs/{ptp.__class__}",
        "logs/{ptp[0]}",
        "logs/{ptp!r}",
        "logs/{ptp:04d}",
        "logs/{}",
        "logs/{0}",
        "logs/{ptp",
    ],
)
def test_log_dir_non_bare_identifier_rejected(schema, template):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [4]},
        log_dir=template,
    )
    errs = validate_params(preset, schema)
    assert any("io.log_dir" in e for e in errs), f"{template!r} not rejected: {errs}"


def test_r3_dead_derived_rejected(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2]},
        derived={"hp": "ptp * 2"},  # computed, never used
    )
    errs = validate_params(preset, schema)
    assert any("derived 'hp' is computed but never used" in e for e in errs)


def test_r4_empty_sweep_rejected(schema):
    errs = validate_params(_base(sweep={"ptp": []}), schema)
    assert any("sweep dim 'ptp' is empty" in e for e in errs)


def test_r5_distinct_configs_rejects_identical():
    # Two candidates with identical config trees, different log_dir only.
    def cand(log_dir):
        return {
            "deployment": "unified",
            "io": {"log_dir": log_dir},
            "pools": {"main": {"groups": [{"gpu": "H200", "arch": {"type": "llama3_dense"}}]}},
        }

    assert validate_distinct_configs([cand("logs/tp1"), cand("logs/tp2")]) is False


def test_r5_distinct_configs_accepts_real_difference():
    def cand(log_dir, tp):
        return {
            "deployment": "unified",
            "io": {"log_dir": log_dir},
            "pools": {
                "main": {
                    "groups": [{"gpu": "H200", "arch": {"type": "llama3_dense_tp", "tp_size": tp}}]
                }
            },
        }

    assert validate_distinct_configs([cand("logs/tp1", 1), cand("logs/tp2", 2)]) is True


def test_main_rejects_log_dir_only_sweep(tmp_path, schema, monkeypatch):
    # The exact broken pattern end-to-end: ptp moves only the log_dir, not config.
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset_path = tmp_path / "preset.json"
    preset_path.write_text(json.dumps(_base(sweep={"ptp": [1, 2]}, log_dir="logs/tp{ptp}")))
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    assert main_module.main([str(preset_path), "--dry-run"]) == 2


def test_main_rejects_empty_expansion(tmp_path, schema, monkeypatch):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset_path = tmp_path / "preset.json"
    preset_path.write_text(
        json.dumps(
            _base(
                arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
                sweep={"ptp": [1, 2]},
                constraints=["ptp > 100"],  # rejects every combo → 0 runs
            )
        )
    )
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    with pytest.raises(SystemExit):
        main_module.main([str(preset_path), "--dry-run"])


# ── type / structural rejection (TE / SE / P) ───────────────────────────────


def test_te_int_type_mismatch_rejected(schema):
    # `tp_size: "not_an_int"` used to crash int() in normalize; now a clean error.
    preset = _base(arch={"type": "llama3_dense_tp", "tp_size": "not_an_int"})
    assert any("is not a valid int" in e for e in validate_params(preset, schema))


def test_te_list_given_scalar_rejected(schema):
    preset = _base()
    preset["workload"]["trace_files"] = "single.csv"  # must be a list
    assert any("not a valid path_list" in e for e in validate_params(preset, schema))


def test_se_missing_pool_role_rejected(schema):
    preset = _base()
    preset["pools"] = {}  # unified requires the 'main' role
    assert any("requires pool role 'main'" in e for e in validate_params(preset, schema))


def test_se_empty_groups_rejected(schema):
    preset = _base()
    preset["pools"]["main"]["groups"] = []
    assert any("non-empty 'groups'" in e for e in validate_params(preset, schema))


def test_p_log_dir_undeclared_placeholder_rejected(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2]},
        log_dir="logs/{ghost}",
    )
    assert any("io.log_dir references {ghost}" in e for e in validate_params(preset, schema))


# ── post-expansion validation: placeholder bypass (choices / type) ──────────


def test_validate_expanded_rejects_bad_choice(schema):
    cand = _base()
    cand["io"]["log_level"] = "verbose"  # not in log_level choices
    assert any("not one of" in e for e in validate_expanded(cand, schema))


def test_validate_expanded_rejects_bad_type(schema):
    cand = _base(arch={"type": "llama3_dense_tp", "tp_size": 4})
    _arch(cand)["fp8"] = "false"  # str, not bool — would coerce to True
    assert any("not a valid bool" in e for e in validate_expanded(cand, schema))


def test_r7_constant_derived_rejected(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2]},
        derived={"aux": "1"},  # references no sweep dim → constant
        log_dir="logs/{aux}",
    )
    assert any("is constant" in e for e in validate_params(preset, schema))


def test_main_rejects_placeholder_bad_choice(tmp_path, schema, monkeypatch):
    # sweep injects an illegal choice via a placeholder that skips pre-expansion.
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset = _base(sweep={"bad_level": ["verbose"]}, log_dir="logs/{bad_level}")
    preset["io"]["log_level"] = "${bad_level}"
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(json.dumps(preset))
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    assert main_module.main([str(preset_path), "--dry-run"]) == 2


def test_main_rejects_placeholder_bad_bool(tmp_path, schema, monkeypatch):
    # sweep injects "false" (str) into a bool param; must reject before coerce.
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset = _base(
        arch={"type": "llama3_dense", "fp8": "${bad}"},
        sweep={"bad": ["false"]},
        log_dir="logs/{bad}",
    )
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(json.dumps(preset))
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    assert main_module.main([str(preset_path), "--dry-run"]) == 2


# ── placeholder bypass via provider tag / dynamic required / dropped block ──


def test_validate_expanded_rejects_placeholder_arch_tag(schema):
    # Attack 1: arch.type arrives via a sweep, so the pre-expansion tag check
    # only sees a `${...}` placeholder. Post-expansion the concrete (bogus) tag
    # must be caught by the structural re-check.
    cand = _base(arch={"type": "not_a_real_arch", "model_config": "m.json"})
    assert any("arch.type='not_a_real_arch'" in e for e in validate_expanded(cand, schema))


def test_main_rejects_placeholder_arch_tag(tmp_path, schema, monkeypatch):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset = _base(
        arch={"type": "${arch_kind}", "model_config": "m.json"},
        sweep={"arch_kind": ["not_a_real_arch"]},
        log_dir="logs/{arch_kind}",
    )
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(json.dumps(preset))
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    assert main_module.main([str(preset_path), "--dry-run"]) == 2


def test_validate_expanded_rejects_dynamic_tag_missing_required(schema):
    # Attack 2: worker.type arrives via a sweep, so pre-expansion the tag-specific
    # required params (chunked_prefill's max_batch_tokens) cannot be checked. With
    # the tag now concrete, the missing required param must be flagged.
    cand = _base(worker={"type": "chunked_prefill"})  # no max_batch_tokens
    assert any("max_batch_tokens" in e and "required" in e for e in validate_expanded(cand, schema))


def test_main_rejects_placeholder_dynamic_tag_missing_required(tmp_path, schema, monkeypatch):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset = _base(
        worker={"type": "${wk}", "attn_gpu_memory_gb": 80.0},
        sweep={"wk": ["chunked_prefill"]},
        log_dir="logs/{wk}",
    )
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(json.dumps(preset))
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    assert main_module.main([str(preset_path), "--dry-run"]) == 2


def test_validate_dropped_workload_block_still_flags_required(schema):
    # Attack 3: dropping the whole `workload` block must not dodge the required
    # `trace_files` check — the walk materializes the absent block so its required
    # leaves are still inspected.
    preset = _base()
    del preset["workload"]
    assert any("trace_files" in e and "required" in e for e in validate_params(preset, schema))
    assert any("trace_files" in e and "required" in e for e in validate_expanded(preset, schema))


def test_validate_nested_underscore_key_rejected(schema):
    # Attack 4: only TOP-LEVEL `_`-keys are launcher internals; a nested `_`-key
    # is an arch/worker payload typo and must be reported, not silently skipped.
    errs = validate_params(_base(arch={"_model_config": "m.json"}), schema)
    assert any("arch._model_config" in e for e in errs)


# ── concrete validator: table-driven attack corpus + invariants ─────────────


def _concrete_attacks():
    """(concrete candidate, expected error substring) — each is a post-expansion
    tree (no live sweep) that `validate_expanded` must reject. One corpus, so the
    single concrete gate is exercised against every bypass class at once."""
    cases = []

    def add(cid, cand, sub):
        cases.append(pytest.param(cand, sub, id=cid))

    add(
        "bad-arch-tag",
        _base(arch={"type": "not_a_real_arch", "model_config": "m.json"}),
        "arch.type='not_a_real_arch'",
    )
    add("bad-worker-tag", _base(worker={"type": "no_such_worker"}), "worker.type='no_such_worker'")
    add("tag-required-missing", _base(worker={"type": "chunked_prefill"}), "max_batch_tokens")
    add("bad-int-type", _base(arch={"type": "llama3_dense_tp", "tp_size": "x"}), "not a valid int")
    add("unknown-key", _base(arch={"bogus_field": 1}), "bogus_field")
    add("nested-underscore-key", _base(arch={"_typo": 1}), "_typo")
    add(
        "whole-placeholder-residue",
        _base(arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"}),
        "placeholder",
    )

    bad_choice = _base()
    bad_choice["io"]["log_level"] = "verbose"
    add("bad-choice", bad_choice, "not one of")

    bad_bool = _base(arch={"type": "llama3_dense_tp", "tp_size": 4})
    _arch(bad_bool)["fp8"] = "false"
    add("bad-bool-type", bad_bool, "not a valid bool")

    dropped = _base()
    del dropped["workload"]
    add("dropped-workload-block", dropped, "trace_files")

    partial = _base()
    _arch(partial)["model_config"] = "m_${x}.json"
    add("partial-placeholder-residue", partial, "residue")

    return cases


@pytest.mark.parametrize("candidate, expected", _concrete_attacks())
def test_concrete_validator_rejects(candidate, expected, schema):
    errs = validate_expanded(candidate, schema)
    assert any(expected in e for e in errs), f"expected {expected!r} in {errs}"


def _valid_presets():
    """Valid presets whose expansions feed the invariant tests below."""
    return [
        _base(),
        _base(
            arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
            sweep={"ptp": [1, 2, 4]},
            log_dir="logs/tp{ptp}",
        ),
    ]


def test_invariant_passing_candidate_has_no_placeholder_residue(schema):
    # Independent oracle: any candidate the concrete validator accepts must carry
    # zero `${` anywhere. The oracle (a raw substring scan) shares no code with the
    # validator, so this is a real post-condition, not a tautology.
    seen = 0
    for preset in _valid_presets():
        assert validate_params(preset, schema) == []
        for cand in expand_sweep_params(preset, schema):
            assert validate_expanded(cand, schema) == []
            assert "${" not in json.dumps(cand), f"placeholder residue in {cand}"
            seen += 1
    assert seen >= 4  # 1 (no sweep) + 3 (ptp sweep)


def test_invariant_normalize_preserves_concrete_validity(schema):
    # A candidate that passes the concrete gate must still pass after normalize —
    # normalize only fills defaults / coerces, it must never make a valid config
    # invalid (the semantic-drift guard that lets us tighten normalize later).
    import copy

    for preset in _valid_presets():
        for cand in expand_sweep_params(preset, schema):
            assert validate_expanded(cand, schema) == []
            normalized = normalize_params(copy.deepcopy(cand), schema)
            assert validate_expanded(normalized, schema) == []


def test_unsupported_grammar_rejected(schema):
    for expr in ["ptp[0] > 0", "ptp.bit_length() > 0", "[x for x in ptp]"]:
        errs = validate_params(_base(sweep={"ptp": [1]}, constraints=[expr]), schema)
        assert errs, f"expected rejection for {expr!r}"


def test_eval_failure_is_loud_not_silent_none(schema):
    preset = _base(sweep={"ptp": [2]}, derived={"hp": "ptp // 0"})
    with pytest.raises(ValueError):
        expand_sweep_params(preset, schema)


def test_undeclared_placeholder_rejected(schema):
    preset = _base(arch={"type": "llama3_dense_tp", "tp_size": "${ghost}"})
    errs = validate_params(preset, schema)
    assert any("${ghost}" in e for e in errs)


# ── sweep expansion ───────────────────────────────────────────────────────


def test_expand_single_no_sweep(schema):
    cands = expand_sweep_params(_base(), schema)
    assert len(cands) == 1


def test_expand_list_sweep_substitutes_typed(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2, 4, 8]},
    )
    cands = expand_sweep_params(preset, schema)
    vals = sorted(_arch(c)["tp_size"] for c in cands)
    assert vals == [1, 2, 4, 8]
    assert all(isinstance(_arch(c)["tp_size"], int) for c in cands)  # typed, not "${ptp}"


def test_expand_dict_sweep_labels(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": {"lo": 1, "hi": 8}},
    )
    cands = expand_sweep_params(preset, schema)
    by_label = {c["_sweep_labels"]["ptp"]: _arch(c)["tp_size"] for c in cands}
    assert by_label == {"lo": 1, "hi": 8}


def test_expand_cartesian_product(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2], "rate": [10, 100]},
    )
    assert len(expand_sweep_params(preset, schema)) == 4


def test_expand_derived_and_constraint(schema):
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
        sweep={"ptp": [1, 2, 4, 8]},
        derived={"ep": "8"},
        constraints=["ptp * ep <= 32"],
    )
    cands = expand_sweep_params(preset, schema)
    assert sorted(_arch(c)["tp_size"] for c in cands) == [1, 2, 4]  # 8*8=64 dropped


def _compound_preset(**extra):
    """A preset whose tp_size + request_rate are driven by compound members."""
    preset = _base(arch={"type": "llama3_dense_tp", "tp_size": "${tp}"}, **extra)
    preset["workload"]["request_rate"] = "${rate}"
    return preset


def test_expand_compound_zip(schema):
    preset = _compound_preset(
        compound={"tp_rate": {"fast": {"tp": 4, "rate": 10}, "slow": {"tp": 8, "rate": 20}}}
    )
    cands = expand_sweep_params(preset, schema)
    assert len(cands) == 2  # zipped rows, NOT a 2x2 grid
    rows = {(_arch(c)["tp_size"], c["workload"]["request_rate"]) for c in cands}
    assert rows == {(4, 10), (8, 20)}
    for c in cands:
        # group name = labeled axis; members live in env; group not crossed
        assert c["_env"]["tp_rate"] in ("fast", "slow")
        assert c["_sweep_labels"]["tp_rate"] == c["_env"]["tp_rate"]
        assert set(c["_compound_members"]) == {"tp", "rate"}


def test_expand_compound_crosses_sweep(schema):
    preset = _compound_preset(
        compound={"tp_rate": {"fast": {"tp": 4, "rate": 10}, "slow": {"tp": 8, "rate": 20}}},
        sweep={"gpu_kind": ["NVIDIA H200", "NVIDIA H100"]},
    )
    preset["pools"]["main"]["groups"][0]["gpu"] = "${gpu_kind}"
    cands = expand_sweep_params(preset, schema)
    assert len(cands) == 4  # 2 compound rows x 2 sweep values = cartesian


# ── compound validation ─────────────────────────────────────────────────────


def test_validate_compound_ok(schema):
    preset = _compound_preset(
        compound={"tp_rate": {"fast": {"tp": 4, "rate": 10}, "slow": {"tp": 8, "rate": 20}}}
    )
    assert validate_params(preset, schema) == []


def test_compound_inconsistent_member_set_rejected(schema):
    preset = _compound_preset(
        compound={"tp_rate": {"fast": {"tp": 4, "rate": 10}, "slow": {"tp": 8}}}
    )  # slow lacks rate
    errs = validate_params(preset, schema)
    assert any("every row must declare the same members" in e for e in errs)


def test_compound_member_collides_sweep_rejected(schema):
    preset = _compound_preset(
        compound={"g": {"a": {"tp": 4, "rate": 10}, "b": {"tp": 8, "rate": 20}}},
        sweep={"tp": [1, 2]},  # collides with compound member 'tp'
    )
    errs = validate_params(preset, schema)
    assert any("'tp' collides with a sweep/derived name" in e for e in errs)


def test_compound_member_not_config_effective_rejected(schema):
    # `extra` is in the rows but never lands in any config leaf → phantom column.
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${tp}"},
        compound={"g": {"a": {"tp": 4, "extra": 1}, "b": {"tp": 8, "extra": 2}}},
    )
    errs = validate_params(preset, schema)
    assert any("compound member 'extra' never reaches a run config value" in e for e in errs)


@pytest.mark.parametrize(
    "compound, needle",
    [
        ([1, 2], "`compound` must be a mapping"),
        ({"g": [1, 2]}, "compound group 'g' must be a non-empty mapping"),
        ({"g": {}}, "compound group 'g' must be a non-empty mapping"),
        ({"g": {"a": 5}}, "compound['g']['a'] must be a non-empty mapping"),
    ],
)
def test_compound_bad_shape_rejected(schema, compound, needle):
    errs = validate_params(_base(compound=compound), schema)
    assert any(needle in e for e in errs), f"{compound!r}: {errs}"


# ── variants manifest (single-axis cross-file) ──────────────────────────────


def test_expand_manifest_single_axis_tags_and_prefixes(tmp_path, schema):
    from launcher.__main__ import _expand_manifest

    (tmp_path / "dense.json").write_text(
        json.dumps(
            _base(
                arch={"type": "llama3_dense_tp", "tp_size": "${tp}"},
                sweep={"tp": [2, 4]},
                log_dir="logs/d{tp}",
            )
        )
    )
    (tmp_path / "big.json").write_text(
        json.dumps(_base(arch={"type": "llama3_dense_tp", "tp_size": 8}, log_dir="logs/b"))
    )
    manifest = {"variants": {"arch": {"dense": "dense.json", "big": "big.json"}}}

    cands = _expand_manifest(manifest, schema, str(tmp_path / "m.json"), [])
    assert cands is not None
    assert len(cands) == 3  # dense (tp sweep → 2) + big (1)
    assert {c["_env"]["arch"] for c in cands} == {"dense", "big"}
    assert all(c["io"]["log_dir"].startswith(("dense/", "big/")) for c in cands)


def test_main_manifest_single_axis_ok(tmp_path, schema, monkeypatch):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    (tmp_path / "dense.json").write_text(
        json.dumps(
            _base(
                arch={"type": "llama3_dense_tp", "tp_size": "${tp}"},
                sweep={"tp": [2, 4]},
                log_dir="logs/d{tp}",
            )
        )
    )
    (tmp_path / "big.json").write_text(
        json.dumps(_base(arch={"type": "llama3_dense_tp", "tp_size": 8}, log_dir="logs/b"))
    )
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps({"variants": {"arch": {"dense": "dense.json", "big": "big.json"}}})
    )
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    assert main_module.main([str(manifest), "--dry-run"]) == 0


def test_main_manifest_multi_axis_rejected(tmp_path, schema, monkeypatch):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps({"variants": {"arch": {"dense": "d.json"}, "hw": {"h200": "h.json"}}})
    )  # two axes → reject
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    assert main_module.main([str(manifest), "--dry-run"]) == 2


def test_manifest_axis_collision_with_inner_env_rejected(tmp_path, schema):
    # A manifest axis must not shadow an inner sweep/compound/derived name, else it
    # overwrites that _env binding (corrupting log_dir + the aggregation axis).
    from launcher.__main__ import _expand_manifest

    (tmp_path / "p.json").write_text(
        json.dumps(
            _base(
                arch={"type": "llama3_dense_tp", "tp_size": "${tp}"},
                sweep={"tp": [2, 4]},
                log_dir="logs/{tp}",
            )
        )
    )
    manifest = {"variants": {"tp": {"dense": "p.json"}}}  # axis 'tp' collides
    assert _expand_manifest(manifest, schema, str(tmp_path / "m.json"), []) is None


def test_manifest_extra_key_rejected(tmp_path, schema, monkeypatch):
    # Strict parse: a stray top-level key in a manifest is an error, not a no-op.
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "variants": {"arch": {"dense": "d.json"}},
                "sweep": {"typo": [1]},  # stray key
            }
        )
    )
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    assert main_module.main([str(manifest), "--dry-run"]) == 2


@pytest.mark.parametrize(
    "preset_kw, needle",
    [
        ({"sweep": {1: [2]}}, "sweep dim name 1 must be a string identifier"),
        ({"compound": {1: {"a": {"tp": 4}}}}, "compound group name 1 must be a string identifier"),
        ({"compound": {"g": {"a": {2: 4}}}}, "compound member name 2 must be a string identifier"),
        (
            {"compound": {"g": {"a/b": {"tp": 4}}}},
            "label 'a/b' must be a non-empty path-safe string",
        ),
    ],
)
def test_symbol_names_must_be_identifiers(schema, preset_kw, needle):
    # Non-identifier names leak into _env and crash _sweep_axes' mixed-type sort.
    errs = validate_params(_base(**preset_kw), schema)
    assert any(needle in e for e in errs), errs


def test_dict_sweep_label_must_be_path_safe(schema):
    # A dict-sweep label reaches io.log_dir as a path segment → no `..` traversal.
    preset = _base(
        arch={"type": "llama3_dense_tp", "tp_size": "${tp}"},
        sweep={"tp": {"../escape": 4}},
        log_dir="logs/{tp}",
    )
    errs = validate_params(preset, schema)
    assert any("label '../escape'" in e for e in errs)


@pytest.mark.parametrize("bad_io", [[], "x", 5])
def test_io_block_must_be_mapping(schema, bad_io):
    # A non-dict `io` would otherwise be silently rebuilt as the default block.
    preset = _base()
    preset["io"] = bad_io
    assert any("'io' must be a mapping" in e for e in validate_params(preset, schema))


def _mock_build(monkeypatch, schema):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)
    return main_module


def test_main_rejects_bad_analyze_subjects(tmp_path, schema, monkeypatch):
    main_module = _mock_build(monkeypatch, schema)
    preset = _base()
    preset["analyze_subjects"] = "slo-general"  # must be a list of strings, not a bare str
    p = tmp_path / "p.json"
    p.write_text(json.dumps(preset))
    assert main_module.main([str(p), "--dry-run"]) == 2


def test_main_rejects_non_mapping_root(tmp_path, schema, monkeypatch):
    main_module = _mock_build(monkeypatch, schema)
    p = tmp_path / "p.json"
    p.write_text("[]")  # top-level is a list, not a mapping
    assert main_module.main([str(p), "--dry-run"]) == 2


def test_main_rejects_duplicate_keys(tmp_path, schema, monkeypatch):
    main_module = _mock_build(monkeypatch, schema)
    p = tmp_path / "p.json"
    p.write_text('{"deployment": "unified", "deployment": "pd"}')  # dup key
    assert main_module.main([str(p), "--dry-run"]) == 2


def test_load_preset_rejects_duplicate_yaml_keys(tmp_path):
    from launcher.__main__ import PresetError, _load_preset

    p = tmp_path / "p.yaml"
    p.write_text("deployment: unified\ndeployment: pd\n")
    with pytest.raises(PresetError):
        _load_preset(p)


def test_readme_worked_example(schema):
    # Drift guard for launcher/README.md "every technique in one preset": the pd
    # sweep must validate clean and expand to exactly 10 runs (2×3×2 − 2 rejected),
    # collapsing to 5 distinct cache keys (batch params are not affects_cache).
    def grp(tp, worker):
        return {
            "gpu": "NVIDIA H200",
            "replicas": tp[1],
            "arch": {
                "type": "llama3_dense_tp",
                "model_config": "model/config/llama3_8b.json",
                "fp8": True,
                "tp_size": tp[0],
            },
            "worker": worker,
        }

    preset = {
        "deployment": "pd",
        "workload": {
            "trace_files": ["trace/aime_long.csv"],
            "input_file_format": "text-generation-independent",
            "arrival_mode": "trace_timed",
            "session_dependency": "independent",
        },
        "io": {"log_dir": "logs/pd_{prefill_tp}_d{decode_tp}tp_r{decode_replicas}_{batch}"},
        "pools": {
            "prefill": {
                "groups": [
                    grp(
                        ("${prefill_tp}", 2),
                        {
                            "type": "chunked_prefill",
                            "attn_gpu_memory_gb": 80.0,
                            "max_batch_tokens": "${max_batch_tokens}",
                            "batch_policy": "${batch_policy}",
                        },
                    )
                ]
            },
            "decode": {
                "groups": [
                    grp(
                        ("${decode_tp}", "${decode_replicas}"),
                        {"type": "barebone", "attn_gpu_memory_gb": 80.0},
                    )
                ]
            },
        },
        "sweep": {"prefill_tp": {"p8": 8, "p4": 4}, "decode_tp": [2, 4, 8]},
        "compound": {
            "batch": {
                "big": {"max_batch_tokens": 16384, "batch_policy": "separate-prefill-priority"},
                "small": {"max_batch_tokens": 8192, "batch_policy": "mix"},
            }
        },
        "derived": {"decode_replicas": "16 // decode_tp"},
        "constraints": ["prefill_tp >= decode_tp"],
    }
    assert validate_params(preset, schema) == []
    cands = [normalize_params(c, schema) for c in expand_sweep_params(preset, schema)]
    assert len(cands) == 10
    assert len({cache_key(c, schema) for c in cands}) == 5


# ── normalization ─────────────────────────────────────────────────────────


def test_normalize_fills_defaults(schema):
    out = normalize_params(_base(), schema)
    assert out["pools"]["main"]["placement"] == "least-queued"
    assert out["pools"]["main"]["groups"][0]["replicas"] == 1
    assert out["pools"]["main"]["groups"][0]["worker"]["attn_gpu_memory_gb"] == 80.0
    assert _arch(out)["fp8"] is False
    assert out["workload"]["duration_ms"] == 5000.0
    assert out["workload"]["request_rate"] == 10.0
    assert out["io"]["log_level"] == "info"
    assert out["io"]["quiet"] is False


def test_normalize_coerces_types(schema):
    preset = _base(arch={"type": "llama3_dense_tp", "tp_size": "2"})
    out = normalize_params(preset, schema)
    assert _arch(out)["tp_size"] == 2 and isinstance(_arch(out)["tp_size"], int)


def test_normalize_omits_absent_optionals(schema):
    out = normalize_params(_base(), schema)
    # num_layers / sim_num_layers have no default → stay absent (serde Option).
    assert "num_layers" not in _arch(out)
    assert "sim_num_layers" not in _arch(out)


# ── argv (writes a concrete config file) ────────────────────────────────────


def test_build_cli_command_writes_config(schema, tmp_path):
    out = normalize_params(_base(), schema)
    cfg_path = tmp_path / "run_config.yaml"
    argv = build_cli_command(out, "/bin/sim", cfg_path, subcommand="run")
    assert argv == ["/bin/sim", "run", str(cfg_path)]
    written = yaml.safe_load(cfg_path.read_text())
    assert written["deployment"] == "unified"
    assert written["pools"]["main"]["groups"][0]["gpu"] == "H200"
    assert not any(k.startswith("_") for k in written)  # internals stripped


def test_build_cache_only_subcommand(schema, tmp_path):
    out = normalize_params(_base(), schema)
    argv = build_cli_command(out, "/bin/sim", tmp_path / "c.yaml", subcommand="build-cache-only")
    assert argv[1] == "build-cache-only"


def test_strip_internal_drops_underscore_keys():
    c = {"deployment": "unified", "_env": {"ptp": 2}, "_sweep_labels": {}}
    assert strip_internal(c) == {"deployment": "unified"}


# ── cache key (registry walk over affects_cache leaves) ─────────────────────


def test_cache_key_ignores_rate_and_logdir(schema):
    a = normalize_params(_base(log_dir="x", request_rate=1.0), schema)
    b = normalize_params(_base(log_dir="y", request_rate=99.0), schema)
    assert cache_key(a, schema) == cache_key(b, schema)


def test_cache_key_distinguishes_tp(schema):
    a = normalize_params(_base(arch={"type": "llama3_dense_tp", "tp_size": 4}), schema)
    b = normalize_params(_base(arch={"type": "llama3_dense_tp", "tp_size": 8}), schema)
    assert cache_key(a, schema) != cache_key(b, schema)


def test_cache_key_from_real_schema():
    try:
        real = load_schema("debug")
    except SchemaNotFound:
        pytest.skip("simulator not built")
    cfg = normalize_params(
        _base(
            arch={"type": "llama3_dense_tp", "model_config": "m.json", "tp_size": 4},
            gpu="NVIDIA H200",
        ),
        real,
    )
    keys = {path for path, _ in cache_key(cfg, real)}
    assert any(k.endswith("arch.model_config") for k in keys)
    assert any(k.endswith("arch.tp_size") for k in keys)
    assert not any("request_rate" in k or "log_dir" in k for k in keys)


# ── log_dir templating ({name} over the resolved env / labels) ──────────────


def test_format_log_dir_from_env():
    out = _format_log_dir(
        {"io": {"log_dir": "logs/tp{ptp}_r{rate}"}, "_env": {"ptp": 4, "rate": 10.0}}
    )
    assert out["io"]["log_dir"] == "logs/tp4_r10.0"


def test_format_log_dir_uses_label():
    out = _format_log_dir(
        {"io": {"log_dir": "logs/{ptp}"}, "_env": {"ptp": 1}, "_sweep_labels": {"ptp": "small"}}
    )
    assert out["io"]["log_dir"] == "logs/small"


def test_format_log_dir_noop_without_template():
    out = _format_log_dir({"io": {"log_dir": "logs/plain"}})
    assert out["io"]["log_dir"] == "logs/plain"


def test_format_log_dir_warns_unknown(capsys):
    out = _format_log_dir({"io": {"log_dir": "logs/{unknown}"}, "_env": {}})
    assert out["io"]["log_dir"] == "logs/{unknown}"
    assert "unknown log_dir placeholder {unknown}" in capsys.readouterr().err


# ── metadata persistence ─────────────────────────────────────────────────────


def test_write_shared_and_run_metadata(tmp_path):
    root_dir = tmp_path / "sweep"
    log_dir = tmp_path / "run"
    preset = {"deployment": "unified", "sweep": {"ptp": [1, 4]}}
    params = {"deployment": "unified", "io": {"log_dir": str(log_dir)}, "_sweep_labels": {"x": "y"}}
    metadata.write_shared_metadata(root_dir, preset)
    metadata.write_run_metadata(log_dir, params, ["/bin/sim", "run", "cfg.json"])

    shared = {p.relative_to(root_dir).as_posix() for p in root_dir.rglob("*") if p.is_file()}
    names = {p.relative_to(log_dir).as_posix() for p in log_dir.rglob("*") if p.is_file()}
    assert {"preset.json", "git_snapshot/commit.txt"} <= shared
    assert {"raw/params.json", "raw/command.txt", "manifest.json"} <= names

    saved = json.loads((log_dir / "raw" / "params.json").read_text())
    assert "_sweep_labels" not in saved  # launcher-internal stripped
    assert "start_ts" in saved


# ── experiment-root (sweep base_dir) ─────────────────────────────────────────


def _root(*paths):
    return str(_experiment_root([{"io": {"log_dir": p}} for p in paths]))


def test_experiment_root_siblings():
    assert _root("/x/logs/a", "/x/logs/b").endswith("/x/logs")


def test_experiment_root_disjoint_reaches_root():
    assert _root("/x/a/r", "/y/b/r") == "/"


# ── log_dir collision validation ─────────────────────────────────────────────


def test_validate_unique_log_dirs_rejects_duplicates(capsys):
    params = [
        {"io": {"log_dir": "logs/same"}},
        {"io": {"log_dir": "./logs/same"}},
    ]
    assert validate_unique_log_dirs(params) is False
    assert "runs share log_dir" in capsys.readouterr().err


def test_run_sweep_aborts_on_log_dir_collision(schema, capsys):
    params = [{"io": {"log_dir": "logs/same"}}, {"io": {"log_dir": "logs/same"}}]
    assert asyncio.run(_run_sweep_async(params, {}, schema, "debug", parallelism=1)) == 2
    assert "aborting sweep" in capsys.readouterr().err


def test_main_validates_log_dirs_before_dry_run(tmp_path, schema, monkeypatch, capsys):
    import launcher.exec as exec_module
    from launcher import __main__ as main_module

    preset_path = tmp_path / "preset.json"
    # Two sweep points that DO change config (tp_size), but write to the same
    # (un-templated) log_dir → collision caught after the plan print.
    preset_path.write_text(
        json.dumps(
            _base(
                arch={"type": "llama3_dense_tp", "tp_size": "${ptp}"},
                sweep={"ptp": [1, 2]},
                log_dir="logs/same",
            )
        )
    )
    monkeypatch.setattr(exec_module, "cargo_build", lambda build_type, **kwargs: True)
    monkeypatch.setattr(main_module, "load_schema", lambda build_type: schema)

    assert main_module.main([str(preset_path), "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert "[plan] 2 run(s) across 1 preset(s)" in captured.out
    assert "aborting sweep" in captured.err


# ── aggregator contract (sweep axes from _env) ──────────────────────────────


def test_sweep_axes_from_env(schema):
    runs = [
        {"_env": {"ptp": 1, "ep": 4, "model": "m"}},
        {"_env": {"ptp": 2, "ep": 4, "model": "m"}},
        {"_env": {"ptp": 1, "ep": 8, "model": "m"}},
    ]
    assert set(_sweep_axes(runs)) == {"ptp", "ep"}  # model constant → not an axis


def test_sweep_axes_folds_compound_members_into_group(schema):
    # tp/rate are compound members of group tp_rate; the group is the single axis,
    # the members must NOT each become an independent axis (no fake tp×rate grid).
    runs = [
        {"_env": {"tp_rate": "fast", "tp": 4, "rate": 10}, "_compound_members": ["rate", "tp"]},
        {"_env": {"tp_rate": "slow", "tp": 8, "rate": 20}, "_compound_members": ["rate", "tp"]},
    ]
    assert _sweep_axes(runs) == ["tp_rate"]


def test_sweep_axes_preserve_dsl_order(schema):
    runs = [
        {"_env": {"tp": 2, "request_rate": 40, "derived_batch": 80}},
        {"_env": {"tp": 4, "request_rate": 60, "derived_batch": 240}},
    ]
    assert _sweep_axes(runs) == ["tp", "request_rate", "derived_batch"]


def test_sweep_manifest_upserts_compatible_invocations(tmp_path):
    first = [
        {
            "io": {"log_dir": str(tmp_path / "r40")},
            "_env": {"request_rate": 40},
            "_sweep_labels": {"request_rate": "r40"},
        },
        {
            "io": {"log_dir": str(tmp_path / "r50")},
            "_env": {"request_rate": 50},
            "_sweep_labels": {"request_rate": "r50"},
        },
    ]
    second = [
        {
            "io": {"log_dir": str(tmp_path / "r50")},
            "_env": {"request_rate": 50},
            "_sweep_labels": {"request_rate": "r50-new"},
        },
        {
            "io": {"log_dir": str(tmp_path / "r60")},
            "_env": {"request_rate": 60},
            "_sweep_labels": {"request_rate": "r60"},
        },
    ]

    manifest_path = _write_sweep_manifest(first, tmp_path)
    _write_sweep_manifest(second, tmp_path)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["axes"] == ["request_rate"]
    assert [member["path"] for member in manifest["runs"]] == ["r40", "r50", "r60"]
    assert manifest["runs"][1]["labels"]["request_rate"] == "r50-new"


def test_sweep_manifest_skips_multi_preset_batch_without_axes(tmp_path):
    manifest_path = _write_sweep_manifest(
        [
            {"io": {"log_dir": str(tmp_path / "preset-a")}, "_env": {}},
            {"io": {"log_dir": str(tmp_path / "preset-b")}, "_env": {}},
        ],
        tmp_path,
    )

    assert manifest_path is None
    assert not (tmp_path / "sweep_manifest.json").exists()


def test_sweep_manifest_rejects_axis_mismatch(tmp_path):
    _write_sweep_manifest(
        [
            {"io": {"log_dir": str(tmp_path / "r40")}, "_env": {"rate": 40}},
            {"io": {"log_dir": str(tmp_path / "r50")}, "_env": {"rate": 50}},
        ],
        tmp_path,
    )
    with pytest.raises(ValueError, match="do not match"):
        _write_sweep_manifest(
            [
                {"io": {"log_dir": str(tmp_path / "tp2")}, "_env": {"tp": 2}},
                {"io": {"log_dir": str(tmp_path / "tp4")}, "_env": {"tp": 4}},
            ],
            tmp_path,
        )


def test_aggregate_calls_rust_sweep_pipeline(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        "launcher.sweep.run_sweep_analysis",
        lambda experiment_dir, build_type: calls.append((experiment_dir, build_type)),
    )

    _aggregate(tmp_path, "release")

    assert calls == [(tmp_path, "release")]


# ── real subprocess plumbing (uses the built binary) ────────────────────────


def test_cargo_build_env_pins_launcher_python_and_drops_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PYTHONHOME", "/poisoned/python-home")
    monkeypatch.setenv("PYTHONPATH", "/poisoned/python-path")

    env = _cargo_build_env()

    assert env["PYTHON"] == sys.executable
    assert env["PYO3_PYTHON"] == sys.executable
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env


def test_logged_process_captures_stdout(tmp_path):
    binary = binary_path("debug")
    if not binary.is_file():
        pytest.skip("simulator not built")
    # A config pointing at a nonexistent trace fails fast; we just verify the
    # wrapper spawns, captures stdout, and reports the non-zero exit.
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "deployment": "unified",
                "workload": {
                    "trace_files": ["does/not/exist.csv"],
                    "duration_ms": 5000.0,
                    "run_to_end": False,
                    "request_rate": 10.0,
                    "arrival_mode": "trace_timed",
                    "session_dependency": "independent",
                },
                "io": {
                    "log_dir": str(tmp_path),
                    "log_level": "info",
                    "quiet": False,
                    "force_cache_build": False,
                },
                "pools": {
                    "main": {
                        "placement": "least-queued",
                        "groups": [
                            {
                                "gpu": "NVIDIA H200",
                                "replicas": 1,
                                "arch": {
                                    "type": "llama3_dense",
                                    "model_config": "model/config/llama3_8b.json",
                                    "fp8": False,
                                },
                                "worker": {"type": "barebone", "attn_gpu_memory_gb": 80.0},
                            }
                        ],
                    }
                },
            }
        )
    )
    argv = [str(binary), "run", str(cfg)]
    result = asyncio.run(
        run_logged_process(
            argv,
            tmp_path,
            env=_build_subprocess_env(),
            name="test-simulator",
        )
    )
    assert result.succeeded is False
    assert (tmp_path / "stdout.log").read_text().strip()


# ── resume / --refresh (.complete marker, INV-5) ────────────────────────────


def test_complete_marker_requires_validated_artifacts(tmp_path):
    assert not _is_complete(tmp_path)
    _mark_complete(tmp_path)
    assert (tmp_path / COMPLETE_MARKER).is_file()
    assert not _is_complete(tmp_path)


def test_resume_skips_completed_run(tmp_path, schema, monkeypatch):
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    _mark_complete(log_dir)
    monkeypatch.setattr(
        "launcher.sweep.validate_simulation_artifacts",
        lambda _log_dir: ArtifactValidation(()),
    )
    params = normalize_params(_base(log_dir=str(log_dir)), schema)
    assert asyncio.run(_run_single_async(params, None, schema, "debug")) is True


def test_public_run_entrypoints_require_explicit_schema():
    with pytest.raises(TypeError, match="requires a loaded Schema"):
        run_single(_base(), None, None)
    with pytest.raises(TypeError, match="requires a loaded Schema"):
        run_sweep([_base()], _base(), None)


def test_failed_run_leaves_no_marker(tmp_path, schema):
    binary = binary_path("debug")
    if not binary.is_file():
        pytest.skip("simulator not built")
    log_dir = tmp_path / "run"
    params = normalize_params(
        _base(
            log_dir=str(log_dir),
            arch={"model_config": "model/config/llama3_8b.json"},
            gpu="NVIDIA H200",
        ),
        schema,
    )
    # No trace file on disk → run fails; marker must not be written.
    assert asyncio.run(_launch_one(params, "debug")) is False
    assert not _is_complete(log_dir)


def test_zero_exit_without_required_artifacts_fails_stage(
    tmp_path, schema, monkeypatch
):
    log_dir = tmp_path / "run"
    params = normalize_params(_base(log_dir=str(log_dir)), schema)

    async def successful_process_without_artifacts(*_arguments, **_keyword_arguments):
        return ProcessResult(
            argv=("simulator", "run"),
            pid=123,
            process_group_id=123,
            exit_code=0,
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(
        "launcher.sweep.run_logged_process",
        successful_process_without_artifacts,
    )

    assert asyncio.run(_launch_one(params, "debug", analyze=False)) is False
    stage = json.loads(
        (log_dir / ".launcher/stages/validate_raw_artifacts.json").read_text()
    )
    assert stage["state"] == "FAILED"
    assert "summary.json" in stage["error"]
    assert not (log_dir / COMPLETE_MARKER).exists()


# ── --override parsing (dotted-path into the tree) ──────────────────────────


def test_apply_overrides_dotted_paths():
    from launcher.__main__ import _apply_overrides

    out = _apply_overrides(
        _base(arch={"type": "llama3_dense_tp", "tp_size": 2}),
        [
            "pools.main.groups.0.arch.tp_size=8",
            "io.log_dir=logs/z",
            "workload.request_rate=2.5",
            "pools.main.groups.0.arch.model_config=a/b.json",
        ],
    )
    assert _arch(out)["tp_size"] == 8 and isinstance(_arch(out)["tp_size"], int)
    assert out["io"]["log_dir"] == "logs/z"
    assert out["workload"]["request_rate"] == 2.5
    assert _arch(out)["model_config"] == "a/b.json"


# ── list-params --human (registry-shaped table) ────────────────────────────


def test_human_table_shows_registry_sections(schema, capsys, tmp_path, monkeypatch):
    from launcher import __main__ as main_module

    # _print_params_table re-reads the raw file only for the non-human dump; the
    # human path formats the Registry directly.
    main_module._print_params_table(schema, human=True, build_type="debug")
    out = capsys.readouterr().out
    assert "== deployments" in out
    assert "main→iter_wise" in out
    assert "arch providers" in out and "llama3_dense_tp" in out
    assert "arch_common" in out and "model_config" in out
    assert "pool_common" in out and "placement" in out


# ── per-kernel backend overrides (backends / backends_file) ──────────────────


def test_backends_block_is_control_not_unknown(schema):
    # `backends` is a control key, so a backends block at the root is not flagged
    # as an unknown key (it is not a schema-walked config node).
    errs = validate_params(_base(backends={"main/unified.attn.qkv": ["fa2"]}), schema)
    assert errs == []


def test_backends_unflatten_and_substitute(schema):
    from launcher.__main__ import _merge_backends_file

    preset = _base(
        backends={
            "main/unified.attn.qkv": "${attn_be}",
            "main/unified.mlp.down": ["torch"],
        },
        sweep={"attn_be": {"fa2": ["fa2"], "both": ["fa2", "fa3"]}},
    )
    preset = _merge_backends_file(preset, "preset.yml")
    # flat `pool/role` keys are un-flattened into Rust's nested shape.
    assert preset["backends"] == {
        "main": {"unified.attn.qkv": "${attn_be}", "unified.mlp.down": ["torch"]}
    }
    cands = expand_sweep_params(preset, schema)
    got = {c["_sweep_labels"]["attn_be"]: c["backends"]["main"]["unified.attn.qkv"] for c in cands}
    # the `${attn_be}` value is substituted per combo (best-of-N candidate list)...
    assert got == {"fa2": ["fa2"], "both": ["fa2", "fa3"]}
    # ...while the pinned literal is unchanged across combos.
    assert all(c["backends"]["main"]["unified.mlp.down"] == ["torch"] for c in cands)


def test_backends_file_merge(tmp_path, schema):
    from launcher.__main__ import _merge_backends_file

    (tmp_path / "backends.yaml").write_text("backends:\n  main/unified.attn.qkv: [fa2, fa3]\n")
    preset = _base(backends_file="backends.yaml")
    preset = _merge_backends_file(preset, str(tmp_path / "preset.yml"))
    assert "backends_file" not in preset
    assert preset["backends"] == {"main": {"unified.attn.qkv": ["fa2", "fa3"]}}


def test_backends_file_wins_over_inline(tmp_path, schema):
    from launcher.__main__ import _merge_backends_file

    (tmp_path / "backends.yaml").write_text("backends:\n  main/unified.attn.qkv: [fa3]\n")
    preset = _base(
        backends={"main/unified.attn.qkv": ["fa2"], "main/unified.mlp.down": ["torch"]},
        backends_file="backends.yaml",
    )
    preset = _merge_backends_file(preset, str(tmp_path / "preset.yml"))
    # file overrides the inline value on collision; non-colliding inline survives.
    assert preset["backends"]["main"]["unified.attn.qkv"] == ["fa3"]
    assert preset["backends"]["main"]["unified.mlp.down"] == ["torch"]


def test_backends_non_pool_prefixed_key_rejected():
    from launcher.__main__ import PresetError, _merge_backends_file

    preset = _base(backends={"unified.attn.qkv": ["fa2"]})  # missing `pool/`
    with pytest.raises(PresetError, match="pool/role"):
        _merge_backends_file(preset, "preset.yml")


def test_backends_missing_file_rejected(tmp_path):
    from launcher.__main__ import PresetError, _merge_backends_file

    preset = _base(backends_file="nope.yaml")
    with pytest.raises(PresetError, match="not found"):
        _merge_backends_file(preset, str(tmp_path / "preset.yml"))


def test_backends_undeclared_placeholder_rejected(schema):
    from launcher.__main__ import _merge_backends_file

    preset = _base(backends={"main/unified.attn.qkv": "${nope}"})
    preset = _merge_backends_file(preset, "preset.yml")
    with pytest.raises(ValueError, match="undefined placeholders"):
        expand_sweep_params(preset, schema)


def test_backends_written_through_to_config(tmp_path):
    # write_config keeps the `backends` block (a non-`_` key) for the Rust binary.
    from launcher.schema.argv import write_config

    cand = _base()
    cand["backends"] = {"main": {"unified.attn.qkv": ["fa2"]}}
    path = write_config(cand, tmp_path / "cfg.yaml")
    written = yaml.safe_load(path.read_text())
    assert written["backends"] == {"main": {"unified.attn.qkv": ["fa2"]}}


# ── integration with the real Rust-generated schema ─────────────────────────


def test_real_schema_shape():
    try:
        real = load_schema("debug")
    except SchemaNotFound:
        pytest.skip("simulator not built; run `uv run cargo build` + list-params first")
    assert "unified" in real.deployments
    assert real.roles("unified") == {"main": "iter_wise"}
    assert "llama3_dense_tp" in real.arch_tags("iter_wise")
    assert "barebone" in real.worker_tags("iter_wise")
    tp = next(p for p in real.arch_params("iter_wise", "llama3_dense_tp") if p["name"] == "tp_size")
    assert tp["affects_cache"] is True
