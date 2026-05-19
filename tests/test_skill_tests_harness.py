from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_harness_module():
    path = Path(__file__).resolve().parent / "skill_tests" / "run_codex_skill_tests.py"
    spec = importlib.util.spec_from_file_location("skill_test_harness", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_case_extracts_runner_prompt_without_rubric() -> None:
    harness = _load_harness_module()
    case_path = (
        Path(__file__).resolve().parent
        / "skill_tests"
        / "profile_query_single_gemm_missing.md"
    )

    case = harness.load_case(case_path)
    runner_prompt = harness.build_runner_prompt(case)

    assert "Use the appropriate repo-local skill" in runner_prompt
    assert "Expected Output Description" not in runner_prompt
    assert "Pass Criteria" not in runner_prompt
    assert "Failure Signals" not in runner_prompt


def test_case_glob_matching_accepts_stem_and_filename() -> None:
    harness = _load_harness_module()
    path = Path("profile_query_single_gemm_missing.md")

    assert harness.matches_case(path, "profile_query_single_gemm_missing")
    assert harness.matches_case(path, "profile_query_single_gemm_missing.md")
    assert harness.matches_case(path, "profile_query_*")


def test_judge_decision_format_is_human_readable() -> None:
    harness = _load_harness_module()

    formatted = harness.format_judge_decision(
        {
            "passed": True,
            "score": 95,
            "summary": "Runner used the expected CLI path.",
            "commands_seen": ["uv run python -m profiling list --json"],
            "missing_requirements": [],
            "failure_signals": [],
            "notes": ["temporary DB used"],
        }
    )

    assert "passed: True" in formatted
    assert "score: 95" in formatted
    assert "commands_seen:" in formatted
    assert "- uv run python -m profiling list --json" in formatted
    assert "missing_requirements:\n- none" in formatted


def test_console_truncation_points_to_artifacts() -> None:
    harness = _load_harness_module()

    truncated = harness.truncate_for_console("abcdef", 3)

    assert truncated.startswith("abc")
    assert "truncated 3 chars" in truncated
    assert "see artifacts" in truncated


def test_codex_approval_flag_precedes_exec() -> None:
    harness = _load_harness_module()
    args = argparse.Namespace(
        codex_bin="codex",
        runner_model=None,
        judge_model=None,
        runner_bypass_sandbox=False,
        runner_sandbox="workspace-write",
        add_dir=["/tmp"],
    )

    runner_command = harness.build_runner_command(args, Path("/tmp/runner.md"))
    judge_command = harness.build_judge_command(args, Path("/tmp/judge.json"))

    assert runner_command[:4] == ["codex", "--ask-for-approval", "never", "exec"]
    assert judge_command[:4] == ["codex", "--ask-for-approval", "never", "exec"]
    assert "--ask-for-approval" not in runner_command[4:]
    assert "--ask-for-approval" not in judge_command[4:]
