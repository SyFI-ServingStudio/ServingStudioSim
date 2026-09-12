#!/usr/bin/env python3
"""Run ServingStudioSim skill-test cases through Codex CLI runner and judge agents."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from shlex import quote

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
DEFAULT_OUT_ROOT = Path("/tmp/vibesim_skill_tests")
SECTION_RE = re.compile(r"^## (?P<name>.+?)\s*$", re.MULTILINE)
REPORT_SEPARATOR = "=" * 80
SECTION_SEPARATOR = "-" * 80


@dataclass(frozen=True)
class SkillTestCase:
    path: Path
    title: str
    agent_input: str
    expected_output_description: str
    pass_criteria: str
    failure_signals: str
    raw_markdown: str


@dataclass(frozen=True)
class CodexRun:
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str
    final_path: str
    timed_out: bool


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = select_cases(args.case)

    if args.list:
        for case in cases:
            print(case.path.name)
        return 0

    if not cases:
        print("No skill-test cases matched.", file=sys.stderr)
        return 2

    run_stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = (args.out_dir or DEFAULT_OUT_ROOT / run_stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for case in cases:
        case_summary = run_case(case, args, out_root)
        summaries.append(case_summary)
        if args.brief:
            status = "PASS" if case_summary["passed"] else "FAIL"
            print(f"{status} {case.path.name} -> {case_summary['out_dir']}")
        else:
            print_case_report(case, case_summary, args.max_output_chars)

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary: {summary_path}")
    return 0 if all(summary["passed"] for summary in summaries) else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run markdown skill-test cases with one Codex CLI invocation as "
            "runner and a second Codex CLI invocation as judge."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Case filename, stem, or glob. May be repeated. Defaults to every "
            "*.md case except README.md."
        ),
    )
    parser.add_argument("--list", action="store_true", help="List selected cases and exit.")
    parser.add_argument("--out-dir", type=Path, help="Artifact directory. Defaults under /tmp.")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI binary.")
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Print only one PASS/FAIL line per case. Full artifacts are still saved.",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=20000,
        help="Maximum characters printed for each large report section.",
    )
    parser.add_argument("--runner-model", help="Optional model override for the runner.")
    parser.add_argument("--judge-model", help="Optional model override for the judge.")
    parser.add_argument(
        "--runner-sandbox",
        default="workspace-write",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help=(
            "Runner sandbox. Use danger-full-access only for trusted GPU profiling "
            "evals that cannot run inside the normal sandbox."
        ),
    )
    parser.add_argument(
        "--runner-bypass-sandbox",
        action="store_true",
        help=(
            "Pass Codex CLI --dangerously-bypass-approvals-and-sandbox for the runner. "
            "Use only inside a trusted external sandbox."
        ),
    )
    parser.add_argument(
        "--runner-timeout-s",
        type=int,
        default=1800,
        help="Runner Codex process timeout.",
    )
    parser.add_argument(
        "--judge-timeout-s",
        type=int,
        default=900,
        help="Judge Codex process timeout.",
    )
    parser.add_argument(
        "--add-dir",
        action="append",
        default=["/tmp"],
        help="Extra writable/readable directory for runner Codex. May be repeated.",
    )
    return parser.parse_args(argv)


def select_cases(patterns: list[str]) -> list[SkillTestCase]:
    case_paths = [
        path
        for path in sorted(CASE_DIR.glob("*.md"))
        if path.name != "README.md"
    ]
    if patterns:
        case_paths = [
            path
            for path in case_paths
            if any(matches_case(path, pattern) for pattern in patterns)
        ]
    return [load_case(path) for path in case_paths]


def matches_case(path: Path, pattern: str) -> bool:
    names = {path.name, path.stem}
    try:
        names.add(str(path.relative_to(CASE_DIR)))
    except ValueError:
        pass
    if any(name == pattern for name in names):
        return True
    return any(fnmatch.fnmatch(name, pattern) for name in names)


def load_case(path: Path) -> SkillTestCase:
    raw_markdown = path.read_text(encoding="utf-8")
    sections = split_sections(raw_markdown)
    missing = [
        section
        for section in (
            "Agent Input",
            "Expected Output Description",
            "Pass Criteria",
            "Failure Signals",
        )
        if section not in sections
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path} is missing required section(s): {joined}")

    title = first_heading(raw_markdown) or path.stem.replace("_", " ").title()
    return SkillTestCase(
        path=path,
        title=title,
        agent_input=sections["Agent Input"].strip(),
        expected_output_description=sections["Expected Output Description"].strip(),
        pass_criteria=sections["Pass Criteria"].strip(),
        failure_signals=sections["Failure Signals"].strip(),
        raw_markdown=raw_markdown,
    )


def split_sections(markdown: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[match.group("name").strip()] = markdown[start:end].strip()
    return sections


def first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def run_case(
    case: SkillTestCase,
    args: argparse.Namespace,
    out_root: Path,
) -> dict[str, object]:
    case_out_dir = out_root / case.path.stem
    if case_out_dir.exists():
        shutil.rmtree(case_out_dir)
    case_out_dir.mkdir(parents=True)

    case_copy_path = case_out_dir / "case.md"
    runner_prompt_path = case_out_dir / "runner_prompt.md"
    judge_prompt_path = case_out_dir / "judge_prompt.md"
    runner_final_path = case_out_dir / "runner_final.md"
    judge_final_path = case_out_dir / "judge_final.json"

    case_copy_path.write_text(case.raw_markdown, encoding="utf-8")
    runner_prompt = build_runner_prompt(case)
    runner_prompt_path.write_text(runner_prompt, encoding="utf-8")

    runner_run = run_codex(
        command=build_runner_command(args, runner_final_path),
        prompt=runner_prompt,
        stdout_path=case_out_dir / "runner_stdout.log",
        stderr_path=case_out_dir / "runner_stderr.log",
        timeout_s=args.runner_timeout_s,
    )
    (case_out_dir / "runner_command.json").write_text(
        json.dumps(asdict(runner_run), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    runner_final = read_optional_text(runner_final_path)
    judge_prompt = build_judge_prompt(case, runner_run, runner_final)
    judge_prompt_path.write_text(judge_prompt, encoding="utf-8")

    judge_run = run_codex(
        command=build_judge_command(args, judge_final_path),
        prompt=judge_prompt,
        stdout_path=case_out_dir / "judge_stdout.log",
        stderr_path=case_out_dir / "judge_stderr.log",
        timeout_s=args.judge_timeout_s,
    )
    (case_out_dir / "judge_command.json").write_text(
        json.dumps(asdict(judge_run), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    judge_result = load_judge_result(judge_final_path)
    passed = (
        runner_run.returncode == 0
        and not runner_run.timed_out
        and judge_run.returncode == 0
        and not judge_run.timed_out
        and judge_result.get("passed") is True
    )
    summary = {
        "case": case.path.name,
        "out_dir": str(case_out_dir),
        "passed": passed,
        "runner_returncode": runner_run.returncode,
        "runner_timed_out": runner_run.timed_out,
        "judge_returncode": judge_run.returncode,
        "judge_timed_out": judge_run.timed_out,
        "judge_result": judge_result,
    }
    (case_out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def print_case_report(
    case: SkillTestCase,
    case_summary: dict[str, object],
    max_output_chars: int,
) -> None:
    """Print the human-facing eval transcript while full logs stay in artifacts."""

    status = "PASS" if case_summary["passed"] else "FAIL"
    out_dir = Path(str(case_summary["out_dir"]))
    runner_output = read_optional_text(out_dir / "runner_final.md").strip() or "(empty)"
    judge_result = case_summary.get("judge_result")

    print()
    print(REPORT_SEPARATOR)
    print(f"{status} {case.title}")
    print(f"case: {case.path.name}")
    print(f"artifacts: {out_dir}")
    print(SECTION_SEPARATOR)
    print("Agent Input")
    print(truncate_for_console(case.agent_input, max_output_chars))
    print(SECTION_SEPARATOR)
    print("Agent Output")
    print(truncate_for_console(runner_output, max_output_chars))
    print(SECTION_SEPARATOR)
    print("Evaluator Decision")
    print(format_judge_decision(judge_result))
    diagnostics = format_process_diagnostics(out_dir, case_summary, max_output_chars)
    if diagnostics:
        print(SECTION_SEPARATOR)
        print("Process Diagnostics")
        print(diagnostics)
    print(REPORT_SEPARATOR)


def format_judge_decision(judge_result: object) -> str:
    if not isinstance(judge_result, dict):
        return f"invalid judge result: {judge_result!r}"

    lines = [
        f"passed: {judge_result.get('passed')}",
        f"score: {judge_result.get('score')}",
        f"summary: {judge_result.get('summary', '')}",
    ]
    for field_name in (
        "commands_seen",
        "missing_requirements",
        "failure_signals",
        "notes",
    ):
        values = judge_result.get(field_name)
        lines.append(f"{field_name}:")
        if isinstance(values, list) and values:
            lines.extend(f"- {value}" for value in values)
        elif isinstance(values, list):
            lines.append("- none")
        else:
            lines.append(f"- {values!r}")
    return "\n".join(lines)


def format_process_diagnostics(
    out_dir: Path,
    case_summary: dict[str, object],
    max_output_chars: int,
) -> str:
    diagnostic_blocks: list[str] = []
    for role in ("runner", "judge"):
        returncode = case_summary.get(f"{role}_returncode")
        timed_out = case_summary.get(f"{role}_timed_out")
        if returncode == 0 and timed_out is False:
            continue

        stderr_text = read_optional_text(out_dir / f"{role}_stderr.log").strip()
        stdout_text = read_optional_text(out_dir / f"{role}_stdout.log").strip()
        lines = [f"{role}: returncode={returncode}, timed_out={timed_out}"]
        if stderr_text:
            lines.append("stderr:")
            lines.append(truncate_for_console(stderr_text, max_output_chars))
        if stdout_text:
            lines.append("stdout:")
            lines.append(truncate_for_console(stdout_text, max_output_chars))
        diagnostic_blocks.append("\n".join(lines))
    return "\n\n".join(diagnostic_blocks)


def truncate_for_console(text: str, max_output_chars: int) -> str:
    if max_output_chars <= 0 or len(text) <= max_output_chars:
        return text
    omitted_chars = len(text) - max_output_chars
    truncation_note = f"... truncated {omitted_chars} chars; see artifacts for full text ..."
    return f"{text[:max_output_chars]}\n{truncation_note}"


def build_runner_prompt(case: SkillTestCase) -> str:
    # The runner intentionally receives only the task prompt, not the expected
    # output or pass/fail rubric. That preserves the test as an evaluation.
    return f"""You are the runner for an ServingStudioSim repo-local skill test.

Follow the task below exactly. Discover and use repo-local instructions/skills
as a normal fresh coding agent would. Do not modify repository files unless the
task explicitly asks for edits.

## Task

{case.agent_input}
"""


def build_judge_prompt(
    case: SkillTestCase,
    runner_run: CodexRun,
    runner_final: str,
) -> str:
    command = shell_join(runner_run.command)
    return f"""You are the judge for an ServingStudioSim repo-local skill test.

Evaluate whether the runner output satisfies the case. Return only JSON matching
the supplied output schema. Do not edit files and do not run repo commands.

## Case

{case.raw_markdown}

## Runner Process

- Command: `{command}`
- Return code: `{runner_run.returncode}`
- Timed out: `{runner_run.timed_out}`

## Runner Final Output

```text
{runner_final}
```
"""


def build_runner_command(args: argparse.Namespace, final_path: Path) -> list[str]:
    command = [args.codex_bin]
    if not args.runner_bypass_sandbox:
        # `--ask-for-approval` is a top-level Codex flag in current CLI builds,
        # while sandbox and output flags belong to `codex exec`.
        command.extend(["--ask-for-approval", "never"])
    command.extend(
        [
            "exec",
            "--cd",
            str(REPO_ROOT),
            "--ephemeral",
            "--output-last-message",
            str(final_path),
        ]
    )
    if args.runner_model:
        command.extend(["--model", args.runner_model])
    if args.runner_bypass_sandbox:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["--sandbox", args.runner_sandbox])
        for directory in args.add_dir:
            command.extend(["--add-dir", directory])
    command.append("-")
    return command


def build_judge_command(args: argparse.Namespace, final_path: Path) -> list[str]:
    command = [
        args.codex_bin,
        "--ask-for-approval",
        "never",
        "exec",
        "--cd",
        str(REPO_ROOT),
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--output-schema",
        str(CASE_DIR / "judge_schema.json"),
        "--output-last-message",
        str(final_path),
    ]
    if args.judge_model:
        command.extend(["--model", args.judge_model])
    command.append("-")
    return command


def run_codex(
    *,
    command: list[str],
    prompt: str,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int,
) -> CodexRun:
    timed_out = False
    try:
        process = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        returncode = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = coerce_output(exc.stdout)
        stderr = coerce_output(exc.stderr)

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    final_path = output_last_message_path(command)
    return CodexRun(
        command=command,
        returncode=returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        final_path=str(final_path),
        timed_out=timed_out,
    )


def output_last_message_path(command: list[str]) -> Path:
    for index, part in enumerate(command):
        if part == "--output-last-message" and index + 1 < len(command):
            return Path(command[index + 1])
    raise ValueError("Codex command is missing --output-last-message")


def load_judge_result(path: Path) -> dict[str, object]:
    raw_text = read_optional_text(path).strip()
    if not raw_text:
        return {
            "passed": False,
            "score": 0,
            "summary": "Judge did not produce a final JSON result.",
            "commands_seen": [],
            "missing_requirements": ["judge_final.json was empty"],
            "failure_signals": [],
            "notes": [],
        }
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "score": 0,
            "summary": f"Judge output was not valid JSON: {exc}",
            "commands_seen": [],
            "missing_requirements": ["valid judge JSON"],
            "failure_signals": [],
            "notes": [raw_text],
        }
    if not isinstance(value, dict):
        return {
            "passed": False,
            "score": 0,
            "summary": "Judge JSON was not an object.",
            "commands_seen": [],
            "missing_requirements": ["object-shaped judge JSON"],
            "failure_signals": [],
            "notes": [raw_text],
        }
    return value


def read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def shell_join(command: list[str]) -> str:
    return " ".join(quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
