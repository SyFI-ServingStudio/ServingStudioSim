"""Build and invoke TraceLab's typed trace frontend for a profiling run."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import LoadGeneratorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACELAB_ROOT = Path(__file__).resolve().parent / "tracelab"
REPLAY_MANIFEST = TRACELAB_ROOT / "replay" / "Cargo.toml"
SESSION_RUNNER = TRACELAB_ROOT / "replay" / "target" / "release" / "session_runner"


@dataclass(frozen=True)
class PreparedReplay:
    """Resolved TraceLab invocation inputs prepared before vLLM starts."""

    trace_path: Path
    text_file: Path
    tokenizer: str
    log_path: Path
    summary_path: Path


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_session_runner() -> Path:
    """Build TraceLab before starting vLLM so compilation is outside nsys."""
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(REPLAY_MANIFEST),
            "--bin",
            "session_runner",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return SESSION_RUNNER


def prepare_replay(config: LoadGeneratorConfig, log_dir: Path) -> PreparedReplay:
    """Resolve frontend inputs before launching the server."""
    trace_path = _repo_path(config.frontend.path)
    if not trace_path.is_file():
        raise FileNotFoundError(f"profiling trace not found: {trace_path}")
    text_file = _repo_path(config.text_file)
    if not text_file.is_file():
        raise FileNotFoundError(f"TraceLab text corpus not found: {text_file}")
    return PreparedReplay(
        trace_path=trace_path,
        text_file=text_file,
        tokenizer=config.tokenizer,
        log_path=log_dir / "load_generator" / "replay.jsonl",
        summary_path=log_dir / "load_generator" / "summary.json",
    )


def run_replay(
    config: LoadGeneratorConfig,
    prepared: PreparedReplay,
    *,
    base_url: str,
    model: str,
) -> dict:
    """Replay the shared trace against the ready OpenAI-compatible server."""
    prepared.log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(SESSION_RUNNER),
        "--trace",
        str(prepared.trace_path),
        "--trace-format",
        config.frontend.type,
        "--text-file",
        str(prepared.text_file),
        "--tokenizer",
        prepared.tokenizer,
        "--base-url",
        f"{base_url.rstrip('/')}/v1",
        "--model",
        model,
        "--log-path",
        str(prepared.log_path),
        "--summary-path",
        str(prepared.summary_path),
        "--stream-idle-timeout-secs",
        str(config.stream_idle_timeout_secs),
    ]
    optional_args = (
        ("--max-items", config.max_items),
        ("--rate", config.rate),
        ("--token-pool-limit", config.token_pool_limit),
        ("--max-concurrency", config.max_concurrency),
        ("--max-model-len", config.max_model_len),
    )
    for flag, value in optional_args:
        if value is not None:
            argv.extend([flag, str(value)])
    if config.fail_on_context_overflow:
        argv.append("--fail-on-context-overflow")
    argv.extend(config.extra_args)

    subprocess.run(argv, cwd=REPO_ROOT, check=True)
    return {
        "source_trace": str(prepared.trace_path.resolve()),
        "frontend_type": config.frontend.type,
        "log_path": str(prepared.log_path),
        "summary_path": str(prepared.summary_path),
    }
