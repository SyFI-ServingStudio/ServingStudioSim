"""Build and invoke req-frontend's typed trace frontend for a profiling run."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import LoadGeneratorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
REQ_FRONTEND_ROOT = Path(__file__).resolve().parent / "req-frontend"
REPLAY_MANIFEST = REQ_FRONTEND_ROOT / "Cargo.toml"
SESSION_RUNNER = REQ_FRONTEND_ROOT / "target" / "release" / "session_runner"


@dataclass(frozen=True)
class PreparedReplay:
    """Resolved req-frontend invocation inputs prepared before vLLM starts."""

    trace_path: Path
    text_file: Path
    tokenizer: str
    log_path: Path
    summary_path: Path


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_session_runner() -> Path:
    """Build req-frontend before starting vLLM so compilation is outside nsys."""
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
        raise FileNotFoundError(f"replay text corpus not found: {text_file}")
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
    measurement_ready: Callable[[], None] | None = None,
    routed_experts_dir: Path | None = None,
) -> dict:
    """Replay the shared trace through its explicitly selected wire backend.

    `routed_experts_dir` turns the replay non-streaming and collects the routes
    the server took per request. The caller decides when that trade is worth
    making; this layer only knows that it costs the per-event timeline.
    """
    prepared.log_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_base_url = (
        f"{base_url.rstrip('/')}/v1"
        if config.backend.type == "openai"
        else base_url.rstrip("/")
    )
    argv = [
        str(SESSION_RUNNER),
        "--trace",
        str(prepared.trace_path),
        "--input-file-format",
        config.frontend.input_file_format,
        "--text-file",
        str(prepared.text_file),
        "--tokenizer",
        prepared.tokenizer,
        "--base-url",
        protocol_base_url,
        "--backend",
        config.backend.cli_value,
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
        ("--arrival-mode", config.arrival_mode),
        ("--token-pool-limit", config.token_pool_limit),
        ("--max-concurrency", config.max_concurrency),
        ("--max-model-len", config.max_model_len),
    )
    for flag, value in optional_args:
        if value is not None:
            argv.extend([flag, str(value)])
    if config.context_limit_skip_enabled:
        argv.append("--skip-when-reaching-limit")
    if config.warmup:
        argv.append("--warmup")
    if measurement_ready is not None:
        argv.append("--measurement-gate")
    if routed_experts_dir is not None:
        argv.extend(["--routed-experts-dir", str(routed_experts_dir)])
    argv.extend(config.extra_args)

    if measurement_ready is None:
        subprocess.run(argv, cwd=REPO_ROOT, check=True)
    else:
        with subprocess.Popen(
            argv, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        ) as proc:
            released = False
            try:
                for line in proc.stdout:
                    if line.rstrip("\n") == "REQ_FRONTEND_MEASUREMENT_READY_V1":
                        if released:
                            raise RuntimeError("duplicate frontend measurement boundary")
                        measurement_ready()
                        proc.stdin.write("continue\n")
                        proc.stdin.flush()
                        released = True
                    else:
                        print(line, end="", flush=True)
                returncode = proc.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, argv)
                if not released:
                    raise RuntimeError("frontend exited without a measurement boundary")
            except BaseException:
                proc.kill()
                proc.wait()
                raise
    return {
        "source_trace": str(prepared.trace_path.resolve()),
        "frontend_type": config.frontend.type,
        "backend_type": config.backend.type,
        "log_path": str(prepared.log_path),
        "summary_path": str(prepared.summary_path),
        "warmup": config.warmup,
        "routed_experts_dir": str(routed_experts_dir) if routed_experts_dir else None,
    }
