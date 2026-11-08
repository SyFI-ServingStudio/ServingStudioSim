"""Build and invoke req-frontend's typed trace frontend for a profiling run."""

from __future__ import annotations

import contextlib
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import LoadGeneratorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
REQ_FRONTEND_ROOT = Path(__file__).resolve().parent / "req-frontend"
REPLAY_MANIFEST = REQ_FRONTEND_ROOT / "Cargo.toml"
SESSION_RUNNER = REQ_FRONTEND_ROOT / "target" / "release" / "session_runner"
# Draws or materializes a canonical trace (`tracegen describe` lists its knobs).
TRACEGEN = REQ_FRONTEND_ROOT / "target" / "release" / "tracegen"


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
    """Build req-frontend before starting vLLM so compilation is outside nsys.

    In an engine image the binary was built at the gitlink when the image was
    (`reference_engines/*/Dockerfile`), and nothing is built at run time.
    """
    from ..image_runtime import in_image

    if in_image():
        if not SESSION_RUNNER.is_file():
            raise FileNotFoundError(f"the image carries no req-frontend binary at {SESSION_RUNNER}")
        return SESSION_RUNNER
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
    server_failure: Callable[[], str | None] | None = None,
) -> dict:
    """Replay the shared trace through its explicitly selected wire backend.

    `routed_experts_dir` turns the replay non-streaming and collects the routes
    the server took per request. The caller decides when that trade is worth
    making; this layer only knows that it costs the per-event timeline.

    `server_failure`, polled every second while the replay runs, returns why the
    server can no longer answer (or None). A dead engine leaves its requests
    open until the stream idle timeout; this stops the replay at once instead.
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

    if measurement_ready is None and server_failure is None:
        subprocess.run(argv, cwd=REPO_ROOT, check=True)
    elif measurement_ready is None:
        with subprocess.Popen(argv, cwd=REPO_ROOT) as proc:
            with _watch_server(proc, server_failure) as failure:
                returncode = proc.wait()
            _raise_server_failure(failure)
            if returncode:
                raise subprocess.CalledProcessError(returncode, argv)
    else:
        with subprocess.Popen(
            argv, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        ) as proc, _watch_server(proc, server_failure) as failure:
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
                _raise_server_failure(failure)
                if returncode:
                    raise subprocess.CalledProcessError(returncode, argv)
                if not released:
                    raise RuntimeError("frontend exited without a measurement boundary")
            except BaseException as error:
                proc.kill()
                proc.wait()
                if failure and not isinstance(error, ServerFailedDuringReplay):
                    raise ServerFailedDuringReplay(failure[0]) from error
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


class ServerFailedDuringReplay(RuntimeError):
    """The serving engine died or stopped while the replay was running."""


@contextlib.contextmanager
def _watch_server(
    proc: subprocess.Popen, server_failure: Callable[[], str | None] | None
) -> Iterator[list[str]]:
    """Kill the replay as soon as `server_failure` reports; yield what it said."""

    failure: list[str] = []
    if server_failure is None:
        yield failure
        return
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(1.0):
            reason = server_failure()
            if reason:
                failure.append(reason)
                proc.kill()
                return

    thread = threading.Thread(target=watch, name="replay-server-watch", daemon=True)
    thread.start()
    try:
        yield failure
    finally:
        stop.set()
        thread.join()


def _raise_server_failure(failure: list[str]) -> None:
    if failure:
        raise ServerFailedDuringReplay(f"server failed during replay: {failure[0]}")
