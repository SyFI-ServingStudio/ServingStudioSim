"""One completion model for all launcher-owned subprocesses.

Completion is defined by the root PID being reaped.  Output is always inherited
or written to a regular file, so pipe EOF is never part of the lifecycle.  Each
root starts a new session; cancellation and timeout therefore target the whole
process group instead of leaving profiler or renderer descendants behind.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO

from .spec import ProcessResult, ProcessSpec

_POLL_INTERVAL_SECONDS = 0.05
_DESCENDANT_SETTLE_SECONDS = 0.15
_TERMINATE_GRACE_SECONDS = 1.0


class ProcessSupervisor:
    """Spawn, reap, and clean one process group per launcher stage."""

    async def run(self, spec: ProcessSpec) -> ProcessResult:
        with ExitStack() as resources:
            stdout_stream, capture_stream = self._open_stdout(spec, resources)
            stdin_stream = self._open_stdin(spec, resources)
            process = self._spawn(spec, stdin_stream, stdout_stream)
            started = time.monotonic()
            termination_reason: str | None = None
            try:
                if spec.timeout_seconds is None:
                    exit_code = await self._wait_root(process)
                else:
                    try:
                        exit_code = await asyncio.wait_for(
                            self._wait_root(process), timeout=spec.timeout_seconds
                        )
                    except TimeoutError:
                        termination_reason = "timeout"
                        await self._terminate_group(process)
                        exit_code = process.wait()
            except asyncio.CancelledError:
                await self._terminate_group(process)
                raise

            leaked_descendants = await self._clean_leaked_descendants(process.pid)
            output = self._read_capture(capture_stream)
            return ProcessResult(
                argv=tuple(str(argument) for argument in spec.argv),
                pid=process.pid,
                process_group_id=process.pid,
                exit_code=exit_code,
                elapsed_seconds=time.monotonic() - started,
                output=output,
                leaked_descendants=leaked_descendants,
                termination_reason=termination_reason,
            )

    def run_sync(self, spec: ProcessSpec) -> ProcessResult:
        """Synchronous adapter with the same spawn/reap/group contract."""

        with ExitStack() as resources:
            stdout_stream, capture_stream = self._open_stdout(spec, resources)
            stdin_stream = self._open_stdin(spec, resources)
            process = self._spawn(spec, stdin_stream, stdout_stream)
            started = time.monotonic()
            termination_reason: str | None = None
            try:
                try:
                    exit_code = process.wait(timeout=spec.timeout_seconds)
                except subprocess.TimeoutExpired:
                    termination_reason = "timeout"
                    self._terminate_group_sync(process)
                    exit_code = process.wait()
            except BaseException:
                self._terminate_group_sync(process)
                raise

            leaked_descendants = self._clean_leaked_descendants_sync(process.pid)
            output = self._read_capture(capture_stream)
            return ProcessResult(
                argv=tuple(str(argument) for argument in spec.argv),
                pid=process.pid,
                process_group_id=process.pid,
                exit_code=exit_code,
                elapsed_seconds=time.monotonic() - started,
                output=output,
                leaked_descendants=leaked_descendants,
                termination_reason=termination_reason,
            )

    @staticmethod
    def _open_stdout(
        spec: ProcessSpec, resources: ExitStack
    ) -> tuple[BinaryIO | int | None, BinaryIO | None]:
        if spec.log_path is not None:
            log_path = Path(spec.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if spec.append_log else "wb"
            stream = resources.enter_context(log_path.open(mode))
            return stream, None
        if spec.capture_output:
            stream = resources.enter_context(
                tempfile.TemporaryFile(prefix="vibesim-process-")
            )
            return stream, stream
        return None, None

    @staticmethod
    def _open_stdin(spec: ProcessSpec, resources: ExitStack) -> BinaryIO | int:
        if spec.input_bytes is None:
            return subprocess.DEVNULL
        stream = resources.enter_context(
            tempfile.TemporaryFile(prefix="vibesim-process-input-")
        )
        stream.write(spec.input_bytes)
        stream.seek(0)
        return stream

    @staticmethod
    def _spawn(
        spec: ProcessSpec,
        stdin_stream: BinaryIO | int,
        stdout_stream: BinaryIO | int | None,
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [str(argument) for argument in spec.argv],
            cwd=spec.cwd,
            env=dict(spec.env) if spec.env is not None else None,
            stdin=stdin_stream,
            stdout=stdout_stream,
            stderr=subprocess.STDOUT if stdout_stream is not None else None,
            start_new_session=True,
            close_fds=True,
        )

    async def _wait_root(self, process: subprocess.Popen[bytes]) -> int:
        pidfd_open = getattr(os, "pidfd_open", None)
        if pidfd_open is None:
            while process.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            return process.returncode

        try:
            pidfd = pidfd_open(process.pid, 0)
        except OSError:
            while process.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            return process.returncode

        loop = asyncio.get_running_loop()
        ready = loop.create_future()

        def mark_ready() -> None:
            if not ready.done():
                ready.set_result(None)

        loop.add_reader(pidfd, mark_ready)
        try:
            await ready
        finally:
            loop.remove_reader(pidfd)
            os.close(pidfd)
        return process.wait()

    async def _terminate_group(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            await self._terminate_descendants_after_root(process.pid)
            return
        self._signal_group(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        if process.poll() is None:
            self._signal_group(process.pid, signal.SIGKILL)
            while process.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        if self._group_exists(process.pid):
            self._signal_group(process.pid, signal.SIGKILL)

    def _terminate_group_sync(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            self._signal_group(process.pid, signal.SIGKILL)
            return
        self._signal_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            self._signal_group(process.pid, signal.SIGKILL)
            process.wait()
        if self._group_exists(process.pid):
            self._signal_group(process.pid, signal.SIGKILL)

    async def _clean_leaked_descendants(self, process_group_id: int) -> bool:
        deadline = time.monotonic() + _DESCENDANT_SETTLE_SECONDS
        while self._group_exists(process_group_id) and time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        if not self._group_exists(process_group_id):
            return False
        await self._terminate_descendants_after_root(process_group_id)
        return True

    def _clean_leaked_descendants_sync(self, process_group_id: int) -> bool:
        deadline = time.monotonic() + _DESCENDANT_SETTLE_SECONDS
        while self._group_exists(process_group_id) and time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL_SECONDS)
        if not self._group_exists(process_group_id):
            return False
        self._signal_group(process_group_id, signal.SIGKILL)
        return True

    async def _terminate_descendants_after_root(self, process_group_id: int) -> None:
        self._signal_group(process_group_id, signal.SIGTERM)
        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        while self._group_exists(process_group_id) and time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        if self._group_exists(process_group_id):
            self._signal_group(process_group_id, signal.SIGKILL)

    @staticmethod
    def _signal_group(process_group_id: int, signal_number: int) -> None:
        try:
            os.killpg(process_group_id, signal_number)
        except ProcessLookupError:
            pass

    @staticmethod
    def _group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _read_capture(stream: BinaryIO | None) -> str:
        if stream is None:
            return ""
        stream.flush()
        stream.seek(0)
        return stream.read().decode(errors="replace")
