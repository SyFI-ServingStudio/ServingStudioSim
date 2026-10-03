"""ROCm roctx iteration annotation shim — the AMD analog of the vLLM fork's NVTX scopes.

The NVIDIA alignment path relies on an *instrumented vLLM fork* that, when
``VLLM_NVTX_SCOPES_FOR_PROFILING=1``, wraps each forward in an NVTX range named
``vllm_iteration(N): <phase>`` and emits a ``VibeSimAlignmentIteration {json}``
model-input record (see ``alignment/profiler/vllm_server.py`` and the NVTX marker
contract in ``README.md``). ``alignment/nsys/parse.py`` keys its kernel-ownership
join on exactly those ``vllm_iteration(N)`` ranges (``ITER_RE``).

On AMD we want the *same* region labels to land in a rocprofv3 rocpd capture so
the existing offline ``alignment/rocpd`` producer — which attributes each kernel
dispatch to the ``vllm_iteration(N): <phase>`` roctx range that contains its
launch — consumes an AMD trace unchanged. This module is that annotation layer,
as a **non-fork, importable shim** rather than a patched vLLM checkout:

- :class:`IterationAnnotator` holds the iteration counter and emits the canonical
  ``vllm_iteration(N): <phase>`` roctx ranges plus the ``VibeSimAlignmentIteration``
  boundary marker. It is pure Python and does the GPU-free work: label formatting
  and sequencing.
- The *roctx backend* is the one GPU-touching seam. On ROCm, PyTorch's
  ``torch.cuda.nvtx.range_push/range_pop/mark`` is retargeted onto roctx, and the
  roctx C API (``librocprofiler-sdk-roctx.so.1`` on the pinned rocprofv3-1.3.2
  stack, legacy ``libroctx64`` on old roctracer stacks) is the direct route;
  either lands the range in the rocpd ``region`` table that
  ``roctx_regions_from_rocpd`` reads. The
  backend is injectable so the annotator is testable with a recording backend and
  no GPU — the recorded labels are what a real capture would store, which a test
  then feeds through the real ``alignment/rocpd`` join.
- :func:`install_vllm_roctx_shim` is the entry point the capture uses: a vLLM
  ``general_plugins`` callable (or a manual import-time patch) that wraps the V1
  GPU model runner's ``execute_model`` so every served forward is bracketed.

Which backend actually populates the rocpd region table on the pinned gfx942
ROCm/torch stack is a GPU-only fact (see the handoff note); the selection order
here prefers the direct roctx C API and falls back to torch, and both emit the
identical label text.
"""

from __future__ import annotations

import ctypes
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol, runtime_checkable

#: Env gate, mirroring the NVIDIA fork's ``VLLM_NVTX_SCOPES_FOR_PROFILING``. The
#: capture driver sets this to ``"1"`` for a timing pass and ``"0"`` to switch the
#: annotation off entirely (a pass that must not carry instrumentation).
ROCTX_SCOPES_ENV = "VLLM_ROCTX_SCOPES_FOR_PROFILING"

#: The ``vllm.general_plugins`` entry-point name this package registers for
#: :func:`install_vllm_roctx_shim` (see ``pyproject.toml``). vLLM auto-loads and
#: calls every entry point in that group in each worker process unless
#: ``VLLM_PLUGINS`` narrows the set; when it does, it must include this name.
#: Where the NVIDIA path bakes the NVTX scopes into the vLLM fork's own source,
#: the AMD path has no fork and instead injects the identical annotation through
#: this stock-vLLM plugin seam.
ROCTX_PLUGIN_ENTRY_POINT_NAME = "vibesim_roctx_shim"

#: The marker family the rocpd/nsys parsers recognize. The default engine prefix
#: is ``vllm``; ``sglang`` is the other value ``ITER_RE`` accepts.
DEFAULT_ENGINE_PREFIX = "vllm"

#: The phases the instrumented path may bracket, same names as the NVTX contract.
#: The eager kernel-only alignment path only needs ``forward``; the rest are here
#: so a finer capture can reuse the exact contract vocabulary.
PHASES = ("preprocess", "forward", "postprocess", "sample", "bookkeep", "eplb")

#: Prefix of the versioned model-input boundary record, emitted as a roctx mark.
#: Kept identical to the NVIDIA fork so a downstream reader recognizes one record.
ITERATION_RECORD_TAG = "VibeSimAlignmentIteration"

#: In-process record schema version carried inside the ``VibeSimAlignmentIteration``
#: payload; bumped only when the payload shape changes.
ITERATION_RECORD_SCHEMA_VERSION = 1


def iteration_label(iteration: int, phase: str, *, prefix: str = DEFAULT_ENGINE_PREFIX) -> str:
    """The canonical ``<prefix>_iteration(N): <phase>`` roctx range label.

    This is the single source of the label spelling. A test asserts it matches
    ``alignment.nsys.parse.ITER_RE`` so the shim and the parser cannot drift.
    """
    if iteration < 0:
        raise ValueError(f"iteration index must be non-negative, got {iteration}")
    if not phase:
        raise ValueError("phase must be a non-empty string")
    return f"{prefix}_iteration({iteration}): {phase}"


def iteration_record_marker(payload: dict, *, iteration: int | None = None) -> str:
    """The ``VibeSimAlignmentIteration {json}`` boundary marker text.

    ``payload`` is the per-iteration model-input shape (query widths, KV lengths,
    etc.) the typed predictor adapter consumes downstream. The shim stamps the
    schema version and, when known, the iteration index so the record is
    self-identifying even though roctx marks carry no structured fields.
    """
    body = {"schema_version": ITERATION_RECORD_SCHEMA_VERSION, **payload}
    if iteration is not None:
        body.setdefault("iteration", iteration)
    return f"{ITERATION_RECORD_TAG} {json.dumps(body, separators=(',', ':'), sort_keys=True)}"


@runtime_checkable
class RoctxBackend(Protocol):
    """The GPU-touching seam: push/pop a named range and drop an instant mark."""

    def range_push(self, message: str) -> None: ...

    def range_pop(self) -> None: ...

    def mark(self, message: str) -> None: ...


@dataclass
class RecordedEvent:
    """One backend call, as a real roctx capture would store it."""

    op: str  # "push" | "pop" | "mark"
    message: str | None


@dataclass
class RecordingBackend:
    """A GPU-free backend that records calls instead of emitting roctx ranges.

    The recorded push/pop pairs are exactly the regions a real capture would
    write to the rocpd ``region`` table, so a test can turn them into a synthetic
    rocpd and run the real ``alignment/rocpd`` ownership join over them.
    """

    events: list[RecordedEvent] = field(default_factory=list)

    def range_push(self, message: str) -> None:
        self.events.append(RecordedEvent("push", message))

    def range_pop(self) -> None:
        self.events.append(RecordedEvent("pop", None))

    def mark(self, message: str) -> None:
        self.events.append(RecordedEvent("mark", message))

    def regions(self) -> list[tuple[str, int, int]]:
        """Fold recorded push/pop pairs into ``(label, start_order, end_order)``.

        Ordinals are monotonic call indices — a stand-in for timestamps a GPU
        capture would assign. Nested ranges pop in LIFO order, matching roctx.
        """
        stack: list[tuple[str, int]] = []
        out: list[tuple[str, int, int]] = []
        for order, event in enumerate(self.events):
            if event.op == "push":
                assert event.message is not None
                stack.append((event.message, order))
            elif event.op == "pop":
                if not stack:
                    raise ValueError("roctx range_pop without a matching range_push")
                label, start = stack.pop()
                out.append((label, start, order))
        if stack:
            raise ValueError(f"unclosed roctx ranges: {[label for label, _ in stack]}")
        return out


class TorchRoctxBackend:
    """Backend over ``torch.cuda.nvtx``, which ROCm PyTorch retargets onto roctx.

    Importing torch is deferred to construction so this module stays importable
    in an interpreter without torch (the GPU-free unit tests never build this).
    """

    def __init__(self) -> None:
        import torch  # noqa: PLC0415 — deferred so the module imports without torch

        self._nvtx = torch.cuda.nvtx

    def range_push(self, message: str) -> None:
        self._nvtx.range_push(message)

    def range_pop(self) -> None:
        self._nvtx.range_pop()

    def mark(self, message: str) -> None:
        self._nvtx.mark(message)


class RoctracerBackend:
    """Backend over the roctx C API (``roctxRangePushA``/``Pop``/``MarkA``).

    The most direct route into the rocpd marker table: rocprofv3's
    ``--marker-trace`` records exactly these roctx ranges. ``ctypes`` loads the
    shared object at construction so module import never requires ROCm present.

    Which shared object matters on the pinned rocprofiler-sdk / rocprofv3-1.3.2
    (gfx942) stack. There, ``--marker-trace`` records roctx ranges ONLY from the
    rocprofiler-sdk roctx library, ``librocprofiler-sdk-roctx.so.1``, which
    rocprofv3 lazily ``dlopen``'s while the tool is live and intercepts in-process.
    The legacy ``libroctx64`` (roctracer) library is NOT the one rocprofiler-sdk's
    marker interception hooks, so ranges pushed through it may never land in the
    rocpd region table on this stack. So the SDK soname is tried first and legacy
    ``libroctx64`` is kept only as a fallback for old roctracer stacks.

    Two hazards, documented here because they are GPU-only facts the on-host unit
    test cannot catch:

    - LD_PRELOAD bypasses interception. rocprofv3's marker capture works by being
      the one that loads/dlopens the roctx library inside the traced process; it
      does not rely on an LD_PRELOAD of the roctx lib. Forcing the roctx library
      in via LD_PRELOAD is not the supported path and can leave the SDK's
      interception seeing no ranges. This shim therefore loads the lib itself with
      ``ctypes`` rather than depending on any preload.
    - Legacy libroctx64 can preempt the SDK MARKER registration. If both
      ``libroctx64`` and ``librocprofiler-sdk-roctx`` are resolvable in the same
      process and the legacy one wins the ``roctx*`` symbol resolution first, the
      MARKER_CORE_RANGE records can be registered against the roctracer path
      instead of the SDK path, and ``--marker-trace`` records nothing. Preferring
      the SDK soname here reduces that risk, but the authoritative check is the
      in-full-vLLM-process ordering on a real GPU (see ROCM_CAPTURE_VALIDATION.md).
    """

    # Most-preferred first: the rocprofiler-sdk roctx library that rocprofv3-1.3.2
    # --marker-trace actually records, then legacy libroctx64 for old stacks.
    _CANDIDATE_SONAMES = (
        "librocprofiler-sdk-roctx.so.1",
        "librocprofiler-sdk-roctx.so",
        "libroctx64.so",
        "libroctx64.so.4",
        "libroctx64.so.1",
    )

    def __init__(self, soname: str | None = None) -> None:
        lib = None
        loaded_soname: str | None = None
        # Prefer the SDK roctx library the sitecustomize preloader already mapped
        # RTLD_GLOBAL before torch (the Gap-2 load-order fix). Reusing that exact
        # handle guarantees our roctx* calls resolve to the SDK lib rocprofv3's
        # MARKER service registered against, not torch's later legacy libroctx64.
        # Skipped when an explicit soname is forced (tests / operator override).
        if soname is None:
            from ._roctx_preload import preloaded_handle, preloaded_soname  # noqa: PLC0415

            handle = preloaded_handle()
            if handle is not None:
                lib = handle
                loaded_soname = preloaded_soname()
        if lib is None:
            names = (soname,) if soname else self._CANDIDATE_SONAMES
            last_error: OSError | None = None
            for candidate in names:
                if candidate is None:
                    continue
                try:
                    lib = ctypes.CDLL(candidate)
                    loaded_soname = candidate
                    break
                except OSError as error:  # not present / not loadable
                    last_error = error
            if lib is None:
                raise OSError(
                    "could not load a roctx shared object "
                    f"(tried {list(names)}): {last_error}"
                )
        lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        lib.roctxRangePushA.restype = ctypes.c_int
        lib.roctxRangePop.argtypes = []
        lib.roctxRangePop.restype = ctypes.c_int
        lib.roctxMarkA.argtypes = [ctypes.c_char_p]
        lib.roctxMarkA.restype = None
        self._lib = lib
        #: The soname actually loaded, so the selection order is observable.
        self.soname = loaded_soname
        # Stamp the resolved soname so it is consistent with (and confirms) what
        # the preloader recorded, whether the handle came from the preloader or a
        # fresh dlopen here.
        if loaded_soname is not None:
            from ._roctx_preload import ROCTX_SONAME_RESOLVED_ENV  # noqa: PLC0415

            os.environ[ROCTX_SONAME_RESOLVED_ENV] = loaded_soname

    def range_push(self, message: str) -> None:
        self._lib.roctxRangePushA(message.encode("utf-8"))

    def range_pop(self) -> None:
        self._lib.roctxRangePop()

    def mark(self, message: str) -> None:
        self._lib.roctxMarkA(message.encode("utf-8"))


#: Backend selection order for ``select_backend("auto")``. The direct roctx C API
#: is tried first (the ``librocprofiler-sdk-roctx.so.1`` the rocprofv3-1.3.2
#: ``--marker-trace`` records, else legacy ``libroctx64``), then torch's
#: roctx-retargeted nvtx.
_AUTO_BACKEND_ORDER: tuple[tuple[str, Callable[[], RoctxBackend]], ...] = (
    ("roctracer", RoctracerBackend),
    ("torch", TorchRoctxBackend),
)


def select_backend(name: str = "auto") -> tuple[str, RoctxBackend]:
    """Resolve a roctx backend by name, returning ``(resolved_name, backend)``.

    ``"auto"`` tries each real backend in :data:`_AUTO_BACKEND_ORDER` and returns
    the first that constructs. ``"roctracer"`` / ``"torch"`` force one. ``"record"``
    returns a :class:`RecordingBackend` (GPU-free; for tests and dry runs).
    """
    if name == "record":
        return "record", RecordingBackend()
    if name == "roctracer":
        return "roctracer", RoctracerBackend()
    if name == "torch":
        return "torch", TorchRoctxBackend()
    if name == "auto":
        errors: list[str] = []
        for resolved, factory in _AUTO_BACKEND_ORDER:
            try:
                return resolved, factory()
            except Exception as error:  # noqa: BLE001 — try the next real backend
                errors.append(f"{resolved}: {error}")
        raise RuntimeError(
            "no roctx backend available (install ROCm torch or libroctx64): "
            + "; ".join(errors)
        )
    raise ValueError(f"unknown roctx backend {name!r}; use auto|roctracer|torch|record")


class IterationAnnotator:
    """Emit the canonical ``vllm_iteration(N)`` roctx ranges and boundary records.

    Holds a monotonic iteration counter; each served forward allocates the next
    index with :meth:`next_iteration`. The label spelling comes from
    :func:`iteration_label`, so every range this emits is guaranteed to match the
    parser's ``ITER_RE``. When disabled (the instrumentation-off pass) every method
    is a no-op and touches no backend.
    """

    def __init__(
        self,
        backend: RoctxBackend,
        *,
        prefix: str = DEFAULT_ENGINE_PREFIX,
        enabled: bool = True,
        start_index: int = 0,
    ) -> None:
        self._backend = backend
        self._prefix = prefix
        self._enabled = enabled
        self._next = start_index

    @property
    def enabled(self) -> bool:
        return self._enabled

    def next_iteration(self) -> int:
        """Allocate and return the next iteration index."""
        index = self._next
        self._next += 1
        return index

    @contextmanager
    def phase(self, phase: str, iteration: int) -> Iterator[int]:
        """Bracket ``phase`` of ``iteration`` in a roctx range."""
        if not self._enabled:
            yield iteration
            return
        self._backend.range_push(iteration_label(iteration, phase, prefix=self._prefix))
        try:
            yield iteration
        finally:
            self._backend.range_pop()

    @contextmanager
    def forward(
        self, *, iteration: int | None = None, record: dict | None = None
    ) -> Iterator[int]:
        """Bracket one forward as ``vllm_iteration(N): forward``.

        Allocates the next index when ``iteration`` is omitted. When ``record`` is
        given, emits the ``VibeSimAlignmentIteration`` boundary marker for the
        iteration just before opening the range, matching the NVIDIA ordering.
        """
        index = self.next_iteration() if iteration is None else iteration
        if self._enabled and record is not None:
            self.emit_iteration_record(record, iteration=index)
        with self.phase("forward", index):
            yield index

    def emit_iteration_record(self, payload: dict, *, iteration: int | None = None) -> None:
        """Emit the ``VibeSimAlignmentIteration {json}`` boundary marker."""
        if not self._enabled:
            return
        self._backend.mark(iteration_record_marker(payload, iteration=iteration))


def roctx_scopes_enabled(environ: dict[str, str] | None = None) -> bool:
    """Whether the env gate turns the iteration annotation on."""
    env = os.environ if environ is None else environ
    return env.get(ROCTX_SCOPES_ENV, "0") == "1"


def install_vllm_roctx_shim(
    *,
    backend_name: str = "auto",
    environ: dict[str, str] | None = None,
) -> bool:
    """Wrap the vLLM V1 GPU model runner's ``execute_model`` with roctx ranges.

    This is the non-fork entry point: register it as a ``vllm.general_plugins``
    callable, or call it once from the server process before serving. When the env
    gate is off it returns ``False`` and patches nothing. Otherwise it monkeypatches
    ``vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model`` so every
    served forward is bracketed by ``vllm_iteration(N): forward`` and preceded by a
    ``VibeSimAlignmentIteration`` record carrying the batch's token/KV shape.

    Imports vLLM lazily so this module stays importable (and unit-testable) without
    vLLM or a GPU; raises a clear error only if called in an environment that gates
    the shim on but cannot provide the runner.
    """
    if not roctx_scopes_enabled(environ):
        return False

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: PLC0415

    _, backend = select_backend(backend_name)
    annotator = IterationAnnotator(backend, enabled=True)
    original_execute_model = GPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):  # type: ignore[no-untyped-def]
        record = _forward_record_from_scheduler_output(scheduler_output)
        with annotator.forward(record=record):
            return original_execute_model(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = execute_model  # type: ignore[method-assign]
    return True


def _forward_record_from_scheduler_output(scheduler_output) -> dict:
    """Best-effort batch-shape record from vLLM's ``SchedulerOutput``.

    The authoritative shape fields live on vLLM internals that vary across
    versions; this extracts the one stable, cheap signal (total scheduled tokens)
    and never fails the forward if a field is absent — a missing field yields an
    empty record, not a crash in the serving hot path.
    """
    record: dict = {}
    total = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if total is not None:
        record["total_num_scheduled_tokens"] = int(total)
    return record
