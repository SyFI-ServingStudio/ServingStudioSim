"""Pre-torch RTLD_GLOBAL load of the rocprofiler-sdk roctx library.

This is the load-order fix for Gap 2 on the pinned ROCm-7 / rocprofv3-1.3.2
(gfx942) stack. There, rocprofv3's ``--marker-trace`` records roctx ranges ONLY
from the rocprofiler-sdk roctx library (``librocprofiler-sdk-roctx.so.1``), which
rocprofv3 registers as its MARKER service by intercepting the ``dlopen`` of that
soname *inside the already-wrapped process*. vLLM's ``torch``/``aiter`` pull in
the legacy ``libroctx64`` on import, and if that resident legacy library wins
``roctx*`` symbol resolution first, it preempts the SDK's MARKER registration and
``--marker-trace`` records zero ``vllm_iteration(N)`` ranges — even though the
shim pushed them (confirmed on GPU).

The fix is to ``dlopen`` the SDK roctx library *first*, from inside the
rocprofv3-wrapped process but before any ``import torch``:

- An ``LD_PRELOAD`` of the roctx lib does NOT work: it loads the library before
  rocprofv3's in-process interception is active, so the MARKER never registers.
- A runtime ``ctypes.CDLL(..., mode=RTLD_GLOBAL)`` that runs inside the wrapped
  process is intercepted by rocprofv3 (MARKER registers) AND, being
  ``RTLD_GLOBAL`` and first, wins global ``roctx*`` symbol resolution over
  torch's later ``libroctx64``.
- The earliest in-process hook that runs before any user/torch import is Python's
  ``sitecustomize`` (imported automatically at interpreter startup). The capture
  driver prepends a generated ``sitecustomize.py`` to the launched server's
  ``PYTHONPATH``; that file imports this module and calls :func:`preload_sdk_roctx`.

The loaded handle is kept module-global so :class:`RoctracerBackend
<alignment.profiler.roctx_shim.RoctracerBackend>` can PREFER it (its ``roctx*``
calls then resolve to the SDK lib, not legacy ``libroctx64``), and the resolved
soname is also stamped into :data:`ROCTX_SONAME_RESOLVED_ENV` so the shim and an
operator can confirm which library won.

This module imports only ``ctypes``/``os``/``logging`` (and the gate constant from
:mod:`alignment.profiler.roctx_shim`, which itself defers its torch import), so
importing it never pulls in torch — preserving the load-order guarantee.
"""

from __future__ import annotations

import ctypes
import logging
import os

from alignment.profiler.roctx_shim import ROCTX_SCOPES_ENV

logger = logging.getLogger(__name__)

#: SDK roctx sonames to try, most-specific first. Only the rocprofiler-sdk roctx
#: library is preloaded here; legacy ``libroctx64`` is deliberately NOT preloaded
#: (preloading it would recreate the very preemption this fix avoids).
SDK_SONAMES: tuple[str, ...] = (
    "librocprofiler-sdk-roctx.so.1",
    "librocprofiler-sdk-roctx.so",
)

#: Env var the resolved soname is stamped into, so the shim backend and an
#: operator can confirm the SDK library was loaded first (and which one).
ROCTX_SONAME_RESOLVED_ENV = "VIBESIM_ROCTX_SONAME_RESOLVED"

#: The ``RTLD_GLOBAL`` handle of the preloaded SDK roctx library, or ``None`` when
#: the gate is off or no SDK library is present. The shim backend prefers this.
_PRELOADED_HANDLE: ctypes.CDLL | None = None
_PRELOADED_SONAME: str | None = None


def preloaded_handle() -> ctypes.CDLL | None:
    """The ``RTLD_GLOBAL`` SDK roctx handle loaded by :func:`preload_sdk_roctx`."""
    return _PRELOADED_HANDLE


def preloaded_soname() -> str | None:
    """The soname of the preloaded SDK roctx library, or ``None``."""
    return _PRELOADED_SONAME


def preload_sdk_roctx(
    environ: dict[str, str] | None = None,
    *,
    sonames: tuple[str, ...] = SDK_SONAMES,
) -> str | None:
    """Load the SDK roctx library ``RTLD_GLOBAL`` before torch imports legacy roctx.

    Gated on :data:`ROCTX_SCOPES_ENV`: returns ``None`` and loads nothing unless the
    capture turned the roctx annotation on. When on, tries each soname in
    ``sonames`` with ``ctypes.CDLL(..., mode=RTLD_GLOBAL)`` and stops at the first
    that loads, recording the handle (for :func:`preloaded_handle`) and stamping the
    resolved soname into ``environ`` (default :data:`os.environ`) under
    :data:`ROCTX_SONAME_RESOLVED_ENV`. A missing library is tolerated — it logs and
    returns ``None`` rather than raising, because the loader runs at interpreter
    startup where a crash would take down the whole server.

    Returns the resolved soname, or ``None`` when gated off or no SDK library loads.
    """
    global _PRELOADED_HANDLE, _PRELOADED_SONAME

    env = os.environ if environ is None else environ
    if env.get(ROCTX_SCOPES_ENV, "0") != "1":
        return None

    last_error: OSError | None = None
    for candidate in sonames:
        try:
            handle = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError as error:  # not present / not loadable on this stack
            last_error = error
            continue
        _PRELOADED_HANDLE = handle
        _PRELOADED_SONAME = candidate
        env[ROCTX_SONAME_RESOLVED_ENV] = candidate
        logger.info(
            "preloaded SDK roctx %s RTLD_GLOBAL before torch; "
            "rocprofv3 --marker-trace MARKER service can now register",
            candidate,
        )
        return candidate

    logger.warning(
        "no rocprofiler-sdk roctx library could be preloaded (tried %s): %s; "
        "the roctx shim will fall back to a runtime dlopen, which on this stack "
        "risks legacy libroctx64 preempting rocprofv3's MARKER service",
        list(sonames),
        last_error,
    )
    return None
