"""The device and runtime checks every runner shares.

Each backend declares its device family and capability requirements in
``BackendSupport``. The worker checks these against the physical device before
loading the runner (``unsupported_device``). Runners retain only checks that
depend on the operation's shape, such as supported head counts.

Loaded only inside the worker subprocess; torch is imported lazily.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.db.registry import BackendSupport
from profiling.runners.exceptions import ProfilerNotImplemented


def require_cuda_toolkit(torch: Any, minimum: tuple[int, int], label: str) -> None:
    """Require torch's CUDA toolkit to be at least ``minimum``: the floor of a
    binding the runner JIT-builds against it."""
    version = _major_minor(getattr(torch.version, "cuda", None))
    if version is None or version < minimum:
        raise ProfilerNotImplemented(
            f"{label} requires CUDA >= {minimum[0]}.{minimum[1]}, "
            f"got {getattr(torch.version, 'cuda', None)}"
        )


def require_cutedsl(label: str) -> None:
    """Require vLLM's CuTe DSL path (``has_cutedsl``), which a CuTe DSL backend
    launches through."""
    try:
        from vllm.utils.import_utils import has_cutedsl
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{label} requires pinned vLLM") from exc
    if not has_cutedsl():
        raise ProfilerNotImplemented(f"{label} requires vLLM's CuTe DSL path")


def _major_minor(version: object) -> tuple[int, int] | None:
    major, _, rest = str(version).partition(".")
    minor = rest.partition(".")[0]
    if version is None or not (major.isdigit() and minor.isdigit()):
        return None
    return int(major), int(minor)


def unsupported_device(supports: BackendSupport, label: str) -> str | None:
    """Return an error when the physical device cannot run the declared backend.

    Neuron identity comes from neuron-ls; CUDA capability comes from Torch.
    A requested profile label never substitutes for the physical device.
    """
    if supports.device_family == "neuron":
        from profiling.exec.neuron import current_neuron_name

        try:
            name = current_neuron_name()
        except (OSError, RuntimeError, ValueError) as exc:
            return f"Neuron is required for {label}: {exc}"
        compute = next(iter(supports.compute or (DType.BF16,)))
        kv = next(iter(supports.kv)) if supports.kv else None
        if supports.allows(compute, kv, gpu=name):
            return None
        return f"{label} does not support {name}"
    try:
        import torch
    except ImportError:
        return f"{label} needs torch to read the device"
    if not torch.cuda.is_available():
        return f"CUDA is required for {label}"
    rule = supports.device_rule()
    if rule is None:
        return None
    major, minor = (int(part) for part in torch.cuda.get_device_capability(0))
    if supports.allows_compute_capability((major, minor)):
        return None
    name = torch.cuda.get_device_name(0)
    return f"{label} needs {rule}, got {name} with SM{major}{minor}"
