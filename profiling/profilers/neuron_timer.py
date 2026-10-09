"""Native Neuron device timing for one NKI invocation on an LNC2 unit.

The verified Neuron 2.32 SDK uses NKI 0.6.0+31049202112.g85070674 and
neuronx-cc 2.27.5334.0+f702b353 (bundled nrtpy and nkilib 3b542be2).
``SpikeModel.benchmark(mode="device")`` averages physical-core events, which
double-counts samples for LNC2. Capture its native SystemTraceSession instead,
group physical-core intervals by exec_id, and measure their union. timestamp_ns
is the runtime's synchronized device timestamp; raw nc_timestamp_ns belongs to
independent core clocks and cannot be unioned across cores.

Source: aws-neuron/nkipy spike/src/{sys_trace.cpp,spike/spike_model.py},
commit 0bc6ed1631a91e0a07f817066cfa9d89a7953409, and the AWS Neuron Profiler
2.0 guide's "Adjusting Hardware Timestamps" section. Private SDK adapters are
confined here and reject unfamiliar versions or trace schemas.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import statistics
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_VERIFIED_VERSIONS = {
    "nki": "0.6.0+31049202112.g85070674",
    "neuronx-cc": "2.27.5334.0+f702b353",
}


def _execution_unions_ms(trace: Mapping[str, Any], expected_executions: int) -> list[float]:
    """Return one active device duration per logical invocation, in trace order."""
    if expected_executions < 1:
        raise ValueError("expected_executions must be positive")
    events = trace.get("events")
    if not isinstance(events, list):
        raise KernelLaunchFailed("Neuron trace has no events list")
    starts: dict[tuple[int, int], tuple[int, int, int]] = {}
    intervals: dict[tuple[int, int], list[tuple[int, int]]] = {}
    physical_cores: dict[tuple[int, int], set[int]] = {}
    for event in events:
        if not isinstance(event, dict):
            raise KernelLaunchFailed("Neuron trace event is not an object")
        if event.get("event_type") != "nc_exec_running":
            continue
        try:
            key = (event["nc_idx"], event["tracking_id"])
            timestamp = event["timestamp_ns"]
            if not all(isinstance(value, int) for value in (*key, timestamp)):
                raise TypeError("noninteger trace identity or timestamp")
            phase = event["phase"]
            if phase == "start":
                exec_id = event["data"]["exec_id"]
                core = event["data"]["device_core_idx"]
                if not isinstance(exec_id, int) or not isinstance(core, int) or key in starts:
                    raise ValueError("invalid or duplicate start event")
                starts[key] = (timestamp, exec_id, core)
            elif phase == "stop":
                start, exec_id, core = starts.pop(key)
                if timestamp <= start:
                    raise ValueError("nonpositive device interval")
                invocation = (key[0], exec_id)
                intervals.setdefault(invocation, []).append((start, timestamp))
                physical_cores.setdefault(invocation, set()).add(core)
            else:
                raise ValueError(f"unexpected event phase {phase!r}")
        except (KeyError, TypeError, ValueError) as exc:
            raise KernelLaunchFailed(
                f"Unsupported or incomplete Neuron device trace: {exc}"
            ) from exc
    if starts or len(intervals) != expected_executions:
        raise KernelLaunchFailed(
            f"Neuron trace captured {len(intervals)} complete logical invocations, "
            f"expected {expected_executions}; {len(starts)} unmatched starts"
        )
    core_sets = {frozenset(cores) for cores in physical_cores.values()}
    if len(core_sets) != 1 or len(next(iter(core_sets))) != 2:
        raise KernelLaunchFailed(
            "Neuron LNC2 trace must contain both physical cores for every invocation"
        )
    durations = []
    for spans in intervals.values():
        spans.sort()
        lower, upper = spans[0]
        active_ns = 0
        for start, stop in spans[1:]:
            if start <= upper:
                upper = max(upper, stop)
            else:
                active_ns += upper - lower
                lower, upper = start, stop
        durations.append((active_ns + upper - lower) / 1e6)
    return durations


def _artifact_directory(stamp: Mapping[str, Any]) -> Path:
    """Place a versioned compile fingerprint under the workspace temporary root."""
    temporary_root = os.environ.get("TMPDIR")
    if not temporary_root or not Path(temporary_root).is_absolute():
        raise ProfilerNotImplemented("Neuron profiling requires an absolute workspace TMPDIR")
    key = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    path = Path(temporary_root) / "neuron-kernels" / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _verified_runtime() -> dict[str, str]:
    """nrtpy is bundled in neuronx-cc; require the verified private adapter ABI."""
    versions = {name: importlib.metadata.version(name) for name in ("nki", "neuronx-cc")}
    if versions != _VERIFIED_VERSIONS:
        raise ProfilerNotImplemented(
            f"Neuron NKI adapter was verified with {_VERIFIED_VERSIONS}, found {versions}; "
            "revalidate the SDK API and native trace schema"
        )
    return versions


def _compile_directory(kernel: Any, inputs: Mapping[str, Any]) -> Path:
    """Reuse compiled NEFFs by source, SDK version, dtype, shape and options."""
    versions = _verified_runtime()
    shapes = {
        name: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
        if hasattr(value, "shape") and hasattr(value, "dtype")
        else repr(value)
        for name, value in sorted(inputs.items())
    }
    stamp = {
        "versions": versions,
        "lnc": 2,
        "target": "trn2",
        "callable": f"{kernel.func.__module__}.{kernel.func.__qualname__}",
        "source": inspect.getsource(kernel.func),
        "inputs": shapes,
    }
    return _artifact_directory(stamp)


def measure_nki(
    kernel: Any,
    inputs: Mapping[str, Any],
    check_output: Callable[[Any], None],
    *,
    output_dtype: Any,
    warmup: int = 5,
    iterations: int = 20,
) -> float:
    """Compile/cache a production callable, check its output, then time device work.

    Inputs and outputs live on device throughout capture. Compilation, copying,
    numerical validation and warmup occur before trace. The caller supplies the
    independent correctness check. Visibility and LNC environment are owned by
    the execution worker; this adapter never chooses devices or mutates env.
    """
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    try:
        from nki.framework.compiled import StandaloneKernel
    except ImportError as exc:
        raise ProfilerNotImplemented("Neuron NKI 0.6 and its nrtpy runtime are required") from exc

    artifacts = _compile_directory(kernel, inputs)
    neff = artifacts / "kernel.neff"
    if not neff.exists():
        # NKI 0.6 public jit dispatches to this same standalone adapter. Pass
        # artifacts explicitly so compilation survives worker restarts without
        # a runner-local NKI_ARTIFACTS_DIR environment mutation.
        standalone = kernel[2]._to_subclass(StandaloneKernel, artifacts_dir=str(artifacts))
        standalone(**inputs)
    return measure_neff(
        neff, inputs, check_output, output_dtype=output_dtype,
        warmup=warmup, iterations=iterations,
    )


def measure_neff(
    neff: Path,
    inputs: Mapping[str, Any],
    check_output: Callable[[Any], None],
    *,
    output_dtype: Any,
    output_names: tuple[str, ...] | None = None,
    expected_aliases: Mapping[str, str] | None = None,
    warmup: int = 5,
    iterations: int = 20,
) -> float:
    """Validate and time a compiled production NEFF on the allocated LNC2 unit.

    output_names declares the callback's exact output ABI for a multi-output
    graph. expected_aliases validates the entire compiled alias map. Stateful
    outputs use nrtpy's public call allocation, which preserves input aliases.
    Single outputs reach the callback as one array; multiple outputs as a list.
    """
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    _verified_runtime()
    try:
        import nrtpy
        from nrtpy._nrtpy import SystemTraceSession
    except ImportError as exc:
        raise ProfilerNotImplemented("Neuron nrtpy native runtime is required") from exc
    model = nrtpy.SpikeModel.load_from_neff(neff)
    ordered_names = _validate_neff_contract(model, output_names, expected_aliases)
    device_inputs = {
        name: nrtpy.SpikeTensor.from_numpy(
            inputs[name.removesuffix(".must_alias_input")], name=name
        )
        for name in model.input_tensors_info
    }
    # Public __call__(inputs) binds aliased outputs to their input buffers.
    # Allocating fresh outputs would break repeated in-place KV cache updates.
    device_outputs = model(device_inputs)
    # nrtpy allocates opaque V2 output buffers because NRT's dtype enum cannot
    # represent all compiler types. Interpret their bytes using the production
    # callable's declared output dtype; astype() would convert opaque records.
    import numpy as np

    output_dtype = np.dtype(output_dtype)
    if any(tensor.dtype.itemsize != output_dtype.itemsize for tensor in device_outputs.values()):
        raise KernelLaunchFailed("Neuron output storage does not match its declared dtype")
    output_arrays = [device_outputs[name].numpy().view(output_dtype) for name in ordered_names]
    check_output(output_arrays[0] if len(output_arrays) == 1 else output_arrays)
    with SystemTraceSession(model.model_ref.core_id) as trace:
        for _ in range(warmup):
            model(device_inputs, device_outputs)
        trace.drain_events()
        for _ in range(iterations):
            model(device_inputs, device_outputs)
        events = json.loads(trace.fetch_events_json())
    (neff.parent / "timing-trace.json").write_text(json.dumps(events))
    durations = _execution_unions_ms(events, iterations)
    result = statistics.median(durations)
    if not math.isfinite(result) or result <= 0:
        raise KernelLaunchFailed(f"Invalid Neuron device latency {result}")
    return float(result)


def _validate_neff_contract(
    model: Any,
    output_names: tuple[str, ...] | None,
    expected_aliases: Mapping[str, str] | None,
) -> tuple[str, ...]:
    """Reject changed output/alias ABIs before an executable can mutate state."""
    actual_names = set(model.output_tensors_info)
    if output_names is not None:
        if len(output_names) != len(set(output_names)) or set(output_names) != actual_names:
            raise KernelLaunchFailed(
                f"Neuron output ABI changed: expected {output_names}, found {sorted(actual_names)}"
            )
    else:
        # Native tensor-info iteration is unordered. Numeric order also handles
        # output10/output2; stateful graphs should declare explicit output roles.
        def numeric_order(name):
            match = re.fullmatch(r"output(\d+)", name)
            return (0, int(match.group(1))) if match else (1, name)

        output_names = tuple(sorted(actual_names, key=numeric_order))
    if expected_aliases is not None and model.alias_info != dict(expected_aliases):
        raise KernelLaunchFailed(
            f"Neuron alias ABI changed: expected {dict(expected_aliases)}, found {model.alias_info}"
        )
    return output_names
