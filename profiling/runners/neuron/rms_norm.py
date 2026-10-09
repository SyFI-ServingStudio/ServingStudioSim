"""Compiled NxDI standalone CustomRMSNorm on one Trainium2 LNC2 unit."""

from __future__ import annotations

import importlib.metadata
import inspect

from profiling.db.args import DType
from profiling.profilers.neuron_timer import _artifact_directory, measure_neff
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.neuron.dense import _metrics


def profile_rms_norm(m: int, hidden: int, dtype: DType | str) -> ComputeMetrics:
    if DType.from_value(dtype) != DType.BF16:
        raise ProfilerNotImplemented("Neuron standalone RMSNorm currently supports BF16 only")
    if m < 1 or hidden < 1:
        raise ValueError("Neuron RMSNorm dimensions must be positive")
    import ml_dtypes
    import numpy as np
    import torch
    import torch_neuronx
    from torch_neuronx.xla_impl.ops import RmsNorm

    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torch-neuronx", "neuronx-cc", "islpy")
    }
    if versions["torch-neuronx"] != "2.6.0.2.10.16998+e9bf8a50" or versions["islpy"] != "2024.2":
        raise ProfilerNotImplemented(
            f"CustomRMSNorm trace was verified with torch-neuronx2.6.0.2.10.16998+e9bf8a50 "
            f"and islpy2024.2; revalidate compiler for {versions}"
        )

    def final_norm(hidden_states, weight):
        # Exact NxDI CustomRMSNorm.forward expression (public RmsNorm op):
        # models' BF16 input is promoted to FP32 and the final result is BF16.
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        return RmsNorm.apply(
            hidden_states, weight, 1e-5, len(hidden_states.shape) - 1
        ).to(original_dtype)

    artifacts = _artifact_directory({
        "callable": "torch_neuronx.xla_impl.ops.RmsNorm.apply",
        "source": inspect.getsource(final_norm),
        "versions": versions,
        "shape": [m, hidden],
        "dtype": "bf16",
        "eps": 1e-5,
        "lnc": 2,
        "target": "trn2",
    })
    rng = np.random.default_rng(42)
    bf16 = ml_dtypes.bfloat16
    a = rng.normal(size=(m, hidden)).astype(bf16)
    weight = rng.normal(size=(hidden,)).astype(bf16)
    neff = artifacts / "graph.neff"
    if not neff.exists():
        tensors = tuple(
            torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
            for value in (a, weight)
        )
        compiled = torch_neuronx.trace(
            final_norm, tensors, compiler_workdir=str(artifacts),
            compiler_args=["--target=trn2", "--logical-nc-config=2"],
        )
        del compiled
    # Independent NumPy oracle outside timing, including both BF16 operands.
    fp32 = a.astype(np.float32)
    reference = (
        fp32 / np.sqrt(np.mean(fp32**2, axis=-1, keepdims=True) + 1e-5)
        * weight.astype(np.float32)
    ).astype(bf16).astype(np.float32)

    def check(output):
        np.testing.assert_allclose(
            output.reshape(reference.shape).astype(np.float32), reference, rtol=0.02, atol=0.02
        )

    # Verified Torch compiler NEFF signature: input0=x, input1=weight.
    # The nrtpy adapter validates this signature by resolving its named inputs.
    time_ms = measure_neff(
        neff, {"input0": a, "input1": weight}, check, output_dtype=bf16,
    )
    return _metrics(time_ms, 4 * m * hidden + 2 * m, 2 * (2 * m * hidden + hidden))
