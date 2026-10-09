"""The TP=1 production Torch embedding call compiled for a Neuron LNC2 unit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

from profiling.db.args import DType
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def _embedding(indices, weight):
    import torch.nn.functional as functional

    return functional.embedding(indices, weight)


def profile_embedding(
    num_tokens: int, hidden: int, vocab: int, dtype: DType | str
) -> ComputeMetrics:
    if DType.from_value(dtype) != DType.BF16:
        raise ProfilerNotImplemented("Neuron embedding currently supports BF16 only")
    if min(num_tokens, hidden, vocab) < 1:
        raise ValueError("Neuron embedding dimensions must be positive")
    import ml_dtypes
    import torch
    import torch_neuronx

    from profiling.profilers.neuron_timer import measure_neff

    temporary_root = os.environ.get("TMPDIR")
    if not temporary_root or not Path(temporary_root).is_absolute():
        raise ProfilerNotImplemented("Neuron profiling requires an absolute workspace TMPDIR")
    stamp = {
        "source": inspect.getsource(_embedding),
        "shape": [num_tokens, hidden, vocab],
        "dtype": "bf16",
        "target": "trn2",
        "lnc": 2,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch-neuronx", "torch-xla", "neuronx-cc", "islpy")
        },
    }
    digest = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    work = Path(temporary_root) / "neuron-kernels" / digest
    work.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randint(vocab, (num_tokens,), generator=generator, dtype=torch.int64)
    weight = torch.randn(vocab, hidden, generator=generator, dtype=torch.bfloat16)
    reference = weight[indices].view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    neff = work / "graph.neff"
    if not neff.exists():
        torch_neuronx.trace(
            _embedding,
            (indices, weight),
            compiler_workdir=str(work),
            compiler_args=["--target=trn2", "--logical-nc-config=2", "--auto-cast=none"],
        )

    def check(output):
        import numpy as np

        np.testing.assert_array_equal(output.reshape(reference.shape), reference)

    time_ms = measure_neff(
        neff,
        {
            "input0": indices.numpy(),
            "input1": weight.view(torch.int16).numpy().view(ml_dtypes.bfloat16),
        },
        check,
        output_dtype=ml_dtypes.bfloat16,
    )
    bytes_accessed = 4 * num_tokens * hidden + 8 * num_tokens
    return ComputeMetrics(time_ms, 0.0, bytes_accessed / (time_ms / 1000) / 1e9)
