"""Focused CPU tests for the MI300X gdn_causal_conv torch_rocm backends.

No GPU, no DB. The GPU correctness check runs inside the runner on the ROCm host
(it reuses the NVIDIA runner's reference comparison); its evidence comes from the
Slurm step. Each test names the defect it guards against.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from profiling.db.registry import find_kernel_profiler_spec


@pytest.mark.parametrize("kind", ["gdn_causal_conv_decode", "gdn_causal_conv_prefill"])
def test_torch_rocm_backend_registered_for_mi300x(kind: str) -> None:
    # The MI300X path is a separate backend of the same kind, gated to MI300X and
    # routed to the vllm_rocm_env image. The NVIDIA rows must be left untouched.
    rocm = find_kernel_profiler_spec(kind, "torch_rocm")
    assert rocm.supports.arch_targets == frozenset({"CDNA3"})
    assert rocm.subprocess_env == "vllm_rocm_env"
    assert rocm.runner_ref.function_name == f"profile_{kind}_torch_rocm"

    nvidia = find_kernel_profiler_spec(kind, "vllm_triton")
    assert nvidia.supports.arch_targets is None


@pytest.mark.parametrize("kind", ["gdn_causal_conv_decode", "gdn_causal_conv_prefill"])
def test_rocm_profile_signature_matches_nvidia(kind: str) -> None:
    # Defect: the ROCm runner drifting from the kind's arg order.
    rocm_mod = f"profiling.runners.attention.{kind}_torch_rocm"
    nv_mod = f"profiling.runners.attention.{kind}_vllm_triton"
    import importlib

    rocm = importlib.import_module(rocm_mod)
    nv = importlib.import_module(nv_mod)
    rocm_fn = getattr(rocm, f"profile_{kind}_torch_rocm")
    nv_fn = getattr(nv, f"profile_{kind}_vllm_triton")
    assert list(inspect.signature(rocm_fn).parameters) == list(
        inspect.signature(nv_fn).parameters
    )


@pytest.mark.parametrize("kind", ["gdn_causal_conv_decode", "gdn_causal_conv_prefill"])
def test_rocprof_run_builder_registered(kind: str) -> None:
    import profiling.profilers.rocprof_run as rr

    assert (kind, "torch_rocm") in rr._BUILDERS


@pytest.mark.parametrize("kind", ["gdn_causal_conv_decode", "gdn_causal_conv_prefill"])
def test_torch_rocm_runner_is_import_light(kind: str) -> None:
    # Importing the ROCm runner must not eager-import torch or vllm.
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import profiling.runners.attention.{kind}_torch_rocm as r; "
            "print('torch' in sys.modules, 'vllm' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False"
