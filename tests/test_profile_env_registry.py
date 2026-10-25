"""Every registered profiling backend names a profile env that exists.

A backend whose ``subprocess_env`` is not registered fails only when it is
profiled (``resolve_profile_env`` raises), long after the registry loaded:
six kinds kept naming ``vllm_upstream_fork_env`` after it was folded into
``vllm_env``.
"""

from __future__ import annotations

import pytest

from profiling.container.images import IMAGES, VENDOR_IMAGES
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.exec.env import ENV_REGISTRY, ProfileEnv, resolve_profile_env

_SPECS = list(iter_kernel_profiler_specs())


def _env(spec) -> str:
    return spec.subprocess_env or "default_env"


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: f"{s.kernel_kind}:{s.backend}")
def test_backend_env_is_registered_and_served_by_an_image(spec) -> None:
    env = _env(spec)
    if spec.supports.device_family == "neuron":
        # A Neuron env is only a pinned vendor image; it never runs on a host interpreter.
        assert any(env in image.envs for image in VENDOR_IMAGES.values()), f"no vendor image serves {env!r}"
        return
    assert env in ENV_REGISTRY, f"{spec.kernel_kind}:{spec.backend} names unknown env {env!r}"
    assert any(env in image.envs for image in IMAGES.values()), f"no image serves {env!r}"


@pytest.mark.parametrize(
    "env",
    sorted({_env(spec) for spec in _SPECS if spec.supports.device_family != "neuron"}),
)
def test_backend_env_resolves_on_the_host_runtime(env: str, monkeypatch) -> None:
    monkeypatch.setenv("VIBESIM_PROFILE_RUNTIME", "host")
    assert isinstance(resolve_profile_env(env), ProfileEnv)
