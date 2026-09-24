"""Worker PYTHONPATH composition for envs that own a separate venv."""

from __future__ import annotations

import os
from pathlib import Path

from profiling.exec.env import ENV_REGISTRY, ProfileEnv, compose_pythonpath

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# What the PyO3 bridge exports to the embedded interpreter: repo root, then the
# project venv's site-packages (launcher.exec._build_subprocess_env).
_BRIDGE_PYTHONPATH = os.pathsep.join(
    [str(_PROJECT_ROOT), str(_PROJECT_ROOT / ".venv/lib/python3.12/site-packages")]
)


def test_isolated_env_drops_inherited_site_packages_but_keeps_repo_source():
    # Catches a fork worker importing the project Torch ahead of its own
    # (undefined symbol torch_from_blob) when the Rust bridge spawns it.
    env = ProfileEnv(
        "fork",
        Path("/fork/.venv/bin/python"),
        additional_python_paths=(Path("/fork"),),
        isolated_site_packages=True,
    )
    entries = compose_pythonpath(env, _BRIDGE_PYTHONPATH + os.pathsep + "/extra/src").split(
        os.pathsep
    )

    assert entries[:2] == [str(_PROJECT_ROOT), "/fork"]
    assert "/extra/src" in entries
    assert not any("site-packages" in entry for entry in entries)


def test_non_isolated_env_forwards_inherited_path_verbatim():
    env = ProfileEnv("plain", Path("/usr/bin/python3"))
    assert compose_pythonpath(env, _BRIDGE_PYTHONPATH) == os.pathsep.join(
        [str(_PROJECT_ROOT), _BRIDGE_PYTHONPATH]
    )


def test_only_the_vllm_fork_env_is_isolated():
    isolated = {
        name
        for name, env in ENV_REGISTRY.items()
        if isinstance(env, ProfileEnv) and env.isolated_site_packages
    }
    assert isolated == {"vllm_fork_env"}
