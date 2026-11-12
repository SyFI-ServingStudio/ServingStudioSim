"""An image's build-time check: its interpreter loads every runner its envs serve.

    python -m profiling.container.selfcheck ENV [ENV ...]

Runs without a GPU, on the source copy baked into the image; a worker loads the
same runners, from the checkout's overlay, at run time.
"""

from __future__ import annotations

import sys

from profiling.db.registry import iter_kernel_profiler_specs


def main(envs: list[str]) -> int:
    if not envs:
        print("usage: python -m profiling.container.selfcheck ENV [ENV ...]", file=sys.stderr)
        return 2
    loaded = 0
    for spec in iter_kernel_profiler_specs():
        if (spec.subprocess_env or "default_env") in envs:
            spec.runner_ref.load()
            loaded += 1
    if not loaded:
        print(f"no backend runs in {' '.join(envs)}", file=sys.stderr)
        return 1
    print(f"{loaded} runners load in {' '.join(envs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
