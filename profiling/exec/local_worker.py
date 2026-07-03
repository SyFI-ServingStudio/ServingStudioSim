"""Subprocess worker entrypoint for local L1 profiling.

Agent note:
- This module is executed as ``python -m profiling.exec.local_worker`` inside
  the selected ``ProfileEnv``.
- Keep GPU selection, temp-file ownership, and subprocess command construction
  in ``profiling.exec.local``. This worker only consumes a JSON payload,
  lazy-loads the registered runner, executes specs, and writes JSON results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from profiling.db.batch import args_to_spec, coerce_args
from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec.payload import metrics_to_payload, resolve_chunk_backend
from profiling.runners.metrics import RunnerResult


def _worker_main(input_path: Path, output_path: Path) -> None:
    worker_request = json.loads(input_path.read_text(encoding="utf-8"))
    kernel_kind: KernelKind = worker_request["kernel_kind"]
    chunk_specs = worker_request["specs"]
    backend = resolve_chunk_backend(kernel_kind, chunk_specs)
    profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)

    # The chunk is homogeneous (one backend/gpu_count — enforced upstream), so the
    # worker dispatch is a single uniform list call. Coercion is hoisted here so
    # every runner receives already-coerced kwargs (the `batched` adapter and the
    # native comm runners stay schema-free). List-native comm runners spawn their
    # rank group once for the whole list; compute runners are wrapped by `batched`.
    runner = profiler_spec.load_list_runner()
    kwargs_list = [
        args_to_spec(coerce_args(profiler_spec.args_schema, _strip_backend(chunk_spec)))
        for chunk_spec in chunk_specs
    ]
    results = runner(kwargs_list)
    gpu_name = _current_gpu_name()
    worker_results = [_to_payload(result, gpu_name) for result in results]

    output_path.write_text(json.dumps({"results": worker_results}), encoding="utf-8")


def _strip_backend(chunk_spec: dict) -> dict:
    """Drop the ``backend`` selector (already resolved to the chunk spec) so the
    remaining keys are exactly the runner's schema args."""
    return {key: value for key, value in chunk_spec.items() if key != "backend"}


def _to_payload(result: RunnerResult, gpu_name: str | None) -> dict:
    """Render one ``RunnerResult`` into the worker's on-disk JSON shape that
    ``chunk_result_from_payload`` consumes (unchanged from the single-spec loop)."""
    if result.error is not None or result.metrics is None:
        return {"ok": False, "error": result.error}
    return {
        "ok": True,
        "metrics": metrics_to_payload(result.metrics),
        "gpu_name": gpu_name,
    }


def _current_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-input", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    args = parser.parse_args(argv)
    _worker_main(args.worker_input, args.worker_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
