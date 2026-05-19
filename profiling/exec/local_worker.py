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


def _worker_main(input_path: Path, output_path: Path) -> None:
    worker_request = json.loads(input_path.read_text(encoding="utf-8"))
    kernel_kind: KernelKind = worker_request["kernel_kind"]
    chunk_specs = worker_request["specs"]
    backend = resolve_chunk_backend(kernel_kind, chunk_specs)
    profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
    runner = profiler_spec.load_runner()
    worker_results = []
    for chunk_spec in chunk_specs:
        runner_spec = {key: value for key, value in chunk_spec.items() if key != "backend"}
        try:
            kernel_args = coerce_args(profiler_spec.args_schema, runner_spec)
            metrics = runner(**args_to_spec(kernel_args))
            worker_results.append(
                {
                    "ok": True,
                    "metrics": metrics_to_payload(metrics),
                    "gpu_name": _current_gpu_name(),
                }
            )
        except Exception as exc:
            _empty_cuda_cache()
            worker_results.append({"ok": False, "error": str(exc)})

    output_path.write_text(json.dumps({"results": worker_results}), encoding="utf-8")


def _current_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return None


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-input", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    args = parser.parse_args(argv)
    _worker_main(args.worker_input, args.worker_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
