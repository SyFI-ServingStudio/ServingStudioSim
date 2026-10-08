"""Parse a vLLM torch-profiler Kineto trace into the normalized alignment files.

The sibling of ``alignment/rocpd/parse.py`` for the torch-profiler capture path.
It reads a per-rank PyTorch Kineto chrome trace (``*.pt.trace.json[.gz]``)
offline and writes the exact same artifacts the nsys and rocpd producers do --
``parsed.json`` + ``parsed.kernels.parquet`` and the folded
``kernel_sequences.json`` -- so the existing downstream (``label`` → ``analyze``
kernel-align → the Rust Check-1 reader) runs on a torch trace unchanged.

Only the capture boundary differs: :func:`alignment.torchprof.kineto.read_kineto_trace`
yields the GPU kernel dispatches and iteration ranges, then
:func:`alignment.rocpd.evidence.build_ranges_from_dispatches` -- the backend-neutral
sentinel/roctx segmentation shared with the rocpd producer -- builds the
``RangeStats``. From there this module reuses the nsys assembly verbatim
(``build_kernel_name_index``, ``summarize_devices``, ``build_iteration_details``,
``align_ranges_into_steps``, ``build_device_kernel_sequences``) and the shared
writers, so a schema change on the nsys side is inherited here for free.

Scope matches the rocpd producer: the eager (no-graph), uniform-routing,
kernel-only path. ``jit_*`` fields are 0 (a Kineto trace carries host module-load
events but this smallest Check-1 input does not need the warm-up signal), and
per-iteration vLLM metrics are absent. The one source field is ``"torch"`` so the
merge and any provenance reader can tell the capture backend apart from rocpd.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..nsys.parse import (
    _uses_shared_iteration_index,
    align_ranges_into_steps,
    build_iteration_details,
    build_kernel_name_index,
    summarize_devices,
    write_kernel_sequences,
)
from ..nsys.parsed_io import kernel_rows_path, write_parsed
from ..nsys.sequence import build_device_kernel_sequences
from ..rocpd.evidence import build_ranges_from_dispatches
from .kineto import read_kineto_trace

#: Kept identical to the nsys/rocpd producers so the three producers' stdout is
#: comparable and ``write_parsed`` lifts the on-disk file to the same version.
_DOCUMENT_SCHEMA_VERSION = 5

#: Provenance tag in the parsed document's ``source`` field (rocpd uses "rocpd").
SOURCE = "torch"


def parse_trace(
    trace_path: Path,
    iteration_start: int | None = None,
    iteration_end: int | None = None,
    *,
    default_stage: str = "all",
    top_n: int = 12,
) -> dict:
    """Parse a torch-profiler Kineto trace into the normalized parsed-document dict.

    ``iteration_start`` / ``iteration_end`` bound the kept iteration indices (both
    open by default). Uniform routing: every observed device is data-parallel rank
    0, so TP ranks share one authoritative iteration index -- the same
    ``iteration_index`` step-alignment rule the nsys shared-index path uses.
    """
    if (
        iteration_start is not None
        and iteration_end is not None
        and iteration_start > iteration_end
    ):
        raise ValueError("iteration_start must not exceed iteration_end")

    dispatches, regions = read_kineto_trace(Path(trace_path))
    ranges = build_ranges_from_dispatches(
        dispatches, regions, default_stage=default_stage, capture_kind="torch"
    )

    def in_window(iteration: int) -> bool:
        return (iteration_start is None or iteration >= iteration_start) and (
            iteration_end is None or iteration <= iteration_end
        )

    ranges = [item for item in ranges if in_window(item.iteration)]
    if not ranges:
        raise ValueError("no iteration ranges remain after the requested window")

    dp_rank_by_device = {
        item.worker.device_id: 0 for item in ranges if item.worker.device_id is not None
    }

    stages = sorted({item.stage for item in ranges})
    kernel_name_ids, kernel_names = build_kernel_name_index(ranges)
    by_stage = {
        stage: summarize_devices([item for item in ranges if item.stage == stage], top_n)
        for stage in stages
    }
    phases = sorted({item.phase for item in ranges})
    by_phase = {}
    for phase in phases:
        phase_ranges = [item for item in ranges if item.phase == phase]
        phase_stages = sorted({item.stage for item in phase_ranges})
        by_phase[phase] = {
            "by_stage": {
                stage: summarize_devices(
                    [item for item in phase_ranges if item.stage == stage], top_n
                )
                for stage in phase_stages
            },
            "all": summarize_devices(phase_ranges, top_n),
        }

    iteration_details = build_iteration_details(
        ranges, {}, kernel_name_ids, {}, dp_rank_by_device
    )
    aligned_steps, unpaired_by_device = align_ranges_into_steps(ranges, dp_rank_by_device)
    iterations = [step.iteration for step in aligned_steps]
    scanned_kernel_rows = sum(item.kernel_count for item in ranges)
    kernel_sequences, device_ids = build_device_kernel_sequences(iteration_details, kernel_names)

    return {
        "schema_version": _DOCUMENT_SCHEMA_VERSION,
        "step_alignment": {
            "rule": (
                "iteration_index"
                if _uses_shared_iteration_index(dp_rank_by_device)
                else "kernel_time_overlap"
            ),
            "reference_device_id": (
                aligned_steps[0].reference_device_id if aligned_steps else None
            ),
            "steps": len(aligned_steps),
            "unpaired_steps_by_device": {
                str(device): count for device, count in sorted(unpaired_by_device.items())
            },
        },
        # The nsys/rocpd producers name this `sqlite`; the torch source is a JSON
        # trace, but the key stays `sqlite` so the Rust manifest and every reader
        # that keys off it are untouched. The `source` field carries the real kind.
        "sqlite": str(trace_path),
        "source": SOURCE,
        "iteration_start": iteration_start,
        "iteration_end": iteration_end,
        "range_mode": "phases",
        "iterations": iterations,
        "scanned_kernel_rows": scanned_kernel_rows,
        # A Kineto trace carries host module-load events, but this smallest
        # Check-1 input does not recover the JIT warm-up signal; 0 is honest here.
        "jit_module_loads": 0,
        "stages": stages,
        "phases": phases,
        "kernel_names": kernel_names,
        "iteration_details": iteration_details,
        "device_ids": device_ids,
        "dp_rank_by_device": {
            str(device): rank for device, rank in sorted(dp_rank_by_device.items())
        },
        "tp_size": 1,
        "kernel_sequences": kernel_sequences,
        "by_phase": by_phase,
        "by_stage": by_stage,
        "all": summarize_devices(ranges, top_n),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a vLLM torch-profiler Kineto trace into the normalized alignment files"
    )
    parser.add_argument(
        "--trace",
        dest="trace",
        type=Path,
        required=True,
        help="captured per-rank torch-profiler Kineto chrome trace (*.pt.trace.json[.gz])",
    )
    parser.add_argument(
        "--iteration-start",
        type=int,
        default=None,
        help="first iteration index to keep (default: start of capture)",
    )
    parser.add_argument(
        "--iteration-end",
        type=int,
        default=None,
        help="last iteration index to keep (default: end of capture)",
    )
    parser.add_argument(
        "--default-stage",
        default="all",
        help="stage for ranges lacking a metrics stage (e.g. 'prefill' for a prefill window)",
    )
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=None, help="Write parsed.json here (else stdout)"
    )
    parser.add_argument(
        "--sequences-output",
        type=Path,
        default=None,
        help="Also write the folded, label-ready kernel-sequence catalog here",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = parse_trace(
        args.trace,
        args.iteration_start,
        args.iteration_end,
        default_stage=args.default_stage,
        top_n=args.top_n,
    )
    if args.output:
        write_parsed(Path(args.output), result)
        print(f"wrote {args.output} and {kernel_rows_path(Path(args.output)).name}")
    else:
        print(json.dumps(result, separators=(",", ":")))
    if args.sequences_output:
        write_kernel_sequences(args.sequences_output, result, args.output)
        print(f"wrote {args.sequences_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
