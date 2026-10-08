"""Generate a deterministic fixed-shape ServingStudioSim request trace.

Use this for load-generator capacity tests where shape variance would obscure
the offered-load boundary. Arrival times are milliseconds, matching the L7 and
req-frontend `independent` frontend contract. `--prefix-len` adds the optional
`prefix_len` column: that many context tokens already resident (a pinned prefix
hit) ahead of the `input_len` fresh tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def write_fixed_shape_trace(
    output_path: Path,
    *,
    requests: int,
    input_len: int,
    output_len: int,
    interarrival_ms: float,
    prefix_len: int = 0,
) -> None:
    if requests <= 0:
        raise ValueError("requests must be positive")
    if input_len <= 0 or output_len <= 0:
        raise ValueError("input_len and output_len must be positive")
    if not interarrival_ms >= 0.0:
        raise ValueError("interarrival_ms must be nonnegative")
    if prefix_len < 0:
        raise ValueError("prefix_len must be nonnegative")

    prefix_column = f",{prefix_len}" if prefix_len else ""
    lines = ["id,input_len,output_len,arrival_time" + (",prefix_len" if prefix_len else "")]
    lines.extend(
        f"{request_index},{input_len},{output_len},{request_index * interarrival_ms:.6f}"
        f"{prefix_column}"
        for request_index in range(requests)
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--interarrival-ms", type=float, default=0.0)
    parser.add_argument("--prefix-len", type=int, default=0)
    arguments = parser.parse_args()
    write_fixed_shape_trace(
        arguments.output,
        requests=arguments.requests,
        input_len=arguments.input_len,
        output_len=arguments.output_len,
        interarrival_ms=arguments.interarrival_ms,
        prefix_len=arguments.prefix_len,
    )


if __name__ == "__main__":
    main()
