"""Generate a deterministic fixed-shape VibeSim request trace.

Use this for load-generator capacity tests where shape variance would obscure
the offered-load boundary. Arrival times are milliseconds, matching the L7 and
req-frontend `independent` frontend contract.
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
) -> None:
    if requests <= 0:
        raise ValueError("requests must be positive")
    if input_len <= 0 or output_len <= 0:
        raise ValueError("input_len and output_len must be positive")
    if not interarrival_ms >= 0.0:
        raise ValueError("interarrival_ms must be nonnegative")

    lines = ["id,input_len,output_len,arrival_time"]
    lines.extend(
        f"{request_index},{input_len},{output_len},{request_index * interarrival_ms:.6f}"
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
    arguments = parser.parse_args()
    write_fixed_shape_trace(
        arguments.output,
        requests=arguments.requests,
        input_len=arguments.input_len,
        output_len=arguments.output_len,
        interarrival_ms=arguments.interarrival_ms,
    )


if __name__ == "__main__":
    main()
