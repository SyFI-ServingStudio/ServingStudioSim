"""The normalized NSYS document on disk: ``parsed.json`` plus a kernel-row sibling.

Kernel rows are 95% of a capture's parse (1.38 GB of 1.45 GB on a 9M-kernel GLM-5.3
TP4 view capture), and the only consumer that reads them row by row is the Rust
alignment analyzer. Written as JSON they were the slowest part of the parse to
encode and of the analysis to load, so schema 6 keeps them out of ``parsed.json``::

    parsed.json              every other field, each range without its "kernels"
    parsed.kernels.parquet   one row per kernel, in range order then ordinal order

Each range also drops its ``metrics`` copy. It repeated the rank's full scheduler
metrics once per phase range — 466 MB of a 4169-iteration TP4 capture — and every
reader takes metrics from the iteration (``metrics`` / ``metrics_by_dp_rank``)
instead; a range still names its ``dp_rank``.

The document names the sibling under ``kernel_rows``. ``detail_index`` and
``range_index`` locate each row's range as positions in ``iteration_details`` and
its ``ranges``; the remaining columns are the old inline kernel fields unchanged.
A document without ``kernel_rows`` (schema 5 and earlier) keeps its kernels
inline, and `read_parsed` accepts either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PARSED_SCHEMA_VERSION = 6

_KERNEL_SCHEMA = pa.schema(
    [
        ("detail_index", pa.uint32()),
        ("range_index", pa.uint32()),
        ("ordinal", pa.uint32()),
        ("name_id", pa.uint32()),
        ("category", pa.string()),
        ("start_ns", pa.uint64()),
        ("end_ns", pa.uint64()),
        ("stream_id", pa.uint64()),
        ("correlation_id", pa.uint64()),
        ("track_index", pa.uint32()),
    ]
)
#: Range fields schema 6 does not write: kernels go to parquet, metrics are the
#: iteration's.
_NOT_IN_RANGES = frozenset({"kernels", "metrics"})
#: The inline kernel object's fields, in its key order.
_KERNEL_FIELDS = _KERNEL_SCHEMA.names[2:]


def kernel_rows_path(parsed_path: Path) -> Path:
    """``parsed.json`` -> ``parsed.kernels.parquet``, in the same directory."""
    return parsed_path.with_name(f"{parsed_path.stem}.kernels.parquet")


def write_parsed(path: Path, parsed: dict[str, Any]) -> None:
    """Write `parsed` (kernels inline, as `parse_trace` returns it) as schema 6.

    `parsed` itself is not modified; the caller may keep using its inline kernels.
    """
    path = Path(path)
    columns: dict[str, list] = {name: [] for name in _KERNEL_SCHEMA.names}
    details = []
    for detail_index, detail in enumerate(parsed["iteration_details"]):
        ranges = []
        for range_index, range_row in enumerate(detail["ranges"]):
            kernels = range_row.get("kernels", [])
            columns["detail_index"].extend([detail_index] * len(kernels))
            columns["range_index"].extend([range_index] * len(kernels))
            for field_name in _KERNEL_FIELDS:
                columns[field_name].extend(kernel[field_name] for kernel in kernels)
            ranges.append(
                {key: value for key, value in range_row.items() if key not in _NOT_IN_RANGES}
            )
        details.append({**detail, "ranges": ranges})

    rows_path = kernel_rows_path(path)
    table = pa.table(columns, schema=_KERNEL_SCHEMA)
    pq.write_table(table, rows_path, compression="zstd")
    document = {
        **parsed,
        "schema_version": PARSED_SCHEMA_VERSION,
        "kernel_rows": {"file": rows_path.name, "format": "parquet", "rows": table.num_rows},
        "iteration_details": details,
    }
    # Compact: nothing reads this document by eye. `indent` would also drop
    # `json` onto its pure-Python encoder, which was half of a large parse.
    path.write_text(json.dumps(document, separators=(",", ":")))


def read_parsed(path: Path, *, kernels: bool = True) -> dict[str, Any]:
    """Read a parsed document of any schema, with each range's kernels inline.

    A schema-6 range has no ``metrics`` (see the module docstring).

    ``kernels=False`` skips the kernel rows of a schema-6 document, for a caller
    that needs only iteration metrics; a range then has no ``kernels`` key.
    """
    path = Path(path)
    parsed = json.loads(path.read_text())
    if not isinstance(parsed, dict):
        raise ValueError(f"parsed NSYS root must be a JSON object: {path}")
    rows = parsed.pop("kernel_rows", None)
    if rows is None or not kernels:
        return parsed
    if rows.get("format") != "parquet":
        raise ValueError(f"unsupported kernel_rows format {rows.get('format')!r} in {path}")
    table = pq.read_table(path.with_name(rows["file"]))
    if table.num_rows != rows["rows"]:
        raise ValueError(
            f"{rows['file']} has {table.num_rows} kernel rows, {path.name} declares {rows['rows']}"
        )
    for detail in parsed["iteration_details"]:
        for range_row in detail["ranges"]:
            range_row["kernels"] = []
    values = {name: table.column(name).to_pylist() for name in _KERNEL_SCHEMA.names}
    details = parsed["iteration_details"]
    for row in range(table.num_rows):
        details[values["detail_index"][row]]["ranges"][values["range_index"][row]][
            "kernels"
        ].append({field_name: values[field_name][row] for field_name in _KERNEL_FIELDS})
    return parsed
