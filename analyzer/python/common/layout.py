"""Run-dir artifact layout — mirror of the Rust `io.rs` convention so the two
sides agree on where payloads live and where PNGs go.

`raw/` (sim parquet) · `reports/` (numbers JSON) · `payloads/` (plot JSON) ·
`plots/` (PNG, written here).
"""

from __future__ import annotations

import json
from pathlib import Path

RAW_DIR = "raw"
REPORTS_DIR = "reports"
PAYLOADS_DIR = "payloads"
PLOTS_DIR = "plots"
#: The `encoding` of a detail shard whose records are independent zstd frames.
ZSTD_FRAMES = "zstd-frames"


def resolve_artifact(log_dir: Path, name: str) -> Path:
    """Find an artifact in the run root or any known subdir (the Rust analyzer
    writes payloads into `payloads/`)."""
    for sub in ("", PAYLOADS_DIR, REPORTS_DIR, RAW_DIR, PLOTS_DIR):
        path = log_dir / name if not sub else log_dir / sub / name
        if path.exists():
            return path
    return log_dir / PAYLOADS_DIR / name


def load_payload(log_dir: Path, name: str) -> dict:
    return json.loads(resolve_artifact(log_dir, name).read_text())


def read_sharded_records(log_dir: Path, shard: dict, keys: list) -> list[dict]:
    """Read named records out of a payload's byte-range-addressed sibling.

    A subject whose per-iteration detail runs to hundreds of megabytes ships an
    index plus a `.jsonl`, and the index carries each record's `[offset, length]`
    (see the Rust `iteration_detail` / `breakdown_detail` sections). Reading only
    the wanted records is the point of that layout: a renderer that samples 128
    of 2,040 iterations must not parse the other 1,912. A `zstd-frames` shard
    (`.jsonl.zst`) makes each range one zstd frame, which decodes to the index's
    `decoded_lengths[key]` bytes.

    Returns records in the order of `keys`; a key the index does not know is
    skipped, because a payload written before its shard existed is a missing
    figure, not a crash.
    """
    byte_ranges = shard.get("byte_ranges") or {}
    decode = _record_decoder(shard)
    path = resolve_artifact(log_dir, shard["file"])
    records = []
    with path.open("rb") as handle:
        for key in keys:
            location = byte_ranges.get(str(key))
            if location is None:
                continue
            offset, length = location
            handle.seek(offset)
            records.append(json.loads(decode(str(key), handle.read(length))))
    return records


def _record_decoder(shard: dict):
    """`(key, raw range) -> JSON bytes` for the shard's `encoding`."""
    if shard.get("encoding") != ZSTD_FRAMES:
        return lambda _key, raw: raw
    import pyarrow as pa

    decoded_lengths = shard["decoded_lengths"]
    return lambda key, raw: pa.decompress(
        raw, decompressed_size=decoded_lengths[key], codec="zstd", asbytes=True
    )


def plot_output_path(log_dir: Path, name: str) -> Path:
    path = log_dir / PLOTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
