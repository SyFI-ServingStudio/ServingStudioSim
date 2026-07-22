"""optimality floors — the two global necessary-work lower bounds, per level.

Called by the Rust `optimality` analyzer subject as a subprocess:

    uv run python -m model.work.floors <log_dir>

The Rust side already aggregates each level's workload (reusing the
`workload-conservation` subject's `groups` parser) and pipes it in on **stdin** as::

    {"levels": {"<level_key>": {matmul_tokens, prefill_tokens, decode_passes,
                                prefill_pairs, prefill_cached, decode_kv,
                                prefill_requests}, ...}}

where a level_key is ``cluster`` | ``<pool_tag>`` | ``<pool_tag>/<worker_id>``
(exactly the `levels.rs` level keys). This
script is the *model-aware* half: it never touches the cost_log / parquet. For
each level it loads that pool's model, assembles ONE aggregate ``Workload`` (the
whole run's tokens as a single mega-forward — weights counted once, the loosest
necessary floor), and emits the two roofline floors in **GPU·seconds** on stdout::

    {"levels": {"<level_key>": {"necessary": s, "segmented": s}, ...},
     "meta": {...}}

- **necessary** = global roofline ``max(ΣFLOPs/peak, Σbytes/bw)`` (`roofline_ms`).
- **segmented** = ``Σ_seg max(compute, memory)`` (`segmented_lower_bound_ms`).

`num_gpus=1`: the labeler computes the FULL / unsharded ``F_min``, so
``total_work / peak`` is already the ×G GPU·seconds the Rust R5 rung is in.

The attention geometry is passed as pre-summed scalars, so a single
``mask="full"`` interaction (``pairs() = q·k`` with ``q=1``, ``k=Σpairs``)
reproduces the per-step sum exactly — no labeler-core change, and no 1.7-billion
-element interaction list.
"""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

from .core import AttnInteraction, Workload
from .registry import load_model


@cache
def _model(config_path: str):
    """Load a model once per config path (levels within a pool share it)."""
    return load_model(config_path)


def _pool_specs(log_dir: Path) -> dict[str, dict]:
    """Map pool_tag -> {config, gpu, dtype} from the run's params.json.

    One group per pool assumed (PD/unified dense today); multi-group EP/HP needs
    the same per-group revisit as the rest of the analyzer.
    """
    params = json.loads((log_dir / "raw" / "params.json").read_text())
    specs: dict[str, dict] = {}
    for pool_tag, pool in params.get("pools", {}).items():
        group = pool["groups"][0]
        arch = group["arch"]
        specs[pool_tag] = {
            "config": arch["model_config"],
            # `gpu` is a group-level field (the arch block carries model/tp/fp8).
            "gpu": group["gpu"],
            # fp8 halves the roofline compute peak; else the model's native bf16.
            "dtype": "fp8" if arch.get("fp8") else "bf16",
        }
    return specs


def _spec_for_level(level_key: str, pool_specs: dict[str, dict]) -> dict:
    """Which (config, gpu, dtype) a level computes under.

    ``<pool_tag>`` / ``<pool_tag>/<worker_id>`` take that pool's spec; ``cluster``
    spans every pool, so it requires them to share one model (true for PD single
    -model runs) — otherwise a per-pool-summed cluster floor is needed (deferred).
    """
    if level_key == "cluster":
        distinct = {(s["config"], s["gpu"], s["dtype"]) for s in pool_specs.values()}
        if len(distinct) != 1:
            raise ValueError(
                f"cluster floor needs one shared model across pools, saw {distinct}"
            )
        return next(iter(pool_specs.values()))
    # Pool level is the bare tag; worker level is "<pool_tag>/<worker_id>".
    pool_tag = level_key.split("/", 1)[0]
    return pool_specs[pool_tag]


def _aggregate_workload(totals: dict) -> Workload:
    """Assemble one giant-batch Workload from a level's pre-summed scalars.

    Prefill and decode attention each collapse to a single ``mask="full"``
    interaction carrying the summed pair / cached-KV counts (identity, since
    ``pairs()`` for full = ``q·k`` and we set ``q=1``). Decode reads all context
    keys from cache each step, so cached == pairs == ``decode_kv``; the self-key
    ``+1`` per step is negligible and omitted.
    """
    matmul_tokens = int(totals["matmul_tokens"])
    sampled = int(totals["decode_passes"]) + int(totals["prefill_requests"])
    attn: list[AttnInteraction] = []
    prefill_pairs = int(totals["prefill_pairs"])
    if prefill_pairs > 0:
        attn.append(
            AttnInteraction(1, prefill_pairs, int(totals["prefill_cached"]), "full")
        )
    decode_kv = int(totals["decode_kv"])
    if decode_kv > 0:
        attn.append(AttnInteraction(1, decode_kv, decode_kv, "full"))
    return Workload(matmul_tokens=matmul_tokens, head_positions=sampled, attn=attn)


def compute_floors(log_dir: Path, levels: dict[str, dict]) -> dict[str, dict]:
    """Per-level {necessary, segmented} in GPU·seconds via the labeler roofline."""
    pool_specs = _pool_specs(log_dir)
    out: dict[str, dict] = {}
    for level_key, totals in levels.items():
        spec = _spec_for_level(level_key, pool_specs)
        model = _model(spec["config"])
        workload = _aggregate_workload(totals)
        label = model.label(workload)
        compute_ms, memory_ms, _bound = label.roofline_ms(spec["gpu"], spec["dtype"])
        segmented_ms = label.segmented_lower_bound_ms(spec["gpu"], spec["dtype"])
        out[level_key] = {
            "necessary": max(compute_ms, memory_ms) / 1e3,  # ms -> GPU·seconds
            "segmented": segmented_ms / 1e3,
        }
    return out


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m model.work.floors <log_dir>  (scalars on stdin)")
    log_dir = Path(argv[0])
    request = json.load(sys.stdin)
    floors = compute_floors(log_dir, request["levels"])
    json.dump({"levels": floors, "meta": {"unit": "gpu_seconds"}}, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
