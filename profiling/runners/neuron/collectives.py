"""List-native Neuron runner for the in-graph collectives (``neuron_cc``).

The Neuron pool reserves exactly one chip per chunk, and a ``neuron_cc`` row declares a span of
4 logical cores, so the worker sees ``NEURON_VISIBLE_DEVICES=<a>-<b>`` (the chip's cores). This
runner spawns one process per rank on the first ``num_gpus`` of those cores
(``flashnext.neuron.sim_l1.collectives.run_group``: each rank pins
``NEURON_RT_VISIBLE_CORES=<core>``, joins a gloo world and traces torch functional collectives
into one compiled region) and times rank 0's NRT device time: the slope of 33 vs 1 chained
collectives, with a CPU barrier before every call, the median of 3 repeats.

Only one-chip groups can run here: fabric ``neuron_c2`` with ``num_gpus == 2`` and
``neuron_c4`` with ``num_gpus == 4``. Multi-chip groups (``neuron_c1``, or ``neuron_c4`` with
more than 4 ranks) need chips the pool never hands one chunk; measure them with the standalone
sweep ``python -m flashnext.neuron.sim_l1.collectives --cores ...`` (holding the chips' pool
locks) and import the JSONL with ``tools/neuron-bench-import`` (family ``collectives``).
"""

from __future__ import annotations

import os

from profiling.runners.metrics import CommMetrics, RunnerResult

CORES_PER_CHIP = 4
ONE_CHIP = {"neuron_c2": 2, "neuron_c4": 4}
MULTI_CHIP_HINT = (
    "multi-chip collective groups cannot run in the Sim Neuron pool (one chip per chunk); "
    "measure them with `python -m flashnext.neuron.sim_l1.collectives --cores ...` "
    "(holding the chips' /tmp/servingstudio-neuron-<chip>.lock) and import the JSONL with "
    "tools/neuron-bench-import/import_bench.py --family collectives"
)


def bus_factor(op: str, world: int) -> tuple[float, float]:
    """(algbw bytes, busbw bytes) per message byte, nccl-tests conventions; all_gather's message
    is the per-rank input shard, the others' the per-rank full buffer."""
    n = world
    if op == "all_reduce":
        return 1.0, 2.0 * (n - 1) / n
    if op == "all_gather":
        return float(n), float(n - 1)
    if op in ("reduce_scatter", "all_to_all"):
        return 1.0, (n - 1) / n
    raise ValueError(op)


def visible_cores() -> list[int]:
    span = os.environ.get("NEURON_VISIBLE_DEVICES")
    if span:
        a, _, b = span.partition("-")
        return list(range(int(a), int(b or a) + 1))
    core = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if core:
        return [int(core)]
    raise RuntimeError(
        "no NEURON_VISIBLE_DEVICES / NEURON_RT_VISIBLE_CORES reservation in the worker"
    )


def check_spec(spec: dict) -> str | None:
    """Why this spec cannot be measured live, or None."""
    n, fabric = int(spec["num_gpus"]), spec["fabric"]
    if str(spec.get("dtype", "bf16")).lower() not in ("bf16", "bfloat16", "dtype.bf16"):
        return f"neuron_cc measures bf16 buffers only, got dtype {spec['dtype']}"
    if fabric not in ("neuron_c1", "neuron_c2", "neuron_c4"):
        return f"fabric {fabric!r} is not a Neuron fabric (neuron_c1 / neuron_c2 / neuron_c4)"
    if ONE_CHIP.get(fabric) != n:
        return f"num_gpus={n} fabric={fabric} is a multi-chip or partial group: {MULTI_CHIP_HINT}"
    return None


def _run(op: str, kwargs_list: list[dict]) -> list[RunnerResult]:
    from flashnext.neuron.sim_l1.collectives import run_group  # worker-only, out of tree

    results: list[RunnerResult | None] = [None] * len(kwargs_list)
    groups: dict[int, list[int]] = {}
    for i, spec in enumerate(kwargs_list):
        why = check_spec(spec)
        if why:
            results[i] = RunnerResult(error=why)
        else:
            groups.setdefault(int(spec["num_gpus"]), []).append(i)
    if groups:
        cores = visible_cores()
    for n, idx in sorted(groups.items()):
        if len(cores) < n:
            for i in idx:
                results[i] = RunnerResult(
                    error=f"{n} ranks need {n} logical cores, the worker sees {cores}"
                )
            continue
        sizes = sorted({int(kwargs_list[i]["message_size_bytes"]) for i in idx})
        rows = run_group(cores[:n], [(op, b) for b in sizes], timeout_s=3600)
        by_bytes = {int(r["bytes"]): r for r in rows if r.get("op") == op}
        algf, busf = bus_factor(op, n)
        for i in idx:
            nbytes = int(kwargs_list[i]["message_size_bytes"])
            row = by_bytes.get(nbytes)
            if row is None:
                results[i] = RunnerResult(
                    error=f"no row for {nbytes} bytes (sizes must be multiples of {n * 256})"
                )
            elif row.get("error"):
                results[i] = RunnerResult(error=row["error"])
            elif not row.get("per_cc_device_ms", 0) > 0:
                results[i] = RunnerResult(
                    error=f"non-positive in-graph slope {row.get('per_cc_device_ms')} ms; re-run"
                )
            else:
                ms = float(row["per_cc_device_ms"])
                seconds = ms / 1e3
                results[i] = RunnerResult(
                    metrics=CommMetrics(
                        time_ms=ms,
                        algbw_gbps=algf * nbytes / seconds / 1e9,
                        busbw_gbps=busf * nbytes / seconds / 1e9,
                    )
                )
    return results  # type: ignore[return-value]


def profile_all_reduce_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _run("all_reduce", kwargs_list)


def profile_all_gather_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _run("all_gather", kwargs_list)


def profile_reduce_scatter_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _run("reduce_scatter", kwargs_list)


def profile_all_to_all_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _run("all_to_all", kwargs_list)
