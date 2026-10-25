"""Align qwen4_exp_neuron_bf16 with the flashnext-neuron engine's serving logs.

The engine (framework/flashnext-neuron, `engine/serve.py`) writes one JSONL per run:
a `header` row with the Sim's knob names, one `step` row per engine step (per-group
decode requests and prefill tokens, padded shapes, per-forward walls, rank-0 device
time in profiled windows) and one `request` row per request (absolute arrival,
TTFT, TPOT, ITLs). Two comparisons:

- `replay`: every step's exact shape (each DP group's decode batch and prefill chunk,
  the chunk's start reconstructed from the per-group FIFO of prompts) through
  `launcher timing-predict`; per-step and per-forward wall against the engine's.
- `e2e`: each run as a simulation with the run's knobs, its requests at their
  engine arrival times, and each request's MTP acceptance read off its ITLs.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from collections import deque
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
ENGINE_LOGS = Path(
    os.environ.get(
        "FLASHNEXT_ENGINE_LOGS",
        str(SIM.parents[1] / "framework" / "flashnext-neuron" / "logs" / "engine"),
    )
)

# The engine runs this alignment uses: (label, serve_<name>.jsonl per replica).
RUNS = [
    ("w64_layer_k0", ["w64_2048_512_layer_k0"]),
    ("w64_layer_k1", ["w64_2048_512_layer_k1"]),
    ("w64_layer_k2", ["w64_2048_512_layer_k2"]),
    ("w64_block_k1", ["w64_2048_512_block_k1"]),
    ("w64_rsag_k1", ["w64_2048_512_layer_rsag_k1"]),
    ("w64_a2a_k1", ["w64_2048_512_layer_a2a_k1"]),
    ("w64_layer_k0_c512", ["w64_2048_512_layer_k0_chunk512"]),
    ("w64_layer_k1_c512", ["w64_2048_512_layer_k1_chunk512"]),
    ("w64_rsag_k1_c512", ["w64_2048_512_layer_rsag_k1_chunk512"]),
    ("w64_2rep_layer_k1", ["w64_2048_512_layer_k1_2rep_r0", "w64_2048_512_layer_k1_2rep_r1"]),
    (
        "w64_2rep_rsag_k2_c512",
        [
            "w64_2048_512_layer_rsag_k2_chunk512_2rep_r0",
            "w64_2048_512_layer_rsag_k2_chunk512_2rep_r1",
        ],
    ),
    ("p1_layer_k0", ["poisson1_512_256_layer_k0"]),
    ("p1_layer_k1", ["poisson1_512_256_layer_k1"]),
    ("p1_layer_k0_c512", ["poisson1_512_256_layer_k0_chunk512"]),
    ("p025_layer_k0", ["poisson025_512_256_layer_k0"]),
    ("p025_layer_k1", ["poisson025_512_256_layer_k1"]),
]

METRICS = ("duration_s", "out_tok_s", "ttft_p50_s", "tpot_p50_ms")


def pct(values, q):
    """The engine's percentile (`serve.py`): nearest rank at q * (n - 1)."""
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] if v else None


def load(name: str, logs: Path = ENGINE_LOGS) -> list[dict]:
    with open(logs / f"serve_{name}.jsonl") as f:
        return [json.loads(line) for line in f]


def acceptance(request: dict, k: int) -> list[float]:
    """Per-position conditional MTP acceptance of one request, from its ITLs.

    Tokens of one verify step share a timestamp (ITL 0.0). The first group is the
    prefill's token and the last may be clipped at the output length, so both are
    dropped. Position j's rate is P(step emits >= j + 1 tokens | >= j)."""
    groups, n = [], 1
    for x in request["itl_ms"]:
        if x == 0.0:
            n += 1
        else:
            groups.append(n)
            n = 1
    groups.append(n)
    groups = groups[1:-1] or [1]
    at_least = lambda m: sum(1 for g in groups if g >= m)  # noqa: E731
    out = []
    for j in range(1, k + 1):
        den = at_least(j)
        out.append(at_least(j + 1) / den if den else 0.0)
    return [min(0.999, max(0.0, p)) for p in out]


def measured(files: list[str], logs: Path = ENGINE_LOGS) -> tuple[dict, list[dict]]:
    """The engine's duration, output tok/s, TTFT p50 and TPOT p50 over every replica."""
    reqs, duration = [], 0.0
    for name in files:
        rows = [r for r in load(name, logs) if r["type"] == "request"]
        reqs += rows
        duration = max(duration, max(r["finish_s"] for r in rows))
    out = sum(r["output_len"] for r in reqs)
    return (
        dict(
            duration_s=duration,
            out_tok_s=out / duration,
            ttft_p50_s=pct([r["ttft_ms"] for r in reqs], 0.5) / 1e3,
            tpot_p50_ms=pct([r["tpot_ms"] for r in reqs if r["tpot_ms"] is not None], 0.5),
        ),
        reqs,
    )


def arch_knobs(header: dict) -> dict:
    """The arch knobs a run's header names (the engine writes the Sim's names)."""
    return dict(
        attn_dp=header["attn_dp"],
        attn_tp=header["attn_tp"],
        gdn_tp=header["gdn_tp"],
        moe_ep=header["moe_ep"],
        moe_tp=header["moe_tp"],
        region_granularity=header["region_granularity"],
        prefill_buckets=header["prefill_buckets"],
        decode_buckets=header["decode_buckets"],
        context_buckets=header["context_buckets"],
        tp_collective=header["tp_collective"],
        moe_dispatch=header["moe_dispatch"],
        moe_a2a_capacity_factor=header.get("moe_a2a_cf", 0.0),
    )


def step_cases(rows: list[dict]) -> list[tuple[dict, dict]]:
    """One timing-predict case per engine step, with its measured times.

    Each DP group runs its own decode batch (KV at the decode forward's context
    bucket) and at most one prefill chunk; the chunk's start is the progress of the
    oldest prompt still in prefill in that group (admission order)."""
    header = rows[0]
    k = header["draft_tokens"]
    reqs = sorted(
        (r for r in rows if r["type"] == "request"), key=lambda r: (r["admit_s"], r["rid"])
    )
    queues = [deque() for _ in range(header["attn_dp"])]
    for r in reqs:
        queues[r["group"]].append([r["prompt_len"], 0])
    out = []
    for st in (r for r in rows if r["type"] == "step"):
        fw = st["forwards"]
        dec = [f for f in fw if f["phase"] == "decode"]
        pre = [f for f in fw if f["phase"] == "prefill"]
        ctx = dec[0]["C"] if dec else 0
        groups = []
        for g, (n, t) in enumerate(zip(st["decode_reqs"], st["prefill_tokens"])):
            grp: dict = {}
            if k:
                grp["decode_requests"] = [[ctx, k + 1]] * n
            elif n:
                grp["decode_kv_lens"] = [ctx] * n
            if t:
                head = queues[g][0]
                grp["prefill_chunk_pairs"] = [[head[1], t]]
                head[1] += t
                if head[1] >= head[0]:
                    queues[g].popleft()
            groups.append(grp)
        split = next((i for i, f in enumerate(fw) if f["phase"] == "prefill"), len(fw))
        meas = dict(
            step=st["step"],
            kind=st["kind"],
            wall=st["wall_ms"],
            device=st.get("device_ms"),
            decode_B=st["decode_B"],
            prefill_T=st["prefill_T"],
            dec_wall=sum(f["wall_ms"] for f in fw[:split]) if dec else None,
            dec_main=dec[0]["wall_ms"] if dec else None,
            pre_wall=sum(f["wall_ms"] for f in fw[split:]) if pre else None,
            pre_main=pre[0]["wall_ms"] if pre else None,
        )
        out.append(({"groups": groups}, meas))
    return out


def timing_predict(
    name: str, arch: dict, cases: list[dict], speculative: bool, root: Path
) -> list[dict]:
    """Run `launcher timing-predict` on `cases`; return the per-case breakdowns."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    a = dict(
        type="qwen4_exp_neuron_bf16_speculative" if speculative else "qwen4_exp_neuron_bf16",
        model_config="model/config/qwen3_8_flash_next_bf16.json",
        fp8=False,
        **arch,
    )
    (d / "cases.json").write_text(json.dumps(cases))
    cfg = {
        "arch": {"speculative_iter" if speculative else "iter": a},
        "gpu": "AWS Trainium2 LNC2",
        "log_dir": str((d / "out").relative_to(SIM)),
        "cases_file": "cases.json",
    }
    (d / "cfg.json").write_text(json.dumps(cfg, indent=1))
    env = dict(os.environ, SERVINGSTUDIO_NO_GPU="1")
    r = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "launcher",
            "timing-predict",
            str(d / "cfg.json"),
            "--no-gpu",
            "--analyzer-essential-only",
        ],
        cwd=SIM,
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise RuntimeError(f"timing-predict {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads((d / "out" / "payloads" / "iter_breakdown.json").read_text())["iterations"]


def forward_times(iteration: dict) -> tuple[float, dict, dict]:
    """(step total, ms per depth-1 node (the forwards and `step`), launch ms per forward)."""
    nodes, launch, cur = {}, {}, None
    for n in iteration["nodes"]:
        if n["depth"] == 1:
            cur = n["label"]
            nodes[cur] = nodes.get(cur, 0.0) + n["ms"]
        if n["kind"] == "leaf" and n["label"].startswith("launch") and cur:
            launch[cur] = launch.get(cur, 0.0) + n["ms"]
    return iteration["nodes"][0]["ms"], nodes, launch


def replay_run(name: str, extra: dict, root: Path, logs: Path = ENGINE_LOGS) -> list[dict]:
    """Every step of run `name`, measured and predicted (unique shapes predicted once)."""
    rows = load(name, logs)
    header = rows[0]
    arch = arch_knobs(header) | extra
    k = header["draft_tokens"]
    if k:
        arch["draft_tokens"] = k
    steps = step_cases(rows)
    index: dict[str, int] = {}
    for case, _ in steps:
        index.setdefault(json.dumps(case, sort_keys=True), len(index))
    cases = [json.loads(key) for key in sorted(index, key=index.get)]
    preds = [forward_times(it) for it in timing_predict(f"replay_{name}", arch, cases, k > 0, root)]
    out = []
    for case, meas in steps:
        total, nodes, launch = preds[index[json.dumps(case, sort_keys=True)]]
        out.append(
            dict(
                meas,
                p_wall=total,
                p_dec=sum(v for key, v in nodes.items() if key.startswith("decode")),
                p_pre=sum(v for key, v in nodes.items() if key.startswith("prefill")),
                p_dec_main=nodes.get("decode forward", 0.0),
                p_pre_main=nodes.get("prefill forward", 0.0),
                p_device=total - sum(launch.values()) - nodes.get("step", 0.0),
            )
        )
    return out


def render_preset(
    label: str,
    files: list[str],
    arch_extra: dict,
    worker_extra: dict,
    out_dir: Path,
    logs: Path = ENGINE_LOGS,
) -> Path:
    """A one-run unified preset with run `files`' knobs and requests."""
    header = load(files[0], logs)[0]
    k = header["draft_tokens"]
    _, reqs = measured(files, logs)
    reqs = sorted(reqs, key=lambda r: (r["arrival_s"], r["rid"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / "trace.csv"
    lines = ["id,input_len,output_len,arrival_time" + (",accept_rate" if k else "")]
    for i, r in enumerate(reqs):
        line = f"{i},{r['prompt_len']},{r['output_len']},{r['arrival_s'] * 1e3:.3f}"
        if k:
            a = acceptance(r, k)
            line += "," + (
                f"{a[0]:.4f}" if k == 1 else '"' + json.dumps([round(x, 4) for x in a]) + '"'
            )
        lines.append(line)
    trace.write_text("\n".join(lines) + "\n")
    arch = dict(
        type="qwen4_exp_neuron_bf16_speculative" if k else "qwen4_exp_neuron_bf16",
        model_config="model/config/qwen3_8_flash_next_bf16.json",
        fp8=False,
        **arch_knobs(header),
    )
    if k:
        arch["draft_tokens"] = k
    arch |= arch_extra
    worker = dict(
        type="speculative" if k else "chunked_prefill",
        attn_gpu_memory_gb=25.769803776,
        max_batch_tokens=header["max_batch_tokens"],
        max_prefill_chunks_per_partition=header.get("prefill_per_group_per_step", 1),
    )
    if k:
        worker["draft_tokens"] = k
    else:
        worker["dp_placement"] = "vllm-least-loaded"
    worker |= worker_extra
    workload = dict(
        trace_files=[str(trace)],
        input_file_format="text-generation-independent",
        arrival_mode="trace_timed",
        run_to_end=True,
        request_rate=1.0,
    )
    if k:
        workload["input_file_tags"] = ["speculative"]
    cfg = dict(
        deployment="unified",
        workload=workload,
        io=dict(log_dir=str(out_dir / "run")),
        pools=dict(
            main=dict(
                placement="round-robin",  # the engine splits replicas by rid parity
                groups=[
                    dict(gpu="AWS Trainium2 LNC2", replicas=len(files), arch=arch, worker=worker)
                ],
            )
        ),
    )
    path = out_dir / "preset.json"
    path.write_text(json.dumps(cfg, indent=1))
    return path


def simulated(run_dir: Path) -> dict:
    """The simulated run's duration, output tok/s, TTFT p50 and TPOT p50."""
    import pyarrow.parquet as pq

    reqs = [
        r
        for r in pq.read_table(run_dir / "raw" / "request_slo.parquet").to_pylist()
        if r["completed"]
    ]
    duration = json.loads((run_dir / "summary.json").read_text())["sim_ms"] / 1e3
    out = sum(r["num_output_tokens"] for r in reqs)
    return dict(
        duration_s=duration,
        out_tok_s=out / duration,
        ttft_p50_s=pct([r["ttft_ms"] for r in reqs], 0.5) / 1e3,
        tpot_p50_ms=pct([r["tpot_mean_ms"] for r in reqs if r["tpot_mean_ms"] is not None], 0.5),
    )


def e2e_run(
    label: str,
    files: list[str],
    arch_extra: dict,
    worker_extra: dict,
    root: Path,
    logs: Path = ENGINE_LOGS,
) -> dict:
    out_dir = root / label
    preset = render_preset(label, files, arch_extra, worker_extra, out_dir, logs)
    env = dict(os.environ, SERVINGSTUDIO_NO_GPU="1")
    r = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "launcher",
            str(preset),
            "--no-gpu",
            "--no-analyze",
            "--refresh",
        ],
        cwd=SIM,
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode:
        raise RuntimeError(f"simulation {label} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return dict(measured=measured(files, logs)[0], simulated=simulated(out_dir / "run"))


def error_pct(sim: float, meas: float) -> float:
    return 100.0 * (sim / meas - 1.0)


def median_abs(values) -> float:
    return statistics.median(abs(v) for v in values)
