"""Close the loop between the Sim's config search and the flashnext-neuron engine.

One *loop* lives in `logs/trn2/tune/<name>/` and goes through four resumable steps:

- `propose`: candidates from a config-search experiment (the top-N engine-runnable
  configs of one workload, variants of the best one, a contrast the Sim predicts to
  be worse), each pinned to the engine's own knobs (buckets, replicas, the W64 load);
- `predict`: each candidate as a Sim run with exactly those knobs; the predictions
  (and the inputs they rest on: MTP acceptance, the fitted a2a packing cost) are
  written before anything runs on the device;
- `run`: the engine's `serve.py` commands with the matching knobs and the same
  workload (two 32-core replicas, comm ports 1000+ apart), written as scripts and
  optionally executed one candidate at a time;
- `compare`: predicted vs measured per metric, ranking agreement, and a verdict:
  `realign` when an error passes the threshold or the ranking flips.

Every step appends to `tune.log` and records itself in `state.json`; a rerun skips
what is done (`--force` redoes a step).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[1]
sys.path.insert(0, str(SIM / "tools" / "trn2-engine-align"))
sys.path.insert(0, str(SIM / "tools" / "trn2-config-search"))

import engine_align as ea  # noqa: E402

ENGINE_ROOT = Path(
    os.environ.get("FLASHNEXT_ENGINE_ROOT", str(SIM.parents[1] / "framework" / "flashnext-neuron"))
)
TUNE_ROOT = SIM / "logs" / "trn2" / "tune"
GPU = "AWS Trainium2 LNC2"
MODEL_CONFIG = "model/config/qwen3_8_flash_next_bf16.json"
CORE_HBM_GB = 25.769803776

# The W64 workload: 64 requests, 2048 in / 512 out, 50 ms apart (engine
# `--arrival fixed:0.05`; Sim trace presets/trn2/trn2_fixed_2k_512_c64.csv).
WORKLOADS = {
    "w64": dict(
        requests=64,
        interval_s=0.05,
        input_len=2048,
        output_len=512,
        engine_args="--requests 64 --arrival fixed:0.05 --input fixed:2048 --output fixed:512",
    ),
}

# The engine's knobs when a search config leaves them out (serve.py defaults and the
# buckets of its measured runs).
ENGINE_DEFAULTS = dict(
    decode_buckets=[1, 2, 4, 8, 16],
    context_buckets=[2048, 4096],
    replicas=2,
)
ENGINE_PREFILL_BUCKETS = (128, 512, 2048)
REPLICA_CORES = 32
# The largest block-granularity prefill the engine loads: attn_dp x T gathered MoE rows
# per rank. 4 x 2048 = 8192 runs (w64 block k1); 8 x 2048 = 16384 fails at NEFF load
# with "Allocation Failure" (tune loop blind_w64, chunk4096). Layer granularity at
# 16384 rows is unmeasured.
BLOCK_PREFILL_ROWS_MAX = 8192

# Sim knobs the engine does not have, with the only value the engine implements
# (None: no engine equivalent at all; any value other than the arch default is a
# what-if). A candidate that sets one to anything else cannot run.
SIM_ONLY = {
    "step_forwards": "separate",
    "spmd_padding": True,
    "max_prefill_chunks_per_partition": 1,
    "ple_table_shards": "replica",
    "launch_model": "pipelined",
    "kind_time_scale": None,
    "host_time_scale": 1.0,
    "region_device_factor": 1.0,
    "moe_a2a_pack_us_per_row": None,
    "lnc": 2,
}
# Sim knobs with no engine flag that only describe the Sim (fitted constants and
# placement): they pass through to the prediction, never to the engine.
SIM_DESCRIPTIVE = {
    "region_host_ms",
    "forward_overhead_ms",
    "mtp_forward_overhead_ms",
    "region_gap_ms",
    "step_overhead_ms",
    "launch_coll_ms",
    "launch_plain_ms",
    "dp_placement",
    "attn_gpu_memory_gb",
    "hbm_per_core_gb",
    "hbm_reserve_gb",
}
# Bookkeeping fields of a search config row (not knobs).
BOOKKEEPING = {"family", "scenario", "label", "core_id", "accept_rate"}
GRANULARITY = {"per_op": "op", "mixer": "mixer", "layer": "layer", "block": "block"}
METRICS = ea.METRICS  # duration_s, out_tok_s, ttft_p50_s, tpot_p50_ms
# The metrics the verdict trusts. TTFT p50 on W64 is the prefill backlog and carries
# the engine's first-use stalls (ALIGNMENT.md: 10-16% low), so it only warns.
GATED = ("duration_s", "out_tok_s", "tpot_p50_ms")
# Two measured durations closer than this are a tie (run-to-run noise from host load).
TIE_PCT = 3.0


def _rel(p: Path) -> str:
    return str(p.relative_to(SIM)) if p.is_relative_to(SIM) else str(p)


# ------------------------------------------------------------------------- loop state
class Loop:
    def __init__(self, name: str, root: Path | None = None):
        self.name = name
        self.dir = (root or TUNE_ROOT) / name
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts) -> Path:
        return self.dir.joinpath(*parts)

    def log(self, msg: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        with open(self.path("tune.log"), "a") as f:
            f.write(line + "\n")
        print(line)

    def state(self) -> dict:
        p = self.path("state.json")
        return json.loads(p.read_text()) if p.is_file() else {}

    def mark(self, step: str, **info) -> None:
        s = self.state()
        s[step] = dict(done=dt.datetime.now().isoformat(timespec="seconds"), **info)
        self.path("state.json").write_text(json.dumps(s, indent=1))

    def done(self, step: str) -> bool:
        return step in self.state()

    def read(self, fname: str):
        return json.loads(self.path(fname).read_text())

    def write(self, fname: str, obj) -> Path:
        p = self.path(fname)
        p.write_text(json.dumps(obj, indent=1) + "\n")
        return p


# ------------------------------------------------------------------- knob translation
def engine_knobs(cfg: dict) -> dict:
    """A search config completed with the engine's defaults (buckets, replicas)."""
    k = {key: v for key, v in cfg.items() if key not in BOOKKEEPING}
    for key, v in ENGINE_DEFAULTS.items():
        k.setdefault(key, v)
    k.setdefault("max_batch_tokens", 2048)
    k.setdefault("draft_tokens", 0)
    if "prefill_buckets" not in k:
        mbt = k["max_batch_tokens"]
        k["prefill_buckets"] = [b for b in ENGINE_PREFILL_BUCKETS if b <= mbt] or [128]
    tp = k.get("attn_tp", 8)
    k.setdefault("attn_tp", tp)
    k.setdefault("gdn_tp", tp)
    k.setdefault("attn_dp", REPLICA_CORES // max(tp, k["gdn_tp"]))
    k.setdefault("moe_tp", 1)
    k.setdefault("moe_ep", REPLICA_CORES // k["moe_tp"])
    k.setdefault("region_granularity", "mixer")
    k.setdefault("tp_collective", "all_reduce")
    k.setdefault("moe_dispatch", "all_gather")
    return k


def engine_blockers(k: dict) -> list[str]:
    """Why the engine cannot run knobs `k` (empty: it can)."""
    out = []
    for key, only in SIM_ONLY.items():
        if key in k and (only is None or k[key] != only):
            out.append(f"{key}={k[key]!r} is Sim-only (engine: {only!r})")
    for key in k:
        if key.endswith("_impls"):
            out.append(f"{key} has no engine flag (engine uses its DEFAULT_IMPLS)")
    if k.get("attn_tp") != k.get("gdn_tp"):
        out.append("engine has one tp for QSA and GDN (attn_tp must equal gdn_tp)")
    if k["attn_dp"] * k["attn_tp"] != REPLICA_CORES:
        out.append(f"engine replicas are {REPLICA_CORES} cores (attn_dp x attn_tp)")
    if (k["moe_ep"], k["moe_tp"]) != (REPLICA_CORES, 1):
        out.append("engine MoE is EP32 x TP1 only")
    if k["attn_tp"] not in (2, 4, 8):
        out.append(f"attn_tp {k['attn_tp']} not supported (24 Q heads; 1 does not fit HBM)")
    if k["replicas"] not in (1, 2):
        out.append("replicas must be 1 or 2 (64 cores)")
    if k["region_granularity"] not in GRANULARITY:
        out.append(f"granularity {k['region_granularity']!r}")
    if k.get("moe_a2a_capacity_factor", 0) and k["moe_dispatch"] != "all_to_all":
        out.append("moe_a2a_capacity_factor needs moe_dispatch all_to_all")
    if not 0 <= k["draft_tokens"] <= 3:
        out.append("draft_tokens 0..3")
    rows = k["attn_dp"] * max(k["prefill_buckets"])
    if k["region_granularity"] == "block" and rows > BLOCK_PREFILL_ROWS_MAX:
        out.append(
            f"block prefill over {rows} gathered MoE rows (attn_dp x largest prefill bucket) "
            f"does not load on the engine (NRT Allocation Failure at dp8 x 2048; "
            f"{BLOCK_PREFILL_ROWS_MAX} loads); the Sim's HBM check has no activation memory"
        )
    if max(k["context_buckets"]) < 2560:
        out.append("context buckets must reach 2048 + 512")
    return out


def serve_args(k: dict) -> str:
    """The `serve.py` flags of knobs `k` (one replica, without cores/port/name)."""
    a = [
        f"--tp {k['attn_tp']}",
        f"--ep {k['moe_ep']}",
        "--dp-mode split",
        f"--granularity {GRANULARITY[k['region_granularity']]}",
        f"--tp-collective {k['tp_collective']}",
        f"--moe-dispatch {k['moe_dispatch']}",
        f"--mtp-k {k['draft_tokens']}",
        f"--max-batch-tokens {k['max_batch_tokens']}",
        "--prefill-buckets " + ",".join(map(str, k["prefill_buckets"])),
        "--decode-buckets " + ",".join(map(str, k["decode_buckets"])),
        "--context-buckets " + ",".join(map(str, k["context_buckets"])),
    ]
    if k.get("moe_a2a_capacity_factor"):
        a.append(f"--moe-a2a-cf {k['moe_a2a_capacity_factor']}")
    return " ".join(a)


def warmup_shapes(k: dict) -> str:
    """Every shape the run can hit, warmed (compiled) before the clock starts."""
    shapes = [f"prefill:{t}:{c}" for t in k["prefill_buckets"] for c in k["context_buckets"]]
    shapes += [f"decode:{b}:{max(k['context_buckets'])}" for b in k["decode_buckets"]]
    return ",".join(shapes)


def sim_mismatches(k: dict, wl: dict) -> list[str]:
    """Where the Sim would not model what the engine runs for knobs `k` on workload `wl`.

    The engine caps a chunk at the largest prefill bucket (`min(remaining, budget -
    decode rows, largest bucket)`); the Sim's chunked-prefill worker does not, and runs
    a chunk above the largest bucket as several bucket calls in one step. The two agree
    only when the budget is at most the largest bucket or every prompt fits one bucket."""
    big = max(k["prefill_buckets"])
    if k["max_batch_tokens"] > big and wl["input_len"] > big:
        return [
            f"max_batch_tokens {k['max_batch_tokens']} > largest prefill bucket {big} with "
            f"{wl['input_len']}-token prompts: the Sim does not cap chunks at the bucket"
        ]
    return []


# ------------------------------------------------------------------------------ propose
def parse_override(text: str) -> tuple[str, dict]:
    """`label:knob=v,knob=v` (values JSON, else strings) -> (label, knobs)."""
    label, _, body = text.partition(":")
    knobs = {}
    for item in filter(None, _split_top(body)):
        key, _, v = item.partition("=")
        try:
            knobs[key] = json.loads(v)
        except json.JSONDecodeError:
            knobs[key] = v
    return label, knobs


def _split_top(text: str) -> list[str]:
    """Split on commas outside brackets."""
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "[{"
        depth -= ch in "]}"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def search_rows(exps: list[Path], workload: str, metric: str = "total_tok_s") -> list[dict]:
    """Finished base-scenario rows of `workload` (`fixed_2k_512` or `fixed_2k_512-a0.67`)
    from the experiments' results.jsonl, best `metric` first."""
    rows = []
    for exp in exps:
        f = exp / "results.jsonl"
        if not f.is_file():
            raise FileNotFoundError(f"{f}: run `search.py collect` first")
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("cfg.scenario", "base") != "base":
                continue
            if workload in (r["workload"], r.get("workload_base")) and r.get(metric) is not None:
                rows.append(dict(r, _exp=str(exp)))
    rows.sort(key=lambda r: -r[metric])
    return rows


def row_config(r: dict) -> dict:
    cfg = {k[4:]: v for k, v in r.items() if k.startswith("cfg.")}
    cfg.setdefault("label", r.get("label"))
    return cfg


def propose(
    loop: Loop,
    exps: list[Path],
    workload: str,
    top: int,
    variants: list[str],
    contrasts: list[str],
    explicit: list[str],
    metric: str = "total_tok_s",
    accept_rate: float | None = None,
    engine_workload: str = "w64",
) -> list[dict]:
    """Pick the candidates: the top-`top` engine-runnable configs, `variants`
    (`label:knob=v,...` applied to the best), `contrasts` (the same, marked as the
    expected loser) and `explicit` configs (`label:knob=v,...` from scratch)."""
    cands, skipped = [], []
    rows = search_rows(exps, workload, metric) if exps else []
    seen = set()
    best = None
    for r in rows:
        cfg = row_config(r)
        k = engine_knobs(cfg)
        key = json.dumps(k, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        why = engine_blockers(k)
        if why:
            skipped.append(dict(label=cfg.get("label"), blockers=why))
            continue
        if best is None:
            best = (cfg, r)
        if len([c for c in cands if c["role"] == "top"]) < top:
            n = len(cands) + 1
            cands.append(
                dict(
                    label=f"top{n}",
                    role="top",
                    family=cfg.get("family", "unified"),
                    knobs=k,
                    accept_rate=accept_rate or cfg.get("accept_rate"),
                    search=dict(
                        exp=r["_exp"],
                        label=cfg.get("label"),
                        workload=r["workload"],
                        **{m: r.get(m) for m in (metric, "ttft_p50", "tpot_p50")},
                    ),
                )
            )
    for spec, role in [(v, "variant") for v in variants] + [(c, "contrast") for c in contrasts]:
        label, over = parse_override(spec)
        if best is None:
            raise ValueError("variants / contrasts need a search experiment with a runnable best")
        cfg = dict(best[0], **over)
        if "max_batch_tokens" in over and "prefill_buckets" not in over:
            cfg.pop("prefill_buckets", None)  # re-derive the buckets from the new budget
        cands.append(
            dict(
                label=label,
                role=role,
                family=_family(cfg),
                knobs=engine_knobs(cfg),
                accept_rate=accept_rate or cfg.get("accept_rate"),
                search=dict(base=best[0].get("label"), override=over),
            )
        )
    for spec in explicit:
        label, cfg = parse_override(spec)
        cands.append(
            dict(
                label=label,
                role=cfg.pop("role", "explicit"),
                family=_family(cfg),
                knobs=engine_knobs(cfg),
                accept_rate=accept_rate or cfg.get("accept_rate"),
                search=None,
            )
        )
    for c in cands:
        if c["family"] == "mtp" and not c["knobs"]["draft_tokens"]:
            c["family"] = "unified"
        if c["knobs"]["draft_tokens"] and c["accept_rate"] is None:
            raise ValueError(f"{c['label']}: MTP candidate needs --accept-rate")
        c["blockers"] = engine_blockers(c["knobs"]) + sim_mismatches(
            c["knobs"], WORKLOADS[engine_workload]
        )
        c["engine_workload"] = engine_workload
    bad = [c for c in cands if c["blockers"]]
    if bad:
        raise ValueError(
            "not engine-runnable: " + "; ".join(f"{c['label']}: {c['blockers']}" for c in bad)
        )
    loop.write(
        "proposal.json",
        dict(
            workload=workload,
            metric=metric,
            exps=[str(e) for e in exps],
            candidates=cands,
            skipped=skipped[:50],
        ),
    )
    loop.mark("propose", candidates=[c["label"] for c in cands])
    loop.log(
        f"propose: {len(cands)} candidates ({', '.join(c['label'] for c in cands)}); "
        f"{len(skipped)} search configs not engine-runnable"
    )
    return cands


def _family(cfg: dict) -> str:
    return "mtp" if cfg.get("draft_tokens") else "unified"


# ------------------------------------------------------------------------------ predict
def accept_draw(files: list[str], k: int, logs: Path | None = None) -> list[list[float]]:
    """Per-request MTP acceptance (per draft position) of a previous engine run, in rid
    order: an empirical draw of the acceptance distribution. On the seeded random
    prompts the per-request values do not carry over between runs (correlation <= 0.26
    over the W64 k=1 runs), but their spread does, and duration follows the slowest
    requests: a uniform mean predicts W64 k=1 ~13% short."""
    _, reqs = ea.measured(files, logs or ENGINE_ROOT / "logs" / "engine")
    return [ea.acceptance(r, k) for r in sorted(reqs, key=lambda r: r["rid"])]


def _accept_cell(a) -> str:
    if isinstance(a, list):
        return f"{a[0]:.4f}" if len(a) == 1 else '"' + json.dumps([round(x, 4) for x in a]) + '"'
    return f"{a}"


def write_trace(path: Path, wl: dict, accept) -> Path:
    """The workload's trace; `accept` is one rate for every request, a per-request list
    (cycled), or None (no MTP)."""
    lines = ["id,input_len,output_len,arrival_time" + (",accept_rate" if accept else "")]
    for i in range(wl["requests"]):
        line = f"{i},{wl['input_len']},{wl['output_len']},{i * wl['interval_s'] * 1e3:.6f}"
        if accept:
            line += "," + _accept_cell(
                accept[i % len(accept)] if isinstance(accept, list) else accept
            )
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    return path


def render_preset(cand: dict, out_dir: Path) -> Path:
    """A one-run unified preset with the candidate's engine knobs and workload, laid
    out as `engine_align.render_preset` lays out a measured run."""
    k = dict(cand["knobs"])
    spec = cand["family"] == "mtp"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = write_trace(
        out_dir / "trace.csv",
        WORKLOADS[cand["engine_workload"]],
        (cand.get("accept_draw") or cand["accept_rate"]) if spec else None,
    )
    replicas = k.pop("replicas")
    worker_keys = {
        "max_batch_tokens",
        "max_prefill_chunks_per_partition",
        "dp_placement",
        "attn_gpu_memory_gb",
    }
    worker = dict(
        type="speculative" if spec else "chunked_prefill",
        attn_gpu_memory_gb=CORE_HBM_GB,
        max_prefill_chunks_per_partition=1,
    )
    arch = dict(
        type="qwen4_exp_neuron_bf16_speculative" if spec else "qwen4_exp_neuron_bf16",
        model_config=MODEL_CONFIG,
        fp8=False,
    )
    for key, v in k.items():
        (worker if key in worker_keys else arch)[key] = v
    if spec:
        worker["draft_tokens"] = arch["draft_tokens"]
    else:
        arch.pop("draft_tokens", None)
        worker.setdefault("dp_placement", "vllm-least-loaded")
    workload = dict(
        trace_files=[str(trace)],
        input_file_format="text-generation-independent",
        arrival_mode="trace_timed",
        run_to_end=True,
        request_rate=1.0,
    )
    if spec:
        workload["input_file_tags"] = ["speculative"]
    cfg = dict(
        deployment="unified",
        workload=workload,
        io=dict(log_dir=str(out_dir / "run")),
        pools=dict(
            main=dict(
                placement="round-robin",  # the engine splits replicas by rid parity
                groups=[dict(gpu=GPU, replicas=replicas, arch=arch, worker=worker)],
            )
        ),
    )
    p = out_dir / "preset.json"
    p.write_text(json.dumps(cfg, indent=1))
    return p


def run_sim(preset: Path, analyze: bool) -> None:
    env = dict(os.environ, SERVINGSTUDIO_NO_GPU="1", UV_NO_SYNC="1")
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "launcher",
        str(preset),
        "--no-gpu",
        "--refresh",
    ]
    if not analyze:
        cmd.append("--no-analyze")
    r = subprocess.run(cmd, cwd=SIM, env=env, capture_output=True, text=True)
    (preset.parent / "launcher.log").write_text(r.stdout[-20000:] + "\n" + r.stderr[-20000:])
    if r.returncode:
        raise RuntimeError(f"simulation {preset} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def ladder(run_dir: Path) -> dict | None:
    """The R0-R7 cluster ladder and gap buckets of a Sim run (None without analysis)."""
    p = run_dir / "reports" / "optimality_report.json"
    if not p.is_file():
        return None
    cl = json.loads(p.read_text()).get("cluster") or {}
    from trn2_search.collect import BUCKET_KEYS, LADDER_KEYS

    return dict(
        ladder={k: cl.get(k) for k in LADDER_KEYS},
        buckets={k: (cl.get("buckets") or {}).get(k) for k in BUCKET_KEYS},
    )


def pred_file(tag: str | None) -> str:
    return f"predictions_{tag}.json" if tag else "predictions.json"


def predict(
    loop: Loop,
    analyze: bool = True,
    force: bool = False,
    simulate=None,
    tag: str | None = None,
    accept_from: list[str] | None = None,
    logs: Path | None = None,
) -> dict:
    """Simulate every proposed candidate; write predictions[_<tag>].json (resumable per
    candidate). `accept_from` replaces each MTP candidate's uniform acceptance by the
    per-request draw of those earlier engine runs (`accept_draw`)."""
    simulate = simulate or run_sim
    prop = loop.read("proposal.json")
    pf = loop.path(pred_file(tag))
    preds = json.loads(pf.read_text()) if pf.is_file() and not force else {}
    preds.setdefault("candidates", {})
    for c in prop["candidates"]:
        if c["label"] in preds["candidates"]:
            continue
        d = loop.path(f"sim_{tag}" if tag else "sim", c["label"])
        if accept_from and c["knobs"]["draft_tokens"]:
            c = dict(c, accept_draw=accept_draw(accept_from, c["knobs"]["draft_tokens"], logs))
        preset = render_preset(c, d)
        loop.log(f"predict {c['label']}: simulating {_rel(preset)}")
        simulate(preset, analyze)
        m = ea.simulated(d / "run")
        preds["candidates"][c["label"]] = dict(
            role=c["role"],
            family=c["family"],
            knobs=c["knobs"],
            accept_rate=c["accept_rate"],
            accept_from=accept_from if c.get("accept_draw") else None,
            predicted=m,
            ladder=ladder(d / "run"),
        )
        loop.write(pred_file(tag), preds)
        loop.log(
            f"predict{'[' + tag + ']' if tag else ''} {c['label']}: "
            + ", ".join(f"{k} {v:.2f}" for k, v in m.items())
        )
    preds["inputs"] = dict(
        workload=prop["candidates"][0]["engine_workload"] if prop["candidates"] else None,
        sim_commit=_git_head(),
        uncertain=[
            "MTP acceptance is an input (per candidate accept_rate), not predicted",
            "moe_a2a_pack_us_per_row 0.8 is fitted on one engine run (all_to_all candidates)",
            "the engine's first-use stalls (2-7 s per run) are not modeled",
        ],
        accept_from=accept_from,
        made=dt.datetime.now().isoformat(timespec="seconds"),
    )
    loop.write(pred_file(tag), preds)
    loop.mark("predict" + (f"_{tag}" if tag else ""), n=len(preds["candidates"]))
    return preds


def _git_head() -> str | None:
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=SIM, capture_output=True, text=True
    )
    return r.stdout.strip() or None


# ----------------------------------------------------------------------------------- run
PORTS = (61000, 63000)  # NEURON_RT_ROOT_COMM_ID per replica process group (>= 1000 apart)
DIST_PORTS = (29900, 29990)


def run_name(loop: Loop, label: str) -> str:
    return f"tune_{loop.name}_{label}"


def engine_script(loop: Loop, cand: dict) -> str:
    """The bash script that serves one candidate on the engine (both replicas)."""
    k = cand["knobs"]
    name = run_name(loop, cand["label"])
    wl = WORKLOADS[cand["engine_workload"]]
    common = (
        f"{wl['engine_args']} {serve_args(k)} --warmup-shapes {warmup_shapes(k)} "
        "--profile-window 150:190"
    )
    lines = [
        "#!/bin/bash",
        f"# tune loop {loop.name}, candidate {cand['label']} ({cand['role']}); by tune.py run",
        "set -u",
        f"cd {ENGINE_ROOT}",
    ]
    reps = k["replicas"]
    if reps == 1:
        lines.append(
            f"scripts/slice_run.sh {PORTS[0]} logs/engine/serve_{name}.log python -m "
            f"flashnext.neuron.engine.serve --cores 0-31 --port {DIST_PORTS[0]} --name {name} "
            f"{common}"
        )
    else:
        lines.append(f"rm -rf /scratch/tmp/sync_{name}")
        common += f" --replicas {reps} --sync-dir /scratch/tmp/sync_{name}"
        for i in range(reps):
            cores = f"{i * REPLICA_CORES}-{(i + 1) * REPLICA_CORES - 1}"
            lines.append(
                f"scripts/slice_run.sh {PORTS[i]} logs/engine/serve_{name}_r{i}.log python -m "
                f"flashnext.neuron.engine.serve --cores {cores} --port {DIST_PORTS[i]} "
                f"--name {name}_r{i} --replica {i} {common} &"
            )
        lines.append("wait")
    return "\n".join(lines) + "\n"


def engine_files(loop: Loop, cand: dict) -> list[str]:
    name = run_name(loop, cand["label"])
    reps = cand["knobs"]["replicas"]
    return [name] if reps == 1 else [f"{name}_r{i}" for i in range(reps)]


def engine_finished(files: list[str], logs: Path) -> bool:
    for f in files:
        p = logs / f"serve_{f}.jsonl"
        if not p.is_file():
            return False
        lines = p.read_text().splitlines()
        if not lines or json.loads(lines[-1]).get("type") != "summary":
            return False
    return True


def busy_containers() -> list[str]:
    """Running engine containers (`slice-<port>`); a run must not share the device."""
    r = subprocess.run(
        ["sudo", "-n", "docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    )
    return [n for n in r.stdout.split() if n.startswith("slice-")]


def run(
    loop: Loop,
    execute: bool,
    only: list[str] | None = None,
    logs: Path | None = None,
    check_device=busy_containers,
) -> list[dict]:
    """Write each candidate's engine script; with `execute`, run the unfinished ones in
    order (blocking). Predictions must exist first: the loop is blind by construction."""
    if not loop.path("predictions.json").is_file():
        raise RuntimeError(
            "predict first: engine runs are compared against predictions made before them"
        )
    logs = logs or ENGINE_ROOT / "logs" / "engine"
    prop = loop.read("proposal.json")
    out = []
    for c in prop["candidates"]:
        if only and c["label"] not in only:
            continue
        script = loop.path("engine", f"{c['label']}.sh")
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(engine_script(loop, c))
        script.chmod(0o755)
        files = engine_files(loop, c)
        status = "finished" if engine_finished(files, logs) else "pending"
        if execute and status == "pending":
            busy = check_device()
            if busy:
                raise RuntimeError(f"engine containers running: {busy}; stop them or wait")
            loop.log(f"run {c['label']}: {_rel(script)}")
            r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
            status = "finished" if engine_finished(files, logs) else f"failed (exit {r.returncode})"
            loop.log(f"run {c['label']}: {status}")
        out.append(dict(label=c["label"], script=str(script), files=files, status=status))
    has = loop.path("runs.json").is_file()
    prev = {r["label"]: r for r in loop.read("runs.json")} if has else {}
    prev.update({r["label"]: r for r in out})
    every = [prev[c["label"]] for c in prop["candidates"] if c["label"] in prev]
    loop.write("runs.json", every)
    if len(every) == len(prop["candidates"]) and all(r["status"] == "finished" for r in every):
        loop.mark("run", n=len(every))
    return out


# ------------------------------------------------------------------------------ compare
def kendall_tau(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return None
    s = sum(
        (1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1)
        for i, j in pairs
        if (a[i] - a[j]) * (b[i] - b[j]) != 0
    )
    return s / len(pairs)


def discordant_pairs(
    labels: list[str], meas: list[float], pred: list[float], tie_pct: float = TIE_PCT
) -> list[tuple[str, str]]:
    """Pairs ordered differently measured and predicted, ignoring measured ties."""
    out = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            gap = abs(meas[i] - meas[j]) / min(meas[i], meas[j]) * 100
            if gap >= tie_pct and (meas[i] - meas[j]) * (pred[i] - pred[j]) < 0:
                out.append((labels[i], labels[j]))
    return out


STALL_STEPS = 3  # the engine's first executions after warm-up (ALIGNMENT.md: steps 0-2)


def first_use_stall_s(files: list[str], logs: Path) -> float:
    """The engine's first-use stalls, read off the timeline alone: each of the first
    `STALL_STEPS` steps' wall above the median wall of the later steps with the same
    shape (kind, padded batch, chunk, contexts). The max over replicas (the run's
    duration is the last replica's finish). The Sim does not model them."""
    worst = 0.0
    for f in files:
        steps = [r for r in ea.load(f, logs) if r["type"] == "step"]

        def shape(st):
            ctx = tuple(fw.get("C") for fw in st["forwards"])
            return (st["kind"], st.get("decode_B"), st.get("prefill_T"), ctx)

        stall = 0.0
        for st in steps[:STALL_STEPS]:
            same = [x["wall_ms"] for x in steps[STALL_STEPS:] if shape(x) == shape(st)]
            if same:
                stall += max(0.0, st["wall_ms"] - sorted(same)[len(same) // 2])
        worst = max(worst, stall / 1e3)
    return worst


def compare(
    loop: Loop,
    threshold: float = 15.0,
    logs: Path | None = None,
    resim: bool = False,
    resim_fn=None,
    tag: str | None = None,
) -> dict:
    """Predicted vs measured for every finished candidate, ranking agreement, verdict.

    Duration and out tok/s are compared against the measured run without its
    first-use stalls (`first_use_stall_s`; the raw errors are kept as `err_raw`).
    `resim` re-simulates each measured run from its own header and per-request
    acceptance (`engine_align.e2e_run`): the error left once the inputs are the
    measured ones."""
    logs = logs or ENGINE_ROOT / "logs" / "engine"
    preds = loop.read(pred_file(tag))["candidates"]
    prop = {c["label"]: c for c in loop.read("proposal.json")["candidates"]}
    rows = []
    for label, p in preds.items():
        files = engine_files(loop, prop[label])
        if not engine_finished(files, logs):
            continue
        raw, reqs = ea.measured(files, logs)
        stall = first_use_stall_s(files, logs)
        meas = dict(raw, duration_s=raw["duration_s"] - stall)
        meas["out_tok_s"] = raw["out_tok_s"] * raw["duration_s"] / meas["duration_s"]
        acc = None
        if p["knobs"]["draft_tokens"]:
            vals = [ea.acceptance(r, p["knobs"]["draft_tokens"])[0] for r in reqs if r["itl_ms"]]
            acc = sum(vals) / len(vals) if vals else None
        row = dict(
            label=label,
            role=p["role"],
            predicted=p["predicted"],
            measured=meas,
            measured_raw=raw,
            stall_s=stall,
            err={m: ea.error_pct(p["predicted"][m], meas[m]) for m in METRICS},
            err_raw={m: ea.error_pct(p["predicted"][m], raw[m]) for m in METRICS},
            accept_pred=p["accept_rate"],
            accept_meas=acc,
        )
        if resim:
            fn = resim_fn or (lambda lab, fs: ea.e2e_run(lab, fs, {}, {}, loop.path("resim"), logs))
            row["resim"] = fn(label, files)["simulated"]
            row["resim_err"] = {m: ea.error_pct(row["resim"][m], meas[m]) for m in METRICS}
            row["resim_err_raw"] = {m: ea.error_pct(row["resim"][m], raw[m]) for m in METRICS}
        rows.append(row)
    labels = [r["label"] for r in rows]
    rank = {}
    for m, better_low in (("duration_s", True), ("out_tok_s", False), ("tpot_p50_ms", True)):
        mv = [r["measured"][m] for r in rows]
        pv = [r["predicted"][m] for r in rows]

        def order(v, lo=better_low):
            return [labels[i] for i in sorted(range(len(v)), key=lambda i: v[i] if lo else -v[i])]

        rank[m] = dict(
            measured=order(mv),
            predicted=order(pv),
            kendall_tau=kendall_tau(mv, pv),
            discordant=discordant_pairs(labels, mv, pv),
            best_agrees=bool(rows) and order(mv)[0] == order(pv)[0],
        )
    over = [
        (r["label"], m, r["err"][m]) for r in rows for m in GATED if abs(r["err"][m]) > threshold
    ]
    warn = [
        (r["label"], m, r["err"][m])
        for r in rows
        for m in METRICS
        if m not in GATED and abs(r["err"][m]) > threshold
    ]
    realign = bool(over or rank["duration_s"]["discordant"])
    result = dict(
        threshold_pct=threshold,
        rows=rows,
        ranking=rank,
        over_threshold=over,
        warnings=warn,
        verdict="realign" if realign else "trust",
        pending=[lab for lab in preds if lab not in labels],
    )
    sfx = f"_{tag}" if tag else ""
    loop.write(f"comparison{sfx}.json", result)
    loop.path(f"comparison{sfx}.md").write_text(markdown(result) + "\n")
    loop.mark("compare" + sfx, verdict=result["verdict"], n=len(rows))
    loop.log(
        f"compare: {len(rows)} measured, verdict {result['verdict']}"
        + (f"; over {threshold}%: {over}" if over else "")
    )
    return result


def markdown(res: dict) -> str:
    cols = ["candidate", "role", "duration raw (err)", "stalls (s)"]
    cols += [f"{m} pred / meas (err)" for m in METRICS] + ["accept pred / meas"]
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for r in res["rows"]:
        raw = r.get("measured_raw", r["measured"])["duration_s"]
        e_raw = r.get("err_raw", r["err"])["duration_s"]
        cells = [r["label"], r["role"], f"{raw:.1f} ({e_raw:+.1f}%)", f"{r.get('stall_s', 0):.1f}"]
        for m in METRICS:
            cells.append(f"{r['predicted'][m]:.1f} / {r['measured'][m]:.1f} ({r['err'][m]:+.1f}%)")
        a = r["accept_pred"]
        cells.append(
            f"{a:.2f} / {r['accept_meas']:.2f}" if a and r["accept_meas"] is not None else "-"
        )
        lines.append("| " + " | ".join(cells) + " |")
    if any("resim" in r for r in res["rows"]):
        lines += [
            "",
            "Re-simulated with each run's measured acceptance and arrivals (errors against "
            "the run without stalls):",
            "",
            "| candidate | " + " | ".join(f"{m} resim (err)" for m in METRICS) + " |",
            "|---" * (len(METRICS) + 1) + "|",
        ]
        for r in res["rows"]:
            lines.append(
                f"| {r['label']} | "
                + " | ".join(f"{r['resim'][m]:.1f} ({r['resim_err'][m]:+.1f}%)" for m in METRICS)
                + " |"
            )
    lines.append("")
    for m, rk in res["ranking"].items():
        tau = rk["kendall_tau"]
        lines.append(
            f"- {m}: measured {' < '.join(rk['measured'])}; predicted "
            f"{' < '.join(rk['predicted'])}; tau {tau if tau is None else round(tau, 2)}; "
            f"best agrees {rk['best_agrees']}; discordant {rk['discordant'] or 'none'}"
        )
    lines.append(
        f"- verdict: **{res['verdict']}** (threshold {res['threshold_pct']}% on "
        f"{', '.join(GATED)}; TTFT warns only; duration and out tok/s without the engine's "
        "first-use stalls)"
    )
    if res["over_threshold"]:
        lines.append(f"- over threshold: {res['over_threshold']}")
    if res["warnings"]:
        lines.append(f"- warnings: {res['warnings']}")
    if res["pending"]:
        lines.append(f"- not measured yet: {', '.join(res['pending'])}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- headroom
# What-ifs the engine would have to build (CONFIG_SEARCH.md stage 4), applied to one
# candidate's exact knobs. Each is Sim-only by construction.
SCENARIOS = {
    "moe_prefill_3x": {
        "kind_time_scale": ["moe_block@256=0.3333", "moe_routed_experts@256=0.3333"]
    },
    "no_pad": {"spmd_padding": False},
    "host_half": {"host_time_scale": 0.5},
    "fused": {"step_forwards": "fused"},
    "multi_chunk": {"max_prefill_chunks_per_partition": 16},
    "a2a_cf125": {"moe_dispatch": "all_to_all", "moe_a2a_capacity_factor": 1.25},
    "moe_prefill_3x+no_pad": {
        "kind_time_scale": ["moe_block@256=0.3333", "moe_routed_experts@256=0.3333"],
        "spmd_padding": False,
    },
}
# R0-R7 rungs of the analyzer's cluster ladder (GPU-seconds summed over ranks).
RUNGS = (
    ("R0 real", "real"),
    ("R1 busy", "busy"),
    ("R2 balanced", "balanced"),
    ("R3 per-config best", "per_config_best"),
    ("R4 ignore network", "ignore_network"),
    ("R4a/b sustained / achievable", "achievable_limit"),
    ("R5 hardware limit", "hardware_limit"),
    ("R6/R7 necessary work", "segmented_necessary"),
)


def headroom(
    loop: Loop, label: str, scenarios: dict | None = None, simulate=None, analyze: bool = True
) -> dict:
    """The candidate's R0-R7 ladder and its predicted gain under each what-if."""
    simulate = simulate or run_sim
    scenarios = scenarios or SCENARIOS
    cand = next(c for c in loop.read("proposal.json")["candidates"] if c["label"] == label)
    base = loop.read("predictions.json")["candidates"][label]
    out = dict(label=label, base=base["predicted"], ladder=base["ladder"], scenarios={})
    for name, over in scenarios.items():
        d = loop.path("headroom", label, name)
        if (d / "run" / "summary.json").is_file():
            m = ea.simulated(d / "run")
        else:
            c = dict(cand, knobs=dict(cand["knobs"], **over))
            simulate(render_preset(c, d), False)
            m = ea.simulated(d / "run")
        gain = {k: ea.error_pct(m[k], base["predicted"][k]) for k in METRICS}
        out["scenarios"][name] = dict(override=over, predicted=m, change_pct=gain)
        loop.log(f"headroom {label} {name}: out tok/s {gain['out_tok_s']:+.1f}%")
    loop.write(f"headroom_{label}.json", out)
    loop.path(f"headroom_{label}.md").write_text(headroom_markdown(out) + "\n")
    loop.mark(f"headroom_{label}", n=len(scenarios))
    return out


def headroom_markdown(h: dict) -> str:
    lines = []
    lad = (h.get("ladder") or {}).get("ladder")
    if lad:
        r0 = lad["real"]
        lines += ["| Rung | GPU-s (all ranks) | of R0 |", "|---|---|---|"]
        for name, key in RUNGS:
            v = lad.get(key)
            if v is not None:
                lines.append(f"| {name} | {v:.0f} | {v / r0:.1%} |")
        b = h["ladder"]["buckets"]
        lines += ["", "| Gap bucket | share of R0 |", "|---|---|"]
        for k, v in sorted(b.items(), key=lambda kv: -((kv[1] or {}).get("frac") or 0)):
            if v and v.get("frac"):
                lines.append(f"| {k} | {v['frac']:.1%} |")
        lines.append("")
    lines += [
        "| What-if | duration (s) | out tok/s | TTFT p50 (s) | TPOT p50 (ms) |",
        "|---|---|---|---|---|",
    ]
    b = h["base"]
    lines.append(
        f"| base | {b['duration_s']:.1f} | {b['out_tok_s']:.0f} | {b['ttft_p50_s']:.1f} | "
        f"{b['tpot_p50_ms']:.1f} |"
    )
    ranked = sorted(h["scenarios"].items(), key=lambda kv: -kv[1]["change_pct"]["out_tok_s"])
    for name, s in ranked:
        m, g = s["predicted"], s["change_pct"]
        lines.append(
            f"| {name} | {m['duration_s']:.1f} ({g['duration_s']:+.0f}%) | {m['out_tok_s']:.0f} "
            f"({g['out_tok_s']:+.0f}%) | {m['ttft_p50_s']:.1f} ({g['ttft_p50_s']:+.0f}%) | "
            f"{m['tpot_p50_ms']:.1f} ({g['tpot_p50_ms']:+.0f}%) |"
        )
    return "\n".join(lines)
