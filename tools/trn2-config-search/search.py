"""Config search for Qwen3.8-Flash-Next BF16 on one trn2.48xlarge (CPU only).

    uv run --no-sync python tools/trn2-config-search/search.py <command> ...

Commands (an experiment directory EXP holds every artifact of one stage):

    plan STAGE.yaml EXP     expand the stage into EXP/configs.jsonl, write traces
    probe EXP               dry-run every config -> EXP/probe.jsonl (ok / invalid / needs_profiling)
    run EXP                 launcher presets for the ok configs, then run them
                            (--no-gpu --no-plot, --parallelism N); resumable
    collect EXP             EXP/results.jsonl (+ .csv) with SLO goodput and the ladder
    report EXP              EXP/report.md: frontiers, best configs, sensitivity, what-ifs
    top EXP OUT.yaml        the best configs per workload as `seeds` for a next stage

See tools/trn2-config-search/README.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from trn2_search import report as rp  # noqa: E402
from trn2_search.collect import collect  # noqa: E402
from trn2_search.presets import render, split_by_keyset  # noqa: E402
from trn2_search.space import expand_stage, knobs  # noqa: E402
from trn2_search.workloads import from_spec, trace_path  # noqa: E402

DEFAULT_SLOS = {
    # name: (TTFT ms, mean TPOT ms). Override or add with --slo name=TTFT,TPOT.
    "chat": (2_000.0, 100.0),
    "chat_strict": (1_000.0, 60.0),
    "agentic": (30_000.0, 150.0),
    "agentic_relaxed": (60_000.0, 250.0),
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def _stage(exp: Path) -> dict:
    return yaml.safe_load((exp / "stage.yaml").read_text())


def _workloads(stage: dict, exp: Path) -> list[tuple[str, str]]:
    """`(label, trace path)` per workload (x accept rate for the mtp family)."""
    specs_file = stage.get("workloads_file")
    specs = {}
    if specs_file:
        for s in yaml.safe_load((ROOT / specs_file).read_text())["workloads"]:
            specs[s["name"]] = s
    for s in stage.get("workload_specs", []):
        specs[s["name"]] = s
    names = stage.get("workloads") or list(specs)
    rates = stage.get("accept_rates") if stage.get("family") == "mtp" else None
    out = []
    for name in names:
        w = from_spec(specs[name])
        for a in rates or [None]:
            label = name if a is None else f"{name}-a{a}"
            path = trace_path(w, exp / "traces", a)
            out.append((label, os.path.relpath(path, ROOT)))
    return out


def cmd_plan(args) -> int:
    stage_path, exp = Path(args.stage), Path(args.exp)
    stage = yaml.safe_load(stage_path.read_text())
    seeds = stage.get("seeds_file")
    if seeds:
        stage["seeds"] = yaml.safe_load((ROOT / seeds).read_text())["seeds"] + stage.get(
            "seeds", []
        )
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "stage.yaml").write_text(yaml.safe_dump(stage, sort_keys=False))
    configs = expand_stage(stage)
    _write_jsonl(exp / "configs.jsonl", configs)
    wls = _workloads(stage, exp)
    print(f"[plan] {len(configs)} configs x {len(wls)} workloads -> {exp}/configs.jsonl")
    return 0


def cmd_probe(args) -> int:
    from trn2_search.probe import probe_configs

    exp = Path(args.exp)
    configs = _jsonl(exp / "configs.jsonl")
    done = {}
    if (exp / "probe.jsonl").is_file() and not args.refresh:
        done = {r["label"]: r for r in _jsonl(exp / "probe.jsonl")}
    todo = [c for c in configs if c["label"] not in done]
    wl = _workloads(_stage(exp), exp)[0]
    res = probe_configs(todo, wl, exp / "probe", jobs=args.jobs) if todo else []
    allres = list(done.values()) + res
    _write_jsonl(exp / "probe.jsonl", allres)
    counts: dict[str, int] = {}
    for r in allres:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"[probe] {counts}")
    for r in allres:
        if r["status"] != "ok" and args.verbose:
            print(f"  {r['status']:16s} {r['label']}: {r['detail']}")
    return 0


def _ok_configs(exp: Path) -> list[dict]:
    configs = _jsonl(exp / "configs.jsonl")
    status = {r["label"]: r["status"] for r in _jsonl(exp / "probe.jsonl")}
    return [c for c in configs if status.get(c["label"]) == "ok"]


def cmd_run(args) -> int:
    exp = Path(args.exp).resolve()
    ok = _ok_configs(exp)
    wls = _workloads(_stage(exp), exp)
    pdir = exp / "presets"
    pdir.mkdir(exist_ok=True)
    for old in pdir.glob("*.yaml"):
        old.unlink()
    paths = []
    rel_exp = os.path.relpath(exp, ROOT)
    for i, (family, cfgs) in enumerate(split_by_keyset(ok)):
        name = f"p{i:02d}_{family}"
        preset = render(family, cfgs, wls, f"{rel_exp}/runs/{name}")
        path = pdir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(preset, sort_keys=False))
        paths.append(path)
    n = len(ok) * len(wls)
    print(f"[run] {len(ok)} ok configs x {len(wls)} workloads = {n} runs in {len(paths)} presets")
    if not paths:
        return 0
    # One launcher batch: every preset's runs share one parallel sweep.
    argv = [sys.executable, "-m", "launcher", *(os.path.relpath(p, ROOT) for p in paths)]
    argv += ["--no-gpu", "--no-plot", "--parallelism", str(args.parallelism)]
    if args.dry_run:
        argv.append("--dry-run")
    print("[run]", " ".join(argv), flush=True)
    env = dict(os.environ, SERVINGSTUDIO_NO_GPU="1", UV_NO_SYNC="1")
    out = subprocess.DEVNULL if args.dry_run else None
    rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=out).returncode
    return rc


def _slos(args) -> dict[str, tuple[float, float]]:
    slos = dict(DEFAULT_SLOS)
    for s in args.slo or []:
        name, vals = s.split("=", 1)
        t, p = (float(x) for x in vals.split(","))
        slos[name] = (t, p)
    return slos


def cmd_collect(args) -> int:
    exp = Path(args.exp)
    configs = {c["label"]: c for c in _jsonl(exp / "configs.jsonl")}
    rows = collect(exp / "runs", configs, _slos(args))
    _write_jsonl(exp / "results.jsonl", rows)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("cfg."), k))
    with (exp / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()})
    print(f"[collect] {len(rows)} runs -> {exp}/results.jsonl")
    return 0


def _rows(exps: list[str]) -> list[dict]:
    rows = []
    for e in exps:
        rows += _jsonl(Path(e) / "results.jsonl")
    return rows


def build_report(
    rows: list[dict], slo_map: dict[str, list[str]], metric_default: str, top_n: int
) -> str:
    """The markdown report over `rows` (one or more experiments)."""
    out = []
    probe_note = "Runs only; configs that need profiling are listed by `probe`."
    out.append(f"{len(rows)} runs. {probe_note}\n")
    for wl, wrows in rp.by_workload(rows).items():
        base = [r for r in wrows if r.get("cfg.scenario", "base") == "base"]
        out.append(f"### {wl}\n")
        slos = [s for prefix, ss in slo_map.items() if wl.startswith(prefix) for s in ss]
        hdr = [
            "config",
            "tok/s",
            "tok/s/chip",
            "TTFT p50 s",
            "TTFT p90 s",
            "TPOT p50 ms",
            "TPOT p90 ms",
        ]
        hdr += [f"goodput req/s @{s}" for s in slos]
        front = rp.pareto(base, "total_tok_s", "tpot_p50")
        out.append("Pareto frontier, throughput vs TPOT p50 (base constants):\n")
        out.append(rp.markdown_table(hdr, [_line(r, slos) for r in front]))
        front_t = rp.pareto(base, "total_tok_s", "ttft_p90")
        out.append("\nPareto frontier, throughput vs TTFT p90:\n")
        out.append(rp.markdown_table(hdr, [_line(r, slos) for r in front_t]))
        out.append("\nBest config per objective:\n")
        objs = [("max tok/s", "total_tok_s")] + [
            (f"max goodput @{s}", f"goodput_req_s@{s}") for s in slos
        ]
        brows = []
        for name, m in objs:
            b = rp.best(base, m)
            if b:
                brows.append(
                    [
                        name,
                        rp.describe(b),
                        rp.fmt(b.get(m), 3),
                        rp.fmt(b["total_tok_s"]),
                        rp.fmt(b["tpot_p50"]),
                        rp.fmt(_num_s(b["ttft_p90"]), 2),
                    ]
                )
        out.append(
            rp.markdown_table(
                ["objective", "config", "value", "tok/s", "TPOT p50 ms", "TTFT p90 s"], brows
            )
        )
        sens = rp.sensitivity(base, metric_default)
        if sens:
            out.append(
                f"\nKnob sensitivity on {metric_default}"
                " (spread of the best reachable value per knob value):\n"
            )
            out.append(
                rp.markdown_table(
                    ["knob", "spread", "best value", "best per value"],
                    [
                        [
                            s["knob"],
                            f"{100 * s['spread']:.0f}%",
                            s["best_value"],
                            "; ".join(f"{k}: {rp.fmt(v)}" for k, v in s["per_value"].items()),
                        ]
                        for s in sens
                    ],
                )
            )
        sc = rp.scenario_ranks(wrows, metric_default)
        if len(sc) > 1:
            out.append(f"\nWhat-if scenarios ({metric_default}; tau = rank agreement with base):\n")
            out.append(
                rp.markdown_table(
                    [
                        "scenario",
                        "configs",
                        "Kendall tau",
                        "best config",
                        "best value",
                        "base winner's rank",
                    ],
                    [
                        [
                            s["scenario"],
                            s["configs"],
                            rp.fmt(s["kendall_tau"], 2),
                            s["best_desc"],
                            rp.fmt(s["best_value"]),
                            s["base_best_rank_here"],
                        ]
                        for s in sc
                    ],
                )
            )
        lad = [
            r
            for r in sorted(base, key=lambda r: -float(r.get(metric_default) or 0))[:top_n]
            if r.get("ladder_real")
        ]
        if lad:
            out.append("\nHeadroom ladder of the top configs (fraction of R0 GPU-time):\n")
            names = [n for n, _ in rp.LADDER_VIEW]
            out.append(
                rp.markdown_table(
                    ["config"] + names,
                    [
                        [rp.describe(r)] + [f"{100 * v:.0f}%" for v in rp.ladder(r).values()]
                        for r in lad
                    ],
                )
            )
        out.append("")
    return "\n".join(out)


def _num_s(ms) -> float:
    try:
        return float(ms) / 1000.0
    except (TypeError, ValueError):
        return float("nan")


def _line(r: dict, slos: list[str]) -> list:
    return [
        rp.describe(r),
        rp.fmt(r["total_tok_s"]),
        rp.fmt(r["tok_s_per_chip"]),
        rp.fmt(_num_s(r["ttft_p50"]), 2),
        rp.fmt(_num_s(r["ttft_p90"]), 2),
        rp.fmt(r["tpot_p50"]),
        rp.fmt(r["tpot_p90"]),
    ] + [rp.fmt(r.get(f"goodput_req_s@{s}"), 3) for s in slos]


def _slo_map(args) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    for s in args.slo_for or [
        "fixed=chat,chat_strict",
        "chat=chat,chat_strict",
        "agentic=agentic,agentic_relaxed",
    ]:
        prefix, names = s.split("=", 1)
        m[prefix] = names.split(",")
    return m


def cmd_report(args) -> int:
    rows = _rows(args.exp)
    text = build_report(rows, _slo_map(args), args.metric, args.top)
    out = Path(args.out) if args.out else Path(args.exp[0]) / "report.md"
    out.write_text(text)
    print(f"[report] {out}")
    return 0


def cmd_top(args) -> int:
    rows = [r for r in _rows(args.exp) if r.get("cfg.scenario", "base") == "base"]
    seeds, seen = [], set()
    for wl, wrows in rp.by_workload(rows).items():
        metrics = args.metric or ["total_tok_s"]
        for m in metrics:
            ranked = sorted((r for r in wrows if r.get(m) is not None), key=lambda r: -float(r[m]))
            for r in ranked[: args.n]:
                if r["cfg.core_id"] in seen:
                    continue
                seen.add(r["cfg.core_id"])
                # `accept_rate` belongs to the workload (an MTP trace), and a
                # knob a row lacks is null in the results table.
                cfg = {
                    k[4:]: v
                    for k, v in r.items()
                    if k.startswith("cfg.") and v is not None and k != "cfg.accept_rate"
                }
                seeds.append(knobs(cfg))
    Path(args.out).write_text(yaml.safe_dump({"seeds": seeds}, sort_keys=False))
    print(f"[top] {len(seeds)} seeds -> {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("stage")
    p.add_argument("exp")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("probe")
    p.add_argument("exp")
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("run")
    p.add_argument("exp")
    p.add_argument("--parallelism", type=int, default=48)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("collect")
    p.add_argument("exp")
    p.add_argument("--slo", action="append", help="name=TTFT_ms,TPOT_ms (adds or replaces)")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("report")
    p.add_argument("exp", nargs="+")
    p.add_argument("--metric", default="total_tok_s")
    p.add_argument("--slo-for", action="append", help="workload-prefix=slo1,slo2")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("top")
    p.add_argument("exp", nargs="+")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--metric", action="append")
    p.set_defaults(fn=cmd_top)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
