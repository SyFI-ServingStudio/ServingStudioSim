"""CLI: the Sim <-> flashnext-neuron engine tuning loop (`tune_loop.py`, README.md).

    tune.py propose <name> --exp DIR [...] --workload W [--top N] [--variant L:k=v,..]
                           [--contrast L:k=v,..] [--config L:k=v,..] [--accept-rate A]
    tune.py predict <name> [--no-analyze] [--force]
    tune.py run     <name> [--execute] [--only LABEL ...]
    tune.py compare <name> [--threshold 15] [--resim]
    tune.py headroom <name> <label> [--scenario S ...]
    tune.py status  <name>

State and logs: logs/trn2/tune/<name>/ (tune.log, state.json, proposal.json,
predictions.json, engine/<label>.sh, runs.json, comparison.{json,md})."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tune_loop as tl  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("propose")
    p.add_argument("name")
    p.add_argument("--exp", nargs="*", type=Path, default=[], help="config-search experiment dirs")
    p.add_argument(
        "--workload",
        default="fixed_2k_512",
        help="search workload label (e.g. fixed_2k_512-a0.67 for MTP rows)",
    )
    p.add_argument("--metric", default="total_tok_s")
    p.add_argument("--top", type=int, default=2)
    p.add_argument("--variant", action="append", default=[], help="label:knob=v,... on the best")
    p.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="label:knob=v,... on the best, expected to lose",
    )
    p.add_argument("--config", action="append", default=[], help="label:knob=v,... from scratch")
    p.add_argument(
        "--accept-rate",
        type=float,
        default=None,
        help="MTP acceptance input of every candidate (default: the search row's)",
    )
    p.add_argument("--engine-workload", default="w64", choices=sorted(tl.WORKLOADS))
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("name")
    p.add_argument("--no-analyze", action="store_true", help="skip the R0-R7 ladder")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("--execute", action="store_true", help="run on the device (blocking; use tmux)")
    p.add_argument("--only", nargs="*")
    p = sub.add_parser("compare")
    p.add_argument("name")
    p.add_argument("--threshold", type=float, default=15.0)
    p.add_argument(
        "--resim",
        action="store_true",
        help="also re-simulate each run with its measured acceptance",
    )
    p = sub.add_parser("headroom")
    p.add_argument("name")
    p.add_argument("label", help="candidate (usually the best measured)")
    p.add_argument("--scenario", nargs="*", help=f"subset of {sorted(tl.SCENARIOS)}")
    p = sub.add_parser("status")
    p.add_argument("name")
    a = ap.parse_args(argv)
    loop = tl.Loop(a.name)
    if a.step == "propose":
        if loop.done("propose") and not a.force:
            print(f"propose already done ({loop.path('proposal.json')}); --force to redo")
            return 0
        exps = [e if e.is_absolute() else tl.SIM / e for e in a.exp]
        tl.propose(
            loop,
            exps,
            a.workload,
            a.top,
            a.variant,
            a.contrast,
            a.config,
            a.metric,
            a.accept_rate,
            a.engine_workload,
        )
    elif a.step == "predict":
        preds = tl.predict(loop, analyze=not a.no_analyze, force=a.force)
        for lab, p in preds["candidates"].items():
            print(lab, json.dumps({k: round(v, 2) for k, v in p["predicted"].items()}))
    elif a.step == "run":
        for r in tl.run(loop, a.execute, a.only):
            print(f"{r['label']}: {r['status']}  bash {r['script']}")
    elif a.step == "compare":
        res = tl.compare(loop, a.threshold, resim=a.resim)
        print(tl.markdown(res))
    elif a.step == "headroom":
        sc = {k: tl.SCENARIOS[k] for k in a.scenario} if a.scenario else None
        print(tl.headroom_markdown(tl.headroom(loop, a.label, sc)))
    else:
        print(json.dumps(loop.state(), indent=1))
        if loop.path("runs.json").is_file():
            for r in loop.read("runs.json"):
                print(f"{r['label']}: {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
