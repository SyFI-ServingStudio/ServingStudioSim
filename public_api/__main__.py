"""``uv run python -m public_api serve --bind HOST --port PORT``."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from profiling.perf_api import DB_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m public_api", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve the read-only public API.")
    serve.add_argument("--bind", default="127.0.0.1", help="Address to listen on.")
    # No default: pick a free port that no other user's service claims.
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="profile.db to read, opened read-only. Defaults to the one the profiler uses.",
    )
    serve.add_argument("--build-type", default="release", help="Simulator build to introspect.")
    serve.add_argument(
        "--jobs", type=int, default=8, help="Presets built, and members checked, at once."
    )
    serve.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "public_api" / "predictions",
        help="Where predictions are kept for Read more. Inside the checkout: the "
        "Analyzer reads a prediction with the checkout that holds it.",
    )
    serve.add_argument(
        "--sims-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "public_api" / "simulations",
        help="Where simulations run and are kept for a day. Inside the checkout, like --runs-dir.",
    )
    serve.add_argument(
        "--workloads-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "public_api" / "workloads",
        help="Where uploaded workloads are kept for a day.",
    )
    # No default, as for --port; it listens on 127.0.0.1 only.
    serve.add_argument(
        "--analyzer-port", type=int, required=True, help="Port of the Analyzer it starts."
    )
    serve.add_argument(
        "--forwarded-allow-ips",
        default=None,
        help="Proxies whose X-Forwarded-For names the client the rate limits count "
        "(uvicorn's option; default 127.0.0.1).",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from alignment.load_generator.runner import TRACEGEN
    from launcher.exec import analyzer_binary_path
    from launcher.schema.loader import schema_from_dict
    from profiling.gpu_policy import disable_gpus
    from public_api import predict, simulate, workloads
    from public_api.app import create_app
    from public_api.deployments import DeploymentIndex
    from public_api.kernels import KernelLibrary, git_commit
    from public_api.sim_preset import SimIndex
    from public_api.sources import Sources

    # Every simulator and launcher call it makes reads this profile.db and
    # measures nothing: a row it lacks is an error, never a profile.
    disable_gpus()
    os.environ["VIBESIM_PROFILE_DB"] = str(args.db.resolve())
    sources = Sources(db_path=args.db, build_type=args.build_type)
    if not sources.binary.exists():
        parser.error(
            f"{sources.binary} is missing; run `uv run cargo build --release -p simulator`"
        )
    if not args.db.exists():
        parser.error(f"{args.db} does not exist")
    if not TRACEGEN.exists():
        parser.error(
            f"{TRACEGEN} is missing; run `cargo build --release --manifest-path "
            "alignment/load_generator/req-frontend/Cargo.toml --bin tracegen`"
        )
    # Built once: the presets, the binary and the captures are fixed for the
    # life of the service. Which rows each member lacks is asked of profile.db
    # now too, so a member the site offers is one it can predict.
    index = DeploymentIndex.build(
        sources.cost_trees, sources.list_params(), sim_commit=git_commit(), jobs=args.jobs
    )
    index.check(lambda member: predict.missing_specs(member, args.build_type), jobs=args.jobs)
    # Each sim member is built as a run builds it, once per capture it can replay.
    registry = schema_from_dict(sources.list_params())
    sims = SimIndex.build(index)
    sims.check(
        lambda member, capture: simulate.missing_rows(
            index, member, capture, registry, args.build_type
        ),
        jobs=args.jobs,
    )
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.sims_dir.mkdir(parents=True, exist_ok=True)
    analyzer = analyzer_binary_path(args.build_type)
    if not analyzer.exists():
        parser.error(f"{analyzer} is missing; run `uv run cargo build --release -p analyzer`")
    bind = f"127.0.0.1:{args.analyzer_port}"
    process = subprocess.Popen(
        [
            str(analyzer),
            "serve",
            "--logs-root",
            str(args.runs_dir.resolve()),
            "--logs-root",
            str(args.sims_dir.resolve()),
            "--bind",
            bind,
        ]
    )
    queue = simulate.Simulations(
        args.sims_dir,
        args.build_type,
        run_id=lambda simulation_id: simulate.analyzer_run_id(f"http://{bind}", simulation_id),
    )
    try:
        uvicorn.run(
            create_app(
                KernelLibrary(sources, index),
                args.runs_dir.resolve(),
                f"http://{bind}",
                simulate.SimulationService(
                    sims, registry, queue, workloads.Workloads(args.workloads_dir, args.build_type)
                ),
            ),
            host=args.bind,
            port=args.port,
            proxy_headers=True,
            **(
                {"forwarded_allow_ips": args.forwarded_allow_ips}
                if args.forwarded_allow_ips
                else {}
            ),
        )
    finally:
        process.terminate()
        process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
