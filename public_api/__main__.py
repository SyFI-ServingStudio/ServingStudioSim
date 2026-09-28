"""``uv run python -m public_api serve --bind HOST --port PORT``."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from profiling.perf_api import DB_PATH


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
    args = parser.parse_args(argv)

    import uvicorn

    from public_api.app import create_app
    from public_api.arch.library import ArchLibrary
    from public_api.kernel.library import KernelLibrary
    from public_api.kernel.sources import KernelSources

    sources = KernelSources(db_path=args.db, build_type=args.build_type)
    if not sources.binary.exists():
        parser.error(
            f"{sources.binary} is missing; run `uv run cargo build --release -p simulator`"
        )
    if not args.db.exists():
        parser.error(f"{args.db} does not exist")
    library = KernelLibrary(sources)
    archs = ArchLibrary(library)

    def warm() -> None:
        library.warm()
        archs.warm()

    # Answer requests while the caches fill; a request that comes first builds its own.
    threading.Thread(target=warm, name="warm", daemon=True).start()
    uvicorn.run(create_app(library, archs), host=args.bind, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
