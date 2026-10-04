# public_api

Read-only data service for the external ServingStudio site. The Intro site's
pages request `/api/public/v1/...` from their own origin, and the site proxies
those paths here: Vite's `PUBLIC_API_PROXY_TARGET` in development, the web
server in front of the site otherwise. The Analyzer serves the internal UI and
is a separate service.

The service never runs a GPU, a profiler or a simulation, and never writes
`profile.db`: every query opens it read-only.

## Run

```bash
uv run cargo build --release -p simulator   # the introspection commands it calls
uv run python -m public_api serve --bind 127.0.0.1 --port <port> [--db <profile.db>]
```

`--port` has no default: check that the port is free and not in another user's
range first. `--db` defaults to the profiler's database (`VIBESIM_PROFILE_DB` or
`profiling/profile.db`). Interactive API docs are at `/api/public/v1/docs`.

### In Docker

For a long-running service, `public_api/docker/run.sh BIND PORT` runs it in a
container that Docker restarts on failure and when the host reboots
(`--restart unless-stopped`):

```bash
uv run cargo build --release -p simulator
public_api/docker/run.sh 10.158.48.50 5220
docker logs -f servingstudio-public-api
```

The image holds only the Python environment, installed from `uv.lock`'s
`public-api` and `launcher` groups, so it is rebuilt only when the lock changes.
The container runs as the calling user and mounts the checkout read-only at the
same path, with its git directory (a worktree's lies outside it). It serves
that checkout's code, `profiling/profile.db` and `target/release/simulator`. To
serve new code or data, update the checkout, rebuild the simulator and rerun
`run.sh`, which replaces the container. `PUBLIC_API_CONTAINER` and `PUBLIC_API_IMAGE` override the
container and image names.

## Routes

All `GET`, under `/api/public/v1`.

| Route | Returns |
| --- | --- |
| `/health` | `status` and the Sim commit served |
| `/kernels` | Catalog: every registered kind with its docs summary, backends, precisions and row coverage per GPU/backend/precision; plus categories, GPUs with spec-sheet peaks, and `models`: each checkpoint of `model/catalog.yaml` in its order, with its `name`, `family` and `config` (the `model/config/` stem of its structure) |
| `/kernels/{kind}` | One kind: docs, measuring method, arguments with type and unit, backends, torch reference, and the chart over several configs the kind declares (`view`, below; null for most kinds) |
| `/kernels/{kind}/rows` | Measured rows, column-oriented with a shared provenance table. Any query parameter filters a column by equality (`gpu`, `backend` or an argument); `format=csv` returns CSV |

A kind's `view` is its `KernelDoc.view` (`ConfigView` in `profiling/db/doc.py`):
a `title`, a `summary`, and two roles, each a Rust config field with a `label`
and a `doc`. `series` draws one line per value over the configs' sweep axis; its
values are positions in an order, 0 first, and position 0 leads. `workload` is
picked with a selector. The fused-MoE kinds declare `EP_RANKS_BY_LOAD`:
`folded_rank_position` (the simulator's order of EP ranks by load) and
`expert_demand` (the routing).

## Sources

`public_api/kernel/library.py` joins only sources that live with the code:

- kernel prose: `DOC`, `arg(...)` and `BackendDoc` in `profiling/kernels/<kind>.py`
  (`profiling/db/doc.py`), and the profiling registry;
- compute dtype columns: `simulator kernel-list`, cached until the binary
  changes (`public_api/kernel/sources.py`);
- checkpoint names and the config each one uses: `model/catalog.yaml`, reread
  when it changes (no restart);
- GPU peaks: `gpu/spec.json`;
- measurements: `profile.db`, aggregates cached until the file changes.

## Agent skill

`skills/servingstudio-kernel-performance/SKILL.md` teaches a coding agent to
answer kernel-performance questions from these routes: find the kind, read what
it measures and how, then filter its rows. Install it with the
`skills` CLI, which finds every `SKILL.md` under the path it is given:

    npx skills add https://github.com/SyFI-ServingStudio/ServingStudioSim/tree/main/public_api

The skill reads the base URL from `SERVINGSTUDIO_API` and defaults to
`https://servingstudio.cs.washington.edu/api/public/v1`.
