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
uv run cargo build --release -p simulator -p analyzer   # the binaries it runs
uv run python -m public_api serve --bind 127.0.0.1 --port <port> --analyzer-port <port> [--db <profile.db>]
```

`--port` and `--analyzer-port` (the Analyzer it starts, on 127.0.0.1) have no
default: check that each port is free and not in another user's range first.
Kept predictions live in `--runs-dir` (default `logs/public_api/predictions`). `--db` defaults to the profiler's database (`VIBESIM_PROFILE_DB` or
`profiling/profile.db`). Interactive API docs are at `/api/public/v1/docs`.

### In Docker

For a long-running service, `public_api/docker/run.sh BIND PORT` runs it in a
container that Docker restarts on failure and when the host reboots
(`--restart unless-stopped`):

```bash
uv run cargo build --release -p simulator -p analyzer
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

Under `/api/public/v1`; all `GET` but `POST /predict`. None writes anything.

| Route | Returns |
| --- | --- |
| `/health` | `status` and the Sim commit served |
| `/kernels` | Catalog: every registered kind with its docs summary, backends, precisions, row coverage per GPU/backend/precision and the public presets that build a config of it (`used_by`); plus categories, GPUs with spec-sheet peaks, and `models`: each checkpoint of `model/catalog.yaml` in its order, with its `name`, `family` and `config` (the `model/config/` stem of its structure) |
| `/kernels/{kind}` | One kind: docs, measuring method, arguments with type and unit, backends (each with its dtypes and device requirement: `min_compute_capability`, `sm_targets`, and the `gpu/spec.json` GPUs that meet it, `gpus`, null for any CUDA GPU), torch reference, and the chart over several configs the kind declares (`view`, below; null for most kinds) |
| `/kernels/{kind}/rows` | Measured rows, column-oriented with a shared provenance table. Any query parameter filters a column by equality (`gpu`, `backend` or an argument); `format=csv` returns CSV |
| `/kernels/{kind}/configs` | Every kernel config a public deployment asks the kind's rows for: its `id`, GPU, cache axes, cells and how many each backend measured, the profile.db args every cell shares (`fixed`) and the ones the axes move (`swept`), the Rust config's scalar fields (`identity`; `structured` names the rest) and its `uses` (preset, member params, roles) |
| `/kernels/{kind}/configs/{id}` | One config with its whole identity and its grid: per cell, coordinates, profile.db args, whether the kernel can run it, and each backend's measured metrics there |
| `/models` | Every checkpoint of `model/catalog.yaml` with its public presets (`presets/public/<checkpoint>/<arch type>.yaml`): axes, and per member its params, GPUs per replica, prediction shape and the profile.db rows it lacks (`missing`, `{kind: count}`); plus `case_fields`, the fields of a prediction case per shape |
| `/models/{checkpoint}/{arch}/tree` | One member's cost tree, structure only; the query gives every axis of the preset. Each leaf names its config by `id`, the key of `/kernels/{kind}/configs/{id}`; `configs` gives each config's identity and the profile.db rows this member asks of it and lacks (`missing`) |
| `POST /predict` | `{preset, params, cases, analyze?}`: each case's time on one member, per section and per tree node, from one `launcher timing-predict` run on the served profile.db. A member that lacks rows answers 409 naming them. With `analyze: true` the prediction is also analyzed (no plots) and kept for six hours, and the answer names it by `prediction_id` |
| `/analyzer/predictions/{prediction_id}/...` | The Analyzer's routes for one kept prediction (`/api/analyzer/v1/predictions/{id}/...`: descriptor, reports, payloads), forwarded read-only to the `analyze serve` the service runs on 127.0.0.1 over its runs directory |

A config's `id` hashes its kind, GPU and identity (`KernelConfig::identity`:
every field but `gpu_name` and `backends`, each `Dim` at its value, and a routing
corpus named by its path in the dataset repo). The service builds every preset
member once at startup (`simulator cost-trees --kernel-configs`, captures read
from the local hub cache) and asks profile.db which rows each lacks
(`timing-predict --dry-run`), so a restart picks up new presets, code or rows.

A kind's `view` is its `KernelDoc.view` (`ConfigView` in `profiling/db/doc.py`):
a `title`, a `summary`, and two roles, each a Rust config field with a `label`
and a `doc`. `series` draws one line per value over the configs' sweep axis; its
values are positions in an order, 0 first, and position 0 leads. `workload` is
picked with a selector. The fused-MoE kinds declare `EP_RANKS_BY_LOAD`:
`folded_rank_position` (the simulator's order of EP ranks by load) and
`expert_demand` (the routing).

## Sources

`public_api/kernels.py` joins only sources that live with the code:

- kernel prose: `DOC`, `arg(...)` and `BackendDoc` in `profiling/kernels/<kind>.py`
  (`profiling/db/doc.py`), and the profiling registry;
- compute dtype columns: `simulator kernel-list`, cached until the binary
  changes (`public_api/sources.py`);
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
