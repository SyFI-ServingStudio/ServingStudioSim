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

## Routes

All `GET`, under `/api/public/v1`.

| Route | Returns |
| --- | --- |
| `/health` | `status` and the Sim commit served |
| `/kernels` | Catalog: every registered kind with its docs summary, backends, precisions, row coverage per GPU/backend/precision, and the models that run it (`used_by`, model-config stems); plus categories, GPUs with spec-sheet peaks, and models (the model catalog in its order, then any model a kind names that the catalog does not, with `name` null) |
| `/kernels/{kind}` | One kind: docs, measuring method, arguments with role (swept or fixed by the model) and unit, backends, torch reference, and each deployment that runs it (from the supported cost trees and the registered configs' uses) with its model, label, sources and the shapes it asks for |
| `/kernels/{kind}/rows` | Measured rows, column-oriented with a shared provenance table. Any query parameter filters a column by equality (`gpu`, `backend` or an argument); `format=csv` returns CSV |
| `/kernels/{kind}/configs` | The kernel configs registered as reading the kind's rows (profile.db's kernel-config registry), one per config and GPU: the profile.db args every cell shares (`fixed`) and the ones the cache axes move (`swept`), the Rust config's scalar values, cache coordinates and axes, cells measured per backend, and each use (pool, role) pointing into shared `sources` and `deployments` tables. No identity: it can run to megabytes |
| `/kernels/{kind}/configs/{config_hash}` | One config's grid on the simulator's cache axes: each cell's coordinates, profile.db args, whether the kernel can run it, and each backend's measured metrics there; plus the full identity and each use with its source, deployments and model. `gpu` is required only when the config is registered on more than one GPU |

A deployment is one arch block on one GPU: `arch`, `gpu`, `model_config` (the
file stem of a `model/config/` file, as `#[supported]` rows spell it), `model`
(its `model/catalog.yaml` entry, or null), `params` and `label`. The params are
the ones the arch's `#[supported]` rows name; an arch without rows falls back to
its own params that change its kernel configs and take a number, a bool or one
of listed choices (file paths stay out). A source is what built a config:
`supported`, `timing_predict`, `alignment` (pack, variant, case count) or
`preset`, with `ref` its path.

## Sources

`public_api/kernel/library.py` joins only sources that live with the code:

- kernel prose: `DOC`, `arg(...)` and `BackendDoc` in `profiling/kernels/<kind>.py`
  (`profiling/db/doc.py`), and the profiling registry;
- which GPUs, models and parallel sizes an arch runs: `#[supported(...)]` on the arch
  variant in `simulator/src/arch/config.rs`, and model names in `model/catalog.yaml`;
- cost trees, compute dtype columns and swept/fixed columns: `simulator
  supported-cost-trees`, `simulator kernel-list` and `simulator kernel-query` (op
  `rows`), and arch params and `#[supported]` rows: `simulator list-params`, all
  cached until the binary changes (`public_api/kernel/sources.py`);
- GPU peaks: `gpu/spec.json`;
- measurements: `profile.db`, aggregates cached until the file changes;
- which configs read which rows: the kernel-config registry in `profile.db`
  (`profiling/db/kernel_config.py`), filled by the launcher when a run or
  prediction builds its kernel cache.
