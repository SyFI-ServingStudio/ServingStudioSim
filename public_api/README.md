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
| `/kernels` | Catalog: every registered kind with its docs summary, backends, precisions, row coverage per GPU/backend/precision, and the models that run it; plus categories, GPUs with spec-sheet peaks, and models |
| `/kernels/{kind}` | One kind: docs, measuring method, arguments with role (swept or fixed by the model) and unit, backends, torch reference, and each supported deployment with the shapes its cost tree asks for |
| `/kernels/{kind}/rows` | Measured rows, column-oriented with a shared provenance table. Any query parameter filters a column by equality (`gpu`, `backend` or an argument); `format=csv` returns CSV |

## Sources

`public_api/kernel/library.py` joins only sources that live with the code:

- kernel prose: `DOC`, `arg(...)` and `BackendDoc` in `profiling/kernels/<kind>.py`
  (`profiling/db/doc.py`), and the profiling registry;
- which GPUs, models and parallel sizes an arch runs: `#[supported(...)]` on the arch
  variant in `simulator/src/arch/config.rs`, and model names in `model/catalog.yaml`;
- cost trees, compute dtype columns and swept/fixed columns: `simulator
  supported-cost-trees`, `simulator kernel-list` and `simulator kernel-query` (op
  `rows`), cached until the binary changes (`public_api/kernel/sources.py`);
- GPU peaks: `gpu/spec.json`;
- measurements: `profile.db`, aggregates cached until the file changes.
