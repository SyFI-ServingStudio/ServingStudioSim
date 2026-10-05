# public_api

Data and simulation service for the external ServingStudio site. The Intro site's
pages request `/api/public/v1/...` from their own origin, and the site proxies
those paths here: Vite's `PUBLIC_API_PROXY_TARGET` in development, the web
server in front of the site otherwise. The Analyzer serves the internal UI and
is a separate service.

The service never runs a GPU or a profiler, and never writes `profile.db`:
every query opens it read-only. It runs simulations only of the sim presets'
members (below), queued, on the served `profile.db`.

## Run

```bash
uv run cargo build --release -p simulator -p analyzer   # the binaries it runs
cargo build --release --manifest-path alignment/load_generator/req-frontend/Cargo.toml --bin tracegen
uv run python -m public_api serve --bind 127.0.0.1 --port <port> --analyzer-port <port> [--db <profile.db>]
```

`--port` and `--analyzer-port` (the Analyzer it starts, on 127.0.0.1) have no
default: check that each port is free and not in another user's range first.
Kept predictions live in `--runs-dir` (default `logs/public_api/predictions`)
and simulations in `--sims-dir` (default `logs/public_api/simulations`); the
Analyzer reads both. Uploaded workloads are kept in `--workloads-dir` (default
`logs/public_api/workloads`). `--db` defaults to the profiler's database
(`VIBESIM_PROFILE_DB` or `profiling/profile.db`). Behind a proxy, pass its
address as `--forwarded-allow-ips` so the rate limits count the client that
`X-Forwarded-For` names. Interactive API docs are at `/api/public/v1/docs`.

### In Docker

For a long-running service, `public_api/docker/run.sh BIND PORT` runs it in a
container that Docker restarts on failure and when the host reboots
(`--restart unless-stopped`):

```bash
uv run cargo build --release -p simulator -p analyzer
cargo build --release --manifest-path alignment/load_generator/req-frontend/Cargo.toml --bin tracegen
public_api/docker/run.sh 10.158.48.50 5220
docker logs -f servingstudio-public-api
```

The image holds only the Python environment, installed from `uv.lock`'s
`public-api` and `launcher` groups, so it is rebuilt only when the lock changes.
The container runs as the calling user and mounts the checkout read-only at the
same path, with its git directory (a worktree's lies outside it). It serves
that checkout's code, `profiling/profile.db`, `target/release/simulator` and
req-frontend's `tracegen`. To
serve new code or data, update the checkout, rebuild the simulator and rerun
`run.sh`, which replaces the container. `PUBLIC_API_CONTAINER` and `PUBLIC_API_IMAGE` override the
container and image names. The rate limits count the client the CSE web host's
proxy names; `PUBLIC_API_FORWARDED_ALLOW_IPS` (`--forwarded-allow-ips`, default
that host's address) lists the proxies trusted to name it.

## Routes

Under `/api/public/v1`; all `GET` but `POST /predict`, `POST /simulate`,
`POST /workloads` and `DELETE /simulations/{id}`.

| Route | Returns |
| --- | --- |
| `/health` | `status` and the Sim commit served |
| `/kernels` | Catalog: every registered kind with its docs summary, backends, precisions, row coverage per GPU/backend/precision and the public presets that build a config of it (`used_by`); plus categories, GPUs with spec-sheet peaks, and `models`: each checkpoint of `model/catalog.yaml` in its order, with its `name`, `family` and `config` (the `model/config/` stem of its structure) |
| `/kernels/{kind}` | One kind: docs, measuring method, arguments with type and unit, backends (each with its dtypes and device requirement: `min_compute_capability`, `sm_targets`, and the `gpu/spec.json` GPUs that meet it, `gpus`, null for any CUDA GPU), torch reference, and the chart over several configs the kind declares (`view`, below; null for most kinds) |
| `/kernels/{kind}/rows` | Measured rows, column-oriented with a shared provenance table. Any query parameter filters a column by equality (`gpu`, `backend` or an argument); `format=csv` returns CSV |
| `/kernels/{kind}/configs` | Every kernel config a public deployment asks the kind's rows for: its `id`, GPU, cache axes, cells and how many each backend measured, the profile.db args every cell shares (`fixed`) and the ones the axes move (`swept`), the Rust config's scalar fields (`identity`; `structured` names the rest) and its `uses` (preset, member params, roles) |
| `/kernels/{kind}/configs/{id}` | One config with its whole identity and its grid: per cell, coordinates, profile.db args, whether the kernel can run it, and each backend's measured metrics there |
| `/models` | Every checkpoint of `model/catalog.yaml` with its public presets (`presets/public/<checkpoint>/<arch type>.yaml`): the arch's reader-facing name (`arch_name`, from `model/arch_catalog.yaml`), axes, and per member its params, GPUs per replica, prediction shape and the profile.db rows it lacks (`missing`, `{kind: count}`); plus `case_fields`, the fields of a prediction case per shape |
| `/models/{checkpoint}/{arch}/tree` | One member's kernels: each section's leaf slots; the query gives every axis of the preset. Each slot names its config by `id`, the key of `/kernels/{kind}/configs/{id}`; `configs` gives each config's identity and the profile.db rows this member asks of it and lacks (`missing`) |
| `POST /predict` | `{preset, params, cases, analyze?}`: each case's time on one member, per section and per node of its cost tree (as `analyze gen-iter-breakdown` reads it), from one `launcher timing-predict` run on the served profile.db. A member that lacks rows answers 409 naming them. With `analyze: true` the prediction is also analyzed (no plots) and kept for six hours, and the answer names it by `prediction_id` |
| `/analyzer/predictions/{prediction_id}/...` | The Analyzer's routes for one kept prediction (`/api/analyzer/v1/predictions/{id}/...`: descriptor, reports, payloads), forwarded read-only to the `analyze serve` the service runs on 127.0.0.1 over its runs directory |
| `/simulations/presets` | Every sim preset: its deployment, pools (arch preset, arch, GPU, worker), axes, the captures its members replay, and per member its params, GPUs, pools and why it cannot run a capture (`unavailable`: `{error}`, `{missing: {role: rows}}`, or `{misfit}` when the capture's requests do not fit its pools); plus the limits and the workload's JSON schema |
| `/workloads` | Where a simulation's requests can come from: a capture; a generated trace, with each offered tracegen generator's arguments (`tracegen describe`); an upload, with the formats and tags it may declare (`simulator trace-formats`) and its size limit |
| `POST /workloads` | The request body, a CSV of the `format` (and comma-separated `tags`) the query gives, kept for 24 h once the simulator reads it: 201 `{workload_id, input_file_format, input_file_tags, requests, expires_at}` |
| `POST /simulate` | `{preset, params, workload}`: queues one member's simulation and answers 202 `{simulation_id, status, routing}` (below) |
| `/simulations/{simulation_id}` | Its status (`queued` with `queue_position`, `running`, `done`, `failed` with `error`, `timed_out`) and request, with its `routing`; once done, `summary` and the Analyzer's `run_id` |
| `DELETE /simulations/{simulation_id}` | Stops it if it has not finished and removes it: `{simulation_id, status: "deleted"}` |
| `/analyzer/runs/{run_id}/...` | The Analyzer's routes for one simulation's run (`/api/analyzer/v1/runs/{id}/...`: descriptor, `subjects/<subject>/report` and `payload`), forwarded read-only. No run catalog |
| `/analyzer/kernel-kinds` | The Analyzer's `/api/analyzer/v1/kernel-kinds`: each kernel kind's title and category from its DOC, by which Read more names and groups a result's kernels |

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

## Simulations

A sim preset (`presets/public_sim/<checkpoint>/<name>.yaml`, `public_api/sim_preset.py`)
is a run config without its workload, in the launcher's sweep language:
`deployment`, and `pools` of one group each, `{replicas, arch, worker}`. A
group's `arch` names an arch preset of the same checkpoint (`preset:`) and
gives that preset's axes but `workload`; its GPU and complete arch block are
that public member's. The expanded list is the support list: a combination is
supported when a sim preset has a member for it. At startup the service builds
every member as a run builds it (`simulator dry-run`: the deployment's
`build_flow`) once per capture it can replay, and records the profile.db rows
each lacks; `tests/test_public_sim_presets.py` asserts that every member builds
and is measured.

The workload is the reader's. Its requests come from one of three `source`s
(`public_api/workloads.py`):

- `capture`: a `workload` row of the pools' arch preset, whose `trace.csv` the
  run replays. A dense member replays any published capture's requests: each
  distinct `trace.csv` once, named by its workload label (with its model's
  directory when one label has two), the one most captures record first.
- `generated`: a trace req-frontend's `tracegen` draws (a
  `session-execution-v2` file). `generator` is `{type, <argument>: value}`,
  each argument one of `tracegen describe`'s for that generator and passed to
  its flag as given; tracegen checks the values. The service offers
  `synthetic` and sets `--out` itself; a generator gets 30 s and 2 GiB.
- `upload`: a CSV kept by `POST /workloads`, named by its `workload_id`.

MoE routing is always the member's capture: an MoE member offers only its
capture rows (never a synthetic routing), and every source routes as the
`capture` the request names, the preset's first by default. The answer's
`routing` says so: `{capture, label}`, the label "requests and routing from
capture X", "requests from your workload, routing from capture X", or for a
dense member "...; the model routes no experts".

```json
POST /api/public/v1/simulate
{
  "preset": "Llama-3.1-8B/llama3_dense_tp_barebone",
  "params": {"tp_size": 1, "replicas": 1},
  "workload": {
    "source": "generated",        // "capture" (default), "generated" or "upload"
    "capture": null,              // capture source: its trace; MoE: its routing. The preset's first by default
    "generator": {"type": "synthetic", "sessions": 100, "rounds": "1", "input_len": "lognormal:1024,0.8"},
    "upload": null,               // upload source: a workload_id from POST /workloads
    "load": {"rate": 4.0},        // or {"concurrency": 32}; none: the trace's own arrival times
    "duration_ms": null,          // required when run_to_end is false
    "run_to_end": true,
    "accept_rate": null           // speculative workers only: one probability, or one per draft position
  }
}
```

Every request of the source runs. `load` is one number: `rate`, the requests
(a session trace's sessions) per second, replays the trace's own spacing
scaled to that mean, so the simulator's trace-timed `request_rate` is the
asked rate over the trace's; `concurrency` makes every request ready at once
and keeps at most that many in flight (sessions, when rounds chain). A trace
whose requests all arrive at once has no rate to scale and takes only a
concurrency. Unknown workload fields are refused (422), not ignored.

What a trace holds is read once, by `simulator workload-plan`: each capture
in `/simulations/presets` and each upload's record carry `requests`,
`sessions` (whether rounds chain), the mean `prompt_tokens` (carried prefix
included) and `output_tokens`, and `rate`, the first rounds' arrivals per
second over their span (null when they all arrive at once). They also
carry `input_file_tags`, the tags the trace's header fits.

A capture's trace is read as the format and tags its header fits, like an
upload. A speculative capture's trace carries the acceptance that capture
recorded, one chain per request (the `speculative` tag's `accept_rate`
column, from the capture pass's own per-request decode progress): a
speculative member replays it and takes no `accept_rate`, and a worker that
drafts nothing runs the same requests without that column. A dense member,
which may replay any capture's requests, lists each request list once by its
requests alone, so a capture's acceptance column does not make a new list.

Before it queues, the request is checked: the member exists (400 with
`choices` otherwise) and builds and is measured on that capture (409), and a
replayed capture's own requests fit (its `misfit`, refused as below); the
source's fields are given (`generator` for `generated`, `upload` for `upload`,
neither for `capture`); there are at most 2000 requests; a speculative worker
gets acceptance from exactly one of `accept_rate` and an upload tagged
`speculative` (its own column), and no other worker takes `accept_rate`; and
the simulator loads the trace as the run will and checks it against the run's
pools (`simulator workload-plan --config` on the run config, the check `run`
makes before its first tick: format, tags, every row, the replay settings, so
chaining a trace without sessions is refused; every request's `prefix_len +
input_len + output_len`, plus the draft tokens for a speculative worker, fits
each pool's `max_model_len`, which is the arch's own or else its checkpoint's
`max_position_embeddings`; and a per-position `accept_rate` has one probability
per draft position) (400 for each; a request that does not fit answers
`detail: {message, too_long: {max_model_len, requests, total}}`, the counts
of the simulator's own check). The run's directory holds the trace
it replays (`workload.csv`: the source's rows, with the `accept_rate`
column), a generated trace as tracegen wrote it
(`generated.csv`, `.manifest.json`, `.plan.json`), the concrete run config
(`simulation.run.json`) and the record (`simulation.json`), whose `workload`
also says what the trace held (`trace`: the facts below). A queued run is
the launcher's standard single run (`launcher.sweep.run_single`, analyzed, no
plots) in a child process (`public_api/simulate_run.py`).

`POST /workloads` reads the body (at most 1 MiB, 413 past it) as the declared
`format` and `tags`, or with neither as the format whose columns, with some of
its tags' columns, are exactly the header's (`simulator trace-formats`; a
header that fits none is refused with the list, and tags without a format are
refused), with the same `simulator workload-plan` and keeps it only
when it loads and has at most 2000 requests. An upload tagged `speculative`
carries its own `accept_rate` column; its width is checked against a member's
worker when a simulation names it.

Once `done`, `summary` is the run's `summary.json` and the Analyzer's
`slo-general` report:

```json
{"cause": "...", "requests": {"total": 32, "finished": 32}, "sim_ms": ..., "num_gpus": 1,
 "throughput": {"total_tok_s": ..., "prefill_tok_s": ..., "decode_tok_s": ...,
                "total_tok_s_per_gpu": ..., "completed_req_s": ...},
 "ttft_ms": {"mean": ..., "p50": ..., "p90": ..., "p99": ..., "max": ..., "n": ...},
 "tpot_ms": {...}, "e2e_ms": {...}}
```

Limits (`public_api/simulate.py`, `public_api/app.py`): two simulations run
at once and up to 32 wait (429 past that); a run is stopped after 10 minutes
of wall clock and marked `timed_out`; a run and its directory are removed 24
hours after it was submitted, and an upload 24 hours after it arrived.
`POST /predict` takes 30, and `POST /simulate` and `POST /workloads` 6,
requests per minute per client address (429 with `Retry-After`), counted in
the process; a request the route refuses (4xx) is not counted. A restart keeps
finished runs and marks unfinished ones failed.

## Sources

`public_api/kernels.py` joins only sources that live with the code:

- kernel prose: `DOC`, `arg(...)` and `BackendDoc` in `profiling/kernels/<kind>.py`
  (`profiling/db/doc.py`), and the profiling registry;
- compute dtype columns: `simulator kernel-list`, cached until the binary
  changes (`public_api/sources.py`);
- checkpoint names and the config each one uses: `model/catalog.yaml`, read
  at start;
- GPU peaks: `gpu/spec.json`;
- measurements: `profile.db`, aggregates cached until the file changes.

## Agent skills

Three skills teach a coding agent to use these routes:

- `skills/servingstudio-kernel-performance/SKILL.md` answers kernel-performance
  questions: find the kind, read what it measures and how, then filter its rows.
- `skills/servingstudio-timing-predict/SKILL.md` predicts iteration times: pick
  a member from `/models`, read its kernels, build cases from its `predict` shape,
  then `POST /predict`.
- `skills/servingstudio-simulate/SKILL.md` simulates a deployment on a request
  stream: pick a member from `/simulations/presets`, choose a capture, a
  generated trace or an upload, `POST /simulate`, then read
  `/simulations/{id}`.

Install them with the `skills` CLI, which finds every `SKILL.md` under the path
it is given:

    npx skills add https://github.com/SyFI-ServingStudio/ServingStudioSim/tree/main/public_api

Each skill reads the base URL from `SERVINGSTUDIO_API` and defaults to
`https://servingstudio.cs.washington.edu/api/public/v1`.
