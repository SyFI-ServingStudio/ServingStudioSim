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
| `/kernels/{kind}` | One kind: docs, measuring method, arguments with role (swept or fixed by the model) and unit, backends, torch reference, the chart over several configs the kind declares (`view`, below; null for most kinds), and each deployment that runs it (from the registered configs' uses, which include every `#[supported]` deployment once registered) with its model, label, and the shapes it asks for (the role, pool, fixed columns and config of each leaf, and the members that ask) |
| `/kernels/{kind}/rows` | Measured rows, column-oriented with a shared provenance table. Any query parameter filters a column by equality (`gpu`, `backend` or an argument); `format=csv` returns CSV |
| `/kernels/{kind}/configs` | The kernel configs registered as reading the kind's rows (profile.db's kernel-config registry), one per config and GPU: the profile.db args every cell shares (`fixed`) and the ones the cache axes move (`swept`), the Rust config's scalar values, cache coordinates and axes, cells measured per backend, a reader's name for structured config values (`config_labels`, below), and each use (pool, role) naming entries of a shared `deployments` table and the members that ask. No identity: it can run to megabytes |
| `/kernels/{kind}/configs/{config_hash}` | One config's grid on the simulator's cache axes: each cell's coordinates, profile.db args, whether the kernel can run it, and each backend's measured metrics there; plus the full identity, its uses, and the `deployments` entries they name. `gpu` is required only when the config is registered on more than one GPU |
| `/archs` | Every arch tag: `name` and `summary` (`model/arch_catalog.yaml`), `contract`, the `models` (model-config stems) and `gpus` its `#[supported]` rows run, their model `families`, the `params` the rows choose between, and how many `param_sets` and `combinations` it supports; plus `models`, the catalog entries they name |
| `/archs/{arch}` | One arch: every param (`list-params`: type, default, choices, description, `set_when_predicting` on a traffic param, and `values`, the values its rows list, null for a param no row chooses), the `query` names of its cost trees, and its supported `param_sets` |
| `/archs/{arch}/cost-tree` | One supported parameter set's cost tree, chosen by the query: `gpu`, `model` (a model-config stem) and every param the arch's rows name, all required. The structure only: no timings |

A deployment entry is one `#[supported]` row on one GPU and model config:
`arch`, `gpu`, `model_config` (the file stem of a `model/config/` file, as
`#[supported]` rows spell it), `model` (its `model/catalog.yaml` entry, or null),
`params`, `varies`, `label` and `members`. The params are the ones the arch's
`#[supported]` rows name; an arch without rows falls back to its own params that
change its kernel configs and take a number, a bool or one of listed choices
(file paths stay out). The rows come from `simulator list-params`, which lists
each row once with its value lists unexpanded, and a deployment belongs to the
first row that covers it (the launcher's `_check_supported` rule). A param the
row lists several values for is in `varies`, its `params` value is the list of
values registered deployments run, and the label reads `max_model_len 8192 ·
65536`; each such deployment is a `member` (`params` holds its varying values).
A use names `{id, members}`: an entry and the member indices that ask for the
config, so a config only some values use says which. A deployment no row covers
is an entry of its own with one member.

A kind's `view` is its `KernelDoc.view` (`ConfigView` in `profiling/db/doc.py`):
a `title`, a `summary`, and two roles, each a Rust config field with a `label`
and a `doc`. `series` draws one line per value over the configs' sweep axis; its
values are positions in an order, 0 first, and position 0 leads. `workload` is
picked with a selector, by each config's `config_labels` entry for that field.
The fused-MoE kinds declare `EP_RANKS_BY_LOAD`: `folded_rank_position` (the
simulator's order of EP ranks by load) and `expert_demand` (the routing).

`config_labels` names a config's `expert_demand` (`public_api/kernel/demand.py`),
which the configs list otherwise leaves out, as `{routing, label, reference,
fingerprint, binding, preference}`. The routing and its file come from the arch
blocks of the config's uses (`routing`, with its `list-params` default, and the
file field `ROUTING_ARTIFACTS` gives it). A hub artifact is labeled by the
`hf://<repo>@<revision>/<path>` reference its preset wrote, which the launcher
records instead of the file it fetched; a file this checkout tracks by its
repo-relative path; any other file by its file name and the `fingerprint`; a synthetic routing by its name (`uniform`). `reference` is
the path or `hf://` reference, null otherwise. The fingerprint is the demand's
content: a corpus manifest's payload checksum, or a hash of the popularity table.
`binding` is how the config reads it (a corpus's `group_size` and layer slice, a
table's layer count); `preference` orders measured routings first, a corpus
before a popularity file, for a page's default pick.

What registered a config (a `#[supported]` row, a
preset, a prediction or an alignment pack, with its path) is not published:
those records name local paths and are bookkeeping. A routing label is not a
registry record: it names the demand the config holds, and only by a tracked
repo path or a hub reference, never by a local path. They stay in profile.db's
registry, read with `profiling.db.kernel_config.registered_configs`. A config's
`identity` and `config_args` are published with every absolute file path cut to
its file name (an expert-demand corpus under a local HF cache, say).

## Archs and cost trees

The Models page reads three routes. None of them costs anything: a tree says
how leaf costs compose, and a leaf's cost depends on the batch, which only a
prediction supplies (below).

A parameter set is one `#[supported]` row on one GPU and model config, in the
shape of a kernel's deployment entry (`arch`, `gpu`, `model_config`, `model`,
`params`, `varies`, `label`, `members`; the same `row_entry` code builds both).
Each member is one combination of the values the row lists and adds `label`,
`query` (the cost-tree query that names it), `gpus_per_replica`, `error` (the
build's, null when it built) and `counts`: `leaves`, distinct kernel `configs`,
and how many of those profile.db's registry holds (`registered`) and has any
measured row for (`measured`). A combination two rows cover belongs to the first.

`cost-tree` rejects a query it would have to guess at: a missing param, one
the rows do not choose (the tree takes its default; the document lists those
under `defaults` or `set_when_predicting`) or a mistyped value is a 400, and a
set no row covers a 404.
Both answer `{"detail": {"message", "choices"}}`, `choices` being every valid
query.

A tree comes from `simulator supported-cost-trees`, which builds every supported
combination structure-only with every other param at its schema default. The
document has the set's `arch`, `name`, `gpu`, `model_config`, `model`, `params`,
`label`, `query`, `defaults`, `set_when_predicting`, `gpus_per_replica`, `counts` and:

- `defaults`: `{name: default}` for each param the rows leave open, and
  `set_when_predicting`: the names of those that `list-params` marks
  `set_when_predicting` (`#[param(set_when_predicting)]` in Rust). Such a param
  describes the traffic, not the deployment: MoE `routing`, whose `uniform`
  default only lets a config omit it and is no choice for a prediction
  (`skills/operate-run-simulation/references/moe-routing.md`), so it is not
  listed as a default. The build still used that default, so a MoE tree's
  fused-MoE leaves name uniform-demand configs (`expert_demand`) until a
  prediction sets the routing;

- `sections`: `[{section, root}]`, one per compiled CostTree (`iter` for an
  iter-wise arch; `attn`, or `prologue`, `pre_attn`, `post_attn`, ... for the
  two sides of a disaggregated deployment);
- a node is `{id, kind, label?, children}`: `kind` is `sum`, `max` (with its
  `overlap` divisor: `max(children) / overlap`), `scale` (with its repeat `n`)
  or `leaf`. `id` is the node's index in the flat `cost_manifest` and `label`
  its composite line (worklet and partition) when Rust gives one. A composite's
  `path` is the dotted role its leaves share;
- a leaf is `{id, kind: "leaf", slot: {index, name, kind, backends,
  config_hash}}`: the slot index (the order of a cost log's `slot_*` columns),
  the dotted leaf name, the kernel kind (its page is `/kernels/{kind}`), the
  backends the config may run, and the config's hash, the key of
  `/kernels/{kind}/configs/{config_hash}`;
- `configs`: `{config_hash: {kind, args, args_omitted, registry}}`, each leaf
  config once (rank copies share one): its scalar identity fields, the names of
  the structured ones, and `registry` (`{cells, infeasible, measured}` per
  backend on this GPU) or null when the registry does not hold it;
- `kernels`: `{kind: {documented, title, category}}` for the kinds the leaves use.

A leaf's config hash is computed from the leaf's own Rust config
(`KernelConfig::identity`: every field but `gpu_name` and `backends`, each
`Dim` reduced to its value, then `content_hash`), so it names the same config
the registry does; a binary-backed test checks it against the records
`supported-cost-trees --kernel-configs` writes.

The node and leaf shapes are the Analyzer's cost-tree transport
(ServingStudioUI `artifacts/schema/costTree`: `{kind, label?, children,
overlap | n}`, a leaf's `slot` with `name` and `kind`) without the numbers a
prediction adds there (`base`, `stats`, and the per-node `ms` and `pct`). Two
differences are deliberate: a leaf carries `config_hash` and `backends` instead
of the whole `kernel_config` (its `Dim` bindings run to megabytes on the GLM
trees; the config route has it), and every node keeps `id` and every leaf its
slot `index`, so a prediction can return times by node and by slot without
resending the tree.

## Later: predictions

The routes stay read-only now. A prediction is "this arch, these params, these
requests": the plan is one write route that takes a supported parameter set
(the same `query` as `cost-tree`) and a batch of requests, and answers in terms
of the tree the reader already has.

- `POST /archs/{arch}/predictions` with `{query, requests, routing}`: `requests`
  as `timing-predict` cases (per group, `prefill_chunk_pairs` of
  `[prefix_len, append_len]` and `decode_kv_lens`), bounded in count and
  length. It answers `202` with a prediction id; `GET /predictions/{id}` returns the status and then, per case,
  the iteration total and a time per node `id` and per slot `index` (with the
  backend chosen), which the page overlays on the tree it drew.
- MoE archs need an explicit routing (`corpus` or `popularity` with its
  artifact); the route never defaults one to `uniform`.
- Public predictions read measured rows only (no JIT profiling and no GPU): a
  case that needs a row profile.db lacks is reported as not predictable, with
  the leaves that lack it, instead of being profiled.
- A prediction runs the release binary's `timing-predict` in a worker pool, off
  the request path, with its own rate limit; nothing it does writes profile.db.

## Sources

`public_api/kernel/library.py` joins only sources that live with the code:

- kernel prose: `DOC`, `arg(...)` and `BackendDoc` in `profiling/kernels/<kind>.py`
  (`profiling/db/doc.py`), and the profiling registry;
- which GPUs, models and parallel sizes an arch runs: `#[supported(...)]` on the arch
  variant in `simulator/src/arch/config.rs`, and model names in `model/catalog.yaml`
  (reread when it changes, no restart);
- compute dtype columns: `simulator kernel-list`; arch params and `#[supported]`
  rows: `simulator list-params`; every supported combination's cost tree:
  `simulator supported-cost-trees`; all cached until the binary changes
  (`public_api/kernel/sources.py`);
- arch names: `model/arch_catalog.yaml`, reread when it changes
  (`public_api/arch/library.py`);
- GPU peaks: `gpu/spec.json`;
- measurements: `profile.db`, aggregates cached until the file changes;
- routing labels: the registry's `hf://` references, and `git ls-files` for the
  files this checkout tracks (`sources.py`);
- which configs read which rows: the kernel-config registry in `profile.db`
  (`profiling/db/kernel_config.py`), filled by the launcher when a run or
  prediction builds its kernel cache.
