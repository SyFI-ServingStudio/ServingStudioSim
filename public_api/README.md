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
that checkout's code, `profiling/profile.db` and `target/release/simulator`, and
reads Hugging Face corpora from the local hub cache (`$HF_HOME/hub`, also
read-only, offline). To serve new code or data,
update the checkout, rebuild the simulator and rerun `run.sh`, which replaces
the container. `PUBLIC_API_CONTAINER` and `PUBLIC_API_IMAGE` override the
container and image names.

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
| `/archs/{arch}/cost-tree` | One supported parameter set's cost tree, chosen by the query: `gpu`, `model` (a model-config stem) and every param the arch's rows name, all required, and optionally the run params of one of the set's registered runs (below). The structure only: no timings |
| `/archs/{arch}/kernel-data` | For the same query, the registry's config document of every config the tree reads, as a simulator kernel-data bridge reads them (below): `format`, `gpu`, `configs` and `missing` (the `config_key`s the registry does not hold) |
| `/files?path=` | The text of one file a prediction reads (`run.predict.files`): a model config git tracks, or the file of a routing `presets/alignment/routings.yaml` names (a hub corpus's manifest from this machine's hub cache, never its payload). 404 for any other path |

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
file field `ROUTING_ARTIFACTS` gives it). `routing` is the category a reader
sees first: `corpus`, `popularity`, `random`, `uniform`, in that order
(`preference`). A measured routing's `label` is its name in
`presets/alignment/routings.yaml`, which names content: a corpus by its payload
checksum, a popularity file by the sha256 of its bytes, however a run named the
file. A name is for readers only and in no config hash. A measured routing the
table does not name has a null label (`tests/test_routing_names.py` keeps every
registered one named); a synthetic routing's label is its kind and seed
(`uniform`, `random, seed 1`). `reference` is the `hf://<repo>@<revision>/<path>`
reference a preset wrote (which the launcher records instead of the file it
fetched) or the repo path git tracks, null otherwise. The fingerprint is the
demand's content: a corpus manifest's payload checksum, or a hash of the
popularity table. `binding` is how the config reads it (a corpus's `group_size`
and layer slice, a table's layer count).

What registered a config (a `#[supported]` row, a
preset, a prediction or an alignment pack, with its path) is not published:
those records name local paths and are bookkeeping. A routing label is not a
registry record: it names the demand the config holds, by the name table,
never by a local path. They stay in profile.db's
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
build's, null when it built) and `counts`: `leaves`, distinct kernel `configs`
(each kind and config hash once), and how many of those profile.db's registry
holds (`registered`), has any measured row for (`measured`), and has what a
prediction fits from (`predictable`: every backend the config's leaves run
has a timed, non-outlier row at every feasible cell of its grid), and `run`:
which registered run the counts are of (`basis` `registry`, with its
`params`, source names and how many distinct runs the set has) or `defaults`.
A combination two rows cover belongs to the first.

### Which run a tree is built as

The params a set leaves open (`max_model_len`, MoE `routing` and its file,
`draft_tokens`, ...) change the tree, and their schema defaults are no run
anyone made: `max_model_len` 1,048,576, routing `uniform`. So a tree is built
as a real run of the set, read from the kernel-config registry in profile.db:

- a run is the arch block of one `_kernel_config_source` group (a preset's
  pool, an alignment case, a `timing_predict` config) whose arch, GPU, model
  config and set params equal the set's (`KernelLibrary._deployment`, defaults
  filled). A `supported` source is not a run: it records the defaults tree;
- a run's params are its block's params besides the set's, each it leaves out
  at its default. A run that leaves out a `set_when_predicting` param (MoE
  `routing`) built uniform demand without choosing it, so it is skipped:
  uniform routing is offered only where a run named it;
- each distinct run is built structure-only with its own block by `simulator
  supported-cost-trees --archs -` (a JSON list of `{gpu, arch}` on stdin),
  its model config and routing files as local paths. An `hf://` corpus is used
  only when the local hub cache holds it (`resolve_reference(local_only=True)`:
  nothing is downloaded); a run naming a file this machine lacks is skipped;
- a run registered from another machine's log names its routing file by that
  path. When the repo holds a copy, the copy's provenance sidecar
  (`<name>.provenance.json` beside it) lists the paths it was `copied_from`
  and its `sha256`, and the run is read as naming the copy: it builds, and is
  the same run as the copy's preset records. A copy whose content no longer
  has that sha256 names nothing;
- a record of a run this machine still cannot build is folded into a built
  run when the registry shows it is that run recorded again: every config it
  asked for is in the built run's tree, one of them folding an MoE routing (a
  config's hash covers its routing). Its source joins that run's `sources`,
  and it is no skipped run. This covers a record whose values its YAML read
  another way (`mtp_mode: off` read as `false`); an arch whose configs fold
  no routing cannot be told apart this way;
- each is counted as the defaults tree is, and the set's runs are ordered best
  first: the routing's category (`corpus`, `popularity`, `random`, `uniform`:
  uniform never leads a routed set), then most configs predictable, then most
  measured, then the higher measured share, then a preset
  over an alignment case over a prediction, then registration order. A run's
  registered config count is no denominator: registration skips configs with
  no measured row;
- a set no run matches (or whose runs were all skipped) is built at its
  defaults by `supported-cost-trees`, as before.

The runs of every set of an arch are built together, in parallel processes,
and kept until profile.db or the binary changes; the server's warm-up thread
builds every arch's.

`cost-tree` rejects a query it would have to guess at: a missing param, a
name neither the rows nor the set's registered runs choose, or a mistyped value
is a 400; a set no row covers, or run params no registered run of the set used,
a 404. Both answer `{"detail": {"message", "choices"}}`, `choices` being every
valid query (for run params, the full query of each run). Run params in the
query need not name every one: the best run holding the given values is built.

The document has the set's `arch`, `name`, `gpu`, `model_config`, `model`, `params`,
`label`, `query`, `defaults`, `set_when_predicting`, `gpus_per_replica`, `counts` and:

- `run`: `basis` (`registry` or `defaults`); `params`, the run params the tree
  was built with, a routing file named as a reader names it (the `hf://`
  reference, the repo path when git tracks it, else its file name and the
  demand's fingerprint; never another machine's path); `sources`, the registry
  sources that recorded that run (`{id, kind, path, name}`, an alignment case
  also `variant` and `cases`; `path` only when this checkout tracks it);
  `routing`, its routing's name as `/kernels/.../configs` gives it (the
  routing picker orders its options by category, then a named routing before
  an unnamed one, then by name); `query`,
  the full cost-tree query of the run; `pickers`, one per run param (the
  routing and its file as one) with `options`: the values the set's runs used,
  each `selected`, `compatible` (a run has it and every other picker's
  selection) and the `counts` of the run choosing it lands on, and `fixed`
  when there is one option; `combinations`, how many runs; and `skipped`, the
  runs not built here with the reason; and `predict`, what a browser predicts
  the run from (below), null at the defaults;
- `defaults`: with a registered run, `{name: default}` for each run param its
  block left out; at the defaults, for each param the rows leave open, and
  `set_when_predicting`: the names of those that `list-params` marks
  `set_when_predicting` (`#[param(set_when_predicting)]` in Rust). Such a param
  describes the traffic, not the deployment: MoE `routing`, whose `uniform`
  default only lets a config omit it and is no choice for a prediction
  (`skills/operate-run-simulation/references/moe-routing.md`), so it is not
  listed as a default. The defaults build still used it, so such a tree's
  fused-MoE leaves name uniform-demand configs (`expert_demand`) until a
  prediction sets the routing. A registered run recorded it, so its list is empty;

- `sections`: `[{section, root}]`, one per compiled CostTree (`iter` for an
  iter-wise arch; `attn`, or `prologue`, `pre_attn`, `post_attn`, ... for the
  two sides of a disaggregated deployment);
- a node is `{id, kind, label?, children}`: `kind` is `sum`, `max` (with its
  `overlap` divisor: `max(children) / overlap`), `scale` (with its repeat `n`)
  or `leaf`. `id` is the node's index in the flat `cost_manifest` and `label`
  its composite line (worklet and partition) when Rust gives one. A composite's
  `path` is the dotted role its leaves share;
- a leaf is `{id, kind: "leaf", slot: {index, name, kind, backends,
  config_hash, config_key}}`: the slot index (the order of a cost log's
  `slot_*` columns), the dotted leaf name, the kernel kind (its page is
  `/kernels/{kind}`), the backends the config may run, the config's hash (with
  the kind, the key of `/kernels/{kind}/configs/{config_hash}`), and
  `config_key`, `"<kind>:<config_hash>"`, its entry in `configs`;
- `configs`: `{config_key: {kind, config_hash, args, args_omitted, registry}}`,
  each leaf config once (rank copies share one): its scalar identity fields,
  the names of the structured ones, and `registry` (`{cells, infeasible,
  measured, usable}`, the last two per backend on this GPU: cells with a row,
  and feasible cells whose row is timed and no outlier) or null when the registry does not hold
  it. A config is keyed by kind and hash, as the registry keys it: the hash is
  of the identity, which leaves the kind out, so two kinds of one shape (a
  prefill and a decode attention, `rms_norm` and `residual_rms_norm`) share a
  hash and are two configs;
- `kernels`: `{kind: {documented, title, category}}` for the kinds the leaves use.

A leaf's config hash is computed from the leaf's own Rust config
(`KernelConfig::identity`: every field but `gpu_name` and `backends`, each
`Dim` reduced to its value, then `content_hash`), so with its kind it names
the same config the registry does; a binary-backed test checks it against the
records `supported-cost-trees --kernel-configs` writes.

The node and leaf shapes are the Analyzer's cost-tree transport
(ServingStudioUI `artifacts/schema/costTree`: `{kind, label?, children,
overlap | n}`, a leaf's `slot` with `name` and `kind`) without the numbers a
prediction adds there (`base`, `stats`, and the per-node `ms` and `pct`). Two
differences are deliberate: a leaf carries `config_hash`, `config_key` and
`backends` instead of the whole `kernel_config` (its `Dim` bindings run to
megabytes on the GLM trees; the config route has it), and every node keeps
`id` and every leaf its slot `index`, so a prediction can return times by node
and by slot without resending the tree.

## Predicting in the browser

The routes stay read-only: the Models page predicts a run in the reader's
browser, with Sim's timing-predict compiled to wasm (`just build-wasm`,
`simulator/wasm/`), from three reads.

- `run.predict` of the cost tree: `arch`, the run's arch block as a run config
  carries it (`type`, `model_config` as `model/config/<stem>.json`, the set's
  params and the run's; the predictor picks its contract, iter, speculative,
  attn or ffn, from `type`); `gpu`; `files`, the paths the block names that
  `/files` serves; `complete`, whether every config the tree reads is
  `predictable` (a reader greys out a run that is not: the gap is filled by
  measuring it, not by the predictor); and `error`, why it
  cannot be predicted (a routing file `routings.yaml` does not name, say).
- `/files?path=` for each of `files`: the predictor reads them from memory,
  by the path the block names. A token corpus binds by its manifest alone.
- `/archs/{arch}/kernel-data`, with the tree's query: per config, the
  registry's grid (`axes`) and, per cell, row-major, `feasible` and the row
  each backend `measured` there (the kind's metrics and `outlier`)
  (`profiling/db/kernel_data.py`, format 1). The predictor fits every kernel's
  cache from these and asks nothing else; a config the registry lacks is a
  gap to fill at the source (register the run), not something to guess.

`/health`'s `sim_commit` and the predictor's `version()` must agree, and the
documents' `format` must be the one the predictor reads, or the page shows no
number. A server-side `POST /archs/{arch}/predictions` stays possible later
behind the same inputs; nothing here writes profile.db.

## Sources

`public_api/kernel/library.py` joins only sources that live with the code:

- kernel prose: `DOC`, `arg(...)` and `BackendDoc` in `profiling/kernels/<kind>.py`
  (`profiling/db/doc.py`), and the profiling registry;
- which GPUs, models and parallel sizes an arch runs: `#[supported(...)]` on the arch
  variant in `simulator/src/arch/config.rs`, and model names in `model/catalog.yaml`
  (reread when it changes, no restart);
- compute dtype columns: `simulator kernel-list`; arch params and `#[supported]`
  rows: `simulator list-params`; every supported combination's cost tree:
  `simulator supported-cost-trees`, and a registered run's with `--archs`;
  all cached until the binary changes
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

## Agent skill

`skills/servingstudio-kernel-performance/SKILL.md` teaches a coding agent to
answer kernel-performance questions from these routes: find the kind, read what
it measures and how, then filter its rows. Install it with the
`skills` CLI, which finds every `SKILL.md` under the path it is given:

    npx skills add https://github.com/SyFI-ServingStudio/ServingStudioSim/tree/main/public_api

The skill reads the base URL from `SERVINGSTUDIO_API` and defaults to
`https://servingstudio.cs.washington.edu/api/public/v1`.
