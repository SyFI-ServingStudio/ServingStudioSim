# Select MoE routing for simulation and timing prediction

Prefer a matching expert-popularity file over synthetic uniform demand. Apply
this to every MoE arch that costs routed experts, including TP-only models.

1. Honor an explicit user routing choice. For uniform demand, set `routing:
   uniform` and remove popularity-file fields from the copied config. An
   inherited preset default is not an explicit user request.
2. Otherwise inspect any configured `expert_popularity_file`, then search the
   source preset/campaign directory, `presets/alignment/`, and relevant existing
   experiments under `logs/` (including ignored artifacts). Use `rg --files
   --hidden --no-ignore <directories>` to locate popularity JSON files and
   inspect nearby configs for referenced files with other names. Check supplied
   artifact locations too; do not launch a new profiling campaign just to search.
3. Match the checkpoint and workload where recorded, and verify expert count,
   MoE layer count, top-k, and EP size against the selected arch. Prefer the
   same workload/campaign when several files match; record any workload
   difference. Do not alter metadata to make an incompatible file pass.
4. Set `routing: popularity` with `expert_popularity_file` on each applicable arch.
   One file serves the whole model: a marginal has no layer axis left, so a
   speculative model's MTP layer folds the same one. Resolve paths for the actual
   launcher invocation after copying configs; absolute paths avoid ambiguity.
5. Use uniform only when the user requests it or no matching file is found
   after searching. Record the searched locations and missing-file reason, and
   remove stale file fields from the copied config. A malformed/unreadable file,
   validation failure, or arch lacking profile support is not a missing file:
   report that issue rather than silently substituting uniform. If the user
   explicitly requires a file/measured routing, a missing file is an error too.

`popularity` requires a valid file; `uniform` and `random` reject popularity
files. `corpus` reads recorded per-token routes from `token_corpus_file` on the
GLM-5.2 NVFP4 archs: prefer it over `popularity` when the deployment drafts
(`draft_tokens > 0`), because a marginal cannot express which experts a verify
block's tokens jointly select. At verify width 1 the two agree and either is
fine. A corpus is also the only source that measures the MTP layer's own
routing, because it keeps the layer axis a marginal has summed away.
The runtime never falls back from a measured source to uniform. The agent chooses any
permitted fallback while preparing the config and reports it with the result.
Check `simulator/src/arch/config.rs` for supported fields and
`simulator/src/arch/build.rs` for profile validation. AFD FFN selectors without
popularity-file fields cannot use measured routing.
