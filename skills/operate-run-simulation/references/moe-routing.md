# Select MoE routing for simulation and timing prediction

Prefer a matching measured routing artifact over synthetic uniform demand — a
token corpus first, an expert-popularity marginal otherwise. Apply this to every
MoE arch that costs routed experts, including TP-only models.

1. Honor an explicit user routing choice. For uniform demand, set `routing:
   uniform` and remove corpus/popularity-file fields from the copied config. An
   inherited preset default is not an explicit user request.
2. Otherwise inspect any configured `token_corpus_file` or
   `expert_popularity_file`, then search the source preset/campaign directory,
   `presets/alignment/`, and relevant existing experiments under `logs/`
   (including ignored artifacts). Use `rg --files --hidden --no-ignore
   <directories>` to locate corpus manifests and popularity JSON files and
   inspect nearby configs for referenced files with other names. Check supplied
   artifact locations too; do not launch a new profiling campaign just to search.
3. Match the checkpoint and workload where recorded, and verify expert count,
   MoE layer count, top-k, and EP size against the selected arch. For a corpus
   also check that its `group_size` equals the deployment's verify width
   (`draft_tokens + 1`). Prefer the same workload/campaign when several files
   match; record any workload difference. Do not alter metadata to make an
   incompatible file pass.
4. Set `routing: corpus` with `token_corpus_file`, or `routing: popularity` with
   `expert_popularity_file`, on each applicable arch. Either file serves the
   whole model: a marginal has no layer axis left, so a speculative model's MTP
   layer folds the same one, while a corpus keeps the layer axis and the MTP
   layer is a slice of it. Resolve paths for the actual launcher invocation after
   copying configs; absolute paths avoid ambiguity.
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

A corpus is hundreds of megabytes, so it is not in the repository.
`token_corpus_file` accepts either a local path — what a fresh capture writes,
under `<log_dir>/token_corpus/manifest.json` — or a hub reference that the
launcher fetches during expansion:

```yaml
routing: corpus
token_corpus_file: hf://uw-syfi/servingstudio-corpora@<commit-sha>/glm53/manifest.json
```

The revision must be a commit sha; a branch or tag is refused. Search for an
existing corpus the same way as a popularity file, and record which one was
used — two corpora of the same model are different recordings.

Both artifacts come from one capture: `profile_kind: token_corpus` in
`operate-run-alignment`, which writes the corpus and, when the expert topology
is declared, the marginal as well. Do not launch one just to satisfy this rule;
that is a GPU campaign, not a config step.

The runtime never falls back from a measured source to uniform. The agent chooses any
permitted fallback while preparing the config and reports it with the result.
Check `simulator/src/arch/config.rs` for supported fields and
`simulator/src/arch/build.rs` for profile validation. AFD FFN selectors without
popularity-file fields cannot use measured routing.
