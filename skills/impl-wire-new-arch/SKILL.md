---
name: impl-wire-new-arch
description: >-
  Use when wiring a just-built VibeSim L4 arch model into the shared dispatch sites
  so it is selectable by tag, predictable via timing-predict, and ready for a
  future worker — the integration step AFTER impl-compose-arch has produced the
  arch file. Covers arch/mod.rs re-exports, the arch/build.rs concrete builder +
  its uniform predictor dispatch (build_iter_model / build_attn_model /
  build_ffn_model), and the deployment build arm (worker pairing or explicit bail).
  The completion proof is a green timing-predict run on the new arch. Does NOT author
  the arch cost file / trait impl (that is impl-compose-arch) or build a worker.
---

# Impl Wire New Arch

You are the implementer for the **cross-file integration** that turns a buildable
L4 arch into a *reachable, predictable* one. `impl-compose-arch` produced the arch
model file (`arch/<family>.rs`: `build_configs → resolve_configs → build`, the
compiled `CostTree`, and the contract-trait impl) and declared its `config.rs`
selector variant. Your job is to connect that model to the shared dispatch sites
so it can be selected by `type:`, costed by `timing-predict`, and later run by a
worker with no further arch surgery.

The through-line of `top-add-new-arch` is **where the timing comes from**. This
phase does not add timing — it *wires the composed timing so a caller can evaluate
it*. The tightest such caller is `timing-predict` (no worker, no DES), which is
why this phase exists before Validate: it makes the new arch predictable the
moment its cost model builds.

## Precondition (from impl-compose-arch)

Do not start until all hold:

- `simulator/src/arch/<family>.rs` builds and its unit tests pass in isolation;
- `pub mod <family>;` is in `simulator/src/arch/mod.rs`;
- the `config.rs` selector variant parses (`just test-cpu` config round-trip green);
- the model impls the right contract trait for its kind:
  - iter → `IterwiseUnifiedModel`; attn → `AttnLayerwiseModel`; ffn → `FfnLayerwiseModel`
  (`arch/contract.rs`). If it does not, that is impl-compose-arch's leaf, not this one.

## Read First

- `simulator/src/arch/README.md` (the L4↔L5 contract) and `arch/contract.rs`;
- `simulator/src/arch/build.rs` (the `build_iter_model` / `build_attn_model` /
  `build_ffn_model` predictor matches + the concrete `dense` / `qwen3_moe` /
  `qwen3_attn` / `qwen3_ffn_moe` builders — copy the closest);
- `simulator/src/timing_predict.rs` (`PredictArchSel` and the `run_iter_cases` /
  `run_attn_cases` / `run_ffn_cases` drivers that call the `build_*_model` seam);
- the matching deployment: `deployment/unified.rs` + `deployment/pd.rs` (iter),
  or `deployment/afd.rs` (attn/ffn); and `deployment/config.rs` (the pool maps).

## The wiring surface — the compiler is your checklist

Adding a selector variant makes the **exhaustive** matches below fail to compile
until you add its arm. Let the build errors drive you; there is no central
registry to update (the launcher `list-params` schema is *derived* from
`#[derive(ProviderSchema)]` + `#[param]`, so a new tag surfaces with no manual arm).

**Compiler-forced (no catch-all — you MUST add an arm):**

| Site | iter | attn | ffn |
|---|:--:|:--:|:--:|
| `arch/config.rs` — `*ArchSel::model()` `\|`-chain | ✔ | ✔ | ✔ |
| `arch/build.rs` — `build_{iter,attn,ffn}_model` match (**the predictor seam**) | ✔ | ✔ | ✔ |
| `deployment/unified.rs` — `match &g.arch` | ✔ | — | — |

The predictor dispatch is **uniform**: all three kinds go through a
`build_*_model` match in `arch/build.rs` (each returns `Box<dyn …LayerwiseModel>` /
`Box<dyn IterwiseUnifiedModel>`). `timing_predict.rs` just *calls* those and needs
**no** per-arch edit. So the rule is one-liner: **one predictor arm in
`arch/build.rs` + one deployment arm — for every arch kind.**

**Catch-all `bail!` (compiles without you; add an arm only to enable that deployment):**
`deployment/pd.rs` (iter, tuple match on prefill/decode) and `deployment/afd.rs`
(attn/ffn, tuple match on attn/ffn). Leaving these unwired is fine for a
timing-predict-only milestone — the arch still predicts; only a real run bails.

> **Two arms, two consumers — wire both.** Every arch is built from TWO places
> that share the same concrete `arch_build::<family>` builder but dispatch
> **independently**: the **predictor** arm (`build_*_model` in `arch/build.rs`,
> boxed, no worker) and the **deployment** arm (`unified.rs`/`pd.rs`/`afd.rs`, a
> concrete-typed worker flow). Wiring only one means a green timing-predict with a
> bailing real run, or vice-versa. This shape is now identical for iter and AFD —
> the predictor seam lives in `arch/build.rs` for all three kinds (it used to be
> split off in `timing_predict.rs` for AFD; that asymmetry was removed so a second
> AFD arch lands as one more match arm, not a signature break).

## Expected Write Scope

- `arch/mod.rs` — `pub use <family>::{<Model>, <Parallel>};` so `build.rs` can name
  the concrete type (the `pub mod` line is already there from the precondition).
- `arch/build.rs` — a concrete builder `pub fn <family>(model_spec, <parallel
  params>, gpu, name, bridge) -> Result<<Model>>` mirroring the closest existing
  one (`ModelSpec` → `dense_model_cfg`/`moe_model_cfg`, build the `*Parallel`
  struct, `resolve_configs(build_configs(...))`, `build(name, resolved, bridge)`
  with a `.context(...)` naming the likely missing-`profile.db`-row failure). Then
  add its **predictor arm** to the matching `build_iter_model` / `build_attn_model`
  / `build_ffn_model` (`Box::new(<family>(...)?)`) — the uniform seam both the
  deployment and the predictor build from.
- `timing_predict.rs` — **no per-arch edit.** It calls `build_{iter,attn,ffn}_model`
  generically, so the `arch/build.rs` predictor arm above is all prediction needs
  (for any kind — iter and AFD alike).
- `arch/config.rs` — extend the `model()` `|`-chain to include the new variant
  (compiler-forced). Confirm the variant's `#[param(cache_key)]` flags are right
  (model identity / dtype / any param that changes which kernels are needed).
- deployment arm — **pave the road for the worker**: add the deployment `build`
  arm that constructs the model via your same `arch_build::<family>` concrete
  builder and pairs it with a worker. If the worker does not exist yet, `bail!`
  with an explicit `"<deployment>: <arch> requires worker <X>, not wired yet"` so
  the future worker dev is a drop-in (it just fills this arm). For `unified.rs`
  (compiler-forced) you must add *some* arm now — a clear `bail!` is acceptable.

Do NOT edit the arch cost file, add a worker impl, change `contract.rs` traits, or
touch the launcher schema derive. Stop and return to the orchestrator if the arch
does not fit an existing contract trait or needs a new `PredictArchSel` family.

## Tests And Smoke

Build + CPU tier (config round-trip, arch unit tests, dispatch compiles):

```bash
uv run cargo build --release -p simulator
just test-cpu
```

Confirm the tag is in the launcher schema (derived — should appear with no arm):

```bash
uv run python -m launcher list-params | grep -i <tag>
```

**The completion proof — timing-predict runs green on the new arch.** Author a
minimal predict config (see `operate-run-timing-predict`; templates in
`presets/predict_qwen3_235b_*`) selecting the new arch and a couple of cases:

```bash
cd /m-coriander/coriander/kanzhu/VibeSim_workspace/main
CUDA_VISIBLE_DEVICES=<idle> uv run python -m launcher timing-predict presets/<new_arch_predict>.json
```

Read `logs/<dir>/reports/iter_breakdown.ans`: every element from the Phase-1
decision table must appear as a leaf with a non-zero, plausibly-scaled timing (no
missing kernel, no zero, no absurd µs). A cold `profile.db` JIT-fills on the idle
GPU; a warm one needs none.

## Report Back

Return: the arch kind (iter/attn/ffn) and selector tag; files + match arms added;
the concrete `build.rs` builder signature; whether the deployment arm is wired or
bails (and which worker it awaits); `list-params` confirmation; and the
timing-predict smoke result (log dir + per-case totals, or the first missing/zero
leaf if the cost model is incomplete — that is an impl-compose-arch gap, not a
wiring one).
