# `model/work/` — optimal necessary-work labeler

The **independent ground-truth accountant** for VibeSim's redundancy analysis. Given a
model `config.json` + a `Workload`, it computes the *theoretical minimum* compute
(FLOPs) and memory traffic (bytes) the forward MUST do, plus total/activated parameter
counts. A real run's *achieved* work (the sim's logged `slot_flops` / `slot_bytes`,
summed with the per-slot physical GPU multiplicity) is then measured against this
minimum; `achieved / minimum` is the redundancy factor.

> **Independence is the whole point.** The minimum is derived only from the model
> config + the workload — never from the simulator's kernel tree. An accountant built
> from the sim's own decomposition could never detect the sim doing redundant work.

## Use

```python
from model.work import load_model, Workload

label = load_model("model/config/llama3_8b.json").label(
    Workload.causal_lm(decode=[4096] * 256, sampled=256)
)
label.flops        # {attn_proj, attn_internal, ffn, router, lm_head}
label.bytes        # {weights, kv}
label.params       # {total, activated, breakdown{...}}
label.segments     # per-op [Segment(name, flops, bytes, count, ...)]
label.roofline_ms("H200", "bf16")               # global floor: (compute, memory, bound)
label.segmented_lower_bound_ms("H200", "bf16")  # realistic: Σ per-op max(compute, memory)
label.work_efficiency(achieved_flops)           # F_min / achieved ∈ (0, 1]
```

For fixed-batch per-location attribution, `segments` are stable semantic work
rows, not simulator shapes. Causal GQA emits separate `attn.prefill` and
`attn.decode` rows plus `kv_cache_append`; the latter counts compulsory K/V
cache writes that remain even when adjacent leaves fuse. Versioned files under
`location_maps/` map these semantic rows to exact CostTree location names. A map
must consume every semantic row exactly once and explicitly list every
non-communication location. An empty semantic list means that location has zero
minimum under the cross-leaf-fusion convention. The mapping never derives
minimum work from simulator shapes.

Learned normalization scales are compulsory model weights and therefore remain
in the minimum at their norm locations. Only the intermediate norm/activation
tensor traffic is fusible away; norm and activation compute remain unpinned.

```
uv run python -m model.work model/config/llama3_8b.json --decode 256x4096
uv run python -m model.work <config> --prefill 8192@0 --gpu B200 --json
```

## Architecture: compose, don't cross-product

Attention mechanism and FFN mechanism vary **independently**, so a model is one
`AttentionSpec` + one `FFNSpec` composed by a thin per-model builder — not one file per
full `(attention × ffn)` combination.

```
core.py            physics: Workload, AttnInteraction, MatmulGroup, Segment, WorkLabel,
                   LayerStack, Model, roofline
attention/base.py  AttentionSpec protocol      ffn/base.py    FFNSpec protocol
attention/gqa.py   MHA / MQA / GQA (+gate)     ffn/dense.py   dense SwiGLU
attention/linear.py  Gated DeltaNet (O(T))     ffn/moe.py     router + routed/shared experts
                   (future: mla, swa)                        (future: fine-grained experts)
models/llama3.py     compose(GQA, dense)       models/qwen3_moe.py  compose(GQA, MoE)
models/qwen3_6.py    hybrid [GDN×3, GQA×1] + dense    registry.py  architectures[0] -> builder
```

- **New attention (MLA/SWA/SSM)** → one new `attention/*.py`; every FFN combination is free.
- **New FFN (shared experts)** → one new `ffn/*.py`; every attention combination is free.
- **New checkpoint of a known family** → zero code (numbers come from its `config.json`).
- **Unknown `architectures[0]`** → hard error (the "not yet labeled" guard).

### Heterogeneous layer stacks (hybrid attention)

A model is not `one attn × N layers` — it is a list of `LayerStack(attn, ffn, count, tag)`
archetypes. Uniform models (`Model.uniform(...)`) are a single stack; a **hybrid** model
lists one stack per archetype. Qwen3.6-27B is the worked example: its 64 layers are
`[linear, linear, linear, full]` × 16 (`full_attention_interval=4`) → **48 Gated DeltaNet
linear-attention + 16 gated full-attention layers**, each over the same dense SwiGLU MLP.
`label()` folds each stack with its own `count`; the stack `tag` prefixes segment names
(`linear.attn`, `full.attn`, …) so the per-op table stays legible. This is what a
dense-then-MoE schedule (DeepSeek's first-k-dense) would use too.

Two attention mechanisms therefore coexist, and their cost models differ in kind:

- **`GQA`** — quadratic softmax attention. `internal_flops` scales with the causal-triangle
  `pairs`; `kv_bytes` grows with cached tokens. `output_gate=True` adds the Qwen3.5/3.6
  gate matmul on the q side (the elementwise gate multiply itself stays out of the denominator).
- **`GatedDeltaNet`** — linear attention. `internal_flops` is **O(T)** (3 `d_k·d_v` products
  per token per value head, counted off HF's `torch_recurrent_gated_delta_rule`: `kᵀS`, the
  rank-1 delta update, and `qᵀS` → `6·num_v_heads·d_k·d_v` per token), with **no** `(q,k)` pair
  blow-up. `kv_bytes` is a **fixed** recurrent state (`num_v_heads·d_k·d_v` per sequence,
  read+write each step) that does **not** grow with context — so at long context the labeler
  shows the 48 linear layers' state flat while the 16 full layers' KV cache dominates.

The only currency between a spec and `core.py` is
`MatmulGroup(name, n, k, activated_mult, total_count, bucket)`; `core.py` applies
`flops = 2·(matmul_tokens·activated_mult)·n·k` and folds `× num_layers` uniformly.

## Modality-agnostic workload

`Workload` is **not** a fixed causal-LM struct. The weight/FFN/param side needs only the
universal `matmul_tokens` (T) and `head_positions` (T_out); all attention-specific
geometry lives in a list of normalized `AttnInteraction(num_query, num_key,
num_cached_key, mask)`. `mask` picks causal-triangle vs full-rectangle pair counts;
`num_cached_key` picks KV-read vs fused-zero. Text decoders use `Workload.causal_lm(...)`;
vision encoders / cross-attention are future constructors — the `label()` signature never
changes.

`attention_step_count` normally equals `len(attn)`. Analyzer inputs may collapse many
GQA interactions into one geometry-preserving aggregate; the explicit count retains the
original recurrent-state transactions required by linear-attention models without
materializing every decode step.

## Two time floors

`label(wl)` also decomposes the forward into `segments` — one kernel-like unit per
matmul + one fused attention unit per layer + embedding + lm_head — each with its own
FLOPs and bytes, hence its own arithmetic intensity and its own bound. Two roofline
readings fall out:

- **global floor** (`roofline_ms`) — `max(ΣFLOPs / peak, Σbytes / bandwidth)`. Assumes
  the whole forward is one fused kernel where all compute and all memory overlap. The
  loosest lower bound.
- **segmented lower bound** (`segmented_lower_bound_ms`) —
  `Σ_seg max(FLOPs_seg / peak, bytes_seg / bandwidth)`. Separate kernels run
  sequentially and cannot overlap each other, so this is tighter (≥ global) and more
  realistic. Still a valid lower bound — each segment counts only necessary work.

The gap between them is the compute a memory-bound global view hides (e.g. decode
attention is KV-bandwidth-bound, but the QKV/FFN GEMMs still add their compute time on
top). Ceiling throughput is reported from the segmented bound.

For a mapped CostTree, the per-location segmented floor is recomputed after
semantic rows are assigned: each location first sums its minimum FLOPs/bytes and
then applies its own `max(FLOPs/peak, bytes/bandwidth)`. Llama's v1 mapping is
one-to-one for non-zero semantic rows, so this reconciles exactly with the
semantic segmented lower bound.

## The four pinned conventions (what defines `F_min`)

1. **Causal prefill uses the exact triangle** `pairs = q·(k − (q−1)/2)`, not the full
   rectangle `q·k`.
2. **`lm_head` counts only sampled positions** (`head_positions`), not every token — so a
   run that projects all prefill tokens through the head shows up as redundancy.
3. **`router` is a real matmul** → counted in `activated_params` and `router` FLOPs.
4. **`norm / rope / softmax / activation` compute and fusible intermediate traffic are
   OUT of the denominator.** Learned norm scales and persistent KV-cache writes are
   weights/state that cannot be fused away, so their bytes remain in the denominator.
   Keeping optional elementwise math out keeps `F_min` exact and oracle-checkable.

## Parameter counts

`MatmulGroup` carries both `activated_mult` (per-token instances, `top_k` for experts) and
`total_count` (all instances, `num_experts`). "Activated" has more than one defensible
convention, so `params["activated"]` exposes **all** of them rather than picking one:

- `activated["layers"]` = `Σ activated_mult·n·k · num_layers` plus declared norm
  weights — the per-token transformer parameters (attn + ffn/experts + router + norm).
- `activated["with_embed_head"]` = the above **+ embedding + lm_head** — the "A-XXB" figure
  most model cards quote (e.g. Qwen3-235B-**A22B**: layers ≈ 20.9 B, with-embed-head ≈ 22.2 B).
- `total` = `Σ total_count·n·k · num_layers` + declared norm weights + embedding + lm_head (respecting
  `tie_word_embeddings`) — the model's headline size (e.g. 235 B).

## Two validation gates

- **Golden** — hand-computed FLOP buckets + `total`/`activated` params for a reference
  model, asserted in `tests/test_model_work.py`. (Llama3-8B: total 8.030 B, activated
  6.979 B; Qwen3-235B-A22B MoE: total ≈ 235 B, activated ≈ 22 B; Qwen3.6-27B hybrid:
  total 26.895 B over 48 GDN + 16 gated-full layers, dense so activated == total.)
- **Oracle** (Phase 3) — cross-check against the sim's *unsharded* per-op `slot_flops`:
  for a sharded, non-replicated matmul, `Σ achieved == labeler F_min` bucket (ratio ≈ 1).

## Adding a model

Derive the matmul inventory + attention/KV math from the **model's true architecture**
(HF modeling code / paper — route through `top-split-model-into-kernels`), *not* by copying
VibeSim's arch shapes. Add/compose the specs, register the `architectures[0]` string, add a
golden. The forthcoming `dev-add-model-work-label` skill formalizes this.
