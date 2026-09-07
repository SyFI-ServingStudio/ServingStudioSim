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

One arch type needs one map, and arch types that model the same model still need
one each. `glm52_dsa_moe_unified.json` (126 locations),
`glm52_vllm_dsa_moe_unified.json` (166), and
`glm52_vllm_nvfp4_dsa_moe_unified.json` (114) share all 82 semantic rows. Their
different decompositions split the same work into different leaves; quantize,
gather, and fill leaves have zero minimum, while communication leaves never
appear in a map.

Learned normalization scales are compulsory model weights and therefore remain
in the minimum at their norm locations. Only the intermediate norm/activation
tensor traffic is fusible away; norm and activation compute remain unpinned.

## Mixed precision comes from the checkpoint, not from a flag

A quantized checkpoint declares its own precision. HF FP8 repos keep the master
`dtype` at `bfloat16` and add one `quantization_config` key, so the two configs
for one model differ by exactly that key — `glm52.json` / `glm52_fp8.json`,
`qwen3_235b.json` / `qwen3_235b_fp8.json`. Both sides are verbatim downloads;
`tests/test_model_work.py` asserts they stay one key apart. **Point
`model_config` at the config the run actually served.** A run whose arch says
`fp8: true` while its config says nothing is rejected by `floors.py` — labeling
FP8 weights at two bytes reports a "minimum" larger than the traffic that moved,
which silently drives redundancy below 1 and is undetectable downstream.
The ModelOpt GLM-5.2 NVFP4 config similarly declares routed-expert-only FP4
weights, group size 16, and an FP8 KV cache; the NVFP4 arch type is checked
against that declaration while retaining BF16 as the unconverted fallback.

Precision is then per segment, never global, because a real checkpoint is mixed:

- `quantization_config.modules_to_not_convert` is the authority for weights.
  `MatmulGroup.module` carries each matrix's layer-relative checkpoint path so it
  can be matched. For GLM-5.2 and Qwen3-235B only `mlp.gate` (the router) and
  GLM's `self_attn.indexers_proj` stay at the master dtype.
- The list is an exclusion over the **Linear** modules the quantizer walks, so it
  says nothing about the embedding table — which is never converted. `lm_head`
  *is* listed by both checkpoints and is honored.
- A converted FP8 matrix also reads its FP32 block scale
  (`ceil(n/128)·ceil(k/128)·4` bytes), which is compulsory traffic.
- GLM-5.2 NVFP4 converts only `mlp.experts`: packed E2M1 weights cost half a
  byte each and read one FP8 E4M3 scale per 16 weights. Dense layers, attention,
  router, shared experts, embedding, and head remain BF16.
- A mechanism can fix its own precision independently of the weights:
  `AttentionSemantic.compute_dtype` pins GLM's DSA index logits to FP8 (the index
  cache is FP8 by construction) and its sparse MLA to BF16 (vLLM's FlashMLA
  kernel), under either checkpoint.

Each `Segment` therefore carries a `compute_dtype`, and both roofline floors
divide segment by segment. Fusing every leaf into one kernel still cannot fuse
across precisions, so even the global fused floor sums `flops_seg / peak_seg`.
On GLM-5.2 FP8 that is not a rounding detail: ~20% of the FLOPs stay on the BF16
tensor cores, and H200's FP8 peak is twice its BF16 peak, so a single global FP8
peak understates the compute floor by a fifth.

```
uv run python -m model.work model/config/llama3_8b.json --decode 256x4096
uv run python -m model.work <config> --prefill 8192@0 --gpu B200 --json
uv run python -m model.work.parameter_counts <config>
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
attention/glm52_dsa.py  GLM MLA/DSA             (future: fine-grained experts)
models/llama3.py     compose(GQA, dense)       models/qwen3_moe.py  compose(GQA, MoE)
models/qwen3_6.py    hybrid [GDN×3, GQA×1] + dense    registry.py  architectures[0] -> builder
models/glm52.py      dense/full-index + sparse/index-share stacks + shared MoE
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

GLM-5.2 MLA/DSA uses the optional multi-row attention contract: each full-index
stack emits indexer prefill/decode rows plus sparse-MLA rows and separate BF16
MLA-cache / FP8-index-cache append rows; index-share stacks omit the indexer rows
and index-cache writes while retaining the MLA cache write. Its q_absorb and v_up groups are the per-head W_UK/W_UV views of the
single learned `kv_b_proj` matrix, so the accountant preserves both execution
work and exact parameter totals.

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

## Speculative execution

For speculative GLM-5.2, R6/R7 are conditioned on the **executed algorithmic
workload**. Rejected draft/verify candidates remain necessary work. Acceptance
changes subsequent rounds, contexts, and batches; it does not remove queries
from a round that already ran. This is not a useful-output-only autoregressive
baseline.

The speculative cost logger preserves raw prefill `(prefix, query)` and decode
`(final_context, query)` geometry plus draft depth and the context limit.
`speculative.py` reconstructs causal target verification, the first MTP pass
with one sampled endpoint per request, and every later MTP endpoint separately.
Sparse top-k saturation is applied per interaction, before aggregation. Geometry
histograms preserve multiplicity without expanding whole-run request lists.
The ordinary scalar affine shortcut is not applied to these staged workloads.

The index-share, full-index, and single-draft location maps cover TP4 and TP8.
Weights follow the accountant's existing read-once lower-bound convention;
serial dependence alone does not prove that a shared weight must leave cache
and be fetched again. Kernel timing and CostTree approximations never determine
the necessary FLOP formulas.

Request-side admission/completion telemetry separately records completed rounds,
emitted outputs, resident KV, and pending work. Conservation uses those facts,
not an average acceptance-rate estimate, including at a mid-iteration stop.
Historical logs without these new observations remain explicitly unsupported.

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
golden. The `impl-add-model-work-label` skill formalizes this.
