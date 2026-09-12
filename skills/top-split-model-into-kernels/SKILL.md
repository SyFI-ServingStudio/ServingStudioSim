---
name: top-split-model-into-kernels
description: >-
  Use as the entry point when the user wants to break a Transformer model's
  forward pass into the sequence of ServingStudio Sim kernels it costs as — the per-op
  boundary decision. First calls top-explore-models to establish the
  architecture, then for each op routes to dev-lookup-transformers-model (what
  math) and dev-explore-kernel (whether a real fused kernel exists at what launch
  granularity), then applies the boundary rules to assign each op a home: reuse
  an existing kernel, a new dedicated kernel (hand to top-add-kernel), an
  `elementwise` byte-placeholder, or fold into a neighbor. Never skips a
  cost-bearing op. Produces a per-op decision table, not code.
---

# Top Split Model Into Kernels

Top-level orchestrator. Turn a model's forward pass into the sequence of ServingStudio Sim
kernels it costs as. You own the boundary judgment; you do not implement kernels
and you do not run the ecosystem search yourself — you sequence the lower-level
skills and decide from their findings. Your output is the input to
`top-add-kernel` for any op that needs a new dedicated kernel.

## Step 1: establish the architecture

Start by calling `top-explore-models` to settle what the model is: parameter
scale, overall architecture, attention type (MHA / GQA / MLA / GDN / …),
layer / head / hidden dims, context length, and MoE layout. Do not begin the
per-op split until the architecture is settled — the arch is what tells you which
reference decomposition is closest and which ops are the deltas that need a fresh
boundary decision.

## Step 2: diff against the nearest reference

Do not judge every op from scratch. ServingStudio Sim already has canonical decompositions —
the dense GQA transformer in `simulator/src/arch/llama3_dense.rs` plus its
`worklet/` and `op/` leaves is the worked example. Find the closest existing
reference decomposition, reuse its verdicts wholesale, and only give the
**delta** ops (what this model does differently) a fresh boundary decision. A GDN
model reuses every dense projection/norm/MLP verdict and only re-decides
attention.

## Per-op investigation

For each op that needs deciding, gather two facts before judging:

- **What math it computes** — via `dev-lookup-transformers-model` (exact forward
  path, QKV / RoPE / cache / MLP / MoE semantics). Read HF/Torch for the math,
  not the granularity: HF eager materializes everything and does NOT reflect
  kernel launch boundaries.
- **Whether a real fused kernel exists, at what launch granularity** — via
  `dev-explore-kernel` (vLLM / SGLang / FlashInfer / flash-attn / cutlass). The
  fact that matters is whether the op is issued as one fused launch or several.

## Assign a home (four verdicts, never skip)

Re-project the HF math onto the launches the optimized serving stack actually
issues, then give each op exactly one home:

1. **Reuse an existing kernel** — maps onto `single_gemm`, `grouped_gemm`,
   `rms_norm`, `all_reduce`, `p2p_*` with its args schema. Covers QKV / O /
   gate / up / down projections, norms, the router GEMM, and collectives.
2. **New dedicated kernel** — a real fused launch whose cost no existing kernel
   captures (GQA / GDN / MLA attention, novel fused ops). Hand it to
   `top-add-kernel`.
3. **`elementwise` byte-placeholder** — a small memory-bound op that runs as its
   own launch and no heavy kernel absorbs (RoPE, KV-append, a standalone copy).
   Size it by input/output bytes per token. This is the no-skip floor: its cost
   is real, just cheap — do not drop it.
4. **Fold into a neighbor** — a standard epilogue the adjacent heavy kernel
   genuinely fuses, so its cost is already in that kernel's measurement:
   bias-add, scale, and the residual-add fold into the preceding GEMM
   (`o_proj` / `down_proj`) as the `+C` epilogue. Name the host and add nothing —
   a placeholder here would double-count. Fold only when you are confident the
   fusion is real; otherwise it is a placeholder (verdict 3), not a fold. Never
   dress up an omit as a fold — "it's inside attention anyway" for a memory-bound
   op like RoPE is really dropping it, which verdict 3 exists to prevent.

## The boundary rules

Two tests decide where verdicts 1–4 split. The launch boundary of the real
serving stack is the ground truth; both failure modes are deviations from it:

- **Measurable, or the boundary is too coarse.** Every unit must be something a
  single microbench can produce a number for. A unit with no kernel on the table
  that is not an `elementwise` placeholder means the boundary was drawn too big —
  shrink it toward the launch boundary.
- **Do not split a fusion.** If the stack issues several math ops as one launch
  and `cost(fused) != sum(cost(parts))` — attention never materializes the score
  matrix — keep them one unit. Assembling GQA attention out of basic GEMM +
  softmax kernels is the classic wrong answer.

No-skip floor: any op too small to earn a bespoke kernel still gets verdict 3,
never dropped. `elementwise` is both the floor (nothing lost) and the guard
against minting N tiny kernels for cheap memory-bound ops.

Separate semantic slots from physical repetition. One inseparable public
operation is one slot even when it launches a fixed sequence of kernels or owns
a runtime chunk loop. Do not expand runtime chunks, ranks, layers, or capacity
into permanent `chunk_00...N`-style slots. Conversely, two independently issued
production operations remain two slots even when they are adjacent or share an
implementation library.

Before verdict 2, record why verdict 1 cannot preserve the operation semantics,
args meaning, dtype/layout, public callable, and logical launch boundary. A
backend is not a license to reinterpret an existing kind's args.

## Output

A per-op decision table for one decoder layer, plus the outside-loop ops (embed,
final norm, lm_head):

| op | math (source anchor) | fused kernel found? | verdict | maps to |
|----|----------------------|---------------------|---------|---------|

For each op record the verdict, the target kernel kind + args axes (or the
`elementwise` byte-rate, or the host kernel it folds into), and the evidence
anchor. Every cost-bearing op must appear — verify completeness before returning.
Route the verdict-2 ops to `top-add-kernel`.
