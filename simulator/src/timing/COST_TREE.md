# CostTree — structure / data separation

`cost_tree.rs`. The mechanism that lets the simulator pay for a cost model's
*structure* once and its *numbers* every iteration.

The structure of a cost query — which L1 primitives run, how they compose, where
a homogeneous layer repeats — is **stable across iterations**; only the leaf
metrics change with batch shape. So we compile the structure once into a
`CostTree`, then stream per-iter leaf `LeafMetrics` through it and fold them to one
total. This file is the code-matching reference for the CostTree.

## The node algebra

The build-time tree is a recursive `CostNode`; the eval-time form is the flat
`FlatCostNode` array that `flatten()` lowers it to.

| `CostNode` | meaning | time | flops / bytes / energy |
|---|---|---|---|
| `Leaf(slot)` | one L1 primitive; `slot` indexes the per-iter buffer | from cache | from cache |
| `Sum(children)` | serial composition | Σ children | Σ children |
| `Scale{n, child}` | homogeneous-layer fold: `n ×` one child subtree | `n ×` | `n ×` |
| `Max{overlap, children}` | fan-out / overlap | `max(child.time)/overlap` | **Σ children** (work never overlaps away) |
| `Labeled{label, child}` | render-only identity wrapper | — | — (dropped at flatten) |

`flops`/`bytes`/`energy` **always sum**; only `Scale` and `Max{overlap}` change
`time` (INV-4). `coverage` flags always OR up the tree, so an off-grid leaf
anywhere surfaces at the root.

`Scale` is the key economy: a 32-layer model evaluates one layer subtree and
multiplies, never materializing 32 copies. Use it **only for provably-identical
subtrees** (INV-3); a heterogeneous fan-out (e.g. uneven EP) expands into `Max`
with N distinct children instead.

## One walk, two visitors

Slot index = **fixed traversal visit order** (INV-2). Compile and eval share the
*same* walk, so the indices line up by construction:

- **`CostTreeBuilder`** (compile): `builder.leaf(name, kind, config)` mints the
  next slot and returns a `CostNode::Leaf(slot)`. Each layer's hand-written
  `compile` assembles its `CostNode` subtree around these, mirroring its lookup.
- **`Evaluator`** (eval): `evaluator.push(metrics, make_input)` writes the next
  leaf's `LeafMetrics` into `buf[cursor]` and advances. `eval` must push leaves in
  the order `compile` minted them. `with_inputs` additionally records each leaf's
  typed `SlotInput` (for the `cost_log` capture); the no-logger path never clones.

Because the two visitors walk identically, `buf[slot]` always holds the right
leaf — no name lookup, no map.

## flatten → a single reverse pass

`flatten()` lowers the recursive tree to `Vec<FlatCostNode>` in **BFS layout**:
each composite reserves a contiguous block for its direct children, so `children`
is a valid `Range<usize>` and **every parent index precedes its children's**.
That layout is what makes `aggregate` a single bottom-up pass with no recursion
and no allocation when the caller reuses the scratch buffer:

```rust
// CostTree::aggregate(flat, buf, scratch) — iterate high→low index.
scratch.resize(flat.len(), LeafMetrics::ZERO);
for i in (0..flat.len()).rev() {
    scratch[i] = match &flat[i] {
        Leaf(slot)            => buf[*slot],
        Sum { children }      => Σ scratch[children],
        Scale { n, children } => (Σ scratch[children]) × n,
        Max { overlap, children } => { time = max/overlap; work = Σ },
    };
}
// scratch[0] (the root) is the answer.
```

Walking high→low guarantees each child's subtree result is ready when its parent
is reached (parent index < child indices). The scratch buffer is caller-owned and
reused by the worker/model across iterations.

## Names stay off the hot path (INV-5)

Identity lives **only** in compile-time products, never in `FlatCostNode`, the
`aggregate` pass, or log rows:

- **`LeafDesc`** (`slots[i]`) — the dotted leaf `name` + the kernel `kind` /
  one-line `config` summary, captured at compile for the shape render.
- **`CostManifest`** — the per-worker
  `cost_manifest/worker_<pool_tag>_<worker_id>.json` sidecar: `slots` + the
  flat `nodes` + `node_labels` (index-aligned to `nodes`; recovers the
  `Labeled` composite identity that `flatten` drops). This is what lets a
  downstream analyzer reproduce `total_time_ms` from a row's per-slot
  `slot_time_ms` (by re-running `aggregate`) and group slots into semantic
  subtrees *structurally*, without parsing dotted names.

`FlatCostNode` carries no `String`; `cost_log` rows carry only `slot_*` lists
keyed by position. Identity is re-attached from the manifest, off the hot path.

## Worked example

`Sum( a, Scale{3}( Sum(b, c) ), d )` — the dense shape in miniature (a fold
wrapping a 2-leaf layer). Slots are minted `[a, b, c, d]` (folded leaves counted
**once**, `n_slots() == 4`). With `buf = [a, b, c, d]`:

```
total = a + 3·(b + c) + d        # time / flops / bytes all fold the same way
```

`describe()` renders it for inspection:

```
Sum
│  Leaf#0 a (ka) x=1
│  Scale{n=3}
│  │  Sum
│  │  │  Leaf#1 b (kb) x=2
│  │  │  Leaf#2 c (kc) x=3
│  Leaf#3 d (kd) x=4
```

## Invariants

- **INV-1** Structure is fixed across iterations; variable fan-out must aggregate
  into fixed slots. `n_slots()` is decided at compile.
- **INV-2** slot = fixed visit order; compile (`CostTreeBuilder`) and run
  (`Evaluator`) share one walk.
- **INV-3** `Scale`/fold only for provably-identical subtrees; heterogeneous
  fan-out uses `Max` with N children.
- **INV-4** `flops`/`bytes`/`energy` always sum; only `Max{overlap}`/`Scale`
  touch `time`; `coverage` always ORs up.
- **INV-5** names/labels live only in compile-time products (`CostManifest`);
  the hot path and log rows are name-free, reconstructed via slot position +
  manifest.
- **INV-6** runtime aggregation yields the scalar that drives the clock; the full
  tree / any taxonomy is replayed by the analyzer from the manifest — the sim
  builds in no taxonomy.

> The dense vertical currently emits only `Leaf` / `Sum` / `Scale`; `Max` is
> defined for the full algebra and lands with the first overlap/fan-out
> (HP / EP) path.
