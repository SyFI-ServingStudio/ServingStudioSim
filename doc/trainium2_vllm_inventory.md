# Stock vLLM Neuron: Llama 3.1 8B TP4 inventory

This is the Phase 1 decision table for the stock vLLM Neuron path. It does not
describe the earlier separately compiled NxDI TP1 decoder. Python profiling,
Rust timing, the full-model predictor and independent necessary-work mapping
are implemented. Worker integration and native/clean HTTP capture are validated;
shared timing and serving alignment artifacts are produced separately.

## Frozen evidence

The source is vLLM Neuron `0.24.0.1.1.0`, commit
`f8abae640a43824c1dc73aed3cf2f67b83bce507`. Paths below are relative to that
checkout, available locally at `../tmp/vllm-neuron-stock/`.
The immutable image is
`sha256:44d2eef799027b5c925af25a1ad9f45a93aae07451bdf9c02857ec8a7635a24a`.
Model weights and KV storage are BF16; the original checkpoint has 32 layers,
hidden width 4096, intermediate width 14336, 32 query heads, 8 KV heads,
head width 128 and vocabulary 128256, with untied embedding/head and Llama 3.1
scaled RoPE. One Trainium2 chip supplies four LNC2 ranks, eight physical cores.

The accepted Phase 0 evidence is in `../tmp/vllm-neuron-phase0-sweep/` and
`../tmp/vllm-neuron-capacity-{128,512,2048}/`. Their frozen `compiled-models/`
directories contain FX graphs, actual compiler commands, input layouts and NEFFs.
`../tmp/vllm-neuron-phase1/inventory.py` verifies every compiler-artifact hash,
the complete 32-layer graph and correspondence to measured graph hashes. Its
`inventory.json` covers 24 distinct graphs in 26 capture/graph pairs and 2148
logical forwards, with no unassigned measured forward. These are the 21 accepted
configurations; seven failed configurations are excluded from timing acceptance.

## Measurement decision

### Later composition experiment (2026-10-09 UTC)

**Update:** a two-region composition at the `LlamaModel` return is now
registered as `neuron_llama_region`. It is accepted because it reproduces
unsplit stock logits; see [Model/head region composition](trainium2.md#modelhead-region-composition-experimental).
The finer per-op decomposition below is still not additive.

An AMD-style decomposition has now been tried against this stock path. Public
`NF.mlp`, `NF.qkv_proj` (including RoPE), `NF.flash_attention`, and `NF.o_proj`
all ran through the stock compiler with independent FP32 numerical checks.
Six shapes passed: MLP rows 1, 16 and 512, plus the three prefill operations at
512 rows. Each has 20 native two-core measurements. The rank-local fused MLP
is registered as `neuron_dense_mlp/vllm_neuron`, with BF16 H4096/I3584 and only
those three row counts. It uses actual layer-0 TP4 rank-0 checkpoint shards and
fixed synthetic RMS-normalized activations. The public stock callable keeps
both column-tiling options enabled. Each profiling run independently checks
FP32 gate/up/SiLU/down math with relative L2 <=0.02 and normalized peak error
<=0.05. The remaining component probes are experimental measurements.
No component's full-forward additivity has been established.

Fresh debug compilations restore source locations for compiled Torch operations
as well as NKI kernels. All three HLO computations match the baseline after
removing debug metadata and verifying relocated NKI binaries by SHA256. The
B16 debug run reproduces all 128 full-vocabulary baseline logit vectors exactly;
the original vendor precision gate also passes. Debug NEFF bytes differ, so
binary identity is not claimed.

Complete rank-0 instruction captures now cover prefill512, decode1 and decode16.
The debug prefill capture has dropped DMA notifications and supports instruction
analysis only. Decode16 has no loss warnings. Decode1 has an exact exporter
schema warning for an unused TimelineAnnotation field, retained in its report;
its instruction coverage passes. Framework/HLO IDs remain absent. Source-tagged
instruction intervals overlap and cannot be summed as component latency.

The stock MLP backend's Rust cache uses exact categorical cells for rows 1, 16
and 512; unsupported shapes fail rather than interpolate. Its public profiling
CLI measured 0.16067, 0.1682305 and 0.6067715 ms, respectively, in a private DB.
These are rank-local measurements, excluding TP communication.

Standalone compiled decode attention also passes independent output, KV-write
and untouched-storage checks at B1/B16. However, its medians (0.5974315 and
0.553355 ms) do not scale into the full-model costs (9.919804 and 55.263739 ms).
The discrepancy changes direction between batches. Matched HLO contraction,
gather/scatter and alias signatures do not explain it; native instruction
counts differ substantially. Controlled runtime position/page-pattern changes
within the same B16 executable change median time by at most 0.32%, leaving
compute instruction counts identical. Lowered compiler records now show a
111,116,288-byte full-pool copy in standalone attention and different gather
tiles: standalone K/V use 32/4 large tiles, while the full model uses 1,024
small tiles each per layer. These explain the observed instruction-count
pattern; why the compiler selects these schedules remains unresolved.

A one-layer production-path diagnostic passes the unchanged vendor numerical
gate at both batches. Its full-forward medians are 2.443813 ms for prefill,
1.4647165 ms for B1 decode and 2.237486 ms for B16 decode. These include model
boundaries and are not per-layer costs or E2E measurements. The four-layer check also passes: prefill 5.8790895 ms, decode B1 2.005268 ms
and B16 7.975922 ms. An affine extrapolation from depths 1 and 4 misses the
historical full32 baseline by -0.93%, -28.93% and +11.35%, respectively. This is
a diagnostic comparison with different activations/histories, not a controlled
compiler-cause attribution. Neither reduced-depth nor standalone measurements justify
replacing the accepted whole-forward leaf. No fitted residual, shared DB rows
or new L4 default were introduced.

A full32 experiment split the original FX graph into prologue, decoder stack
and epilogue, preserving all original operations and 64 KV aliases. All 192
logit vectors passed the unchanged vendor BC criterion (the RMS branch failed).
The sums of independently measured region medians differ from stock by +0.75%
for prefill512, +5.39% for decode1 and +0.77% for decode16. Promotion was rejected
because decode1 exceeds the fixed 5% gate. An extra early decode1 execution in
one public B16 request bracket is retained pending scheduler attribution.
No region costs have been registered. A subsequent two-region experiment follows
the existing public model/head boundary: embedding, all layers and final norm
remain together; the head includes row selection, projection, collectives and
sampling. Its sums differ from stock by +0.38%, +1.76% and +0.31% for
prefill512, decode1 and decode16, passing the same 5% gate. All 192 logit vectors
pass the unchanged vendor criterion. Scheduler observations now account for
all 262 forwards, including the extra early decode1, across all four ranks.
Independent equal-input checks use the captured native cache history as input
to HF FP32 and BF16 references. All head checks and untouched-cache checks pass.
The stricter, predeclared component criterion requires RMS error no greater
than the eager-BF16 reference error. Hidden-state ratios are 1.169 for prefill,
0.929 for decode1 and 1.008 for decode16; prefill K/V also fail. This rejects
region registration despite the separately passed full-logit criterion. The
experiment changes compilation boundaries and is not the unmodified stock
executable. These accumulated differences do not identify a faulty primitive. A separate
trial widened only the 64 layer collectives to FP32, retaining BF16 outputs,
embedding/head behavior and all KV aliases. It failed both original full-logit
shape checks and the component checks (prefill hidden/K/V and decode1 hidden).
It was rejected without timing; no precision threshold was changed.

Local evidence, reproducible probes, rejected captures and the remaining
contract decisions are recorded in
[the composition report](../../tmp/trainium-composition/REPORT.md).

### Initial supported boundary

`vllm_neuron/vllm/worker/neuron_model_runner.py:1432` compiles the model with
`fullgraph=True`; `model/llama3/model.py:1610` includes on-device sampling in the
forward. The compiler inlines NKI bodies into the parent graph (runner line1422).
The synchronized native trace measures that graph on each participating core.
It does not provide separately attributable durations for FX nodes.

The initial L1 unit is therefore **one full compiled forward, including sampling**.
Its duration is the union of native execution intervals across all four ranks
and eight physical cores. Host packing, scheduling, compilation, loading and
input/output transfers are outside this duration. Summing rank durations would
count concurrent work repeatedly. No layer-count multiplier applies: all32
layers already execute inside the measured operation.

The nearest existing kind, `neuron_llama_decoder`, is a separately compiled
synthetic-weight NxDI TP1 decoder without embedding, final head or sampling.
Its cache ABI and compiler boundary differ. GEMM, norm and MLP kinds also cannot
represent the whole stateful TP4 forward. A new `neuron_llama_forward` kind with
the `vllm_neuron` backend preserves those meanings without reinterpreting old rows.

## Decision table

The measurable full forward owns 100% of captured graph execution time. Internal
rows have no separately measured share; their listing order is semantic order,
not an invented performance ranking. Each internal row folds into the full
forward and contributes no additional timing.

| Operation | Production evidence | Actual execution / timing home |
| --- | --- | --- |
| Full stateful forward and sampling | runner fullgraph compilation; `model/llama3/model.py:1610–1732` | New L1 `neuron_llama_forward/vllm_neuron`; 100% of captured graph time |
| Vocab-sharded embedding and rank communication | `model/llama3/model.py:1445–1506`; FX embedding and collective nodes | Fold into full forward |
| Input RMSNorm and residual handling | `model/llama3/model.py:159–164,1223–1345` | Compiled Torch FP32 normalization with BF16 output; fold |
| Prefill QKV and scaled RoPE | `model/llama3/model.py:703–815`; prefill FX `arg_names: [hidden,...]` | Fused NKI QKV/RoPE, 32 calls; fold |
| Prefill KV updates | `model/llama3/model.py:593–624`; FX cache scatter nodes | Separate semantic cache scatter inside graph; fold |
| Prefill attention | same prefill path; FX `arg_names: [q,...]` | NKI FlashAttention, 32 calls; fold |
| Prefill output projection | same prefill path; FX `arg_names: [attention,...]` | NKI output projection, 32 calls; fold |
| Decode mask | `model/llama3/model.py:911–928`; `functional/attention/attention_decode_mask.py:239–267` | One shared NKI mask generation per forward; fold |
| Decode QKV, RoPE, cache gather, GQA attention, cache update and output projection | `functional/attention/attention_decode.py:1210–1258` eligibility and fallback; decode FX | Compiled Torch fallback, not an attention megakernel; fold |
| Attention TP/SP collectives and residual | `model/llama3/model.py:532,989,1223–1345` | Prefill sequence-parallel communication; decode TP all-reduce; fold |
| Post-attention RMSNorm | `model/llama3/model.py:1223–1345` | Outside fused MLP, compiled inside graph; fold |
| Gate/up projections, SiLU product, down projection | `functional/mlp.py:65–106,169–216`; model1091–1113 | 32 fused NKI MLPs, `NormType.NO_NORM`, no quantization; fold |
| MLP TP/SP communication and residual | `model/llama3/model.py:1113,1223–1345` | Outside MLP NKI, inside full graph; fold |
| Final RMSNorm and prefill sequence reconstruction | `model/llama3/model.py:1445–1506` | Fold into full forward |
| Selected-position gather and vocab-sharded head | `model/llama3/model.py:1660–1732` | Fold into full forward |
| On-device token selection and sampling collectives | same forward; FX top-k, cumulative sum and sampling nodes | Retained even for greedy request settings; fold |

All accepted decode graphs contain 65 all-reduces, three all-gathers, 32 Torch
softmax nodes and 36 NKI wrapper calls: 32 MLP, one attention mask, two top-k and
one cumulative-sum call. Accepted prefill graphs contain 68 all-gathers and131
NKI wrapper calls: 32 each QKV/RoPE, attention, output projection and MLP, plus
three sampling calls. These are frontend graph counts, not device launch counts.

## Physical shape and cache requirements

Each rank has 8 query heads, 2 KV heads and intermediate width3584. QKV weights
are `[4096,1536]`, output weights `[1024,4096]`, gate/up `[4096,3584]` and down
`[3584,4096]`. Each layer's K and V arrays are separately
`[6782,2,32,128]` BF16 per rank. Binding is in
`model/llama3/model.py:1781–1812`. The pool has217024 token slots.

Decode block tables are `[compiled_batch,max_blocks_per_sequence]`. With two KV
heads and a two-dimensional table, `_can_use_attention_block_kernel` rejects
the fused path. The compiled fallback gathers the fixed context width, repeats
KV heads for GQA, and applies the runtime mask. Occupied logical context alone
therefore does not identify the compiled work. Prefill and decode also use
different sequence-parallel arrangements.

The profiling key must distinguish phase, compiled token/batch bucket, compiled
context width, pool blocks, page size, TP and dtype. LNC2, fixed model semantics,
greedy request configuration and pinned runtime/compiler belong to this
backend's supported contract and provenance. Preserve the actual compiler flags
from `command.txt`; do not silently replace them with current source defaults.

Logical request lengths remain separate inputs to necessary-work accounting.
Bucket padding and full-context reads are executed work, not compulsory work.

## Plumbing, exclusions and later refinement

### Implemented path and validation boundary

`profiling/kernels/neuron_llama_forward.py` registers the Python kind;
`profiling/runners/neuron/vllm_forward.py` runs the pinned backend. Rust's
`timing/kernels/neuron_llama_forward.rs` selects exact compiled variants without
interpolating between graphs. `arch/llama3_vllm_neuron.rs` composes one atomic
operation at `unified.forward`. Its measured forward already includes every
layer, rank collective and sampling operation, so there is no additional layer
or rank timing multiplier.

The initial fresh profile database covers C512 prefill512 and decode buckets1
and16. The predictor selects bucket16 for logical batches2–16 only when the
architecture explicitly configures `[1,16]`; this is not the engine's default
bucket inventory. Prefill504 and decode1/9/16 have passed the public predictor
path. A separate balanced B9 native capture verified the bucket16 reuse.
Historical rows for C128/512/2048 are retained in a separate temporary combined
database with frozen-import provenance; they do not imply complete coverage of
every architecture configuration. Shared profile database rows have not been
published as part of these validations.

The independent map `model/work/location_maps/llama3_vllm_neuron_unified.json`
assigns all12 Llama semantics to the full-forward slot, using the existing
Llama necessary-work builder. Batch-locked predictor analysis consumes this
map. Exact simulation-iteration analysis also consumes it with replication1,
all23 iterations and all8 unique logical shapes in the initial serving run.

Numerical acceptance is limited to the validated corpus under the unchanged
vendor OR criterion. Matching generated tokens alone is insufficient. Fresh
balanced bucket1 passed the aggregate RMS branch; bucket16 passed the
Bhattacharyya-coefficient branch. Earlier failed attempts remain recorded.
HTTP capture must establish its prompt and graph identities before reusing
this evidence, and it does not itself repeat a full-logit reference comparison.

Native submission/wait, allocation, host input preparation, copies and idle gaps
are framework/runtime overhead. They remain in duty-cycle and E2E evidence,
not additional mathematical L1 leaves. The current offline capture cannot
establish HTTP TTFT/TPOT or scheduler per-request iteration geometry.

The initial backend must retain Phase0's numerical exclusions and the stock
short-context batch256 slot-mapping failure. It cannot claim support for those
configurations from a successful compile or matching generated tokens alone.

Public `NF.mlp`, `NF.qkv_proj`, `NF.flash_attention`, `NF.o_proj` and
`NF.gen_attention_decode_mask` are candidates for finer profiling. Promoting
them to additive L1 units requires matched inputs/layout/compiler options and
comparison with regions of the original instruction trace. Isolated timing
alone does not prove unchanged fusion, residency or overlap. Until that proof
exists, the full-forward leaf is the complete measured model; internal shares
remain unavailable rather than estimated from FLOPs.

## Work metrics: source-derived estimates

Work accounting must remain independent of the necessary-work labeler. The
following counts come from the saved graph shapes and production kernel source,
not from `model.work`. They describe tensor contractions and persistent operand
volume, **not hardware-issued FLOPs or measured HBM traffic**.

All quantities below are per TP rank, covering both of its physical LNC2 cores.
Use `L=32`, `H=4096`, `I=3584`, `Q=8`, `D=128`, `V=32064`.
Compiled prefill has `T=S`, `R=1` sampled row; decode has `T=B`, `R=B`.

```
dense_flops = 2*L*T*(H*1536 + 1024*H + 3*H*I) + 2*R*H*V
decode_attention_flops = L*4*B*Q*C*D
```

Prefill MLP gathers all S rows before its fused call; using S/4 would undercount
its projections. Decode uses the full compiled context C for QK/PV, irrespective
of the runtime mask.

For causal prefill, NKI's QK key tiles are512 and PV key tiles128, with query
groups128. For `q=0,128,...<S`:

```
nq = min(128, S-q)
QK_pairs = sum(nq * min(S, 512*(floor(q/512)+1)))
PV_pairs = sum(nq * min(S, q+128))
prefill_attention_flops = L*2*Q*D*(QK_pairs + PV_pairs)
```

The installed `nkilib/core/attention/attention_cte.py` source is preserved in
`../tmp/vllm-neuron-phase1/stock-nkilib/`; SHA256
`465ecb5e18d80ce89f5e15eee83ddc8e28544fae4c06ed57c1f47945be2c75fe`.
Lines178–184 define tile sizes,3558–3620 select QK tiles,3842–3893 select PV
tiles, and3919–3941 implement the causal skip predicate. This includes masked
arithmetic within retained diagonal tiles. Across32layers, attention counts
are2,147,483,648 /27,917,287,424 /317,827,579,904 FLOPs per rank at
S128/512/2048 respectively. These formulas do not cover prefix/SWA/CP paths.

Persistent operand-volume estimates per rank are:

```
matrix_and_norm_weights = 3,752,861,696 bytes
embedding_operand_bytes = 2*T*H
prefill_cache_write_bytes = 32768*S
decode_cache_read_write_bytes = 32768*B*(C+1)
```

The fixed footprint includes all projection/head matrices and65 norm scales.
Embedding executes masked lookups on every rank, often repeating local row0;
operand volume therefore differs from unique reads. GQA repeat does not multiply
persistent cache volume by4. Multiply per-rank quantities by4 for total chip work,
never by8 physical cores.

FLOP counts omit normalization, activation, softmax, RoPE and sampling arithmetic,
collectives, compiler-introduced work and hardware padding. Bytes omit intermediate
tensors, sampling buffers, communication, spills and NKI weight rereads, and do
not model residency or address coalescing. Any R7 derived from these fields must
be described as an algebraic graph-work estimate with these limitations.

### Roofline partition limitation

The accepted full-forward leaf aggregates all executed contractions and persistent
operand bytes before taking R5's compute/memory maximum. R6 intentionally takes
a separate maximum for each independent semantic segment and then sums them.
These different partitions can yield R6 > R5 even when both executed FLOPs and
bytes exceed their independent minimum totals. In the initial stock alignment,
executed FLOPs and bytes exceed the minimum at every iteration. The positive
R6-minus-R5 difference comes from prefill, partially offset by decode. It is not evidence of a
missing contraction or compulsory-weight count.

Keep the raw diagnostic and R6/R7 definitions. Do not increase estimated traffic,
change the independent work label or fit a residual to force an ordered ladder.
Interpreting R5 minus R6 as redundant work requires compatible execution/semantic
partitions; the current whole-graph estimate does not establish that compatibility.
Omitted non-contraction/intermediate work remains a separate limitation.

## Server-metric repair validation

The stock scheduler observer now retains actual EngineCore queue/schedule events
and token-output timestamps, requiring native `log_stats` and complete eligible
request records. Clean and native HTTP captures both passed for16 requests with
eight outputs;69 focused CPU regressions cover eligibility, legacy resume and
the profile-result/analysis-manifest handoff. Shared kernel and E2E analyses
produce all default campaign metrics without changing profile values, the prior
duty multiplier or tolerances. The original pack-less comparison remains
UNJUDGED. The portable
[`llama31_8b_bf16_trainium2_tp4` campaign](../presets/alignment/llama31_8b_bf16_trainium2_tp4/README.md)
now reproduces this configuration and applies unchanged generic acceptance
limits, selected retrospectively. The neutral simulation fails E2E and output
throughput; the separately duty-calibrated simulation passes all thirteen
metrics using the previously measured factor. No factor was fitted to the
new E2E evidence. Current artifacts are under
`logs/20261009_2_trainium_stock_server_timings/`, with separate policy judgments
under `campaign_policy/`. This completes server evidence for the bounded
whole-forward path, not additive component validation.

### Short-context capacity repair

The original context 128/batch 256 failure was an undersized CPU slot-mapping
buffer, not KV exhaustion. The stock runner capped its prefill token budget at
128 and reused that value for the buffer, while its scheduler could admit more
than 128 decode tokens. A patch for the exact plugin revision sizes `InputBatch` for the larger
of the prefill budget and configured decode buckets. It preserves scheduling,
model operations and the compiled bucket inventory. The replay tool writes a
separate candidate file and rejects unknown sources or existing outputs:
[replay instructions](../tools/trainium2/README.md).

The isolated patched TP4 path passed batch 256/context 128/prompt 120/output 8:
2,048 positions, all static checks, aggregate RMS ratio 0.983163. The subsequent
same-engine capacity sweep also passed the unchanged vendor gate:

| Submitted requests | Maximum resident | Minimum free KV blocks | Validated positions | RMS ratio |
| --- | --- | --- | --- | --- |
| 512 | 512 | 4733 | 4096 | 0.983163 |
| 1024 | 1024 | 2685 | 8192 | 0.983163 |
| 1695 | 1695 | 1 | 13560 | 0.963167 |
| 1696 | 1695 | 1 | 13568 | 0.963201 |

All 39,416 positions pass the static branch. The BC branch fails for 512/1024 and
passes for 1695/1696; every shape also passes the RMS branch. All requests finish,
with no preemptions. The 1696 case explicitly records 1695 running and 1 waiting:
one free 32-token block cannot admit another 128-token request. All 31 cached NEFF
binaries remain byte-identical across loading and generation. Reference reuse
binds exact generated histories; 17 distinct native-array/history comparisons
retain every slot's multiplicity in the vendor aggregates.

This validates the allocation fix and the observed capacity boundary, not new
L1 timing rows or generalized simulator support. The registered stock serving
preset remains context 512 with decode buckets 1/16. Evidence and preserved
harness failures are in
[the capacity report](../../tmp/trainium-composition/capacity-diagnosis/REPORT.md).

### Checkpoint and executable identity; clean engine shutdown

Fresh numerical validation now records hashes of the checkpoint and actual cached
NEFF executables, with loader evidence from all four ranks. Fresh native and
unprofiled HTTP runs verify those identities before and after execution. The
combined audit passes; earlier graph-key-only receipts remain historical and are
not retroactively upgraded. HTTP prompt/request joins are verified, but HTTP
generated token IDs/full logits are not independently checked by this audit.
Numerical acceptance comes from the separately captured, byte-bound native run.

Two shutdown defects were reproduced and fixed: the launcher now gives vLLM a
ten-second graceful drain, and a narrowly scoped, source-pinned compatibility
adapter handles the lite Neuron allocator's unsupported generic caching-allocator
flush. Inspection of the actual allocator shows direct native allocation/free
without a PyTorch cache. Distributed teardown, garbage collection, host-cache
cleanup and other errors remain intact. Unknown versions fail closed. The final
real HTTP run completed all sixteen requests and exited cleanly; the exception
gate was not weakened. Historical failing captures remain available.

Evidence: `tmp/stock-neuron-http-binary-provenance-v1/final-audit-v2.json`
(relative to the workspace root), including frozen source identities and both
capture receipts. New alignment reports are under
`logs/20261009_3_trainium_stock_binary_identity/`. They reuse the explicitly
recorded prior simulations and calibration, without new fitting or profile rows.

Additive composition remains experimental: the two-region candidate passes
whole-output vendor precision and timing checks, but not the stronger independent
hidden-state/KV-cache check. The FP32 collective candidate and subsequent CPU
rounding diagnostics do not resolve that failure. No region timing costs are
registered. This is a remaining limitation, not completed generalized hardware
support.

The fresh fixed-policy comparison rejects the neutral control (eleven of thirteen
checks within bounds; E2E mean and output throughput fail) and accepts the prior
duty-calibrated control (all thirteen within bounds, server TTFT near its bound).
The existing factor and generic thresholds are unchanged. The complete scope and
reuse limitations are retained in the campaign's `REPORT.md` and
`simulation-reuse-provenance.json`.
