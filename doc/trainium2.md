# Trainium2 profiling and model support

## Stock vLLM Neuron TP4 path

The current serving target is Llama3.1-8B BF16 on one Trainium2 chip: four LNC2
units, eight physical cores. Select `llama3_vllm_neuron`. It measures the pinned
stock vLLM Neuron full32-layer compiled forward, including communication and
sampling, as one `neuron_llama_forward/vllm_neuron` leaf. The source, numerical
acceptance boundary and work-accounting limitations are in the
[stock inventory](trainium2_vllm_inventory.md).

The validated serving preset configures context512, decode buckets `[1,16]`,
page32 and6782 KV pages. It admits one prefix-free prefill per iteration with
prefill priority, at most16 resident requests, and no prefix reuse. A32-request
simulation verifies queueing, two cohorts and final KV release. Other compiled
contexts have historical measurements, but are not the initial serving preset.

From the workspace root, source `.env`, then run in `ServingStudioSim`:

```bash
export UV_NO_SYNC=1
export VIBESIM_PROFILE_DB="$TMPDIR/neuron-forward-cli-smoke/profile.db"
uv run --no-sync python -m launcher timing-predict \
  presets/predict_llama3_8b_vllm_neuron.json --no-gpu
uv run --no-sync python -m launcher \
  presets/unified_llama3_8b_vllm_neuron.yaml --dry-run --no-gpu
uv run --no-sync python -m launcher \
  presets/unified_llama3_8b_vllm_neuron.yaml --no-gpu
```

That private database is a local validation artifact, not a checked-in dependency.
Cold profiling requires the declared pinned container environment and checkpoint;
`--no-gpu` deliberately fails if required rows are missing. The shared database
has not been modified. Native and clean stock HTTP captures use the
[stock Neuron alignment adapter](../alignment/neuron/README.md).

The independent `llama3-vllm-neuron-unified-v1` necessary-work map is consumed by
actual simulation-iteration analysis. Its roofline analysis uses the catalog's
explicit bandwidth assumption and source-derived graph work, not measured HBM
traffic or hardware-issued FLOPs. R5 applies one roofline to the compiled forward,
whereas R6 sums separate semantic rooflines. This partition mismatch produces the
observed under-accounted-floor diagnostic despite the recorded total FLOPs and
bytes exceeding the independent minimum at every checked iteration. The diagnostic
remains visible; it does not establish missing work or negative redundancy.
Non-contraction and intermediate work are still omitted from the estimates.

### Model/head region composition (experimental)

Set `composition: model_head_regions` on `llama3_vllm_neuron` to cost each
iteration as two measured `neuron_llama_region/vllm_neuron_fx_regions` leaves.
The profiler splits the stock FX graph at the public `LlamaModel.forward` return
and compiles both parts with the stock backend and original flags. Every stock
operation is kept unchanged. The `model` region holds the embedding, all 32
layers, the final norm, the SP gather and all 64 KV updates. The `head` region
holds row selection, `lm_head`, the logit all-gather and greedy sampling. Each
profiling run must pass four gates before it writes rows:

1. The structural partition proof holds for every rank and shape.
2. The unchanged vendor full-logit check passes on both the split and the stock
   engine.
3. The split is equivalent to stock: every case generates the same tokens, and
   RMS(split−FP32) ≤ 1.10 × RMS(stock−FP32) per case.
4. Per shape, the model and head medians sum to within 5% of the stock
   whole-forward median measured in the same run.

The first run passed all four gates. Logits were bit-identical in 22 of 24
cases; the two that differ change at one position each, with a worst error
ratio of 1.0000018. The region sums differed from the whole forward by −0.31%
(prefill512), +1.76% (decode1) and +0.25% (decode16). An earlier
HF-relative hidden-state gate is not used, because unsplit stock also fails it.
The split reproduces the stock output, which is what this gate needs to show.

The scope is C512 with decode buckets [1,16] only. Use
`presets/predict_llama3_8b_vllm_neuron_regions.json` and
`presets/unified_llama3_8b_vllm_neuron_regions.yaml`. Rows live in the private
`$TMPDIR/neuron-region-smoke/profile.db`, which also holds the whole-forward
rows. The `llama3-vllm-neuron-regions-v1` work map assigns `lm_head` to the head
region and all remaining semantics to the model region. Whole-forward costing
remains the default.

Replaying the stock native and clean captures through the shared alignment
pipeline with this composition (`logs/20261009_4_trainium_regions_alignment/`,
local) gives:

- Kernel error: +0.48%, with 100% map coverage.
- Neutral simulation: 12/13 checks pass. E2E mean is -7.02% against a 7% bound;
  stock whole-forward costing was -7.59%.
- Simulation with the native duty-cycle multiplier (1.0637, taken from the same
  native kernel analysis): 13/13 pass. Server TTFT and iteration cycle are close
  to their bounds.

### Collective-delimited layer segments (experimental)

Adding compiled region cuts costs device time at decode1: about 0.17 ms for
the model/head split, and +0.53 ms for a three-region trial. Per-layer cuts
would therefore fail the 5% gate. `composition: layer_segments` instead keeps
the unchanged stock executable and cuts its native instruction-trace timeline
at the reduction collectives that close each sublayer:

- the first reduction ends the embedding;
- each layer has an attention block and an MLP block, each ending at one of
  the layer's two reductions;
- everything after the last reduction is the head.

Decode uses AllReduce; prefill uses ReduceScatter with AllGather between. The
segments sum exactly to the forward with no added launches. Block rows are
per-layer means, and the L4 folds them as 32 x (attention + MLP).

`neuron_llama_segment/vllm_neuron_collective_segments` gates rows on:

- traced NEFFs byte-identical to the stock forward, plus the unchanged vendor
  logit check;
- complete two-core traces, allowing at most 10 us of trailing control
  instructions past the execution end;
- interior layers 1-30 within 10% of the layer mean, measured as per-layer
  medians;
- the composed forward within 5% of the untraced whole forward.

Systematic first- and last-layer offsets come from where the compiler places
partition boundaries; they are recorded rather than gated. Examples are
decode1 layer-0 attention (+36%) and prefill layer-31 MLP (-10.5%).

Offline replay of a 4-rank, 39-forward capture passes every gate:

- worst interior layer 1.8% from the mean;
- composed versus whole forward +1.78% (decode1), +0.18% (decode16) and
  +0.76% (prefill512).

Per layer, decode16 is about 1.47 ms of attention and 0.22 ms of MLP; prefill512
is about 0.28 ms of attention and 0.87 ms of MLP.

A fresh public-CLI run wrote the 12 rows to the private
`$TMPDIR/neuron-segment-smoke/profile.db`, and every gate passed:

- interior layers within 1.1-2.9% of the layer mean;
- composed versus whole forward +1.83% (decode1), +0.19% (decode16) and +0.58%
  (prefill512).

Replaying the stock native and clean captures with `composition: layer_segments`
(`logs/20261010_1_trainium_segments_alignment/`, local) gives kernel error +0.49%
with 100% map coverage. Both the neutral simulation (E2E -6.93%) and the
duty-calibrated one (x1.0637) are within all 13 campaign bounds. Use
`presets/*_vllm_neuron_segments.*`.

Trace export peaks near 47 GB per rank. Export ranks one at a time, and run
trace analysis under a memory cap.

## Earlier NxDI TP1 path and experiment history

The sections below preserve the separate NxDI implementation and its numerical
experiments. Their unfinished full-model trials do not describe stock TP4
acceptance, and their timing rows are not used by `llama3_vllm_neuron`.

The first target is **AWS Trainium2 LNC2**, one logical execution unit containing
two physical NeuronCores. The trn2.3xlarge validation host reports four such units
and 96 GiB chip memory (24 GiB per logical unit). Profiling reserves the whole
chip while running on one unit to avoid shared-resource contention. Concurrent
units, tensor parallelism and NeuronLink collectives are outside this first path.

Setup entry points live in the parent workspace:
`just setup-neuron-host --install` and `just setup-neuron-python`.
They use AWS packages and separate native NKI / Torch NeuronX environments;
see the parent `reproduce.md`. The AL2023 Torch 2.6 tracing stack is verified
against pinned NxDI dense source rather than its package's Torch 2.9 default.

## Measurement boundary

The Neuron executor discovers physical identity with `neuron-ls`, validates
Trainium2 and LNC2, and launches an isolated worker. CUDA backends cannot run on
Neuron and Neuron backends cannot use CUDA cache labels. Inherited
`NEURON_RT_VISIBLE_CORES` is rejected so allocation never widens caller visibility.
The executor owns a nonblocking whole-chip lock and the selected logical core.
The lock lives in `/tmp` so separate workspace temporary roots share it.

The timer compiles and validates outside its measurement window. It captures
20 native `nc_exec_running` invocations after five warmups, matches event pairs,
and unions synchronized `timestamp_ns` intervals across both physical cores of
each invocation. Its reported time is the median of those 20 logical durations.
Raw core clocks and NRT's average over physical-core samples are unsuitable for
this LNC2 aggregation. Private SDK adapters reject unverified NKI/compiler
versions and incomplete traces. Energy is unavailable and recorded as zero.

Compiled artifacts persist under workspace `TMPDIR/neuron-kernels`, keyed by
source, shape, dtype, compiler configuration and package versions. Stateful
decoder outputs use explicit names and verified MUST_ALIAS metadata; dictionary
iteration order is not an ABI.

## Registered BF16 paths

| Kernel kind | Backend | Boundary |
| --- | --- | --- |
| `single_gemm` | `neuron_nki_qkv` | Public NKI QKV TKG matmul, without normalization or RoPE; 1..96 rows |
| `rms_norm` | `neuron_torch_rms` | NxDI standalone FP32 RMS normalization with BF16 output |
| `neuron_embedding` | `neuron_torch` | Compiled TP1 vocabulary embedding gather |
| `neuron_dense_mlp` | `nki_library` | Public fused SiLU MLP, initially only m=1/H=4096/I=14336 |
| `neuron_llama_decoder` | `nxdi_compiler` | Complete compiled NxDI layer and per-layer aliased KV updates |

The full-width NKI MLP's default column tiling exceeds SBUF capacity. Its one
validated decode shape uses the public transpose option. Prefill is supplied by
the complete compiler decoder path; unsupported standalone MLP shapes fail
before compilation.

The standalone NKI QKV matmul switches from TKG to CTE above 96 query rows.
Its BF16 CTE implementation requires output width at most 4096, and the default
LNC2 allocation also exceeds SBUF at H=4096/n=4096. This backend is limited to
the verified TKG domain of 1..96 rows; the full decoder uses the independent
compiler path and supports prefill through 128 rows.

The decoder fixes Llama 3.1 8B semantics: H=4096, I=14336, 32 query heads,
eight KV heads, head dimension 128, BF16 compute and KV, RMS epsilon 1e-5,
theta 500000 and Llama 3.1 scaled RoPE. Initial support is TP1, batch one,
physical KV capacity 512, decode q=1, and prefix-free prefill q=1..128.
Decode still reads the allocated capacity when its runtime mask hides entries.
Weights are deterministic synthetic tensors with nonuniform norm weights.

Example standalone measurement, using a separate database:

```bash
uv run --no-sync python -m launcher kernel-profile run rms_norm \
  --backend neuron_torch_rms --gpu-name 'AWS Trainium2 LNC2' \
  --spec '{"m":1,"hidden":4096,"dtype":"bf16"}' \
  --db "$TMPDIR/trainium-proof.db" --output-dir "$TMPDIR/trainium-rms-proof" \
  --no-energy --json
```

## Prediction and simulation

The `llama3_neuron` selector composes embedding, 32 separately compiled decoder
layers, final normalization and a one-token vocabulary projection. Prefill and
decode occupy separate manifest slots; only the active phase contributes to an
iteration. The vocabulary projection consumes the final query row in prefill.

From the parent workspace, source `.env`, then run in `ServingStudioSim`:

```bash
export VIBESIM_PROFILE_DB="$TMPDIR/trainium-proof.db"
uv run --no-sync python -m launcher timing-predict \
  presets/predict_llama3_8b_trainium2.json --dry-run
uv run --no-sync python -m launcher timing-predict \
  presets/predict_llama3_8b_trainium2.json
uv run --no-sync python -m launcher \
  presets/unified_llama3_8b_trainium2.yaml --dry-run
uv run --no-sync python -m launcher presets/unified_llama3_8b_trainium2.yaml
```

The first run compiles and measures missing profiles on the local accelerator.
Use a named tmux session for that run. The simulation preset uses the existing
small smoke trace and the barebone worker. Its token budget is one: an idle
worker admits one whole prefill, then decodes that request before admitting the
next. Prefix reuse, mixed batches, additional replicas and requests outside the
validated query/cache limits are rejected.

Private validation profiles on this server are in
`$TMPDIR/trainium-kernel-sources/registered-smoke.db`; set `VIBESIM_PROFILE_DB` to
that path to reuse them. The tracked shared profile database is unchanged.

## Validation scope

### Full checkpoint proof

The workspace command `just validate-neuron-checkpoint` runs the public NxDI
full-model path independently of the simulator's layer composition. It accepts
a local Llama 3.1 8B checkpoint and separate compiled-artifact/report directories:

```bash
# From the workspace root, inside a named tmux session:
just validate-neuron-checkpoint compile "$TMPDIR/models/llama3.1-8b" \
  "$TMPDIR/neuron-full-llama3.1-8b" "$TMPDIR/trainium-checkpoint-proof"
just validate-neuron-checkpoint validate "$TMPDIR/models/llama3.1-8b" \
  "$TMPDIR/neuron-full-llama3.1-8b" "$TMPDIR/trainium-checkpoint-proof"
```

Compilation uses all 32 layers, BF16, TP1/LNC2, a 128-token context bucket and
512-token KV capacity. The public `debug="none"` mode disables optional HLO
metadata whose newer API is absent from the AL2023 Torch 2.6 stack. Model math
and SDK source remain unchanged. Existing compiled outputs are protected from
overwrite; use a new output directory to recompile.

Validation reserves an idle chip through the same executor used for profiling,
loads real checkpoint weights through NxDI, and compares greedy token IDs with
the full Transformers CPU BF16 model. Ten cases cover code, numbers, JSON,
short prompts, 128-token prefill, a longer continuation and a repeated prompt
after intervening requests. Reports include shard SHA256 hashes, package
versions, config, first divergent token and diagnostic logit differences.
The helper performs no checkpoint download and writes no profile database rows.

On the trn2.3xlarge validation host, the full checkpoint compiled, loaded and
generated all ten cases; nine token sequences match CPU BF16 exactly, including
128-token prefill and the repeated-request cache-reset check. The arithmetic
case differs at generated token three: CPU BF16 ties comma and period at logit
10.5 and chooses the lower token ID, while native decode/prefill prefer period
(10.5 versus 10.4375). An independent CPU FP32 evaluation of the identical
prefix also prefers period. This supports a rounding-related difference.
The strict helper retains that mismatch and exits unsuccessfully; full token
identity is not claimed. Local evidence is under
`$TMPDIR/trainium-kernel-sources/full-checkpoint/`, including
`validation.json` and `rounding-diagnostic.json`.

### Simulator and kernel proofs

Validation covers the full Rust library suite, focused Python profiling tests,
public kernel-profile commands on the physical accelerator, timing prediction,
and a drained simulation smoke with passing workload-conservation checks.
Rust cache interpolation passed the 15% fidelity bar against native measurements
for decoder prefill, embedding, normalization and two TKG matmul widths. Small
normalization shapes are measured at every integer through 16 because the
compiler's native timings are discontinuous there.

The registered layer invokes the public NxDI decoder and KV manager with both
norms, all projections, scaled RoPE, attention, SiLU MLP, both residual additions,
and actual cache writes. Its independent oracle checks hidden output and the
entire updated K/V buffers, including untouched cache positions. Both decode
and prefill have executed on the physical Trainium2 host.

A separately compiled layer can be composed into a measured simulator model.
It is a custom execution boundary: weights are inlined, per-layer cache updates
are enabled, and RoPE is computed in each layer. Default whole-model NxDI
compilation has different weight separation and fusion opportunities. Checkpoint
generation is exercised by the separate proof above, with its documented BF16
tie. Serving overhead, whole-model timing alignment and end-to-end latency have
not been validated. The captured whole-model boundary and its alignment work
are documented in [the NxDI inventory](trainium2_nxdi_inventory.md).

The catalog now derives 158 dense BF16/FP16 TFLOPS from the two physical Tensor
Engines of an LNC2 unit. Its 725-GB/s HBM value is an explicit equal-bank
analysis estimate from the published chip bandwidth, not an independently
published per-unit guarantee. These rates enable hardware-bound analysis under
that assumption; they never replace measured kernel timing. The experimental
separate-layer graph still has no complete semantic location map and does not
claim R6/R7 readiness. Final whole-model maps and Analyzer validation remain
pending.

### Native serving evidence

The [NxDI alignment adapter](../alignment/neuron/README.md) ran a native trace
pass and a separate clean workload pass through the req-frontend client: ten
requests and 136 forwards each, with identical prompt/generated token IDs,
checkpoint hashes, executable, configuration and package versions. Both retain
the synchronized union of physical cores 4 and 5 as logical device 0. Evidence
is in `logs/20261007_2_trainium2_nxdi_alignment/`. Simulator prediction and
Analyzer acceptance were not completed for this path, because its whole-forward
numerics never passed (below).

The container executor has also profiled a registered NKI kernel through the
public CLI with a host-owned chip reservation, read-only source and the image's
runtime. That run exposed a database fallback that recorded the controller's
CUDA version for a Neuron worker; persistence now selects the row's backend
family, so Neuron rows keep NULL CUDA/driver fields and record the SDK, image
and host driver instead.

### Whole-forward numerical investigation (paused)

The NxDI full-model check feeds the same 32 FP32-generated teacher tokens for
each of the ten prompts to CPU FP32, CPU BF16 and the full 32-layer Neuron model
(320 positions), plus four short/long prefill/decode endpoints. The guard is
maximum absolute error 0.25 **and** cosine at least 0.9998 against CPU BF16;
it has never been loosened. All three references use the stored BF16
checkpoint. Short endpoints pass; the long-context endpoints always fail, so no
whole-forward NxDI timing rows have been accepted. The stock vLLM Neuron path
above is validated separately with the vendor's own criterion and does not
depend on this investigation.

| Full-model candidate | Guards passed | Long prefill / decode max error | Verdict |
| --- | ---: | --- | --- |
| Original NxDI | 201/320 | 2.625 / 0.6875 | baseline |
| Public `layer_boundary_markers` | not passed | 1.656 / 3.063 | rejected |
| Torch/NeuronX 2.9 container stack | bit-identical to original | — | environment not the cause |
| HF `LlamaRMSNorm` at all 65 sites | 200/320 | 2.969 / 3.813 | rejected |
| HF eager attention | 209/320 | 3.688 / 2.844 | rejected |
| HF norm + HF attention | 199/320 | 2.719 / 3.563 | rejected |
| Maintained NKI RMSNorm + HF attention | 214/320 | 4.313 / 4.313 | rejected |
| + maintained NKI RoPE at all 32 sites | 213/320 | 3.813 / 1.549 | rejected; best so far |
| + `--disable-mixed-precision-accumulation` | 208/320 | 2.219 / 2.156 | rejected |
| + direct BF16 NKI residual add (64 sites) | 208/320 | 3.938 / 3.875 | rejected |

Equal-input first-layer component probes, each compiled through the public
NxDI/NKI path at q128 and q1 with original weights:

- **Q/K/V projection**: within 0.0078 of CPU BF16; error matches CPU BF16 rounding.
- **Normalization**: native HF norm multiplies gamma before the final BF16 cast,
  differing from CPU HF by up to 0.031. The maintained `rmsnorm_tkg` subkernel
  reproduces HF rounding (q1 bit-exact, q128 max 0.002).
- **Self-attention**: hidden output within 0.00098 of CPU BF16, including a
  128-token prefix with 31 teacher decode steps.
- **RoPE**: the public primitive's rotation differs from HF; the maintained NKI
  `rope_hf` subkernel matches HF BF16 bit for bit.
- **FFN suffix**: within 0.0156 (prefill) and 0.00098 (last decode). Exporting
  intermediate outputs changed the final hidden bits at 26 decode steps, so
  internal attribution was rejected.
- **Residual add**: the maintained `foreach` add turns `-0 + -0` into `+0` and was
  rejected; a direct `tensor_tensor` add passes all 136 storage comparisons.

Each component was correct in isolation, but no repair fixed the full model.
Debug-tensor capture (`capture_debug_tensor`) fails at load with an unsupported
peer ID on this stack. NKI `device_print` preserves output bits for the first
tile and for a complete norm boundary, which makes it the remaining route to
localize the error across all 33 layer boundaries. Per-trial closeouts, raw
vectors and independent audits are kept under the workspace's
`tmp/trainium-kernel-sources/`.
