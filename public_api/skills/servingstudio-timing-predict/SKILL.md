---
name: servingstudio-timing-predict
description: >-
  Predict how long one LLM serving iteration takes for given batch shapes on a
  published deployment (model, GPU, parallelism, MoE routing) through the
  ServingStudio public API, with a per-kernel breakdown. Not for request
  queueing or end-to-end latency.
---

# ServingStudio Timing Prediction

A ServingStudio timing prediction estimates how long one deployment takes to
run batches you specify. A deployment is one member of a public preset: a
checkpoint, an arch type, a GPU, and one value of each parameter the preset
sweeps, such as `tp_size` or the MoE `workload`. You give it cases, each a batch: which
requests prefill how many tokens and which requests decode at what context
length. The service composes measured kernel times along the deployment's cost
tree and returns each case's time, split by section and by tree node. Use this
skill to answer questions such as "how long is one decode step of 64 requests
at 8k context for Qwen3-235B-A22B-FP8 on 8 H200s?"

The API is JSON over HTTPS. Set the base URL once:

```bash
API=${SERVINGSTUDIO_API:-https://servingstudio.cs.washington.edu/api/public/v1}
curl -s "$API/health"
```

If `/health` does not return `{"status": "ok", ...}`, ask the user for the base
URL; do not guess another host.

## 1. Find a deployment

`/models` lists every checkpoint with its presets, one per arch type. A preset
has `axes` (each swept parameter and its values) and `members` (one per
combination of axis values). List the members:

```bash
curl -s "$API/models" | jq -r '.checkpoints[] | .presets[] | .id as $id | .members[]
  | [$id, (.params | tojson), .gpus_per_replica, (.predict | tojson), (.missing | tojson)] | @tsv'
```

Each member gives:

- `params`: its axis values. A prediction names the member by these, exactly.
- `gpus_per_replica`: the GPUs one replica of this deployment occupies.
- `predict`: the case shape it accepts. `selector` is `iter`,
  `speculative_iter`, `attn` or `ffn`; `groups` is how many groups each case
  must have; a speculative member also gives `query_width` and `max_model_len`.
- `missing`: profile.db rows its kernels lack, `{kind: count}`. Only a member
  with an empty `missing` can be predicted.
- `error`: why the member does not build, or null.

Read the preset's `gpu` and `axes` too. For an MoE preset, the `workload`
axis's `rows` show what each workload binds: `routing`, and the capture it
reads (`expert_popularity_file` or `token_corpus_file`, an `hf://datasets/...`
reference). Prefer a workload that reads a capture; use `uniform` or `random`
only when the user asks for it, and name the workload in your answer.

```bash
curl -s "$API/models" | jq '.checkpoints[].presets[]
  | select(.id == "Qwen3-235B-A22B-FP8/qwen3_moe_fp8_dp_attn_ep_ffn") | {gpu, axes}'
```

`case_fields` in the same response documents every field of a case for each
selector:

```bash
curl -s "$API/models" | jq '.case_fields'
```

## 2. Read the member's kernels

The tree route takes the preset's `id` from `/models`, which has the form
`{checkpoint directory}/{arch}`, and every axis as a query parameter:

```bash
curl -s "$API/models/Llama-3.1-8B/llama3_dense_tp/tree?tp_size=2" > tree.json
jq '{arch, gpu, gpus_per_replica, predict, missing}' tree.json
jq -r '.sections[] | .section as $s | .slots | to_entries[]
  | [$s, .key, .value.name, .value.kernel, .value.config] | @tsv' tree.json
```

`sections` holds one entry per section (`iter`, `attn`, or the five `ffn`
sections). Its `slots` are the section's kernel leaves: slot `j` gives its
`name`, its `kernel` kind, the `backends` it may use, and its `config`. The
tree that combines them comes with each prediction (step 4), where every leaf
names its slot.

`configs` maps each config id to its kernel `kind`, its scalar `identity`, the
names of its `structured` fields, and `missing`, the rows this member asks of
it and lacks. `/kernels/{kernel}/configs/{config}` serves the same config,
with the args it sweeps (`swept`), how many cells each backend measured (`measured`), and each
cell's metrics (`points`):

```bash
curl -s "$API/kernels/single_gemm/configs/5604a34006bb60d2" | jq '{kind, gpu, identity, swept, measured}'
```

## 3. Build the cases

`cases` is a list of 1 to 64 cases. Each case's form depends on the member's
`predict.selector`, and its group count must equal `predict.groups`; read both
from `/models` or the tree rather than deriving them from parallel sizes.

| `selector` | Case | Each group |
| --- | --- | --- |
| `iter` | `{"groups": [...]}` | `prefill_chunk_pairs` `[[prefix_len, append_len], ...]`, plus either `decode_kv_lens` `[kv_len, ...]` or `decode_count` with `average_decode_length` |
| `speculative_iter` | `{"groups": [...]}` | `prefill_chunk_pairs`, plus `decode_requests` `[[kv_len, query_width], ...]` |
| `attn` | `{"groups": [...]}` | as for `iter` |
| `ffn` | `{"tokens_per_group": [n, ...]}` | one token count per group |

A prefill pair is the tokens already in the KV cache and the tokens this
iteration computes; `[0, 2048]` is a fresh 2048-token prompt. A decode KV
length is the context the new token attends to. A group may hold only
prefills, only decodes, or both. The examples below use members of the public
presets; each case list goes into the request body of step 4.

**Dense, whole iteration** (`Llama-3.1-8B/llama3_dense_tp`, `tp_size: 2`;
`iter`, 1 group). Three decodes, then a 2048-token prefill batched with 32
decodes at 4096:

```json
[{"groups": [{"decode_kv_lens": [1024, 2048, 4096]}]},
 {"groups": [{"prefill_chunk_pairs": [[0, 2048]], "decode_count": 32, "average_decode_length": 4096}]}]
```

**MoE with a capture workload** (`Qwen3-235B-A22B-FP8/qwen3_moe_fp8_dp_attn_ep_ffn`,
`attn_tp_size: 4, ep_size: 8, workload: ctx8k_out1k`; `iter`, 2 groups, one
per attention DP group). Both groups decode 64 requests; then one
group prefills an 8k prompt while the other decodes one request:

```json
[{"groups": [{"decode_count": 64, "average_decode_length": 8192},
             {"decode_count": 64, "average_decode_length": 8192}]},
 {"groups": [{"prefill_chunk_pairs": [[0, 8192]]}, {"decode_kv_lens": [8192]}]}]
```

**Speculative** (`GLM-5.3-NVFP4/glm53_vllm_nvfp4_dsa_moe_dflash2`,
`max_model_len: 8192, workload: diverse_100`; `speculative_iter`, 1 group,
`query_width` 8). Each decode request is `[kv_len, query_width]`: its KV length
once the verify step has run, and the member's `query_width` (draft tokens + 1):

```json
[{"groups": [{"decode_requests": [[4096, 8], [4096, 8], [6000, 8], [2000, 8]]}]},
 {"groups": [{"prefill_chunk_pairs": [[0, 2048]], "decode_requests": [[4096, 8]]}]}]
```

**Attention side** (`Qwen3-235B-A22B/qwen3_attn_tp`, `attn_tp_size: 4`;
`attn`, 1 group):

```json
[{"groups": [{"decode_count": 64, "average_decode_length": 4096}]},
 {"groups": [{"prefill_chunk_pairs": [[0, 4096]]}]}]
```

**FFN side** (`Qwen3-235B-A22B-FP8/qwen3_fp8_ffn_moe`, `attn_tp_size: 4, ep_size: 8`;
`ffn`, 2 groups). Tokens each attention group sends to the FFN this pass:

```json
[{"tokens_per_group": [64, 64]}, {"tokens_per_group": [4096, 128]}]
```

A member with a `max_model_len` (in `predict` for a speculative member, in
`params` for others that sweep it) rejects a prefill whose
`prefix_len + append_len` exceeds it, and a decode KV length above it. A
speculative `kv_len` must also be at least `query_width`.

## 4. Predict

Post the preset id, the member's `params`, and the cases. Each `params` value
must equal the member's value; a number given as a string also matches.

```bash
curl -s -X POST "$API/predict" -H 'content-type: application/json' -d '{
  "preset": "Llama-3.1-8B/llama3_dense_tp",
  "params": {"tp_size": 2},
  "cases": [{"groups": [{"decode_kv_lens": [1024, 2048, 4096]}]},
            {"groups": [{"prefill_chunk_pairs": [[0, 2048]], "decode_count": 32, "average_decode_length": 4096}]}]
}' > prediction.json
jq -c '.cases | to_entries[] | .key as $c | .value.sections[] | [$c, .section, .layer, .total_ms]' prediction.json
```

The answer has `sim_commit` and `cases`, one entry per case in request order.
Each entry's `sections` gives, per section:

- `total_ms` and `energy_j`, the section's time and modeled energy;
- `nodes`, the case's cost tree as the Analyzer reads it (`analyze
  gen-iter-breakdown`), root first in display order. Each node has its
  `node` id, `kind` (`sum`, `max`, `scale` or `leaf`), `depth`, `label`, `ms` (one call), `total_ms` (`ms` times
  every enclosing `scale` node's n), `pct` (`total_ms` over the root's), and
  `critical` (on the chain of slowest children from the root, the rows
  `iter_breakdown.ans` marks with ▸). A leaf names its `slot`; runs of
  identical siblings show one node with `copies` (and, under a `max`,
  `avg_total_ms`);
- `slot_backend[j]`, the backend chosen for kernel leaf `j`; null means the
  leaf did not run for this case (for example the prefill kernel in a
  decode-only batch);
- `coverage`, the slot indices whose time did not come from inside the
  measured grid, by flag: `extrapolated` (outside the grid), `jit` or
  `no_coverage`.

Where the time goes:

```bash
jq -r '.cases[1].sections[0].nodes[]
  | [.ms, .total_ms, (.pct * 10 | round / 10), ("  " * .depth) + .label] | @tsv' prediction.json
```

Errors carry `detail`:

| Status | Cause | `detail` |
| --- | --- | --- |
| 400 | `params` miss an axis, name an unknown one, or match no member | `{message, choices}`; `choices` lists every member's `params` |
| 400 | Bad cases: empty, more than 64, wrong group count, unknown field, a length past `max_model_len`, a wrong query width | the simulator's message, naming the case |
| 404 | Unknown preset id | `no public preset '...'` |
| 409 | The member does not build, lacks profile.db rows, or a case needs a row nobody measured | the build error or the missing rows; the service never profiles, so report it and pick another member or shape |

```bash
curl -s -X POST "$API/predict" -H 'content-type: application/json' \
  -d '{"preset": "Llama-3.1-8B/llama3_dense_tp", "params": {"tp_size": 3}, "cases": [{"groups": [{"decode_kv_lens": [1024]}]}]}'
# {"detail":{"message":"Llama-3.1-8B/llama3_dense_tp has no member {'tp_size': '3'}","choices":[{"tp_size":1},{"tp_size":2},{"tp_size":4},{"tp_size":8}]}}
```

### Analyze and keep a prediction

Add `"analyze": true` to the request. The answer gains a `prediction_id`, and
the Analyzer serves the prediction under `$API/analyzer/predictions/{id}/` for
six hours.

```bash
ID=$(curl -s -X POST "$API/predict" -H 'content-type: application/json' -d '{
  "preset": "Qwen3-235B-A22B-FP8/qwen3_moe_fp8_dp_attn_ep_ffn",
  "params": {"attn_tp_size": 4, "ep_size": 8, "workload": "ctx8k_out1k"},
  "cases": [{"groups": [{"decode_count": 64, "average_decode_length": 8192},
                        {"decode_count": 64, "average_decode_length": 8192}]}],
  "analyze": true}' | jq -r .prediction_id)
P="$API/analyzer/predictions/$ID"
curl -s "$P/descriptor" | jq '{selector, case_count, gpu, lifecycle}'
```

Case ids are positions in `cases` (`0`, `1`, ...). Operation ids are positions
in a case's `operations`, one per section: `0` for `iter`, `0` to `4` for the
`ffn` sections. The useful routes under `$P`:

| Route | Gives |
| --- | --- |
| `subjects/cases/payload` | each case's input, sections (`operations`) and total |
| `cases/{case}/operations/{op}/subjects/cost-tree/payload` | the section's tree with each leaf's kernel config, backend, exact input, time, FLOPs, bytes and achieved TFLOPS or GB/s; `time_share` gives each kernel position's share of the section's time on its critical path |
| `cases/{case}/operations/{op}/leaves/{leaf}/subjects/kernel-throughput-analysis/payload` | one leaf's measured grid points around its exact input; `{leaf}` counts the cost-tree payload's nodes in preorder, root 0 |
| `cases/{case}/subjects/optimality-waterfall/payload` | the case's GPU-seconds (time × GPUs) split into necessary hardware work and the gaps: batching, communication, imbalance, hardware gap |
| `cases/{case}/subjects/optimality-kernel-ladder/payload` | the same rungs per kernel position |
| `subjects/kernel-input-distribution/payload` | the inputs each kernel position saw across all cases |
| `subjects/scoped-optimality/report?path=iter/root/1` | optimality of one subtree; `path` is the section and child ordinals from the root (or pass `label`) |

```bash
curl -s "$P/subjects/cases/payload" | jq -c '.cases[] | {case_id, total_time_ms}'
curl -s "$P/cases/0/subjects/optimality-waterfall/payload" | jq '.level | {total, buckets}'
```

An unknown or expired id answers 404 `prediction_not_found`.

## 5. Interpret and report

What a time is depends on the selector:

- `iter`: one forward iteration of the whole model on one replica
  (`gpus_per_replica` GPUs), every group in the case included. Section `iter`,
  `layer` -1.
- `speculative_iter`: one verify iteration of the target model together with
  the proposer pass the tree shows (for example `unified.dflash2`).
- `attn`: one layer of attention on one attention shard (`layer` 0). Every
  layer sees the same batch, so the attention work of an iteration is this
  time times the layer count, which the root label states (`94 layers`).
- `ffn`: building blocks of one iteration on the FFN pool: `prologue`
  (embedding), `pre_attn` (layer 0's QKV), `post_attn` for one representative
  middle layer (which stands for each layer but the last), `post_attn_last`,
  and `epilogue` (final norm and LM head). One iteration is
  `prologue + pre_attn + (layers - 1) × post_attn + post_attn_last + epilogue`.

The `attn` and `ffn` presets model the two pools of an attention-FFN
disaggregated deployment. The transfer between the pools is not in either
tree, so report the two sides separately. The Analyzer's optimality subjects
sum the sections they are given, so for `ffn` they cover the five building
blocks once, not a full iteration.

A predicted time is the sum, max and repeat of kernel times along the tree.
It includes in-tree communication (all-reduce, MoE dispatch and combine) but
no CPU scheduling or launch gaps between kernels. It is not a request latency:
there is no queue, no scheduler, no arrival process, and no TTFT or TPOT.
Those need a simulation of the request stream, which `/predict` does not run.

When you answer, name the preset, the member's `params` (with the MoE
`workload` and whether it reads a capture), the GPU and `gpus_per_replica`, the
exact cases, each case's `total_ms`, and `sim_commit`. Mention any slots under
`coverage.extrapolated`, since their times lie outside the measured grid. Quote
only times the API returns; label any arithmetic on them, such as scaling an
`attn` time by the layer count, as your own.
