# feat(trn2): compiled prefill-bucket inventories for stock vLLM Neuron

## Purpose

Let the stock vLLM Neuron target compile and cost more than one prefill
graph. vllm-neuron is a static-graph engine. It compiles one prefill graph for
each `num_batched_tokens_buckets` entry, and it pads each prompt to the
smallest graph that holds it. Its default inventory is `[128, 256, ..., C]`
(`vllm_neuron/utils/bucket_utils.py`). The profiler and the simulator pinned
the inventory to the whole-context graph `[C]`, so every prompt cost a
full-context prefill. A hardware replay on 2026-10-20 showed that three
prefill buckets alone cut TPOT p90 by 15% at 8 req/s on the chat trace. The
simulator could not model that inventory, and the profiler refused any
prefill row below C. This PR adds the inventory end to end. Decode buckets
stay powers of two, the vllm-neuron default.

- **Whole-forward kind** (`simulator/src/timing/kernels/neuron_llama_forward.rs`):
  `prefill_buckets` (empty = `[C]`), validated as ascending multiples of 128
  (`PREFILL_BUCKET_STEP`) that end at the context. The categorical cache
  (`Cache1DDirect`) indexes each prefill bucket, then each decode bucket. A
  bucket outside the inventory fails, as before. Region, segment and
  segmented-prefill kinds keep `[C]`.
- **Arch** (`simulator/src/arch/llama3_vllm_neuron.rs`, selector field in
  `arch/config.rs`, pass-through in `arch/build.rs` and
  `deployment/unified.rs`): `lower_input` pads a prompt to the smallest
  compiled prefill bucket that holds it, as
  `NeuronScheduler._calculate_padded_count` does. An inventory other than
  `[C]` requires `composition: whole_forward` and no `prefill_segment_size`
  (`validate_selected`). `[C]` builds the same kernel config as the empty
  default.
- **Profiler** (`profiling/runners/neuron/{stock_contract,vllm_forward,vllm_forward_engine,vllm_forward_trace,vllm_identity}.py`):
  the engine compiles every requested prefill bucket plus C
  (`prefill_buckets_of`). A bucket T below C is validated (full logits against
  the CPU FP32/BF16 references, vendor criterion) and timed on its own
  T-8-token prompts. Those requests are marked `prefill_only`, so their
  decodes (shorter KV) are not decode samples. The binary seal covers every
  compiled prefill graph. Region, segment and per-operation captures keep the
  whole-context graph (`validate_whole_context_shape`).
- **Rows**: 8 new stock prefill rows in `profiling/profile.db` (C512
  128/256/384; C1024 128/256/384/640/896). Two shapes failed the vendor
  numerical criterion and are not written (C1024 512 and 768).
- **Record**: `doc/pr/trainium2-prefill-buckets/README.md` (rows, sweep,
  hardware replay, measurement notes).

Not changed: the alignment pipeline (`alignment/neuron/`) still serves `[C]`.
Its stock graph check and iteration metric assume it, so the hardware replay
below started the server directly with the same engine flags. Every
existing preset and row is unchanged: an empty `prefill_buckets` is `[C]`.

Doc-backed: `doc/trainium2.md` (updated: inventory rules), `doc/invariants.md`
(categorical lookup; unmeasured shapes fail), `doc/detailed_design/L1.md`.

## Results

C512, 600-request chat trace (in+out <= 512), decode `[1, 2, 4, 8, 16, 32, 128]`,
hardware replay of the same schedules (0 errors):

| prefill inventory | rate | TTFT p90 ms | TPOT p90 ms | tok/s |
| --- | ---: | ---: | ---: | ---: |
| `[512]` (previous best) | 8 | 218 | 141.2 | 2,296 |
| `[256, 384, 512]` (new best) | 8 | 198 | 119.9 | 2,304 |
| `[256, 384, 512]` (new best) | 10 | 566 | 130.4 | 2,783 |
| `[512]` (previous best) | 12 | 12,033 | 149.4 | 2,628 |

On hardware, the new best meets the SLO (TTFT p90 < 1 s, TPOT p90 < 150 ms)
at 10 req/s. The simulator puts the previous best at 8 req/s. For the new
best, simulated TPOT p90 is within 2% of hardware at 8 and 10 req/s.
Simulated TTFT p90 is within 2% at 8 req/s. At 10 req/s (near saturation) it
is 2.5x too low. Evidence: `doc/pr/trainium2-prefill-buckets/README.md`.

### With the GQA decode patch (decode attention on the NKI kernel)

The numbers above are the stock engine: at TP4, Llama3 decode attention runs
the Torch fallback, because each rank holds 2 KV heads and the model passes a
2-D block table. A local patch to `vllm_neuron/model/llama3/model.py` passes a
per-head block table, as `qwen3_vl/model_bf16.forward_decode` does. With it,
decode attention runs the NKI `attention_block_tkg` kernel. The patch is not
part of this PR, and its rows are not in this PR's `profile.db`. It was
bind-mounted read-only into the same image; every patched row passed the same
vendor full-logit check.

Patched C512 decode rows (ms, stock in brackets): B16 12.9 (55.3), B64 20.3
(114.7), B128 33.7 (84.3), B256 51.4 (153.3), B512 99.4 (303.1). Prefill rows
do not change.

Hardware replay with the patch, prefill `[256, 384, 512]` (this PR's
inventory), decode `[1, 2, 4, 8, 16, 32, 64, 128, 256]`, 256 resident, same
schedules, 0 errors:

| rate | TTFT p90 ms | TPOT p90 ms | tok/s |
| ---: | ---: | ---: | ---: |
| 10 | 97 | 32.6 | 3,111 |
| 16 | 133 | 81.1 | 4,734 |
| 18 | 156 | 128.5 | 5,055 |
| 20 | 1,200 | 145.6 | 5,275 |

With the patch, the SLO rate is 18 req/s (stock best: 10), and 5,055 tok/s at
the SLO (1.8x). At 10 req/s, TTFT p90 is 83% lower and TPOT p90 is 75% lower
than the stock best. The prefill inventory of this PR is part of that
configuration.

## Test Plan

```bash
uv run cargo test -p simulator --lib
uv run cargo build --release -p simulator
uv run pytest tests/test_neuron_full_forward.py tests/test_stock_neuron_alignment.py \
    tests/test_stock_neuron_identity.py tests/test_neuron_segmented_forward.py \
    tests/test_neuron_llama_segment.py tests/test_neuron_nki_ops.py \
    tests/test_neuron_ops_capture.py tests/test_kernel_docs.py -q
```

Formatting: `ruff check` and `ruff format` on the touched Python files,
`rustfmt` per file on the touched Rust files.

Tiers not run: `just test-gpu` (no CUDA device) and `just test-presets`. The
tests were run on macOS. Files that import torch at collection
(`tests/test_neuron_llama_region.py` and others) and the Linux-only process
tests (`os.waitid`, `/proc`) did not run there; the full cpu tier should be
re-run on a Linux host before merge.

## Test Result

- Rust: `test result: ok. 1640 passed; 0 failed; 8 ignored`
  (`cargo test -p simulator --lib`).
- Python: the eight files above, 714 passed, 6 skipped (frozen tokenizer and
  device captures are not on the host). Includes the needs_binary
  Rust/Python agreement on `PREFILL_BUCKET_STEP`.
- On the chip: 8 prefill rows measured through `profile_forward_batch`.
  Existing rows measured again agree within 1% (C512 prefill 512 38.39 vs
  38.23 ms, decode 16 55.31 vs 55.26, decode 128 84.34 vs 84.26; C1024 prefill
  1024 66.43 vs 65.80, decode 16 38.97 vs 38.85).
- Simulator sweep: 96 runs (4 prefill inventories x 6 decode inventories x 4
  rates); best `[256, 384, 512]` + `[1, 2, 4, 8, 16, 32, 128]`, 10 req/s.
- Hardware replay of the new best at 8 and 10 req/s: 600 requests each,
  0 errors (table above).

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none (vllm-neuron behavior is
cited, not vendored).

---
<details>
<summary>Checklist</summary>

- [x] **Doc-backed — name the file.** `doc/trainium2.md` (updated here),
      `doc/invariants.md`, `doc/detailed_design/L1.md`.
- [x] **Formatting radius matches the change.** `rustfmt` per touched Rust
      file; `ruff format` / `ruff check` on the touched Python files.
- [x] **`profiling/profile.db`:** 8 new `neuron_llama_forward` rows
      (`vllm_neuron`, AWS Trainium2 LNC2); no existing row changes.
- [x] **Unmeasured shapes still fail.** A prefill bucket outside
      `prefill_buckets` panics in `cache_coords` (tested); a missing row is a
      `MissingEntry` build error as for every kind.
- ~~**Goldens.** No modeled number moves for an existing preset.~~

</details>
