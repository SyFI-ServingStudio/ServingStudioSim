# feat(l1): Trainium2 standalone NKI per-op leaves for Llama 3.1 8B TP4

## Purpose

Give the Trainium2 target GPU-style per-op L1 leaves. PR #96 costs the stock
vLLM Neuron forward as one leaf (`neuron_llama_forward`) and records that
"standalone probes of QKV, attention and O-projection were not additive, so no
finer rows exist". That finding was traced to one unit: at TP4 the Llama3
decode attention never reaches the NKI `attention_block_tkg` kernel (vllm-neuron
passes a 2-D block table with 2 KV heads per rank, and the kernel router sends
GQA without a per-head table to its torch fallback, which neuronx-cc lowers
differently inside the whole graph than alone). Every other unit of the layer is
an NKI custom call with a fixed launch boundary, and the sum of their standalone
times tracks the in-model layer. Prefill at T512 is 2% high. Decode at B16, with
decode attention through the kernel and a per-head table, is 13% high raw and
11% low with the empty-NEFF floor removed from each unit: the in-model per-launch
cost is between zero and the floor. This PR adds those units as leaves. It
does not change the composition: `whole_forward`, `model_head_regions` and
`layer_segments` stay as they are, and assembling the new leaves into a layer
(`composition: nki_ops`) is the next step.

Depends on #96: this branch is based on its head (`a030ec69`), so review the
diff against that commit.

- **Six L1 kinds, one backend `vllm_neuron_nki`**:
  `neuron_nki_qkv`, `neuron_nki_attention_cte`, `neuron_nki_output_projection`
  (prefill), `neuron_nki_attention_tkg` (decode, per-head block table),
  `neuron_nki_mlp` and `neuron_glue` (both phases). Args are the compiled
  variant (`phase`/`token_bucket`), the KV geometry where it matters, `tp_size`
  and `dtype`.
- **Runner** `profiling/runners/neuron/nki_bench.py` with the launch units in
  `nki_units.py`: each unit is one production launch at the per-rank shapes of
  TP4, compiled through the production path on the one reserved LNC2 logical
  core (`neuron_logical_cores=1`), warmed, executed 50 times under the NRT
  system profile; `time_ms` is the median device execution, raw. The empty-NEFF
  floor (0.026 ms) is measured once per batch and kept in the row's provenance,
  not subtracted: the decode glue at B1 runs below it and the in-model kernels
  pay about half of it, so the per-launch charge is the composition's decision.
- **Rust kinds** share one config and input (`timing/kernels/neuron_nki.rs`):
  `prefill_buckets` and `decode_buckets` name the exact inventory, the cache is
  `Cache1DDirect` over its index (no interpolation), an inventory needs at least
  two variants, decode attention is capped by its context's KV-resident
  capacity and the shape-only decode kinds by the ladder's largest bucket.
- **Measured grid** (not in `profile.db`, per the Neuron rule): 153 standalone
  points over the stock contract, in `doc/pr/trainium2-nki-ops/nki_leaf_device_times.json`
  (98 leaf rows keyed by kind args, plus the floor and two reference units) and
  summarized in `README.md` beside it.
- **Docs**: `doc/trainium2.md` gains the leaves' section and corrects the
  overlap/additivity wording to what the traces show (engines overlap inside a
  compiler region; launch units do not).

Doc-backed: `doc/detailed_design/L1.md` (one measured layer, KIND = table =
facade stem, `gpu_name` in the config), `doc/invariants.md` (an op's internal
composition is fixed and zero-overlap), `.claude/skills/top-split-model-into-kernels`
boundary rules (the launch boundary of the serving stack is the ground truth;
do not split a fusion). The decode-attention leaf deviates from "the stock
binary": it models the per-head-table decode path that the stock Llama3 at TP4
does not take, and its doc says so.

## Test Plan

```bash
uv run cargo test -p simulator --lib                 # 1466 passed (36 neuron tests, 8 new)
uv run cargo build --release -p simulator            # kernel-query grid/rows smoke on the new kinds
uv run pytest tests/test_neuron_nki_ops.py tests/test_kernel_docs.py -q   # 365 passed
uv run pytest tests/test_neuron_nki_ops.py -q        # 9 passed, incl. needs_binary Rust/Python inventory agreement
uv run python -m profiling list --json | grep neuron_nki   # 6 entries, facade symbols resolve
```

Formatting: `ruff check` and `ruff format` on the new Python files, `rustfmt`
per file on the new Rust files.

Tiers not run: `just test-gpu` (no CUDA device), `just test-presets`, and the
Neuron profiling smoke through the public CLI (`python -m profiling run
neuron_nki_mlp --backend vllm_neuron_nki ...`), which needs the Neuron host
configuration; the runner was instead exercised inside the vendor image on the
chip directly (see Test Result). The cpu tier was run on macOS without the
`profiling` dependency group (CUDA-only wheels): 3426 passed, 36 failed,
59 errors, all environmental and present without this change: `No module
named 'torch'` at collection (`test_neuron_llama_region.py`,
`test_speculative_alignment_records.py`, the kernel reference tests), the
Linux-only Triton runner (`test_elementwise.py`, `test_gdn_*.py`) and
`/proc/self/fd` (`test_launcher_process_supervisor.py`, `test_remote_pieces.py`,
`test_public_simulate.py`). The full cpu tier should be re-run on a Linux host
before merge.

## Test Result

- Rust: `test result: ok. 1466 passed; 0 failed; 8 ignored` (`cargo test -p simulator --lib`).
- Python: `tests/test_neuron_nki_ops.py` 9 passed; with `tests/test_kernel_docs.py` 365 passed.
- `kernel-query` `{"op":"rows","kind":"neuron_nki_mlp", ... "prefill_buckets":[512],"decode_buckets":[16]}`
  returns two rows with exactly `backend, phase, token_bucket, tp_size, dtype`;
  `{"op":"grid","kind":"neuron_nki_attention_tkg", ...}` returns the
  `compiled_variant` axis.
- Runner on the chip (worker side, inside the vendor image on one LNC2 logical
  core, `profiling/` mounted; 17 s for the batch):
  `profile_neuron_nki_mlp_batch` decode B16 → `time_ms=0.1694` (n=50, p90
  0.1704, floor 0.0258 in provenance), the B3 spec → `error="decode buckets are
  powers of two"`; `profile_neuron_glue_batch` decode B1 → `time_ms=0.0216`
  (below the floor, kept raw); `profile_neuron_nki_attention_tkg_batch` B1 C512
  → `time_ms=0.1017`. Every execution had one span on each of the two physical
  cores; exports are named per unit; provenance carries
  `vllm-neuron=0.24.0.1.1.0 neuronx-cc=2.27.5334.0+f702b353 torch=2.11.0` and
  the run directory.
- Modeled numbers: none change; no arch consumes the new kinds yet and
  `profiling/profile.db` is untouched.
- Measurement evidence: `doc/pr/trainium2-nki-ops/README.md` (grid tables, the floor,
  the in-model check at C512).

## Contribution licensing

- [x] I have read the project's CLA.
- [x] I have the right to submit this contribution.
- [x] I have disclosed any third-party code or licensing restrictions.
- [x] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none (the launch units call
vllm-neuron 0.24.0.1.1.0 public functions; no kernel source is vendored).

---
<details>
<summary>Checklist</summary>

- [x] **Doc-backed — name the file.** `doc/detailed_design/L1.md`,
      `doc/invariants.md`, `doc/trainium2.md` (updated here).
- [x] **Formatting radius matches the change.** Rust formatted per file with
      `rustfmt --edition 2021 --config skip_children=true`; `ruff format` /
      `ruff check` on the explicit list of new Python files.
- ~~**`profiling/profile.db`:** unchanged; Neuron rows live in private databases.~~
- [x] **Unmeasured shapes still fail.** A bucket outside `prefill_buckets` /
      `decode_buckets` panics in `cache_coords` (tested), and a missing row is a
      `MissingEntry` build error as for every kind.
- ~~**Goldens.** No modeled number moves.~~

</details>
