# Trainium2 standalone NKI per-op leaves: evidence

Measured 2026-10-18/19 on one trn2.3xlarge (one Trainium2 chip, LNC2), vendor
image `pytorch-inference-vllm-neuronx:0.24.0.1.1.0` (vllm-neuron 0.24.0.1.1.0,
neuronx-cc 2.27.5334.0), Llama 3.1 8B BF16 at the per-rank shapes of TP4, one
logical core. Every number is device execution time (`nc_exec_running`, union
of the core's two physical cores), median of 50 calls after 10 warm-ups,
standalone and raw: the empty-NEFF floor is not subtracted (see "The floor").
`nki_leaf_device_times.json` holds every measured point, keyed by each kind's
args as the Rust kinds enumerate them.

## Maintained artifacts

| File | What |
| --- | --- |
| `nki_leaf_device_times.json` | source and method, the empty-NEFF floor, 98 leaf rows (args, n, device ms median/min/p90), 54 non-leaf reference rows, 10 superseded failures |
| `../../trainium2.md` | design record: "Standalone NKI per-op leaves" |
| `tests/test_neuron_nki_ops.py` | shape refusal, raw rows with the floor in provenance, trace union, Rust/Python inventory agreement |

## The floor

`noop` (one add on 128 elements, its own NEFF): **0.026 ms** per execution.
The runner records it in every row's provenance and does not subtract it: the
decode glue at B1 (0.0215 ms) runs below it, and the in-model check below
shows the in-model kernels pay about half of it themselves. What to charge
per launch is the composition's decision.

## Prefill units per layer per rank (T = context), ms

| Unit | T128 | T512 | T1024 | T2048 | T4096 | T8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `neuron_nki_qkv` | 0.066 | 0.091 | 0.136 | 0.225 | 0.411 | 0.764 |
| `neuron_nki_attention_cte` | 0.036 | 0.077 | 0.135 | 0.276 | 0.695 | 2.272 |
| `neuron_nki_output_projection` | 0.034 | 0.050 | 0.079 | 0.134 | 0.247 | 0.466 |
| `neuron_nki_mlp` (prefill) | 0.293 | 0.612 | 1.241 | 2.481 | 4.961 | 9.920 |
| KV write (torch `index_put_`, inside `neuron_glue`) | 0.026 | 0.035 | 0.047 | 0.071 | 0.121 | 0.222 |
| `neuron_glue` (prefill) | 0.055 | 0.179 | 0.240 | 0.398 | 0.814 | 1.771 |

MLP and the projections are linear in T above T512 (MLP 1.21 us/token); the
attention kernel is quadratic from T2048 on; below T512 every unit sits on a
fixed cost of 30-70 us that the floor only partly explains.

## Decode shape-only units per layer per rank (B requests), ms

| Unit | B1 | B2 | B4 | B8 | B16 | B32 | B64 | B128 | B256 | B512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neuron_nki_mlp` (decode) | 0.161 | 0.164 | 0.167 | 0.169 | 0.170 | 0.170 | 0.173 | 0.292 | 0.293 | 0.612 |
| `neuron_glue` (decode) | 0.022 | 0.021 | 0.042 | 0.044 | 0.044 | 0.039 | 0.041 | 0.041 | 0.061 | 0.105 |

Decode MLP is weight-bandwidth bound and flat up to B64 (88 MB of BF16
weights per rank in 0.17 ms = 0.52 TB/s), then steps at B128 and B512 where the
kernel changes tiling; B512 equals the prefill T512 row, as it should.

## Decode attention block (`neuron_nki_attention_tkg`), B x context, ms

| C \ B | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.094 | 0.089 | 0.085 | 0.097 | 0.134 | 0.193 | 0.304 | 0.497 | | |
| 512 | 0.102 | 0.099 | 0.102 | 0.125 | 0.135 | 0.206 | 0.318 | 0.542 | 0.935 | 1.956 |
| 1024 | 0.105 | 0.099 | 0.098 | 0.129 | 0.139 | 0.208 | 0.313 | 0.529 | 1.012 | |
| 2048 | 0.102 | 0.102 | 0.104 | 0.135 | 0.185 | 0.263 | 0.402 | 0.848 | | |
| 4096 | 0.111 | 0.102 | 0.117 | 0.151 | 0.220 | 0.348 | 0.609 | | | |
| 8192 | 0.114 | 0.113 | 0.144 | 0.201 | 0.336 | 0.551 | | | | |

Empty cells exceed the context's KV-resident capacity (`largest_decode_bucket`)
and are not compiled by the stock engine. Below B8 the block is fixed-cost
(about 0.1 ms: QKV and output projection weights, 12.6 MB per rank); above it
the paged KV gather dominates and the time is close to linear in B x C
(B64 C8192 would read 64 x 8192 x 2 heads x 128 x 2 B x 2 = 537 MB).

The stock engine's decode mask (`gen_attention_decode_mask`, 0.020-0.089 ms)
is not a leaf: the per-head-table path generates the mask inside the kernel.

## In-model check (C512, from the 2026-10-18 device traces)

| Phase | Raw standalone sum | Floor removed per unit | In-model per layer |
| --- | ---: | ---: | ---: |
| prefill T512: four NKI kernels + the KV write inside glue | 0.865 ms (+2%) | 0.735 ms (-13%) | 0.847 ms of kernel spans in a 1.169 ms layer |
| decode B16, per-head table path: attention + mlp + glue + 2 all_reduce | 0.371 ms (+13%) | 0.293 ms (-11%) | 0.328 ms layer span |

The decode mask (0.022 ms) is once per forward, not per layer. The in-model
per-kernel fixed cost sits between zero and the standalone floor, about half
of it; the leaf keeps the raw time and the composition decides the charge.
The stock decode attention (torch fallback) is 0.563 ms standalone against a
1.469 ms GpSimd-bound region in the model; it is not a leaf.

## Known gaps

- The decode-attention leaf exists only with the per-head block-table change
  to Llama3 `forward_decode`; the stock binary never reaches the kernel at
  TP4. Costing a stock deployment with it assumes that change.
- `neuron_glue` bounds the layer's non-kernel work; the model fuses the same
  ops differently (prefill in-model non-kernel time at C512 is about 0.32 ms
  per layer including collectives, against 0.179 + 0.035 standalone).
- Collectives (all_gather, reduce_scatter, all_reduce via nccom-test) are not
  kinds yet; the values measured at C512 are in `trainium2.md`.
- No composition assembles the leaves into a layer yet (`composition: nki_ops`
  is the next step); no row is in the shared `profile.db`.
- The public profiling CLI (host side: registry, reservation, container
  launch, DB write) was not run against the chip; the grid rows come from the
  standalone script the runner was ported from (`bench_kernels.py`, kept with
  the measurement scripts outside the tree; the JSON keeps its
  `decode.gen_mask` and `prefill.kv_write` units under `not_leaves` and its ten
  `FailOnRecompileLimitHit` rows from before the per-unit Dynamo reset under
  `superseded`, each re-measured). The runner itself (`nki_bench.profile_*_batch`) was executed inside
  the vendor image on one logical core and reproduced the grid: decode MLP B16
  raw 0.1699 ms against the grid's 0.1695, decode attention B1 C512 raw 0.1017
  against 0.1016, floor 0.0256 against 0.026.

## Reproduce

```bash
# through the registry (needs the Neuron host configuration of trainium2.md);
# the grid CSV itself was produced by the out-of-tree measurement scripts:
uv run python -m profiling run neuron_nki_mlp --backend vllm_neuron_nki --gpu-name 'AWS Trainium2 LNC2' \
  --db "$TMPDIR/nki_smoke.db" --spec '{"phase":"decode","token_bucket":16,"tp_size":4,"dtype":"bf16"}' --json
```
