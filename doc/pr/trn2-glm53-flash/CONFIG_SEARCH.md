# Config search: GLM-5.3-Flash FP8 on one trn2.48xlarge, InferenceX 8k1k

**These are simulator predictions from partially imported L1 rows.** At the time of the search the
glm5-trn2 benchmarks had produced 477 measured rows (KDA tp 4/8/16, DSA tp 4/8 up to batch 64 and
context 16384, MoE / dense FFN up to 16384 gathered rows, every mHC / lm_head / embed row, and the
trn2-flashnext collectives); the other 393 grid cells are derived (`L1_CONTRACT.md`, "Import and the
derived grid fill"; counts per kind in `l1_coverage.json`). Rows a prediction leans on that are
derived: KDA and DSA at attn_tp 2 and 16 (DSA) (kind-median ratio), DSA decode above 64 rows and
context 32768, MoE above 16384 gathered rows (prefill at dp >= 8 with chunk 4096), `moe_partial`
impl `auto` (the arch uses the measured best of `dense_local` / `cte_hybrid` instead). Host time is
the trn2-flashnext engine fit (`region_host_ms` 1.3 per layer region, `forward_overhead_ms` 19,
`region_gap_ms` 0.17, `step_overhead_ms` 2.5), **not yet aligned on the GLM engine**.

Tool: `tools/trn2-config-search/search.py` (stages `main`, `whatif`, `kernels`); raw results:
`search/results_{main,whatif,kernels}.json` (one InferenceX point per layout and C),
`search/breakdown_main.json`. Workload: 8k1k closed loop, input U[6553, 8192], output 1024, fixed
concurrency C in {4, 8, 16, 32, 64, 128, 256} over the whole instance (2 replicas x 32 ranks),
2C warm-up + 10C measured requests. Metric: total (input + output) tok/s per chip (16 chips) over
the measured window; interactivity = 1000 / median TPOT.

## Search space

| axis | values |
|---|---|
| `attn_tp` (dp = 32 / tp per replica) | 1, 2, 4, 8, 16 |
| prefill chunk (= the one prefill bucket) | 512, 1024, 2048, 4096 |
| `dmax` = largest decode bucket = requests per DP group (slots dmax + 1) | 8, 16, 32, 64 |
| C | 4 ... 256 |

48 layouts x 7 C = 336 runs. Refused by the HBM check (24 GiB per LNC2 core, 2 GB reserve; FP8
FFN weights, BF16 attention / mHC / head, the engine's preallocated `slots x cmax` pools):

| layout | weights / rank | pools / rank |
|---|---|---|
| tp1, any dmax | 24.47 GB | 2.7 - 19.7 GB |
| tp2 d32 / d64 | 17.36 GB | 7.6 / 14.9 GB |
| tp4 d64 | 13.80 GB | 12.5 GB |

(tp8: 12.0 GB weights; tp16: 11.1 GB. Attention weights are BF16 and scale with 1 / tp; the FFN's
FP8 experts are 9.6 GB per rank at EP32 whatever the tp.)

## Answer: best tok/s/chip subject to median interactivity >= X

| X (tok/s/user) | engine as it is (layer regions, fitted host) | + one region per forward, host x 0.25 | + also MoE / mHC x 0.25, DSA / KDA x 0.5 | + also MoE / mHC x 0.1, DSA / KDA x 0.25 |
|---|---|---|---|---|
| 20 | **none** (max 9.2) | none (max 10.5) | 658 (tp16, chunk 2048, d16, C 64) | 2682 (tp4, chunk 2048, d16, C 256) |
| 30 | none | none | none (max 29.8) | 1927 (tp8, chunk 2048, d16, C 256) |
| 40 | none | none | none | 704 (tp16, chunk 2048, d16, C 32) |
| 50 | none | none | none | 388 (tp8, chunk 1024, d16, C 16) |

No layout of the seed engine reaches 20 tok/s/user: its best interactivity is **9.2** (tp8, chunk
2048, d8, C 4: 18.2 tok/s/chip). Without an interactivity floor the throughput winner is **tp4,
chunk 4096 (or 2048), d16, C 256: 555 (552) tok/s/chip at 4.7 tok/s/user**; at >= 8 tok/s/user it
is tp8, chunk 4096, d8, C 128: 248 tok/s/chip. Pareto front (layer regions, fitted host):

| layout | C | tok/s/chip | intvty | median TTFT (s) |
|---|---|---|---|---|
| tp8 chunk2048 d8 | 4 | 18.2 | 9.22 | 3.5 |
| tp8 chunk2048 d8 | 8 | 35.2 | 9.08 | 4.0 |
| tp8 chunk4096 d8 | 16 | 67.5 | 8.51 | 4.0 |
| tp8 chunk1024 d8 | 32 | 127.9 | 8.18 | 5.1 |
| tp8 chunk4096 d8 | 128 | 247.8 | 8.08 | 131.9 (queued: 64 slots per replica) |
| tp4 chunk4096 d8 | 256 | 393.5 | 6.46 | 167.5 (queued) |
| tp4 chunk4096 d16 | 256 | 554.7 | 4.74 | 7.1 |

Rules the search shows:

- **TP8 or TP4.** TP1 does not fit; TP2 fits only at d <= 16 and loses on both ends (best 465
  tok/s/chip, best interactivity 6.8: its larger attention shards cost more per row than its extra
  DP groups save); TP16 peaks at 423 tok/s/chip / 8.8 tok/s/user and becomes the TP of choice only
  when the kernels get much faster. Best per TP (throughput / interactivity): tp4 555 / 7.9,
  tp8 527 / 9.2.
- **dmax caps throughput.** Requests in flight per instance are 2 x dp x dmax; past that C only
  queues (TTFT in the minutes). d8 keeps decode at the 8-row bucket (best interactivity); d16 at tp4
  holds 256 requests (best throughput).
- **Larger chunks win at high C** (tp4 d16 C256: chunk 512 / 1024 / 2048 / 4096 = 451 / 507 / 552 /
  555 tok/s/chip): fewer prefill steps, each amortizing the MoE upcast; at low C the chunk barely
  moves interactivity.

## Bottlenecks

Mean time per engine step by L1 kind (ms, folded over layers; `search/breakdown_main.json`):

| step | tp8 chunk2048 d8, C 4 | tp4 chunk4096 d16, C 256 |
|---|---|---|
| decode-only step | **106.7** | **133.9** |
| - moe_partial (42 layers) | 50.2 | 61.1 |
| - dsa decode (11) | 12.0 | 19.1 |
| - mhc_pre + mhc_post (90) | 16.5 | 20.0 |
| - kda decode (34) | 9.7 | 15.1 |
| - launch + host beyond device | 12.1 + 2.5 | 11.1 + 2.5 |
| - collectives (all) | 3.1 | 4.4 |
| step with a prefill chunk | **778.5** | **2359.5** |
| - prefill moe_partial | 313.2 | 886.6 |
| - prefill kda | 186.9 | 681.1 |
| - prefill dsa | 91.0 | 327.4 |
| - prefill world AG + RS | 48.4 | 170.2 |

- **Decode is device-bound on the MoE partial.** Every decode layer pays ~1.2 ms of
  `moe_partial` even at 8 gathered rows: the region upcasts all 9 local experts' FP8 weights
  (~226 MB) to BF16 per call (`ops/moe.py` notes the same floor for `cte_hybrid`). That alone is
  ~50 ms of a ~105 ms decode step and caps interactivity near 10 tok/s/user. Next are the mHC
  regions (0.1 ms floor x 90 per step), DSA decode and KDA decode.
- **Host is not the decode limit at layer granularity with these constants**: 48 regions x 1.3 ms
  + 19 ms = 81 ms of host per decode forward overlaps ~92 ms of device. Collapsing to one region
  and quartering the host (what-if) moves interactivity only from 9.2 to 10.5.
- **Prefill steps stall decode.** A step that carries a prefill chunk costs 0.8-2.4 s (MoE over
  dp x chunk gathered rows, KDA chunked delta rule), so every request's TPOT absorbs the
  interleaved prefills of the others; this is why interactivity falls with C.
- **To reach 20 / 30 / 40 / 50 tok/s/user** (kernels stage): both host and kernels must move. With
  one region per forward and a quarter of the host time, MoE/mHC 4x faster plus DSA/KDA 2x faster
  reaches 20 (658 tok/s/chip at tp16); 30-50 needs MoE/mHC ~10x and DSA/KDA ~4x faster (e.g. a
  native FP8 expert GEMM without the per-call upcast, fused mHC, NKI KDA/DSA), where tp4 C256
  gives 2682 tok/s/chip at 21.9 and tp8 C256 1927 at 31.5.

## MTP placeholder (k = 3, golden acceptance)

`presets/trn2/glm53_flash_fp8_trn2_8k1k_mtp3.yaml` (seed layout, per-draft acceptance 0.8105, AL
3.00): interactivity 16.9 at C 4 (31.8 tok/s/chip), 13.7 at C 16 (105.7), 6.0 at C 128 (348.7). The
MTP pass is modeled as one DSA + MoE layer plus embed and head; verify rows use the KDA / DSA decode
rows at B x (k + 1) rows. The engine has no MTP yet: treat this as a ceiling sketch.
