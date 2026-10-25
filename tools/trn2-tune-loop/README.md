# trn2-tune-loop

Closes the loop between the Sim's config search (`tools/trn2-config-search/`) and the
flashnext-neuron engine (`framework/flashnext-neuron`, override with
`FLASHNEXT_ENGINE_ROOT`) for Qwen3.8-Flash-Next BF16 on one trn2.48xlarge: pick
configs, predict them, run them on the engine with the same knobs and load, and check
the predictions. Results of the first loop: `doc/pr/trn2-flashnext/TUNE_LOOP.md`;
skill: `operate-tune-trn2-with-engine`.

```bash
source ../neuron_env.sh; export PATH=$HOME/.cargo/bin:$PATH
T="uv run --no-sync python tools/trn2-tune-loop/tune.py"
$T propose blind_w64 --exp logs/trn2/search_v2/s3_mtp --workload fixed_2k_512-a0.67 --top 1 \
    --variant 'chunk4096:max_batch_tokens=4096' --variant 'tp8:attn_dp=4,attn_tp=8,gdn_tp=8' \
    --contrast 'a2a:moe_dispatch=all_to_all'
$T predict blind_w64                 # Sim only (--no-gpu), ~5 s per candidate with the ladder
$T predict blind_w64 --tag accdist --accept-from w64_2048_512_block_k1   # drawn acceptance
$T run blind_w64                     # writes logs/trn2/tune/blind_w64/engine/<label>.sh
$T run blind_w64 --execute           # on the device, blocking: inside a tmux session tune-*
$T compare blind_w64 --resim         # comparison.md; --resim re-simulates with measured acceptance
$T compare blind_w64 --tag accdist   # comparison_accdist.md
$T headroom blind_w64 top1          # R0-R7 ladder + what-ifs on one candidate's knobs (Sim only)
$T status blind_w64
```

| Step | Reads | Writes | Notes |
|---|---|---|---|
| `propose` | `<exp>/results.jsonl` of one or more search experiments | `proposal.json` | Top-N base-scenario rows of the workload that the engine can run (`engine_blockers`), then `--variant` / `--contrast` overrides of the best and `--config` candidates from scratch. Each is completed with the engine's knobs (`ENGINE_DEFAULTS`: decode buckets 1-16, context 2048/4096, 2 replicas; prefill buckets of `128, 512, 2048` up to the chunk budget). Refuses Sim-only knobs and the case the Sim does not model (`sim_mismatches`) |
| `predict` | `proposal.json` | `sim/<label>/` (preset, trace, run), `predictions.json` | One simulation per candidate, on the engine workload's trace (W64: 64 x 2048/512, 50 ms apart, uniform `accept_rate` for MTP), 2 replicas round-robin (the engine's rid parity). Records the R0-R7 ladder and the uncertain inputs. Resumes per candidate |
| `run` | `predictions.json` (required: the loop is blind) | `engine/<label>.sh`, `runs.json` | `serve.py` on cores 0-31 and 32-63, comm ports 61000 / 63000, every shape warmed first, a profiled window at steps 150-190. `--execute` runs unfinished candidates one at a time and refuses while a `slice-*` container is up. A candidate is finished when every replica's `serve_<run>.jsonl` ends with its summary row |
| `compare` | predictions + `logs/engine/serve_tune_<name>_<label>_r*.jsonl` | `comparison.{json,md}` | Error per metric (duration, out tok/s, TTFT p50, TPOT p50; the engine's nearest-rank percentiles), measured MTP acceptance, ranking per metric (Kendall tau, best agrees, pairs that flip beyond a 3% measured tie), verdict |

`headroom <name> <label>` reruns one candidate under the stage-4 what-ifs of the config
search (`SCENARIOS`: a 3x faster MoE prefill kernel, idle-DP padding skip, 2x faster
host, fused decode + prefill forward, 16 chunks per group, all-to-all at capacity
factor 1.25, and the first two together) and writes `headroom_<label>.{json,md}` with
the base run's R0-R7 cluster ladder (GPU-seconds over every rank) and gap buckets.

**Acceptance draws.** `--accept-from` gives each request the per-request MTP acceptance
of earlier engine runs (`accept_draw`, cycled in rid order). On W64, a uniform rate
predicts 10-15% short because duration follows the slowest requests. Per-request values
do not carry over between runs (correlation <= 0.26); their spread does.

**First-use stalls.** `compare` subtracts the engine's first-use stalls
(`first_use_stall_s`). Each of the first 3 steps is compared with its expected wall:
the medians of later forwards of the same shape, plus the later steps' median host
time. It reports the raw errors (`err_raw`) and the raw-duration ranking beside the
gated ones.

**Engine HBM.** `engine_blockers` refuses block prefill over 8192 gathered MoE rows.
The engine failed to load dp8 x 2048 with `Allocation Failure`.

**Verdict.** `realign` when any of duration, out tok/s or TPOT p50 is off by more than
`--threshold` (15%) or the duration ranking flips on a pair more than 3% apart measured;
TTFT p50 only warns (on W64 it is the prefill backlog plus the engine's first-use
stalls, which the Sim does not model). `realign` means: run `tools/trn2-engine-align`
on the new runs (add them to `engine_align.RUNS`) before trusting the next search.

**Every step logs** to `logs/trn2/tune/<name>/tune.log` and records itself in
`state.json`. Rerunning a step resumes (`--force` redoes `propose` / `predict`).

## Knob mapping

| Sim knob | Engine flag |
|---|---|
| `attn_tp` = `gdn_tp`, `attn_dp` (`attn_dp x attn_tp = 32`) | `--tp`, `--dp-mode split` (32 cores per replica) |
| `moe_ep` 32, `moe_tp` 1 | `--ep 32` |
| `region_granularity` per_op / mixer / layer / block | `--granularity op / mixer / layer / block` |
| `tp_collective`, `moe_dispatch`, `moe_a2a_capacity_factor` | `--tp-collective`, `--moe-dispatch`, `--moe-a2a-cf` |
| `draft_tokens` | `--mtp-k` |
| worker `max_batch_tokens` | `--max-batch-tokens` |
| `prefill_buckets`, `decode_buckets`, `context_buckets` | the same flags |
| `replicas` 1 / 2 | `--replicas 2 --replica i` on cores 0-31 / 32-63 |

Sim-only (refused when set to anything but the engine's behaviour): `step_forwards`
(separate), `spmd_padding` (true), `max_prefill_chunks_per_partition` (1),
`ple_table_shards` (replica), `launch_model` (pipelined), `kind_time_scale`,
`host_time_scale`, `region_device_factor`, `moe_a2a_pack_us_per_row`, `*_impls`, P/D
deployments, 64-core replicas, `gdn_tp != attn_tp`, EP < 32.

**Not modeled by the Sim** (refused by `sim_mismatches`): a chunk budget above the
largest prefill bucket with prompts longer than that bucket. The engine caps a chunk at
the largest bucket; the Sim's chunked-prefill worker runs the whole budget as several
bucket calls in one step.

## Tests

`tests/test_trn2_tune_loop.py` (CPU tier): a stub `scripts/slice_run.sh` writes the
timeline a real run would, so `run --execute` and `compare` are exercised without a
device.
