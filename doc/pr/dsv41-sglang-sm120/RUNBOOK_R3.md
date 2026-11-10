# Rental R3: 4x RTX PRO 6000. Realistic-text replays, prefill CUDA graph, 1M-context captures

**Machine:** the same as 2026-11-05 / B2: Massed Compute `RTXPro6000x4` VM (no GPU P2P), $8.76/h via Shadeform.
**Budget:** a hard cap of **2 h ($17.50)**, set 2026-11-08 by the user. The instance is deleted before reporting.
**Sampling** for the live replays: `temperature 1.0`, `top_p 0.95` (the user's generation config).

## Why this rental

Three things the 2026-11-07 alignment could not settle without a GPU (`KERNEL_ALIGN.md`, second pass):

1. **Live replays used random token ids.** DSpark accepted 5.0-5.45 tokens per verify on them, against about 2.5 on
   real text. So the measured Decycle TTFT/TPOT are optimistic, and the e2e comparison had to be run at an
   artificial acceptance.
2. **The prefill CUDA graph was off** (`--disable-prefill-cuda-graph`, copied from sglang#40877). SGLang v0.5.23's
   CUDA default is the breakable prefill graph, which bounded replay supports. Eager prefill idles 15-20%
   (`prefill_gpu_time_multiplier` 1.17) and stalls decodes on busy days.
3. **The dense-blocks decode indexer cost is calibrated, not measured** (`DENSE_INDEXER_PASSES = 10`). Its
   tensors are sized by `--context-length` / ratio, and every B2 capture ran at 8K-16K context length. The
   simulator is still 25% fast at 8 running requests with 1M context length.

## What is ready on kisnis

**Live replay** (airbox-cost-calculator `docs/clients/decycle/sim/kt_bench/`):
- `corpus/agent_corpus.txt`: 17,684 agent trajectories, about 25.6M tokens. Built from tau-bench Sonnet traces
  (agent side, with tool schemas) and Simia AgentBench SFT; message text only, no request headers.
  Rebuild with `build_agent_corpus.py`.
- `corpus/agent_stream.npz`: already tokenized on kisnis with the checkpoint's own `tokenizer.json` (28.3M tokens,
  vocab 129,280, round-trips to the text), so the rental does not need a tokenizer step. Copy this file (113 MB)
  to the rental. `tokenize_corpus.py` rebuilds it.
- `replay_trace_sglang.py --token-stream agent_stream.npz [--temperature T --top-p P]`:
  - each session reads its own trajectory, in order;
  - later rounds still reuse the previous prompt plus the model's real output, so prefix-cache hits behave as
    before;
  - records `spec_accept_length` / `spec_verify_ct` per request.

  Tested against a mock server.
- `serve_sglang_4xpro6000.sh`: `PREFILL_GRAPH=1` drops `--disable-prefill-cuda-graph`; the default 0 keeps the
  2026-11-05 server.

**Alignment campaign** (pack `presets/alignment/dsv41_flash_rtxpro6000_sglang_tp4_ep4`, `check`: 0 errors):
- new variant `tp4_ep4_sglang_dspark_bcg`: the DSpark server with the breakable prefill graph;
- new cases:

| case | variant | context length | what it measures |
|---|---|---:|---|
| 15 `decode_midbatch_ctx1m_c4` | dspark | 1M | decode at 4 running requests, 32-64K contexts |
| 16 `decode_midbatch_ctx1m_c8` | dspark | 1M | decode at 8 running requests (where the sim is 25% fast) |
| 17 `balanced_anchor_c16_ctx1m` | dspark | 1M | case 07 at 1M context length: isolates the `lmax` cost |
| 18 `balanced_anchor_c16_bcg` | dspark_bcg | 16K | case 07 with prefill graphs: duty cycle and kernels |
| 19 `prefill_spectrum_c8_bcg` | dspark_bcg | 64K | chunk boundary and long prompts with prefill graphs |

## Order on the rental (2 h cap)

| time | step |
|---|---|
| 0:00-0:50 | Boot, provenance, host settings, image pull; 510 GB download in the background. Meanwhile: copy the replay files and `agent_stream.npz`; set up ServingStudio in the container (`uv sync`, rustup, mold) for the optional capture. |
| 0:50-0:58 | L1 server: `MEMFRAC=0.86 MAXRUN=16 PREFILL_GRAPH=0`. |
| 0:58-1:13 | **L1: busy day, hour 13 dense part, 15 min**: agent text, `--temperature 1.0 --top-p 0.95`. |
| 1:13-1:21 | L2 server: `PREFILL_GRAPH=1`. Record `max_total_num_tokens` and the prefill-graph capture lines. |
| 1:21-1:36 | **L2: the same busy window.** If L2 cannot start (OOM), use this slot for L1's typical window (first 15 min of hour 0). |
| 1:36-1:52 | Optional, only if started by 1:38: case `16_decode_midbatch_ctx1m_c8`, `profile_nsys` only. Warm start: the L2 server's JIT caches are already built. |
| 1:52-2:00 | Rsync results to kisnis (`kt_bench/results_r3_4xpro6000/`, `logs/r3/`); delete the instance; confirm via `/instances/<id>/info`. |

Busy-window replay command (the same 31 sessions as 2026-11-05, `--hours 13.6 14`, cut at 15 min by `--max-wall-s 900`):

```bash
python3 replay_trace_sglang.py decycle-peak.csv --hours 13.6 14 --url http://127.0.0.1:30000 \
  --token-stream agent_stream.npz --temperature 1.0 --top-p 0.95 --max-wall-s 900 --out replay_L1_busy.jsonl
```

Deferred to a later rental: cases 15, 17, 18 and 19, and both typical windows.

## Afterwards (kisnis, no GPU)

1. **Live L1 vs the 2026-11-05 random-id run:** the real-text acceptance, and Decycle TTFT/TPOT at realistic
   acceptance. These replace the numbers in the client comparison.
2. **L2 vs L1:** what the prefill graph buys on the busy day. If it helps, it becomes the recommended server
   config.
3. **Cases 16 and 17:**
   - label (new kernels may need rules), then kernel-align;
   - replace `DENSE_INDEXER_PASSES` with a fit to the measured indexer kernels, or with profiled rows if they
     need a runner;
   - refit `gpu_time_multiplier` at 1M context length.
4. **Case 18:** `prefill_gpu_time_multiplier` with prefill graphs. If L2 is adopted, the presets need a prefill-graph
   switch in the worker.
5. Re-simulate the windows at the measured acceptance and compare with L1/L2.

## Stop rules

- **Hard cap 2 h.** If the download is slow (under 150 MB/s after 10 min), drop the optional capture; if the weights
  are not ready by 1:05, run L1 only.
- Never `pkill -f`/`pgrep -f` a pattern inside an ssh command that contains it; use PID files.
