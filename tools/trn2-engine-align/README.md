# trn2-engine-align

Checks `qwen4_exp_neuron_bf16[_speculative]` against the flashnext-neuron engine's
serving runs (`framework/flashnext-neuron/logs/engine/serve_*.jsonl`; override the
directory with `FLASHNEXT_ENGINE_LOGS`). Simulator only: every launch is `--no-gpu`.
Results and their reading: `doc/pr/trn2-flashnext/ALIGNMENT.md`.

```bash
source ../neuron_env.sh; export PATH=$HOME/.cargo/bin:$PATH
A="uv run --no-sync python tools/trn2-engine-align/align.py"
$A replay logs/align/replay            # every engine step through timing-predict
$A e2e    logs/align/e2e               # every run as a simulation
$A e2e    logs/align/e2e_whatif --arch '{"region_host_ms": 1.0}' --runs w64_layer
```

| Mode | Input per run | Compares |
|---|---|---|
| `replay` | each step's per-group decode batch (KV at the step's decode context bucket) and prefill chunk, its start rebuilt from the group's FIFO of prompts; the header's knobs | step wall, the decode and prefill forwards (with their MTP passes), rank-0 device time in the profiled window; the summed time with and without first-use stalls |
| `e2e` | the run's requests at their engine arrival times (`request_rate: 1`), per-request MTP acceptance per draft position read off the ITLs, the header's knobs, one prefill chunk per DP group, two replicas split round-robin (the engine's rid parity) | duration, output tok/s, TTFT p50, TPOT p50 (the engine's nearest-rank percentile), and the W64 ranking by duration |

A first-use stall is a step at least 1.3x and 200 ms over its prediction: the
engine's first execution of a shape after warm-up (steps 0-2 of every run).
