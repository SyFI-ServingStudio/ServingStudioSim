# DeepSeek-V4-Flash idle KV / GPU-cache experiments

Replay of last week's **keep / always-handoff / JIT / migrate-on-idle** cells,
with two changes required for this week:

1. **Model:** DeepSeek-V4-Flash (0731), not Qwen2.5-Coder-7B.
2. **Text:** frozen public **real code + prose**, tokenized to IDs. No `"alpha "` filler.

This package lives **in the VibeSim tree** so it can be opened as a PR/branch on
the VibeSim repo. It does **not** vendor VibeServe or a personal experiment dump.

## What VibeSim can and cannot do today

The discrete-event engine on **master** runs DeepSeek V4 via L4 arch
`deepseek_v4_vllm`, but session lifecycle columns (`tool_wait_after_ms`,
`preserved_prefix_kv`, …) are still not wired for exact keep/handoff/JIT/migrate
in the Rust engine.

So:

| Layer | This package |
|---|---|
| Session-level cache policy sim | **Runnable now** (`sim_cache_policy.py`) using V4 KV bytes/token (584 B MLA token from vLLM's V4 cache spec) |
| DES timing (8×H200, 2 replicas) | **`presets/dsv4_agent/`** — keep≈opportunistic prefix cache, handoff≈disabled |
| Full multi-turn migrate/JIT in Rust | **Not here** — see `feat/agent-workload-fidelity` |
| Real GPU harness | [tracelab-vllm-experiments](https://github.com/sheetal104/tracelab-vllm-experiments) `docs/runbooks/dsv4-8xh200.md` |

## Hardware

Verified vLLM recipe for V4-Flash-0731 on H200: **4×H200 TP4+EP per server**;
**8×H200** = two replicas. Policy sim defaults to **141 GB HBM** (`HBM_GB=141`).

## Reproduce (policy sim, no GPU)

```bash
cd experiments/dsv4_idle_kv
python3 gen_realtext_workload.py          # writes workloads/*.jsonl
python3 sim_cache_policy.py --cell all    # keep / handoff / jit / migrate
```

Reports go to `results/` as markdown tables (no bulky JSON dumps).

## Reproduce (DES launcher, no GPU)

```bash
cd experiments/dsv4_idle_kv
./make_traces.sh                          # workloads/*.jsonl → trace/dsv4_*.csv

cd ../..
uv run python -m launcher presets/dsv4_agent/idle24_keep.yaml --dry-run
for p in idle24_keep idle24_handoff mem64_keep fork_scatter; do
  uv run python -m launcher "presets/dsv4_agent/${p}.yaml"
done
```

## Mapping to last week's cells

| Last week (Qwen 7B, alpha filler) | This package |
|---|---|
| keep / always-handoff / JIT @ 24×8, mem 0.35 | `--cell idle_kv` |
| wait-sweep migrate 1s/4s/8s/30s, skew 18/6 | `--cell wait_sweep` |
| mem-bound migrate 64 sess, mem 0.25 | `--cell membound` |
| Nixl warm transfer | not run (failed last week; still experimental) |
| fork binpack | `--cell fork` (shared real-text stems) |
