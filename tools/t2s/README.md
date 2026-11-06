# Text2SQL RL rollouts: measured vs simulated, and B vs oracle

The slime Text2SQL runs (Qwen2.5-Coder-7B-Instruct, GRPO 256 prompts x 5,
multi-turn with SQL tool calls, 8x H200, SGLang 0.5.5.post1) replayed in the
simulator: the colocated arm for inference alignment, the streaming
`bt_tuner_B8_step8` arm for the fixed-B streaming path, then a fixed-B grid and
the slime tuners replayed against its per-rollout oracle. Findings live in the
worktree's `progress.md`; this directory is the plumbing.

| | |
| --- | --- |
| `extract_arm.py` | **runs where the arms are** (stdlib, python3.9): one arm's report / tuner / chunk / span / migration records → `data/arms/<arm>/` |
| `extract_engine_iters.py` | **runs where the arms are**: every SGLang scheduler record of one arm (`sglang_metrics`) → `data/<arm>_iters.csv.gz` (not tracked, 13-17 MB) |
| `build_t2s_trace.py` | the handoff's per-turn export → one session trace per rollout, `trace/t2s/<arm>/r{N}.csv`, plus a tracked `manifest.json` pinning the input's sha256 and per-rollout totals. Its docstring explains every column and why the measured cache hit is pinned. |
| `align_colocated.py` | colocated arm: per-rollout makespan, decode iteration time by batch bucket (measured from consecutive scheduler timestamps), engine-side latency p50/p90/p99 with the t=0 burst apart |
| `align_streaming.py` | streaming arm: rollout r of `bt_tuner_B8_step8` ran at the tuner's `b_before`, so it is compared with grid cell `r{r}_{B_r}` |
| `t2s_cells.py` | reading one simulated cell: training window, inference end, slime's interior-idle observation, per-engine queue peaks |
| `plot_t2s_vs_oracle.py` | fixed B and the slime tuners (CUBIC, idle ±16, KV cap + ±16, from `../slime-b-sweep/replay_cubic.py`) relative to the per-rollout oracle |

## Reproducing

```bash
H=<handoff dir>   # text2sql35_colocated_handoff: per_turn.csv.gz, per_turn_streaming.csv.gz
uv run python tools/t2s/build_t2s_trace.py $H/per_turn.csv.gz trace/t2s/colocated \
    --arm colocate_idle_measurement
uv run python tools/t2s/build_t2s_trace.py $H/per_turn_streaming.csv.gz trace/t2s/streaming \
    --arm bt_tuner_B8_step8
# each manifest.json must come out unchanged (git diff trace/t2s)

# measured per-iteration records, once (on the host holding the arms)
python3 extract_engine_iters.py <arms>/..._colocate_idle_measurement | gzip > coloc_iters.csv.gz

uv run python -m launcher presets/t2s_colocated.yaml
uv run --with pandas python tools/t2s/align_colocated.py logs/t2s/runs/colocated \
    --per-turn $H/per_turn.csv.gz --trace-dir trace/t2s/colocated

uv run python -m launcher presets/t2s_streaming_grid.yaml      # 35 x 32 cells
uv run --with pandas python tools/t2s/align_streaming.py logs/t2s/runs/streaming_grid
uv run python tools/t2s/plot_t2s_vs_oracle.py logs/t2s/runs/streaming_grid --out t2s_b_vs_oracle.png
```

The oracle and the tuner replay read each cell's `raw/cost_log`, so keep it until
the figure is rendered.
