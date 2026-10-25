---
name: operate-search-trn2-configs
description: Use when asked to find good serving configurations, a Pareto frontier, knob sensitivity or optimization headroom for Qwen3.8-Flash-Next BF16 on a trn2.48xlarge with the simulator (tools/trn2-config-search). CPU only; for a single preset run use operate-run-simulation instead.
---

# Search Trainium2 serving configs

The tool is `tools/trn2-config-search/` (its README has the stage-file grammar). The
current results and conclusions are `doc/pr/trn2-flashnext/CONFIG_SEARCH.md`. Read both
before planning a new stage.

## Rules

- **Never touch a Neuron device.** Export `SERVINGSTUDIO_NO_GPU=1`; the tool also passes
  `--no-gpu` to the launcher. If configs come back `needs_profiling`, record them as
  such (the kernels are named in `probe.jsonl`) and do not profile them.
- Keep concurrency at `--parallelism 48` or below; the host is shared with Neuron
  compiles.
- Run long stages in a tmux session named `search-<stage>`, and record the session,
  command and log in the worktree's `progress.md`.

## Steps

1. Environment, from the Sim worktree root:
   `source ../neuron_env.sh; export PATH=$HOME/.cargo/bin:$PATH SERVINGSTUDIO_NO_GPU=1`
   and `S="uv run --no-sync python tools/trn2-config-search/search.py"`.
2. Write or reuse a stage file under `tools/trn2-config-search/stages/`. Start coarse
   (a `grid` over layouts and the big switches), then refine with `seeds` from `top`
   and `oat` variations. Put what-if constants under `scenarios`.
3. `$S plan <stage.yaml> logs/trn2/search/<stage>`. Check the config count.
4. `$S probe logs/trn2/search/<stage> -v`. Only `ok` configs run. `invalid` is the
   simulator's refusal (for example the HBM check). `needs_profiling` lists missing kernels.
5. `$S run logs/trn2/search/<stage>`. Rerunning resumes; after a code change that
   moves costs, delete the stage's `runs/` (or plan into a new directory).
6. `$S collect logs/trn2/search/<stage>`. Pass `--slo name=TTFT_ms,TPOT_ms` for other
   SLOs; this does not re-simulate.
7. `$S report <exp> [<exp> ...] --out <file>` gives frontiers, best configs,
   sensitivity, what-if rank agreement and ladders.
8. Update `CONFIG_SEARCH.md` with the new numbers. Name the constants a conclusion
   depends on (the fitted host constants, the MoE prefill kernel, the all-to-all
   capacity model), and keep "today's engine knobs" apart from what-ifs the engine
   would have to build (`step_forwards`, `spmd_padding`, `moe_a2a_capacity_factor`).
   After an arch change that moves costs, check `tools/trn2-engine-align` first
   (`doc/pr/trn2-flashnext/ALIGNMENT.md`).

## Done when

Every planned config is `ok` and has a row in `results.jsonl` per workload, or is listed
as `invalid` / `needs_profiling`. The report and `CONFIG_SEARCH.md` state the best
config per workload and SLO, and say what moves under the what-if scenarios.
