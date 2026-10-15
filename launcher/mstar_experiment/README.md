# `mstar-experiment` launcher hook (POC)

Delegates to an external lab checkout via **`MSTAR_EXPERIMENT_LAB`** (default `~/mstar-sim-tier-a-qwen3tts`). No M\* or vibesim sim logic in this module.

```bash
export MSTAR_EXPERIMENT_LAB=/path/to/mstar-sim-tier-a-qwen3tts
cd /path/to/servingstudiosim
uv run python -m launcher mstar-experiment --help
```

Subcommands: `run-experiment`, `run-campaign`, `evaluate` — see lab `docs/shared-experiment-layer.md`.

**Evidence:** https://github.com/sheetal104/mstar/blob/fix/simulate-cli-json-asdict/docs/sim/qwen3-tts-tier-a-results.md
