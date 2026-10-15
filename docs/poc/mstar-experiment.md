# M\* experiment manifest POC

**Code:** `launcher/mstar_experiment/` + registration in `launcher/cli.py` (~90 lines).

**Purpose:** One operator entrypoint (`python -m launcher mstar-experiment …`) for shared manifests while keeping vibesim and M\*-Sim decoupled.

**What to review in the PR**

| Area | Change |
|------|--------|
| CLI | New subcommand; delegates to `MSTAR_EXPERIMENT_LAB` |
| Runtime | Lab scripts call existing M\* simulate/validate or SSS launcher dry-run |
| Evidence | External lab doc: Qwen3-TTS gates (V2 pass, V4 concordant), SSS smoke acceptance pass |

**What is out of scope for this PR:** Qwen3-TTS arch in vibesim, merged DES, cross-backend throughput comparison.

**Lab:** https://github.com/sheetal104/mstar-sim-tier-a-qwen3tts — `docs/qwen3-tts-tier-a-results.md`, `docs/shared-experiment-layer.md`. Set `MSTAR_EXPERIMENT_LAB` to a clone to reproduce acceptance on captured artifacts.
