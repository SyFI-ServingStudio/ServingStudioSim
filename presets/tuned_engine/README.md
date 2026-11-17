# Tuned engine presets

The rest of `presets/` is for exploring configurations: parallelism, backends, schedulers, KV and tier settings.
This directory holds a few engine configurations chosen on purpose because they are competitive, each the
simulator's replica of a real engine. They are meant as performance baselines, for example the engine a paper
compares against.

One directory per model checkpoint; in it, one preset per deployment, named `<gpu>_<parallelism>_<role>`:

- `<name>.yaml`: the preset;
- `<name>.md`: where each key comes from, the ablations, the comparison with the real engine, and the scope;
- `<name>.backends.yaml`: its kernel backends (`backends_file`), when it sets any.

## What a preset needs to be here

1. **A real implementation behind it.** The preset replicates an engine that exists and was run. Every key in it
   names its source in that engine (file, config value or design note), and the `.md` has a table comparing the
   simulator with the real engine's measured runs: the engine commit, the trace, and the deviation per setting.
2. **A stated scope.** The `.md` names the model, the hardware, and the workload type the comparison covers, and the
   settings where the simulation is not trusted, with the measured gap that shows it.
3. **Built by the tests.** `just test-presets` builds every member through the binary
   (`tests/test_glm53_flash_prefill_presets.py` covers this directory).
4. **Ablations as keys, not files.** An ablation is the preset with named keys changed; the `.md` lists which key
   each ablation changes and to what. No copy of the preset is stored per ablation.

## Presets

| Preset | Engine | What it runs |
| --- | --- | --- |
| `GLM-5.3-Flash-NVFP4/b200_pp4_prefill.yaml` | SSEngine (pp-impl `lanes/release`; measured engine a78dc6f) | GLM-5.3-Flash NVFP4 prefill on one B200 PP4 pipeline, coding-agent sessions (TraceLab v3 M6), N = 300, HBM 100.18 + DRAM 50 + SSD 128 GB per GPU |

The bring-up and study presets for the same deployment (single requests, saturation, scheduler and tier checks,
concurrency and tier sweeps) stay in `presets/glm53_flash_prefill/` and share this preset's backends file.
