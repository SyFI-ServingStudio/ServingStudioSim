<p align="center">
  <img src="doc/assets/vibesim-logo.svg" alt="VibeSim logo" width="64">
</p>

<h1 align="center">VibeSim</h1>

<p align="center">
  <strong>A fast LLM serving simulator grounded in real GPU kernel measurements.</strong>
</p>

<p align="center">
  <a href="#key-features">Features</a> ·
  <a href="#repository-map">Repository map</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="doc/README.md">Documentation</a>
</p>

---

VibeSim is a discrete-event simulator and performance-modeling toolkit for LLM
serving systems. How much traffic can a deployment sustain? Which parallel layout
works best? Where is GPU time being spent? VibeSim helps answer these questions
by composing **measured GPU kernel timings** into predictions of serving
throughput and latency.

Define an experiment in YAML, replay a workload, and trace the results from the
whole deployment down to individual requests and kernels. Explore configurations
before provisioning a cluster, then use the same analysis to guide improvements
in a real serving framework.

For the complete browser + Agent + Analyzer deployment, use the
[`VibeSimWorkspace`](https://github.com/SyFI-VibeSim/VibeSimWorkspace)
meta-repository.

## 📣 News

- **September 2026:** VibeSim is now available!

---

<a id="key-features"></a>

## ✨ Key features

- 🧩 **Flexible configuration.** Explore dense and MoE models across GPU layouts,
  precisions, and serving strategies, including unified serving, prefill/decode
  disaggregation, attention/FFN disaggregation, and speculative decoding.
  YAML sweeps make it easy to compare configurations. See the
  [architecture compatibility matrix](doc/architecture_compatibility.md) for
  supported combinations.
- ⚡ **Fast simulation.** A Rust event engine and reusable, compiled cost trees
  evaluate changing batch shapes efficiently, making long workloads and broad
  configuration sweeps practical.
- 🎯 **Accurate predictions.** Kernel costs come from measurements on real GPUs.
  Alignment workflows compare predictions with vLLM and SGLang at kernel,
  iteration, and end-to-end levels for supported execution graphs.
- 🔎 **Full observability.** Inspect throughput and latency alongside request
  lifecycles, batch composition, KV occupancy, and per-kernel costs. Analyzer
  produces structured reports, plots, and Perfetto timelines from run artifacts.
- 📊 **Optimization insights.** Attribute simulated GPU time to idle time, load
  imbalance, batching, communication, and kernel efficiency. Independent
  model-work bounds help explain the remaining gap to theoretical performance.

---

<a id="repository-map"></a>

## 🗂️ Repository map

```text
VibeSim/
├── simulator/    Rust simulation engine, model execution, and scheduling
├── profiling/    GPU kernel measurements and reusable timing database
├── launcher/     Experiment CLI, parameter sweeps, and timing predictions
├── analyzer/     Performance reports, plots, and trace exports
├── alignment/    Validation against vLLM and SGLang measurements
├── model/        Model configurations and independent FLOP/byte bounds
├── gpu/          GPU hardware specifications
├── trace/        Workload generators and sample traces
├── presets/      Simulation, prediction, and alignment configurations
├── skills/       Agent workflows for experiments and development
├── tests/        Simulation, profiling, and integration tests
└── doc/          Architecture, design, and compatibility documentation
```

See the [architecture guide](doc/architecture.md) for how the components fit
together, or the [documentation index](doc/README.md) for detailed references.

---

<a id="quick-start"></a>

## 🚀 Quick start

### Requirements

- **Linux** and **Git**, including submodule support.
- **Rust stable and Cargo** to build the simulator and Analyzer.
- **Python 3.12**, managed by **uv**, for the launcher and kernel profiler.
- **just** to run the repository's dependency setup and test commands.
- **A C/C++ build toolchain** for native dependencies and linking, plus **mold**
  on Linux x86-64, as required by [the Cargo configuration](.cargo/config.toml).
- Enough writable disk space for dependencies, build artifacts, and run logs.
- A compatible NVIDIA GPU and CUDA environment when profiling missing kernel
  measurements. The simulation itself runs on the CPU using cached costs.

The repository includes `profiling/profile.db`. Cache coverage depends on the
selected GPU, kernel backend, and input shapes; filling missing measurements
requires the corresponding hardware. Model configuration files are included;
the simulation does not load model weights.

### 1. Build

> [!TIP]
> **Recommended: set up through [VibeSimWorkspace](https://github.com/SyFI-VibeSim/VibeSimWorkspace).**
> It pins compatible versions of the simulator, Agent, and browser UI and provides
> shared setup and build commands. Follow its
> [setup guide](https://github.com/SyFI-VibeSim/VibeSimWorkspace/blob/main/reproduce.md)
> for the complete environment.

For a standalone simulator checkout, clone this repository and initialize its
pinned submodules:

```bash
git clone --recurse-submodules https://github.com/SyFI-VibeSim/VibeSim.git
cd VibeSim

just sync
uv run cargo build --release --workspace
```

`just sync` installs the Python environment in the required dependency order.
Run Python and Cargo commands through `uv run` from the repository root so the
simulator's embedded Python uses that environment.

### 2. Run Llama 3 8B

The included [smoke preset](presets/unified_smoke.yaml) models Llama 3 8B in BF16
on one NVIDIA H200. It replays three small requests from
[`trace/smoke.csv`](trace/smoke.csv) and runs until all requests finish.

```bash
# Validate the preset and inspect the expanded configuration.
uv run python -m launcher presets/unified_smoke.yaml --dry-run

# Run the simulation and generate analysis.
uv run python -m launcher presets/unified_smoke.yaml
```

### 3. Explore the results

The launcher records the experiment inputs and runs the simulator and Analyzer.
The example should complete all **three requests**. Its output lives in
`logs/unified_smoke/`:

| Output | Contents |
| --- | --- |
| `summary.json` | Run completion and simulation summary |
| `raw/` | Request records, kernel costs, and resolved configuration |
| `reports/` | Analysis summaries, including throughput and latency |
| `payloads/` | Structured data behind the analysis |
| `plots/` | Rendered figures |

To explore additional settings and experiments:

```bash
uv run python -m launcher list-params --human
uv run python -m launcher --help
```

See [`launcher/README.md`](launcher/README.md) for sweeps and timing predictions,
[`profiling/README.md`](profiling/README.md) for kernel measurements, and
[`analyzer/README.md`](analyzer/README.md) for interpreting results. Run
`just test-cpu` for the CPU test suite.

---

## 📄 Licensing

This project is source-available under a dual community licensing
model.

You may use it under whichever of the following licenses applies to
your use:

- **PolyForm Noncommercial License 1.0.0** — for noncommercial,
  research, educational, and other uses permitted by that license.
- **PolyForm Internal Use License 1.0.0** — for internal business use,
  including internal use and modification by commercial organizations.

The two are alternatives; you do not need to satisfy both.

Uses not permitted by either community license, including external
commercial productization and redistribution, require a separate
commercial license.

See [`LICENSING.md`](LICENSING.md) for details and
[`COMMERCIAL-LICENSING.md`](COMMERCIAL-LICENSING.md) for commercial licensing
information. Third-party components remain under their own licenses — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Contributors keep their
copyright and grant the rights in [`CLA.md`](CLA.md); see
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.
