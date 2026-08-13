"""VibeSim alignment — validate the L1–L4 cost model against real vLLM.

The simulator predicts how long each kernel/op takes (`profiling` measures L1
kernels; L2–L4 compose them). This package closes the loop the other way: it
drives a real **vLLM** server on real hardware under **Nsight Systems (nsys)**,
parses the trace into per-op GPU kernel busy time, and exposes a typed builder
for VibeSim `timing-predict` cases. The repository-level analyzer then
produces iteration and end-to-end report/payload pairs.

First milestone: **Llama3-8B dense, single GPU, no parallel** (see
`../README.md`). The design record is `doc/` (this package has no design-doc
layer of its own yet); the practical reference is `alignment/README.md`.

Launching enters through the explicit `python -m launcher alignment` stage
subcommands; this package's CLI is
artifact inspection only. The
heavy runtime deps (vLLM / torch / CUDA) never enter this package's interpreter:
`profiler/vllm_server.py` shells out to the fork venv, mirroring how `profiling`
never imports vLLM. req-frontend under `load_generator/` owns typed trace frontends
and replay. Alignment-local code stops at artifact normalization; Rust computes
statistics and the analyzer's Python half renders payload JSON.
"""
