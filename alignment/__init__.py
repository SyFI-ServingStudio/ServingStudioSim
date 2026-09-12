"""Double-sided evidence exchange between ServingStudioSim and real frameworks.

Framework-to-simulator analysis validates L1–L4 predictions with real serving
profiles. Simulator-to-framework analysis attributes real implementation probes
against a ServingStudioSim target. Both consume the same normalized Nsight evidence; only
their comparison policy differs.

The design record is `doc/`; the practical reference is `alignment/README.md`.

Launching enters through the explicit `python -m launcher alignment` stage
subcommands; this package's CLI is
artifact inspection only. The
heavy runtime dependencies never enter this package's interpreter: engine drivers
shell out to their fork environments. req-frontend under `load_generator/` owns
typed server load and client-side measurements. Alignment-local code stops at
artifact normalization; Rust computes cross-source statistics and Analyzer's
Python half renders payloads.
"""
