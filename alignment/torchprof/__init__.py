"""Torch-profiler (Kineto chrome-trace) producer of the normalized alignment files.

A sibling of ``alignment/rocpd`` and ``alignment/nsys``. Where the rocpd producer
reads a rocprofv3 SQLite database, this one reads vLLM's NATIVE torch-profiler
output -- a per-worker PyTorch Kineto chrome trace (``*.pt.trace.json[.gz]``),
written by each TP/EP worker's own ``torch.profiler.profile`` session when the
engine is launched with ``--profiler-config.profiler=torch
--profiler-config.torch_profiler_dir=<dir>`` (the structured replacement, on the
vendored vLLM build, for the classic ``VLLM_TORCH_PROFILER_DIR`` env var).

Only the capture boundary differs. :func:`alignment.torchprof.kineto.read_kineto_trace`
turns a trace's GPU ``kernel`` events into the same :class:`RocpdDispatch` records
the rocpd reader yields and its ``vllm_iteration(N)`` user-annotation ranges (when
present) into :class:`RoctxRegion`; from there
:func:`alignment.rocpd.evidence.build_ranges_from_dispatches` runs the identical
sentinel/roctx iteration-segmentation and timestamp-containment attribution, and
the backend-neutral nsys assembly + shared writers emit the byte-identical
``parsed.json`` / ``parsed.kernels.parquet`` / ``kernel_sequences.json``. Nothing
is forked, so the Rust Check-1 reader consumes a torch trace unchanged.
"""
