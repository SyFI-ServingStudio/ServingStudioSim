"""Reduced-depth TP4 decode-forward runner (worker only): the decode leaves
``neuron_decoder_layer`` and ``neuron_decode_fixed``.

One ``torch.multiprocessing`` spawn per chunk, one process per LNC2 logical
core of the reserved chip (``collective_bench.NeuronMpLauncher``). Each rank
compiles ``nki_units.DecodeForward``: the engine's decode forward with
``depth`` layers. The graph holds the vocab-sharded embedding and its
all-reduce, the RoPE tables, the decoder layers with both block all-reduces,
the final norm, the lm_head shard and the engine's sampler. Each rank compiles
it at ``DEPTHS = (8, 32)`` and runs each depth ``ITERS`` times under the NRT
system profile. For each execution the row keeps the shortest of the four
ranks. The last rank to launch does not wait for the others, so its time is
the graph's own. With ``t(n) = fixed + n * layer``:

- ``neuron_decoder_layer`` = ``(t(32) - t(8)) / 24``, one decoder layer with
  its TP collectives inside a graph the compiler schedules as a whole;
- ``neuron_decode_fixed`` = ``t(8) - 8 * layer``, the per-step rest of the
  forward (embedding, RoPE, final norm, lm_head, sampling, the graph's own
  fixed cost).

The deep pair is deliberate: stacks of 1 and 4 layers under-read the layer
by 2-6% (the compiler schedules a short graph differently from the engine's
32-layer one), and the 32-layer forward alone is 2-5% below the engine's
measured decode step at every bucket from B1 to B128 (C512, kernel path;
doc/pr/llama31-8b-trn2/README.md, "Check 1").

Both kinds come from the same two measurements. The medians of one (path,
bucket, context) are written to
``<cache>/forward-runs/<tag>_<path>_b<B>_c<C>.json``, where ``<cache>`` is
the host's Neuron cache (``SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR``, shared by
every worker and kept across runs). The second kind and later batches read
them instead of measuring again. The tag is
``v<FORWARD_VERSION>_d<depths>_<digest>`` (``_measurement_tag``): the digest
covers the measuring code and the image's engine package versions, so a
changed image or runner measures again.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from pathlib import Path

from profiling.runners.metrics import ComputeMetrics, RunnerResult
from profiling.runners.neuron.collective_bench import NeuronMpLauncher, visible_logical_cores
from profiling.runners.neuron.stock_contract import HIDDEN, TP_SIZE, VOCAB_PER_RANK

DEPTHS = (8, 32)
KINDS = ("neuron_decoder_layer", "neuron_decode_fixed")
# Bump when the measured forward changes outside the files the tag digests (for
# example the constants of stock_contract.py), so a cached measurement is not
# reused: the cache outlives a run.
FORWARD_VERSION = "1"
_PROVENANCE: dict[str, str] = {}


def _key(spec):
    return json.dumps(spec, sort_keys=True, default=str)


def row_provenance(**spec):
    """The provenance note of a row this process measured, or None."""
    return _PROVENANCE.get(_key(spec))


def _cache_dir():
    root = Path(os.environ["SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR"]) / "forward-runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _per_rank(*, rank, world_size, shapes, path, root):
    """Inside each rank: every (bucket, context) of the chunk at both depths.
    Rank 0 returns {"bucket:ctx": {depth: median ms}}."""
    # own_core_only already set LOCAL_RANK=0; kept for direct callers.
    os.environ["LOCAL_RANK"] = "0"
    import gc

    import torch
    import torch.distributed as dist

    from profiling.runners.neuron import nki_bench, nki_units

    attention_cls = nki_units.AttnDecodeNki if path == "nki" else nki_units.AttnDecodeTorch
    # dist.barrier() is unusable on the neuron backend (no accelerator hooks);
    # a one-element all-reduce is the fence.
    fence = lambda: dist.all_reduce(torch.ones(1, device=nki_units.DEV))  # noqa: E731
    out = {}
    for bucket, ctx in shapes:
        x, _cos, _sin, table, pos_ids, slot = nki_units._decode_attention_args(
            attention_cls, bucket, ctx
        )
        ids = torch.randint(
            0,
            VOCAB_PER_RANK * world_size,
            (bucket,),
            generator=torch.Generator().manual_seed(bucket),
        )
        positions = torch.full((bucket,), ctx - 1, dtype=torch.int64)
        args = [ids.to(nki_units.DEV), positions.to(nki_units.DEV), table, pos_ids, slot]
        medians = {}
        for depth in DEPTHS:
            module = nki_units.DecodeForward(depth, attention_cls, path)
            fn = nki_units.compile_unit(module)
            nki_units.sync(fn(*args))
            for _ in range(nki_bench.WARM):
                nki_units.sync(fn(*args))
            fence()
            prof = Path(root) / f"fw_{path}_b{bucket}_c{ctx}_n{depth}_r{rank}"
            with nki_bench.runtime_profile(prof):
                for _ in range(nki_bench.ITERS):
                    nki_units.sync(fn(*args))
            mine = nki_bench.exec_times_ms(nki_bench.load_events(nki_bench.export_trace(prof)))
            if len(mine) != nki_bench.ITERS:
                raise ValueError(f"rank {rank}: {len(mine)} executions for {nki_bench.ITERS} calls")
            tag = Path(root) / f"times_{path}_b{bucket}_c{ctx}_n{depth}"
            Path(f"{tag}_r{rank}.json").write_text(json.dumps(mine))
            fence()
            if rank == 0:
                files = [Path(f"{tag}_r{r}.json") for r in range(world_size)]
                deadline = time.time() + 300
                while not all(f.exists() for f in files) and time.time() < deadline:
                    time.sleep(0.5)
                time.sleep(0.5)
                per_rank = [json.loads(f.read_text()) for f in files]
                medians[depth] = statistics.median(
                    min(times[i] for times in per_rank) for i in range(nki_bench.ITERS)
                )
            del fn, module
            torch._dynamo.reset()
            gc.collect()
        out[f"{bucket}:{ctx}"] = medians
    return out if rank == 0 else None


def _measurement_tag():
    """What a cached measurement must match to be reused: the runner version,
    the depths, a digest of the code that builds and times the forward, and the
    engine packages of this image. A change to any of them measures again."""
    import hashlib
    from importlib import metadata

    here = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("forward_bench.py", "nki_units.py", "nki_bench.py", "collective_bench.py"):
        digest.update((here / name).read_bytes())
    for package in ("vllm-neuron", "libtorch-neuronx-lite", "neuronx-cc", "torch"):
        try:
            digest.update(f"{package}={metadata.version(package)}".encode())
        except metadata.PackageNotFoundError:
            digest.update(f"{package}=absent".encode())
    depths = "-".join(str(d) for d in DEPTHS)
    return f"v{FORWARD_VERSION}_d{depths}_{digest.hexdigest()[:16]}"


def _read_cached(path):
    """The cached {depth: median ms} at ``path``, or None when it is absent or
    unreadable (a partial write or another layout): the shape is measured again."""
    try:
        medians = {int(k): float(v) for k, v in json.loads(path.read_text()).items()}
    except (OSError, ValueError, AttributeError):
        return None
    return medians if set(medians) == set(DEPTHS) else None


def _write_atomic(path, text):
    """Write through a temporary file and rename, so a reader never sees a
    partial file."""
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _estimate(kind, bucket, ctx):
    """(flops, bytes) per rank for the rate columns; estimates, not counters."""
    from profiling.runners.neuron import nki_bench

    if kind == "neuron_decoder_layer":
        return nki_bench.estimate_work("neuron_decoder_layer", "decode", bucket, ctx)
    return 2 * bucket * HIDDEN * VOCAB_PER_RANK, 2 * (HIDDEN * VOCAB_PER_RANK + 2 * bucket * HIDDEN)


def profile_batch(kind, kwargs_list, path="nki"):
    """One result per spec, in order, on attention ``path`` (``nki`` or
    ``torch``, the kind's backend). A bad shape is that row's error; a failed
    rank group is every measured row's error."""
    from profiling.runners.neuron import nki_bench

    if kind not in KINDS:
        raise ValueError(kind)
    shapes, results = [], [None] * len(kwargs_list)
    for i, spec in enumerate(kwargs_list):
        try:
            _, bucket, ctx = nki_bench.validate_shape(kind, spec)
        except ValueError as error:
            results[i] = RunnerResult(error=str(error))
            continue
        shapes.append((i, bucket, ctx))
    if not shapes:
        return results
    cache = _cache_dir()
    tag = _measurement_tag()
    todo = []
    measured = {}
    from_cache = set()
    for _, bucket, ctx in shapes:
        cached = _read_cached(cache / f"{tag}_{path}_b{bucket}_c{ctx}.json")
        if cached is not None:
            measured[(bucket, ctx)] = cached
            from_cache.add((bucket, ctx))
        elif (bucket, ctx) not in todo:
            todo.append((bucket, ctx))
    if todo:
        try:
            launcher = NeuronMpLauncher(TP_SIZE, visible_logical_cores(), own_core_only=True)
            root = cache / f"run-{uuid.uuid4().hex}"
            root.mkdir()
            got = launcher.run(_per_rank, shapes=todo, path=path, root=str(root))
        except Exception as error:  # the whole rank group failed
            for i, _, _ in shapes:
                results[i] = RunnerResult(error=f"{type(error).__name__}: {error}")
            return results
        for key, medians in got.items():
            bucket, ctx = (int(x) for x in key.split(":"))
            medians = {int(k): v for k, v in medians.items()}
            measured[(bucket, ctx)] = medians
            _write_atomic(cache / f"{tag}_{path}_b{bucket}_c{ctx}.json", json.dumps(medians))
    lo, hi = DEPTHS
    for i, bucket, ctx in shapes:
        t = measured[(bucket, ctx)]
        layer = (t[hi] - t[lo]) / (hi - lo)
        fixed = t[lo] - lo * layer
        time_ms = layer if kind == "neuron_decoder_layer" else fixed
        if time_ms <= 0:
            results[i] = RunnerResult(error=f"non-positive {kind} time {time_ms:.4f} ms from {t}")
            continue
        flops, moved = _estimate(kind, bucket, ctx)
        results[i] = RunnerResult(
            metrics=ComputeMetrics(
                time_ms=time_ms,
                tflops=flops / (time_ms * 1e9),
                memory_bandwidth_gbps=moved / (time_ms * 1e6),
            )
        )
        _PROVENANCE[_key(kwargs_list[i])] = (
            f"forward_bench depth{lo}_ms={t[lo]:.4f} depth{hi}_ms={t[hi]:.4f} "
            f"layer_ms={layer:.4f} fixed_ms={fixed:.4f} min over {TP_SIZE} ranks, median of "
            f"{_iters()} executions; measurement={tag}"
            + (" (reused from this worker's cache)" if (bucket, ctx) in from_cache else "")
        )
    return results


def _iters():
    from profiling.runners.neuron import nki_bench

    return nki_bench.ITERS


def profile_neuron_decoder_layer_batch(kwargs_list):
    """One result per spec for kind ``neuron_decoder_layer``; see ``profile_batch``."""
    return profile_batch("neuron_decoder_layer", kwargs_list)


def profile_neuron_decode_fixed_batch(kwargs_list):
    """One result per spec for kind ``neuron_decode_fixed``; see ``profile_batch``."""
    return profile_batch("neuron_decode_fixed", kwargs_list)


def profile_neuron_decoder_layer_torch_batch(kwargs_list):
    """``neuron_decoder_layer`` on the stock call's torch fallback attention."""
    return profile_batch("neuron_decoder_layer", kwargs_list, path="torch")


def profile_neuron_decode_fixed_torch_batch(kwargs_list):
    """``neuron_decode_fixed`` on the stock call's torch fallback attention."""
    return profile_batch("neuron_decode_fixed", kwargs_list, path="torch")
