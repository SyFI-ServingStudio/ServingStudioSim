"""DeepGEMM ``fp8_fp4_mega_moe`` runner (kind ``mega_moe``, backend ``deepgemm_sm100``).

Mirrors DeepInfer's production call (``v4_modules/mlp.py`` ``mega_moe_forward``):
MXFP8 tokens and their top-k slots already sit in the symmetric buffer, the
shared expert is fused (``shared_l1_weights``/``shared_l2_weights``), the
EPLB expert counter is passed, SwiGLU clamp 10, ``fast_math=True``. Inputs are
built as DeepGEMM's own ``tests/test_mega_moe.py`` builds them, except the top-k
slot ids, which realize the spec's ``per_slot_rows`` exactly
(:func:`slot_topk`).

The rank group is spawned once per chunk. Specs with the same geometry share
one symmetric buffer and one set of weights; the buffer's capacity
(``num_max_tokens_per_rank``) is the largest token count of the group unless a
caller pins it (the A/B that showed it does not change the time).

Timing. The kernel synchronizes the EP group, so a rank that launches late
makes every other rank's kernel look longer. Each timed launch therefore
follows a GPU sleep and a one-element NCCL all-reduce (as DeepGEMM's
``bench_kineto(barrier=...)`` does): the host has queued the kernel long before
the all-reduce releases it, so all ranks start within the all-reduce's skew.
Every rank records its own CUPTI kernel time per launch; the row keeps the
median over launches of the per-launch maximum over ranks, a robust statistic a
single straggler launch cannot move.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from profiling.db.args import DType
from profiling.runners.comm._batch import all_error
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics, RunnerResult

KERNEL_NAME = "mega_moe"
MMA_TYPE = "fp8xfp4"
# DeepSeek-V4's swiglu_limit; a template constant of the kernel, it only clips values.
ACTIVATION_CLAMP = 10.0
_WARMUP = 3
_ITERS = 20
# GPU-side sleep before each timed launch (~5 ms on B200): longer than the host
# needs to queue the all-reduce and the kernel behind it.
_SLEEP_CYCLES = 10_000_000
# Tokens per rank for the one correctness check per geometry; the Torch
# reference is a per-expert loop over every rank's tokens.
_CHECK_TOKENS = 256
# calc_diff (1 - cosine-like similarity) bound between kernel and reference. The
# kernel's bf16 rounding inside SwiGLU leaves ~1e-4; a routing or layout error is ~1.
_CHECK_MAX_DIFF = 2e-3
_SEED = 20261115


def profile_mega_moe_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return run_batch(kwargs_list)


def run_batch(kwargs_list: list[dict], capacities: list[int] | None = None) -> list[RunnerResult]:
    """Profile every spec in one rank group. ``capacities`` pins each spec's
    symmetric-buffer capacity (diagnostics only); by default a geometry group
    uses its largest token count."""
    if not kwargs_list:
        return []
    try:
        for spec in kwargs_list:
            _validate(**spec)
        ep_sizes = {int(spec["ep_size"]) for spec in kwargs_list}
        if len(ep_sizes) != 1:
            raise ProfilerNotImplemented("all batch specs must use the same ep_size")
        if capacities is not None and len(capacities) != len(kwargs_list):
            raise ValueError("capacities must match kwargs_list")
    except Exception as exc:  # noqa: BLE001 - one bad spec fails the lockstep group
        return all_error(len(kwargs_list), str(exc))
    try:
        payloads = TorchMpLauncher(ep_sizes.pop()).run(
            _per_rank_batch, specs=kwargs_list, capacities=capacities
        )
    except Exception as exc:  # noqa: BLE001 - the rank group tore down; fail the chunk
        return all_error(len(kwargs_list), str(exc))
    if payloads is None or len(payloads) != len(kwargs_list):
        return all_error(len(kwargs_list), "mega_moe rank group returned no payloads")
    return [
        RunnerResult(
            metrics=ComputeMetrics(
                time_ms=p["time_ms"],
                tflops=p["tflops"],
                memory_bandwidth_gbps=p["memory_bandwidth_gbps"],
                energy_j=p["energy_j"],
            )
        )
        for p in payloads
    ]


def _validate(
    ep_size: int,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    activation_dtype: DType | str,
    weight_dtype: DType | str,
    per_slot_rows: list[int] | tuple[int, ...],
) -> None:
    if DType.from_value(activation_dtype) is not DType.MXFP8_E4M3:
        raise ProfilerNotImplemented("deepgemm_sm100 mega_moe needs activation_dtype=mxfp8_e4m3")
    if DType.from_value(weight_dtype) is not DType.MXFP4_E2M1:
        raise ProfilerNotImplemented("deepgemm_sm100 mega_moe needs weight_dtype=mxfp4_e2m1")
    # ep_size 1 is DeepGEMM's single-process mode (tests/test_mega_moe.py
    # --num-processes 1): every expert local, no NVLink traffic.
    if ep_size < 1:
        raise ValueError(f"mega_moe needs ep_size >= 1, got {ep_size}")
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if num_experts % ep_size:
        raise ValueError(f"num_experts={num_experts} must divide by ep_size={ep_size}")
    # deep_gemm get_symm_buffer_size_for_mega_moe: SF layouts need multiples of 128.
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    if num_shared_experts < 0 or not 1 <= top_k <= num_experts:
        raise ValueError("bad num_shared_experts or top_k")
    rows = [int(r) for r in per_slot_rows]
    if len(rows) != num_experts:
        raise ValueError(f"per_slot_rows has {len(rows)} entries, num_experts={num_experts}")
    group_tokens = ep_size * num_tokens
    if min(rows) < 0 or sum(rows) != group_tokens * top_k:
        raise ValueError("per_slot_rows must be non-negative and sum to ep_size*num_tokens*top_k")
    if max(rows) > group_tokens:
        raise ValueError("a slot cannot receive more rows than the group has tokens")


def slot_topk(per_slot_rows: list[int], ep_size: int, num_tokens: int, top_k: int) -> Any:
    """``[ep_size, num_tokens, top_k]`` distinct slot ids whose histogram is
    exactly ``per_slot_rows``.

    The ``ep_size * num_tokens * top_k`` (token, k) pairs are filled column by
    column (pair ``j`` is token ``j % N``, choice ``j // N``) with every slot
    repeated by its row count. A slot never covers more than ``N`` consecutive
    pairs, so a token never meets the same slot twice, whatever order the slots
    come in. The slot order and the group tokens are shuffled (fixed seed), so
    a slot's rows come from scattered tokens of every source rank and a
    token's slots are unrelated, as with a real router; filling in slot order
    instead gives each slot one contiguous run of tokens.
    """
    import numpy as np

    rng = np.random.default_rng(_SEED)
    n = ep_size * num_tokens
    order = rng.permutation(len(per_slot_rows))
    flat = np.repeat(order.astype(np.int64), np.asarray(per_slot_rows, dtype=np.int64)[order])
    pairs = flat.reshape(top_k, n).T[rng.permutation(n)]  # [group token, k]
    return pairs.reshape(ep_size, num_tokens, top_k)


def _geometry(spec: dict) -> tuple:
    return (
        int(spec["hidden_size"]),
        int(spec["intermediate_size"]),
        int(spec["num_experts"]),
        int(spec["num_shared_experts"]),
        int(spec["top_k"]),
    )


def _per_rank_batch(
    *, rank: int, world_size: int, specs: list[dict], capacities: list[int] | None
) -> list[dict] | None:
    import torch

    groups: OrderedDict[tuple, list[int]] = OrderedDict()
    for index, spec in enumerate(specs):
        groups.setdefault(_geometry(spec), []).append(index)

    payloads: list[dict | None] = [None] * len(specs)
    for geometry, indices in groups.items():
        if capacities is None:
            buckets = {max(int(specs[i]["num_tokens"]) for i in indices): indices}
        else:
            buckets: dict[int, list[int]] = {}
            for i in indices:
                buckets.setdefault(int(capacities[i]), []).append(i)
        layer = _Layer(rank, world_size, *geometry)
        try:
            layer.check(min(_CHECK_TOKENS, min(int(specs[i]["num_tokens"]) for i in indices)))
            for capacity, members in buckets.items():
                layer.allocate(capacity)
                for i in members:
                    payloads[i] = layer.measure(
                        int(specs[i]["num_tokens"]), specs[i]["per_slot_rows"]
                    )
        finally:
            layer.destroy()
        torch.cuda.empty_cache()
    return payloads if rank == 0 else None


class _Layer:
    """One rank's weights and symmetric buffer for one MoE geometry."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        hidden: int,
        intermediate: int,
        num_experts: int,
        num_shared: int,
        top_k: int,
    ) -> None:
        import torch

        self.torch = torch
        self.rank = rank
        self.world_size = world_size
        self.hidden = hidden
        self.intermediate = intermediate
        self.num_experts = num_experts
        self.num_shared = num_shared
        self.top_k = top_k
        self.local_experts = num_experts // world_size
        self.buffer = None
        torch.manual_seed(_SEED + rank)
        self._make_weights()

    # ---- weights -------------------------------------------------------
    def _make_weights(self) -> None:
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp4

        torch = self.torch
        hidden, inter, local = self.hidden, self.intermediate, self.local_experts

        def fp4_stack(n: int, k: int) -> tuple[Any, Any, Any]:
            # randn / sqrt(k) keeps GEMM outputs O(1), so the clamp seldom binds.
            w = torch.empty((local, n, k // 2), device="cuda", dtype=torch.int8)
            sf = torch.empty((local, n, k // 32), device="cuda", dtype=torch.float)
            for e in range(local):
                bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / k**0.5
                w[e], sf[e] = per_token_cast_to_fp4(bf16, use_ue8m0=True, gran_k=32)
            packed_sf = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), local)
            return w, sf, packed_sf

        l1, l1_sf, l1_packed = fp4_stack(2 * inter, hidden)
        l2, l2_sf, l2_packed = fp4_stack(hidden, inter)
        # Kept for the reference check only, in checkpoint ([gate; up]) order.
        self._raw_routed = (l1, l1_sf, l2, l2_sf)
        self.l1, self.l2 = deep_gemm.transform_weights_for_mega_moe(
            (l1, l1_packed), (l2, l2_packed)
        )
        self.shared_kwargs: dict[str, Any] = {}
        self._raw_shared = None
        if self.num_shared:
            shared_inter = inter * self.num_shared
            s1 = torch.randn((2 * shared_inter, hidden), device="cuda", dtype=torch.bfloat16)
            s2 = torch.randn((hidden, shared_inter), device="cuda", dtype=torch.bfloat16)
            s1_fp8, s1_sf, s1_tma = _cast_fp8(torch, s1 / hidden**0.5)
            s2_fp8, s2_sf, s2_tma = _cast_fp8(torch, s2 / shared_inter**0.5)
            self._raw_shared = (s1_fp8, s1_sf, s2_fp8, s2_sf)
            t1, t2 = deep_gemm.transform_weights_for_mega_moe((s1_fp8, s1_tma), (s2_fp8, s2_tma))
            self.shared_kwargs = {"shared_l1_weights": t1, "shared_l2_weights": t2}
        self.recv_stats = torch.zeros((local,), device="cuda", dtype=torch.int)

    # ---- buffer and inputs ---------------------------------------------
    def allocate(self, capacity: int) -> None:
        import deep_gemm
        import torch.distributed as dist

        self._free_buffer()
        self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            self.num_experts,
            capacity,
            self.top_k,
            self.hidden,
            self.intermediate,
            num_shared_experts=self.num_shared,
            mma_type=MMA_TYPE,
        )

    def _inputs(self, num_tokens: int, per_slot_rows: Any = None) -> dict[str, Any]:
        torch = self.torch
        generator = torch.Generator(device="cuda").manual_seed(_SEED * 7 + self.rank)
        x = torch.randn(
            (num_tokens, self.hidden), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        x_fp8, x_sf, _ = _cast_fp8(torch, x)
        if per_slot_rows is None:
            # Uniform distinct slots (the correctness check only).
            topk_idx = (
                torch.rand((num_tokens, self.num_experts), device="cuda", generator=generator)
                .topk(self.top_k, dim=1)
                .indices
            )
        else:
            ids = slot_topk(
                [int(r) for r in per_slot_rows], self.world_size, num_tokens, self.top_k
            )[self.rank]
            topk_idx = torch.from_numpy(ids.copy()).to(device="cuda", dtype=torch.int64)
        topk_weights = torch.rand(
            (num_tokens, self.top_k), device="cuda", dtype=torch.float, generator=generator
        )
        return {"x_fp8": x_fp8, "x_sf": x_sf, "topk_idx": topk_idx, "topk_weights": topk_weights}

    def _write_inputs(self, inputs: dict[str, Any], num_tokens: int, block_m: int) -> None:
        buffer = self.buffer
        buffer.x[:num_tokens].copy_(inputs["x_fp8"])
        buffer.x_sf[:num_tokens].copy_(inputs["x_sf"])
        buffer.topk_idx[:num_tokens].copy_(inputs["topk_idx"])
        buffer.topk_weights[:num_tokens].copy_(inputs["topk_weights"])
        if self.num_shared:
            _write_shared_sf(self.torch, buffer.shared_l1_acts_sf, inputs["x_sf"], block_m)

    def _block_m(self, num_tokens: int) -> int:
        import deep_gemm

        return deep_gemm.get_block_m_for_mega_moe(
            self.world_size,
            self.num_experts,
            self.buffer.num_max_tokens_per_rank,
            num_tokens,
            self.top_k,
            MMA_TYPE,
        )

    def _call(self, y: Any) -> None:
        import deep_gemm

        deep_gemm.fp8_fp4_mega_moe(
            y,
            self.l1,
            self.l2,
            self.buffer,
            **self.shared_kwargs,
            cumulative_local_expert_recv_stats=self.recv_stats,
            activation_clamp=ACTIVATION_CLAMP,
            fast_math=True,
        )

    # ---- correctness ---------------------------------------------------
    def check(self, num_tokens: int) -> None:
        """Compare one call against the Torch reference (outside any timing)."""
        import torch.distributed as dist
        from deep_gemm.testing import calc_diff
        from deep_gemm.utils import cast_back_from_fp4, cast_back_from_fp8

        from profiling.runners.moe import mega_moe_reference as ref

        torch = self.torch
        self.allocate(num_tokens)
        inputs = self._inputs(num_tokens)
        self._write_inputs(inputs, num_tokens, self._block_m(num_tokens))
        y = torch.empty((num_tokens, self.hidden), device="cuda", dtype=torch.bfloat16)
        self._call(y)
        torch.cuda.synchronize()

        def gather(t: Any) -> Any:
            parts = [torch.empty_like(t) for _ in range(self.world_size)]
            dist.all_gather(parts, t.contiguous())
            return torch.cat(parts)

        x = cast_back_from_fp8(inputs["x_fp8"], inputs["x_sf"], gran_k=32, use_packed_ue8m0=True)
        all_x = gather(x)
        all_idx = gather(inputs["topk_idx"])
        all_w = gather(inputs["topk_weights"])
        l1, l1_sf, l2, l2_sf = self._raw_routed

        def dequant(w: Any, sf: Any) -> Any:
            return torch.stack(
                [cast_back_from_fp4(w[e], sf[e], gran_k=32) for e in range(w.shape[0])]
            )

        partial = ref.expert_partial(
            torch,
            all_x,
            all_idx,
            all_w,
            dequant(l1, l1_sf),
            dequant(l2, l2_sf),
            self.rank * self.local_experts,
            ACTIVATION_CLAMP,
        )
        dist.all_reduce(partial)
        expected = partial[self.rank * num_tokens : (self.rank + 1) * num_tokens]
        if self._raw_shared is not None:
            s1, s1_sf, s2, s2_sf = self._raw_shared
            expected = expected + ref.shared_experts(
                torch,
                x,
                cast_back_from_fp8(s1, s1_sf, gran_k=32, use_packed_ue8m0=True),
                cast_back_from_fp8(s2, s2_sf, gran_k=32, use_packed_ue8m0=True),
                ACTIVATION_CLAMP,
            )
        diff = float(calc_diff(y.float(), expected))
        diffs = [None] * self.world_size
        dist.all_gather_object(diffs, diff)
        if max(diffs) > _CHECK_MAX_DIFF:
            raise KernelLaunchFailed(
                f"mega_moe differs from the Torch reference: calc_diff={diffs}"
            )
        self.check_diff = max(diffs)

    # ---- timing --------------------------------------------------------
    def measure(self, num_tokens: int, per_slot_rows: Any) -> dict:
        import torch.distributed as dist

        from profiling.profilers.cupti_kernel_profiler import (
            CuptiKernelProfiler,
            match_kernel_records,
        )
        from profiling.profilers.energy import energy_enabled

        torch = self.torch
        if num_tokens > self.buffer.num_max_tokens_per_rank:
            raise ValueError("num_tokens exceeds the symmetric buffer")
        block_m = self._block_m(num_tokens)
        inputs = self._inputs(num_tokens, per_slot_rows)
        y = torch.empty((num_tokens, self.hidden), device="cuda", dtype=torch.bfloat16)
        token = torch.zeros((1,), device="cuda")

        def launch() -> None:
            torch.cuda._sleep(_SLEEP_CYCLES)
            dist.all_reduce(token)
            self._call(y)

        # The first call JIT-compiles the kernel for this tile config.
        for _ in range(_WARMUP):
            self._write_inputs(inputs, num_tokens, block_m)
            launch()
            torch.cuda.synchronize()
        dist.barrier()

        profiler = CuptiKernelProfiler(device=torch.device("cuda", torch.cuda.current_device()))
        per_launch_ms = []
        for _ in range(_ITERS):
            self._write_inputs(inputs, num_tokens, block_m)
            records = profiler.capture_once(launch, clear_l2_before_run=True)
            matched = match_kernel_records(records, kernel_name_contains=KERNEL_NAME)
            if len(matched) != 1:
                raise KernelLaunchFailed(f"expected one mega_moe launch, got {len(matched)}")
            per_launch_ms.append(matched[0].duration_ns / 1e6)

        local = torch.tensor(per_launch_ms, device="cuda", dtype=torch.float64)
        ranks = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(ranks, local)
        table = torch.stack(ranks)  # [rank, launch]
        time_ms = float(table.max(dim=0).values.median())

        energy_j = (
            self._energy(time_ms, inputs, num_tokens, block_m, y) if energy_enabled() else 0.0
        )
        flops, nbytes = self._work(inputs["topk_idx"], num_tokens)
        return {
            "time_ms": time_ms,
            "tflops": flops / (time_ms * 1e-3) / 1e12,
            "memory_bandwidth_gbps": nbytes / (time_ms * 1e-3) / 1e9,
            "energy_j": energy_j,
            # Diagnostics (not stored): per-rank medians and the launch spread.
            "rank_median_ms": [float(r.median()) for r in table],
            "launch_min_ms": float(table.min()),
            "launch_max_ms": float(table.max()),
            "block_m": block_m,
            "capacity": int(self.buffer.num_max_tokens_per_rank),
        }

    def _energy(self, time_ms: float, inputs: dict, num_tokens: int, block_m: int, y: Any) -> float:
        """Mean per-GPU energy of back-to-back calls; every rank runs the same count."""
        import torch.distributed as dist

        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError:  # same image on every rank, so all ranks skip together
            return 0.0

        from profiling.profilers._duration import DEFAULT_MIN_DURATION_MS, iters_for_duration
        from profiling.profilers.energy import (
            _measure_total_energy_counter_once,
            _nvml_handle_for_cuda_device,
        )

        torch = self.torch
        iters = iters_for_duration(DEFAULT_MIN_DURATION_MS, time_ms)
        self._write_inputs(inputs, num_tokens, block_m)
        pynvml.nvmlInit()
        handle = _nvml_handle_for_cuda_device(pynvml, torch, torch.cuda.current_device())
        dist.barrier()
        attempt = _measure_total_energy_counter_once(
            pynvml, torch, handle, lambda: self._call(y), iters
        )
        per_call = torch.tensor([attempt.energy_j / iters], device="cuda", dtype=torch.float64)
        dist.all_reduce(per_call)
        return float(per_call) / self.world_size

    def _work(self, topk_idx: Any, num_tokens: int) -> tuple[float, float]:
        """Rank-local FLOPs and HBM bytes, counted like DeepGEMM's test_mega_moe."""
        import torch.distributed as dist

        torch = self.torch
        counts = torch.bincount(topk_idx.flatten(), minlength=self.num_experts)
        dist.all_reduce(counts)
        local = counts[self.rank * self.local_experts : (self.rank + 1) * self.local_experts]
        recv = int(local.sum())
        touched = int((local > 0).sum())
        hidden, inter = self.hidden, self.intermediate
        shared_inter = inter * self.num_shared
        flops = 6.0 * hidden * (recv * inter + num_tokens * shared_inter)
        nbytes = (
            touched * 3 * hidden * inter * 0.5  # FP4 routed weights
            + recv * (hidden + 2 * inter + 2 * hidden)  # fp8 in, fp8 mid (w+r), bf16 out
            + (3 * hidden * shared_inter if self.num_shared else 0)  # fp8 shared weights
            + (num_tokens * (hidden + 2 * shared_inter + 2 * hidden) if self.num_shared else 0)
        )
        return flops, float(nbytes)

    def _free_buffer(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None

    def destroy(self) -> None:
        self._free_buffer()


def _cast_fp8(torch: Any, x: Any) -> tuple[Any, Any, Any]:
    """MXFP8 rows with packed UE8M0 scales, plus the TMA-aligned scale copy
    (DeepGEMM test_mega_moe ``_cast_fp8_for_mega_moe``)."""
    from deep_gemm.utils import align, per_token_cast_to_fp8

    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    rows, packed_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (rows, packed_k), (1, align(rows, 4)), dtype=x_sf.dtype, device=x_sf.device
    )
    x_sf_tma.copy_(x_sf)
    return x_fp8, x_sf, x_sf_tma


def _write_shared_sf(torch: Any, dst: Any, sf: Any, block_m: int) -> None:
    """The shared expert's activation scales in the kernel's per-block transposed
    layout (vectorized DeepGEMM test ``_to_shared_mega_moe_sf_layout``; DeepInfer's
    fused norm-cast writes it with ``extra_sf_layout="mega_moe"``)."""
    rows = sf.shape[0]
    token = torch.arange(rows, device=sf.device)
    block, m = token // block_m, token % block_m
    aligned_block_m = (block_m + 127) // 128 * 128
    target = block * aligned_block_m + (m // 128) * 128 + (m % 32) * 4 + (m % 128) // 32
    dst.zero_()
    dst[target] = sf
