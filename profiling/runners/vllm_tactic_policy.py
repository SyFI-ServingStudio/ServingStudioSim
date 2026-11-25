"""Which FlashInfer MoE tactic a vLLM server runs for one call, and how an L1
runner reproduces it.

A vLLM server does not autotune the call it serves. At start-up
(``vllm/model_executor/warmup/kernel_warmup.py``,
``_run_flashinfer_autotune_dummy_runs``) it runs one dummy forward of
``max_num_batched_tokens`` tokens under::

    autotune(tuning_buckets=get_hybrid_num_tokens_buckets(max_tuning_tokens),
             round_up=True)

so the tactic table holds exactly the hybrid buckets up to the per-rank batch
budget (1, 2, 4, ..., 256, 512, ..., 2048, 2560, ..., 4096, then powers of two,
plus the budget itself). It is written to and reloaded from
``flashinfer_autotune_cache/.../autotune_configs_dp*_rank*.json``. While
serving, the MoE call runs outside any ``autotune()`` context, so FlashInfer
looks the call up with the op's own mapper,
``make_hybrid_bucket_mapper(tune_max_num_tokens)`` (round up, clamp at
``tune_max_num_tokens``; ``flashinfer/fused_moe/core.py``
``MoERunner.refine_tuning_config``), where vLLM passes
``tune_max_num_tokens = fi_moe_largest_bucket = max(max_num_tokens * dp_size,
8192)``, capped at the chunk size for the precomputed-routing call
(``trtllm_nvfp4_moe.py``). A bucket outside the tuned set is a cache miss, and
a miss runs ``runners[0]`` with tactic -1, the kernel's built-in default
(``flashinfer/autotuner/autotuner.py``, "No tuned config covers ...; falling
back to runner=MoERunner tactic=-1").

So for a call of M tokens the served tactic is:

- ``bucket = map_to_hybrid_bucket(M, tune_max)``;
- ``bucket in get_hybrid_num_tokens_buckets(max_tuning_tokens)``: the tactic
  the warm-up tuned for that bucket;
- otherwise: tactic -1. With DP the MoE sees ``dp_size`` ranks' tokens, so a
  full prefill (M = dp_size * budget) is a miss.

``POLICY_VLLM`` reproduces this. ``POLICY_RUNNER_AUTOTUNE``, the default,
keeps the runner's existing behaviour, an in-process ``autotune()`` of the
measured shape itself, so rows refreshed without the environment below are
unchanged. A deployment selects ``POLICY_VLLM`` explicitly, with its own batch
budget and DP size.

The deployment numbers are not kernel arguments and not part of a DB key, so
they come from the environment:

- ``VIBESIM_FLASHINFER_TACTIC_POLICY``: ``runner_autotune`` (default) or
  ``vllm``.
- ``VIBESIM_VLLM_MAX_NUM_BATCHED_TOKENS``: the server's
  ``--max-num-batched-tokens``; unset means vLLM's own default for the GPU
  (``vllm/engine/arg_utils.py``: 16384 at >= 160 GiB, 8192 at >= 70 GiB other
  than A100, else 2048, the OpenAI-server values). Drafter tokens are not
  added; set the value vLLM logs in "Running FlashInfer autotune with N
  tokens" when they matter.
- ``VIBESIM_VLLM_DP_SIZE``: the server's data-parallel size (default 1).

This module imports nothing heavy, so the parent (tactic file names) and the
tests can use it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

POLICY_ENV = "VIBESIM_FLASHINFER_TACTIC_POLICY"
BATCH_TOKENS_ENV = "VIBESIM_VLLM_MAX_NUM_BATCHED_TOKENS"
DP_SIZE_ENV = "VIBESIM_VLLM_DP_SIZE"
#: What a container worker must be handed (``profiling/exec/local.py``).
ENV_NAMES = (POLICY_ENV, BATCH_TOKENS_ENV, DP_SIZE_ENV)

POLICY_VLLM = "vllm"
POLICY_RUNNER_AUTOTUNE = "runner_autotune"
POLICIES = (POLICY_VLLM, POLICY_RUNNER_AUTOTUNE)

#: FlashInfer's floor for the MoE tuning bound, kept by vLLM's
#: ``fi_moe_largest_bucket``.
FI_MOE_MIN_TUNE_MAX = 8192

#: The FlashInfer op every TRT-LLM NVFP4 MoE call (monolithic and routed) tunes as.
NVFP4_MOE_CUSTOM_OP = "flashinfer::trtllm_fp4_block_scale_moe"

_PHASE1_END = 256
_PHASE2_END, _PHASE2_STEP = 2048, 256
_PHASE3_END, _PHASE3_STEP = 4096, 512


def tactic_policy() -> str:
    value = os.environ.get(POLICY_ENV, "").strip() or POLICY_RUNNER_AUTOTUNE
    if value not in POLICIES:
        raise ValueError(f"{POLICY_ENV}={value!r}; expected one of {POLICIES}")
    return value


def configured_batch_tokens() -> int | None:
    """``VIBESIM_VLLM_MAX_NUM_BATCHED_TOKENS``, or ``None`` for vLLM's default."""

    value = os.environ.get(BATCH_TOKENS_ENV, "").strip()
    if not value:
        return None
    tokens = int(value)
    if tokens < 1:
        raise ValueError(f"{BATCH_TOKENS_ENV} must be positive, got {tokens}")
    return tokens


def configured_dp_size() -> int:
    value = os.environ.get(DP_SIZE_ENV, "").strip()
    dp = int(value) if value else 1
    if dp < 1:
        raise ValueError(f"{DP_SIZE_ENV} must be positive, got {dp}")
    return dp


def vllm_default_batch_tokens(device_memory_bytes: int, device_name: str) -> int:
    """vLLM's default ``max_num_batched_tokens`` for an OpenAI API server
    (``EngineArgs._set_default_max_num_seqs_and_batched_tokens_args``)."""

    gib = 1 << 30
    if device_memory_bytes >= 160 * gib:
        return 16384
    if device_memory_bytes >= 70 * gib and "a100" not in device_name.lower():
        return 8192
    return 2048


def label_suffix() -> str:
    """The tactic-file name part that keeps policies and budgets apart.

    The default policy (``runner_autotune``) keeps the existing file names. ``nauto`` is resolved by the
    worker from its device, and tactic files are already keyed by device.
    """

    if tactic_policy() == POLICY_RUNNER_AUTOTUNE:
        return ""
    tokens = configured_batch_tokens()
    return f".vllmtac-n{tokens if tokens is not None else 'auto'}"


def next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def hybrid_buckets(max_num_tokens: int) -> tuple[int, ...]:
    """``flashinfer.fused_moe.utils.get_hybrid_num_tokens_buckets``."""

    buckets: list[int] = []
    m = 1
    while m <= min(max_num_tokens, _PHASE1_END):
        buckets.append(m)
        m *= 2
    m = _PHASE1_END + _PHASE2_STEP
    while m <= min(max_num_tokens, _PHASE2_END):
        buckets.append(m)
        m += _PHASE2_STEP
    m = _PHASE2_END + _PHASE3_STEP
    while m <= min(max_num_tokens, _PHASE3_END):
        buckets.append(m)
        m += _PHASE3_STEP
    m = _PHASE3_END * 2
    while m <= max_num_tokens:
        buckets.append(m)
        m *= 2
    if not buckets or buckets[-1] != max_num_tokens:
        buckets.append(max_num_tokens)
    return tuple(sorted(set(buckets)))


def _ceil_to(x: int, step: int) -> int:
    return (x + step - 1) // step * step


def map_to_hybrid_bucket(x: int, max_num_tokens: int) -> int:
    """``flashinfer.fused_moe.utils.map_to_hybrid_bucket``: round up, clamp."""

    if x <= 0:
        return 1
    if x >= max_num_tokens:
        return max_num_tokens
    if x <= _PHASE1_END:
        return next_power_of_2(x)
    if x <= _PHASE2_END:
        return min(_ceil_to(x, _PHASE2_STEP), max_num_tokens)
    if x <= _PHASE3_END:
        return min(_ceil_to(x, _PHASE3_STEP), max_num_tokens)
    return min(next_power_of_2(x), max_num_tokens)


def trtllm_routed_chunk_size(top_k: int, num_experts: int) -> int:
    """``TrtLlmNvFp4ExpertsBase._get_chunk_size``: the CUDA grid.Y bound, at
    most 300000 tokens."""

    max_grid_y, max_tile_tokens = 65535, 128
    bound = (num_experts + (max_grid_y - num_experts + 1) * max_tile_tokens - 1) // top_k
    return min(300000, bound)


def vllm_moe_tune_max(
    batch_tokens: int, dp_size: int, *, routed_chunk: int | None = None
) -> int:
    """The ``tune_max_num_tokens`` vLLM passes: ``fi_moe_largest_bucket``, and
    for the precomputed-routing call that capped at the chunk size."""

    largest = max(batch_tokens * dp_size, FI_MOE_MIN_TUNE_MAX)
    return largest if routed_chunk is None else min(largest, routed_chunk)


@dataclass(frozen=True)
class VllmTactic:
    """The served tactic of one call: ``tuned`` (the warm-up's tactic for
    ``bucket``) or the miss fallback, tactic -1."""

    num_tokens: int
    batch_tokens: int
    dp_size: int
    tune_max: int
    bucket: int
    tuning_buckets: tuple[int, ...]

    @property
    def tuned(self) -> bool:
        return self.bucket in self.tuning_buckets

    def describe(self) -> str:
        choice = "tuned bucket" if self.tuned else "miss -> tactic -1"
        return (
            f"M={self.num_tokens} budget={self.batch_tokens} dp={self.dp_size} "
            f"tune_max={self.tune_max} bucket={self.bucket} ({choice})"
        )


def vllm_tactic(
    num_tokens: int,
    *,
    batch_tokens: int,
    dp_size: int = 1,
    routed_chunk: int | None = None,
) -> VllmTactic:
    tune_max = vllm_moe_tune_max(batch_tokens, dp_size, routed_chunk=routed_chunk)
    return VllmTactic(
        num_tokens=num_tokens,
        batch_tokens=batch_tokens,
        dp_size=dp_size,
        tune_max=tune_max,
        bucket=map_to_hybrid_bucket(num_tokens, tune_max),
        tuning_buckets=hybrid_buckets(batch_tokens),
    )
