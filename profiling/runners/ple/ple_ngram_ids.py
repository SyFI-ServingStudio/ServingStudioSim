"""Profile the PLE n-gram hash ids (``ple_ngram_ids``).

``vllm_triton`` times the fork's ``ple_ngram_ids`` on ``num_tokens`` random
token ids split into ``num_reqs`` near-equal requests, with random per-request
``ngram_context``. Multipliers, head sizes and offsets come from the fork's own
``Qwen4ExpNGramEmbedding`` helpers for a 248,320-token vocabulary and
20M-row heads (the values only change hash results, not the work).

The reference is the fork's eager path: ``compute_ngram_ids`` takes it when the
inputs are CPU tensors. Ids must match bit for bit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "ple_ngram_ids"
_KERNEL_NAME = "_ple_ngram_ids_kernel"
_VOCAB = 248_320
_EOS = 248_044
_HEAD_ROWS = 20_000_000
_MAX_TOKENS = 1 << 17


def validate_args(num_tokens: int, num_reqs: int, ngram_context_len: int, heads_per_ngram: int):
    if type(num_tokens) is not int or not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be an integer in [1, {_MAX_TOKENS}]")
    if type(num_reqs) is not int or not 1 <= num_reqs <= num_tokens:
        raise ValueError("num_reqs must be an integer in [1, num_tokens]")
    if type(ngram_context_len) is not int or not 1 <= ngram_context_len <= 8:
        raise ValueError("ngram_context_len must be an integer in [1, 8]")
    if type(heads_per_ngram) is not int or not 1 <= heads_per_ngram <= 32:
        raise ValueError("heads_per_ngram must be an integer in [1, 32]")


def query_start_loc(num_tokens: int, num_reqs: int) -> list[int]:
    """Near-equal request boundaries."""
    return [num_tokens * r // num_reqs for r in range(num_reqs + 1)]


def profile_ple_ngram_ids_vllm_triton(
    num_tokens: int, num_reqs: int, ngram_context_len: int, heads_per_ngram: int
) -> ComputeMetrics:
    """Time one n-gram id launch."""
    validate_args(num_tokens, num_reqs, ngram_context_len, heads_per_ngram)
    try:
        import torch
        from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding as Emb
        from vllm.models.qwen4_exp.nvidia.ops.ple import ple_ngram_ids
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{KIND}:vllm_triton requires the vLLM fork") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        ngram_size = ngram_context_len + 1
        heads = ngram_context_len * heads_per_ngram
        mult = Emb._make_layer_multipliers(
            ngram_size=ngram_size, unigram_vocab_size=_VOCAB, seed=1234, ple_dense_layer_id=0
        )
        sizes = [_HEAD_ROWS + 2 * h + 1 for h in range(heads)]
        offsets = [sum(sizes[:h]) for h in range(heads)]
        gen = torch.Generator().manual_seed(0x1D5)
        tokens = torch.randint(0, _VOCAB, (num_tokens,), generator=gen)
        tokens[torch.rand(num_tokens, generator=gen) < 0.01] = _EOS
        context = torch.randint(0, _VOCAB, (num_reqs, ngram_context_len), generator=gen)
        qsl = torch.tensor(query_start_loc(num_tokens, num_reqs), dtype=torch.int32)
        host = dict(
            layer_multipliers=torch.tensor(mult, dtype=torch.long),
            ngram_heads_vocab_sizes=torch.tensor(sizes, dtype=torch.long),
            ngram_heads_offsets=torch.tensor(offsets, dtype=torch.long),
        )
        module = SimpleNamespace(
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            eos_token_id=_EOS,
            _shift_precompute=Emb._shift_precompute,
            _shift_apply=Emb._shift_apply,
            **host,
        )
        ref = Emb.compute_ngram_ids(module, tokens, qsl, context)
        dev = {k: v.to(device) for k, v in host.items()}
        d_tokens, d_qsl, d_ctx = tokens.to(device), qsl.to(device), context.to(device)
        out = torch.empty(num_tokens, heads, dtype=torch.int64, device=device)

        def launch() -> None:
            ple_ngram_ids(
                d_tokens, d_qsl, d_ctx, dev["layer_multipliers"], dev["ngram_heads_vocab_sizes"],
                dev["ngram_heads_offsets"], _EOS, heads_per_ngram, output=out,
            )

        launch()
        torch.cuda.synchronize()
        mismatches = int((out.cpu() != ref).sum())
        if mismatches:
            raise KernelLaunchFailed(f"{KIND} ids differ from the eager path: {mismatches}")
        time_ms = Timer.cupti(launch, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{KIND}:vllm_triton ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:vllm_triton failed: {exc}") from exc
    nbytes = num_tokens * (4 + 8 * heads)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(nbytes / (time_ms / 1000.0) / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_ple_ngram_ids_vllm_triton", "query_start_loc", "validate_args"]
