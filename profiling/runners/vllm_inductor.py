"""Compile a block of vLLM's native PyTorch code the way vLLM's compiled graph runs it.

vLLM serves a model as one Inductor graph per piece between its attention ops,
compiled once with the token dimension symbolic and every Dynamo guard dropped
(``vllm/compilation/wrapper.py``), so the same generated Triton kernel serves
every batch size, decode included. Inductor's ``CachingAutotuner`` benchmarks a
kernel's candidate launch configs on its first call and keeps the winner for
all later calls. vLLM's first call is the profile run at
``max_num_batched_tokens``, so the served launch config is the one tuned at a
large token count, not one tuned per batch size.

``compile_like_vllm`` reproduces that: Inductor options from vLLM's own
``CompilationConfig``, the token dimension marked dynamic, guards skipped, and a
first call at ``TUNE_TOKENS``, or at the measured token count when that is
larger, because a deployment that runs such a step has a larger
``max_num_batched_tokens``. The size matters for correctness there: Inductor
picks 32-bit indexing from the first call's sizes, and a kernel tuned at 8192
tokens indexes out of bounds once a step holds more than 2^31 elements (65536
tokens of a 17408-wide SiLU-and-mul). Compiling per shape (``dynamic=False``) instead
picks other configs below a few hundred tokens: on B200 the static compile of
SiLU-and-mul launches 136 programs at one token where the served kernel
launches 17, and the static FP8 quant 24 where it launches 6.

The tuning size is not a deployment knob: tuning at 2048 and at 8192 tokens
selected the same configs for every kernel kind built here (B200), and those
reproduce the grid and block of every served launch in a chunk-2048 capture.

Only blocks whose source is vLLM's own native implementation belong here; the
caller passes that callable unchanged so the generated kernel is the one vLLM's
graph contains.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# First-call token count that selects the launch config (see module docstring).
TUNE_TOKENS = 8192


def vllm_inductor_options(torch: Any) -> dict[str, Any]:
    """The Inductor options vLLM compiles with, plus its guard filter."""
    from vllm.config import CompilationConfig

    options = dict(CompilationConfig().inductor_compile_config)
    options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
    return options


def compile_like_vllm(
    torch: Any,
    block: Callable[..., Any],
    make_inputs: Callable[[int], Sequence[Any]],
    token_dims: dict[int, int],
    num_tokens: int,
) -> Callable[..., Any]:
    """Compile ``block`` with a symbolic token dimension, as vLLM serves ``num_tokens``.

    ``make_inputs(tokens)`` builds the block's positional arguments;
    ``token_dims`` maps an argument index to the dimension that holds tokens.
    The returned callable accepts inputs of up to the tuning token count.
    """
    # With guards dropped, Dynamo would hand an earlier spec's graph (another
    # width or output set) to this one, because both trace the same code object.
    torch._dynamo.reset()
    compiled = torch.compile(block, backend="inductor", options=vllm_inductor_options(torch))
    tune_inputs = make_inputs(max(TUNE_TOKENS, num_tokens))
    for index, dim in token_dims.items():
        torch._dynamo.mark_dynamic(tune_inputs[index], dim)
    compiled(*tune_inputs)
    torch.cuda.synchronize()
    return compiled


__all__ = ["TUNE_TOKENS", "compile_like_vllm", "vllm_inductor_options"]
