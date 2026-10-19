"""Routing inputs shared by the TRT-LLM fused-MoE runners that route in-kernel.

The NVFP4 and FP8 block-scale calls take router logits and select experts
themselves, so a row's per-expert histogram is realized by forcing the logits.
This module owns the parts both runners must agree on: which group arguments
each routing method's measured call takes, and the forced logits.
"""

from __future__ import annotations

from typing import Any

# Routing methods without a correction bias or expert groups: vLLM passes
# routing_bias=None and n_group = topk_group = 0.
UNBIASED_ROUTING_METHODS = frozenset({"renormalize_naive"})
# n_group = topk_group each method's measured call takes. Grouped DeepSeekV3
# routing only admits experts from the best topk_group groups, so forced logits
# could not realize an arbitrary histogram: only the ungrouped form (1) is
# measured. Unbiased methods take no groups (0). Methods left out (MiniMax2)
# accept any positive group counts.
_REQUIRED_GROUPS = {"deepseek_v3": 1, "renormalize_naive": 0}

# Forced router logits: selected experts get distinct high scores, the rest a
# low score, so sigmoid or softmax + top-k picks exactly the requested ids. The
# selected values step down by 0.5 from 8 and stay above the unselected -8 (and
# exact in BF16) for at most 32 of them.
_SELECTED_LOGIT = 8.0
_SELECTED_STEP = 0.5
_UNSELECTED_LOGIT = -8.0
MAX_FORCED_TOP_K = 32


def validate_routing_groups(routing_method: str, n_group: int, topk_group: int) -> None:
    """Reject group arguments the measured call for ``routing_method`` would not honor."""

    required = _REQUIRED_GROUPS.get(routing_method)
    if required is None:
        if n_group < 1 or topk_group < 1:
            raise ValueError(f"{routing_method} routing needs n_group and topk_group >= 1")
    elif n_group != required or topk_group != required:
        raise ValueError(
            f"only ungrouped {routing_method} routing "
            f"(n_group=topk_group={required}) is supported"
        )


def forced_routing_logits(
    torch: Any, ids: list[list[int]], num_experts: int, device: Any, dtype: Any = None
) -> Any:
    """Router logits whose top-k is exactly ``ids``, under DeepSeekV3 (zero
    bias), MiniMax2 or softmax scoring alike. FP32 unless ``dtype`` says
    otherwise."""

    dtype = torch.float32 if dtype is None else dtype
    selected = torch.tensor(ids, dtype=torch.int64, device=device)
    if selected.shape[1] > MAX_FORCED_TOP_K:
        raise ValueError(
            f"forced routing keeps selected logits distinct only for top_k <= "
            f"{MAX_FORCED_TOP_K}, got {selected.shape[1]}"
        )
    logits = torch.full(
        (selected.shape[0], num_experts), _UNSELECTED_LOGIT, dtype=dtype, device=device
    )
    ranks = torch.arange(selected.shape[1], dtype=dtype, device=device)
    values = (_SELECTED_LOGIT - _SELECTED_STEP * ranks).expand_as(selected).contiguous()
    logits.scatter_(1, selected, values)
    return logits


__all__ = [
    "MAX_FORCED_TOP_K",
    "UNBIASED_ROUTING_METHODS",
    "forced_routing_logits",
    "validate_routing_groups",
]
