"""Independent Torch reference for the sink-renormalized gate select.

Written from the routing semantics in the alignment fork's module docstring
and ``InklingGate`` (``vllm/models/inkling/nvidia/moe.py``: sigmoid gate plus
selection bias, top-k over the routed experts, log-sigmoid renormalization over
the k routed plus S sink logits, times route_scale * global_scale), not from
``_inkling_gate_select_kernel``. It shares no algorithm with the kernel: the
selection is one stable descending sort (equal scores keep ascending expert
order, i.e. ties go to the lowest id) instead of K iterative argmax passes, and
the weights are ``sigmoid(l) / sum(sigmoid(l))`` in float64 instead of a
max-shifted softmax of fp32 log-sigmoids.

``compare`` accepts a selection that differs from the reference only where the
scores are within ``score_atol`` (ulp-level sigmoid differences between Triton
and Torch); exact ties (equal logit and equal bias) must follow the lowest-id
rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateShape:
    num_routed_experts: int
    num_shared_experts: int
    top_k: int
    logits_row_stride: int

    @property
    def num_gate_experts(self) -> int:
        """Columns that carry logits: routed then sinks (the rest is GEMM padding)."""
        return self.num_routed_experts + self.num_shared_experts

    @property
    def active(self) -> int:
        """Output columns per token: the routed top-k, then the sinks."""
        return self.top_k + self.num_shared_experts


def selection_scores(torch: Any, shape: GateShape, logits: Any, bias: Any | None) -> Any:
    """fp32 ``sigmoid(l) + bias`` over the routed columns ``[T, R]``."""
    routed = logits[:, : shape.num_routed_experts].float()
    scores = torch.sigmoid(routed)
    if bias is not None:
        scores = scores + bias.float()
    return scores


def reference_select(
    torch: Any,
    shape: GateShape,
    logits: Any,
    bias: Any | None,
    route_scale: float,
    global_scale: Any | None,
) -> tuple[Any, Any]:
    """Return ``(ids int64 [T, K + S], weights float64 [T, K + S])``."""
    R, K, S = shape.num_routed_experts, shape.top_k, shape.num_shared_experts
    scores = selection_scores(torch, shape, logits, bias)
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :K]
    sinks = torch.arange(R, R + S, device=logits.device).expand(logits.shape[0], S)
    ids = torch.cat([order, sinks], dim=1)
    weights = renormalized_weights(torch, logits, ids, route_scale, global_scale)
    return ids, weights


def renormalized_weights(
    torch: Any, logits: Any, ids: Any, route_scale: float, global_scale: Any | None
) -> Any:
    """``route_scale * gs * sigmoid(l_a) / sum_a sigmoid(l_a)`` over the given ids, float64."""
    raw = logits.double().gather(1, ids.long())
    gates = torch.sigmoid(raw)
    scale = float(route_scale)
    if global_scale is not None:
        scale *= float(global_scale.double().item())
    return gates / gates.sum(dim=1, keepdim=True) * scale


def compare(
    torch: Any,
    shape: GateShape,
    logits: Any,
    bias: Any | None,
    route_scale: float,
    global_scale: Any | None,
    ids: Any,
    weights: Any,
    *,
    score_atol: float = 1e-6,
    weight_rtol: float = 1e-5,
) -> dict[str, Any]:
    """Check production ``ids``/``weights`` against the reference; return a report.

    Tokens whose routed ids equal the reference's (same experts, same order)
    pass outright. Any other token passes only as a near-tie: its selection
    must still be a valid top-k up to ``score_atol`` (no excluded expert scores
    more than ``score_atol`` above a selected one, and the order is
    non-increasing within ``score_atol``), and it must not break an exact tie
    the wrong way (an excluded expert with the same logit and bias as a
    selected one and a lower id, or two such experts listed in descending id
    order). Weights are checked against the production ids, so a tolerated
    near-tie does not cascade into a weight error.
    """

    K, R = shape.top_k, shape.num_routed_experts
    ref_ids, _ = reference_select(torch, shape, logits, bias, route_scale, global_scale)
    ids = ids.long()
    routed = ids[:, :K]
    in_range = bool(((routed >= 0) & (routed < R)).all().item())
    distinct = bool((routed.sort(dim=1).values.diff(dim=1) != 0).all().item()) if K > 1 else True
    sinks_exact = bool(torch.equal(ids[:, K:], ref_ids[:, K:]))
    mismatch = (routed != ref_ids[:, :K]).any(dim=1)
    unexplained_tokens = int(mismatch.sum().item())
    if in_range and distinct and unexplained_tokens:
        unexplained_tokens = int(
            (mismatch & ~_valid_near_tie(torch, shape, logits, bias, routed, score_atol))
            .sum()
            .item()
        )
    near_tie_tokens = int(mismatch.sum().item()) - unexplained_tokens
    max_rel = float("inf")
    if in_range:
        expected = renormalized_weights(torch, logits, ids, route_scale, global_scale)
        max_rel = float(((weights.double() - expected).abs() / expected.abs()).max().item())
    report = {
        "tokens": int(ids.shape[0]),
        "routed_ids_in_range": in_range,
        "routed_ids_distinct": distinct,
        "sink_ids_exact": sinks_exact,
        "exact_id_match_tokens": int((~mismatch).sum().item()),
        "near_tie_tokens": near_tie_tokens,
        "unexplained_mismatch_tokens": unexplained_tokens,
        "weights_max_rel_err": max_rel,
    }
    report["ok"] = bool(
        in_range and distinct and sinks_exact and unexplained_tokens == 0 and max_rel <= weight_rtol
    )
    return report


def _valid_near_tie(
    torch: Any, shape: GateShape, logits: Any, bias: Any | None, routed: Any, score_atol: float
) -> Any:
    """Per token: is ``routed`` a top-k selection up to ``score_atol`` that keeps exact ties?"""
    R = shape.num_routed_experts
    T, K = routed.shape
    scores = selection_scores(torch, shape, logits, bias).double()
    chosen = scores.gather(1, routed)
    excluded = torch.ones((T, R), dtype=torch.bool, device=routed.device)
    excluded.scatter_(1, routed, False)
    best_excluded = torch.where(excluded, scores, torch.full_like(scores, float("-inf"))).amax(1)
    top_k_ok = chosen.amin(1) >= best_excluded - score_atol
    order_ok = (chosen[:, :-1] >= chosen[:, 1:] - score_atol).all(1)

    # Equal logit and equal bias give equal scores in any implementation, so
    # the lowest-id rule decides those ties exactly.
    routed_logits = logits[:, :R]
    same = routed_logits[:, None, :] == routed_logits.gather(1, routed)[:, :, None]  # [T, K, R]
    if bias is not None:
        same &= bias[None, None, :] == bias[routed][:, :, None]
    lower = torch.arange(R, device=routed.device)[None, None, :] < routed[:, :, None]
    skipped_lower_twin = (same & lower & excluded[:, None, :]).any(dim=(1, 2))
    consecutive_twins = same[:, :-1, :].gather(2, routed[:, 1:, None]).squeeze(2)
    descending_twins = (consecutive_twins & (routed[:, :-1] > routed[:, 1:])).any(1)
    return top_k_ok & order_ok & ~skipped_lower_twin & ~descending_twins


__all__ = [
    "GateShape",
    "compare",
    "reference_select",
    "renormalized_weights",
    "selection_scores",
]
