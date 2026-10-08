"""Independent draft/verify workload reconstruction from request geometry.

No kernel shapes or achieved work enter these formulas. The target verifies
k+1 causal rows under every proposer; what the proposer runs depends on its
mechanism, chosen by the stages the model declares:

- **MTP** first processes the target's rows and then advances each request
  endpoint once per remaining draft position.
- **DFlash2** runs once, with no recurrence: it projects the context K/V of the
  target's rows, forwards one non-causal block of ``k + 1`` query rows per
  request against its context, and scores the ``k`` drafted positions.
"""

import json
import math
from collections import defaultdict
from dataclasses import replace

from .core import AttnInteraction, Workload
from .models import dflash2
from .models.glm52 import DRAFT_FIRST_STAGE, DRAFT_RECURRENT_STAGE

#: Proposer -> the stages its stacks run against.
PROPOSER_STAGES = {
    "mtp": frozenset({DRAFT_FIRST_STAGE, DRAFT_RECURRENT_STAGE}),
    "dflash2": frozenset(dflash2.STAGES),
}


def proposer_for(declared_stages) -> str:
    """The proposer a model's staged stacks belong to."""
    declared = frozenset(declared_stages)
    matches = [name for name, stages in PROPOSER_STAGES.items() if declared and declared <= stages]
    if len(matches) != 1:
        raise ValueError(f"no single speculative proposer declares stages {sorted(declared)}")
    return matches[0]


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def aggregate_workload(totals: dict, proposer: str = "mtp") -> Workload:
    if proposer not in PROPOSER_STAGES:
        raise ValueError(f"unknown speculative proposer {proposer!r}")
    geometries = totals.get("speculative_geometry")
    if not geometries:
        raise ValueError("speculative floors require per-stage workload geometry")
    target = defaultdict(float)
    recurrent = defaultdict(float)
    # DFlash2: one block per request, keyed by the context it drafts against.
    endpoints = defaultdict(float)
    depth = None
    for encoded, count in geometries.items():
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not math.isfinite(count)
            or count <= 0
        ):
            raise ValueError("geometry multiplicity must be finite and positive")
        group = json.loads(encoded)
        k = _integer(group["draft_tokens"], "draft_tokens", 1)
        maximum = _integer(group["max_model_len"], "max_model_len", k + 1)
        if depth is not None and depth != k:
            raise ValueError("cannot combine different draft depths in one workload")
        depth = k
        for phase, requests in (("prefill", group["prefill"]), ("decode", group["decode"])):
            for left, right in requests:
                if phase == "prefill":
                    cached = _integer(left, "prefill prefix")
                    q = _integer(right, "prefill query", 1)
                    final = cached + q
                else:
                    final = _integer(left, "decode context", k + 1)
                    q = _integer(right, "decode query", 1)
                    if q != k + 1:
                        raise ValueError("decode query must equal draft_tokens + 1")
                    cached = final - q
                if final > maximum:
                    raise ValueError("request context exceeds max_model_len")
                target[(phase, q, cached)] += count
                endpoints[final] += count
                for advance in range(1, k):
                    endpoint = min(final + advance, maximum)
                    recurrent[("decode", 1, endpoint - 1)] += count

    def workload(histogram, *, endpoint_head=False):
        interactions = [
            AttnInteraction(q, cached + q, cached, "causal", phase, count)
            for (phase, q, cached), count in sorted(histogram.items())
        ]
        tokens = {
            phase: sum(i.num_query * i.multiplicity for i in interactions if i.phase == phase)
            for phase in ("prefill", "decode")
        }
        steps = {
            phase: sum(i.multiplicity for i in interactions if i.phase == phase)
            for phase in ("prefill", "decode")
        }
        sampled = sum(steps.values()) if endpoint_head else steps["prefill"] + tokens["decode"]
        return Workload(
            matmul_tokens=sum(tokens.values()),
            head_positions=sampled,
            attn=interactions,
            attention_step_count=sum(steps.values()),
            attention_step_count_by_phase=steps,
            attention_tokens_by_phase=tokens,
        )

    result = workload(target)
    checks = {
        "matmul_tokens": result.matmul_tokens,
        "prefill_tokens": result.attention_tokens_by_phase["prefill"],
        "decode_passes": result.attention_step_count_by_phase["decode"],
        "prefill_requests": result.attention_step_count_by_phase["prefill"],
        "prefill_pairs": sum(i.pairs() for i in result.attn if i.phase == "prefill"),
        "prefill_cached": sum(
            i.num_cached_key * i.multiplicity for i in result.attn if i.phase == "prefill"
        ),
    }
    for name, expected in checks.items():
        actual = totals.get(name)
        if not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, rel_tol=1e-10, abs_tol=1e-8
        ):
            raise ValueError(f"speculative geometry disagrees with {name}: {actual} != {expected}")
    if proposer == "dflash2":
        stages = _dflash2_stages(result, endpoints, depth)
    else:
        stages = {DRAFT_FIRST_STAGE: workload(target, endpoint_head=True)}
        if depth > 1:
            stages[DRAFT_RECURRENT_STAGE] = workload(recurrent, endpoint_head=True)
    return replace(result, stages=stages)


def _dflash2_stages(target: Workload, endpoints: dict, depth: int) -> dict[str, Workload]:
    """The three DFlash2 stages of one step (see :mod:`model.work.models.dflash2`).

    Every scheduled request drafts, prefilling or verifying alike, against the
    context it ends the step with. The context stage consumes every row the
    target ran; the draft stage forwards ``k + 1`` rows per request; the
    selector scores the ``k`` drafted ones.
    """
    width = depth + 1
    requests = sum(endpoints.values())
    blocks = [
        AttnInteraction(width, context + width, context, "full", None, count)
        for context, count in sorted(endpoints.items())
    ]

    def stage(tokens, heads=0, attn=()):
        return Workload(
            matmul_tokens=tokens,
            head_positions=heads,
            attn=list(attn),
            attention_step_count=requests,
        )

    return {
        dflash2.CONTEXT_STAGE: stage(target.matmul_tokens),
        dflash2.DRAFT_STAGE: stage(requests * width, attn=blocks),
        dflash2.SELECT_STAGE: stage(requests * depth, heads=requests * depth),
    }
