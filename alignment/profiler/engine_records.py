"""What each serving engine's alignment log lines look like, and what they must carry.

The alignment records themselves are engine-independent by design: an iteration
record describes a batch shape, a request-timing record describes one request's
latency split. But the *log* they arrive on is not. Each engine wraps its lines
with its own rank prefix and wraps request ids in its own envelope, and each one
can only report the phases it actually has.

So the parsers take an `EngineRecords` and stay otherwise identical. Adding an
engine means adding a descriptor, not another copy of the extraction code.

The descriptor also owns scheduler-record multiplicity. Some engines emit one
copy per replica; others run one scheduler per tensor-parallel rank and repeat
the same iteration, request completion, and reduced expert counts on every TP
rank. The extractor must keep one owner per DP group or it multiplies scheduler
evidence by ``tp_size``.

The field sets are the interesting part. `API_CORE_DURATION_FIELDS` is the split
every engine can produce -- prepare, wait for first output, receive span, tail --
and is required of all of them. Everything else is one engine's internal
plumbing: vLLM's output collector and its per-request generator have measurable
handoffs, and SGLang, which hands off through its own IPC instead, has nothing
to report there. Requiring those of every engine would mean either rejecting
SGLang rows or inventing numbers for them, so they are declared per engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

ExpertCountReductionScope = Literal["expert_parallel", "tensor_parallel"]

#: Phases of the API-server span that any engine can time, because each is
#: bounded by two events every OpenAI-compatible frontend necessarily has.
API_CORE_DURATION_FIELDS = frozenset(
    {
        "api_frontend_prepare_ms",
        "api_first_output_wait_ms",
        "api_token_output_receive_span_ms",
        "api_terminal_tail_ms",
    }
)

#: Phases between accepting a request and the engine seeing it. Both engines
#: dispatch, so both report these.
API_DISPATCH_DURATION_FIELDS = frozenset(
    {
        "api_stream_activation_ms",
        "api_add_request_ms",
    }
)

#: vLLM's `RequestOutputCollector` / per-request async generator handoffs, and
#: the SSE serialization split around them. No SGLang counterpart: it moves
#: outputs over its own IPC and the phases are not separable from outside.
_VLLM_ONLY_API_FIELDS = frozenset(
    {
        "api_first_output_serialize_ms",
        "api_token_sse_yield_span_ms",
        "api_collector_wait_ms",
        "api_collector_wakeup_ms",
        "api_generator_resume_ms",
        "api_engine_output_wait_ms",
        "api_output_fanout_ms",
        "token_events",
        "first_token_event_tokens",
    }
)


@dataclass(frozen=True)
class EngineRecords:
    """How one engine's alignment records appear in its server log."""

    #: The `input_adapter` tag its records carry.
    adapter: str
    #: Pulls a data-parallel rank out of a log line's prefix. `None` when the
    #: engine writes no rank prefix, in which case every line is rank 0 and a
    #: `dp_size > 1` capture cannot be attributed.
    rank_prefix_re: re.Pattern | None
    #: How the prefix is spelled, for the error when a line is missing one.
    rank_prefix_label: str = "rank"
    #: Captures the local rank among processes that repeat one scheduler event.
    #: Rank 0 owns the canonical copy. ``None`` means records are already unique.
    scheduler_record_rank_re: re.Pattern | None = None
    #: Unwraps the engine's request-id envelope back to the id the client sent.
    #: Tried in order; the first full match wins, otherwise the id is used as-is.
    request_id_unwrappers: tuple[re.Pattern, ...] = ()
    #: API-timing fields this engine reports beyond the core + dispatch sets.
    extra_api_duration_fields: frozenset[str] = field(default_factory=frozenset)
    #: Whether each worker process states its own device and rank. When it does,
    #: that statement is the device <-> rank mapping; when it does not, the
    #: mapping is recovered by joining a pid<->rank banner with the profiler's
    #: pid<->device knowledge.
    emits_worker_records: bool = False
    #: Group whose already-summed counts one expert-load record represents.
    #: This is independent of expert sharding: SGLang's recorder reduces on
    #: the default process group of one TP replica, while vLLM's EPLB recorder
    #: reduces on its EP group. The plausibility ceiling uses this population.
    expert_count_reduction_scope: ExpertCountReductionScope = "expert_parallel"
    #: Whether popularity profiling must state the actual expert sharding
    #: degree instead of using vLLM's historical full-world default.
    requires_explicit_expert_parallel_size: bool = False

    def expert_count_reduction_group_size(
        self,
        *,
        tensor_parallel_size: int,
        expert_parallel_size: int,
    ) -> int:
        if self.expert_count_reduction_scope == "tensor_parallel":
            return tensor_parallel_size
        return expert_parallel_size

    def api_required_fields(self, schema_version: int) -> frozenset[str]:
        required = {
            "schema_version",
            "api_request_id",
            "output_tokens",
            *API_CORE_DURATION_FIELDS,
        }
        if schema_version >= 2:
            required |= API_DISPATCH_DURATION_FIELDS
        if schema_version >= 3:
            required |= self.extra_api_duration_fields
        return frozenset(required)

    def unwrap_request_id(self, request_id: str) -> str:
        for pattern in self.request_id_unwrappers:
            matched = pattern.fullmatch(request_id)
            if matched is not None:
                return matched.group(1)
        return request_id

    def owns_scheduler_record(self, line: str) -> bool:
        """Whether this emitter owns the one scheduler-side record to retain."""
        if self.scheduler_record_rank_re is None:
            return True
        matched = self.scheduler_record_rank_re.match(line)
        if matched is None or matched.group(1) is None:
            # A tp_size=1 run writes no TP tag, so its sole scheduler owns it.
            return True
        return int(matched.group(1)) == 0

    def rank_of(self, line: str, *, dp_size: int) -> int:
        if self.rank_prefix_re is None:
            if dp_size > 1:
                raise ValueError(
                    f"{self.adapter} writes no data-parallel rank prefix, so a "
                    f"dp_size={dp_size} capture cannot attribute its records"
                )
            return 0
        prefix = self.rank_prefix_re.match(line)
        if prefix is None or prefix.group(1) is None:
            if dp_size > 1:
                raise ValueError(
                    f"alignment record carries no {self.rank_prefix_label} log prefix, so "
                    f"its DP rank cannot be established in a dp_size={dp_size} capture: "
                    f"{line[:120]!r}"
                )
            return 0
        return int(prefix.group(1))


VLLM_RECORDS = EngineRecords(
    adapter="vllm_text",
    # `(EngineCore_DP3 pid=123)`; the plain `(EngineCore pid=123)` form is a
    # single-engine run, which lands on rank 0.
    rank_prefix_re=re.compile(r"^\(EngineCore(?:_DP(\d+))?\s+pid=(\d+)\)"),
    rank_prefix_label="EngineCore_DP<k>",
    # OpenAI completions wraps the caller's X-Request-Id as `cmpl-<id>-0` on the
    # engine side and `cmpl-<id>` on the API side; the tokens endpoint uses its
    # own `generate-tokens-<id>`.
    request_id_unwrappers=(
        re.compile(r"^cmpl-(.+)-0$"),
        re.compile(r"^cmpl-(.+)$"),
        re.compile(r"^generate-tokens-(.+)$"),
    ),
    extra_api_duration_fields=_VLLM_ONLY_API_FIELDS,
)

SGLANG_RECORDS = EngineRecords(
    adapter="sglang_text",
    # `[2026-08-11 16:18:16 DP1] ...`. The rank tag is only added when the run
    # has data parallelism, so a single-rank log carries a bare timestamp and
    # lands on rank 0 -- the same shape as vLLM's unnumbered `(EngineCore ...)`.
    rank_prefix_re=re.compile(r"^\[[^\]]*\sDP(\d+)\b[^\]]*\]"),
    rank_prefix_label="[... DP<k>]",
    # SGLang prefixes scheduler processes with their rank inside each TP group.
    # All ranks repeat the same scheduler records, so TP0 owns the retained copy.
    scheduler_record_rank_re=re.compile(r"^\[[^\]]*\sTP(\d+)\b[^\]]*\]"),
    # SGLang passes the caller's request id through unwrapped.
    request_id_unwrappers=(),
    extra_api_duration_fields=frozenset(),
    emits_worker_records=True,
    expert_count_reduction_scope="tensor_parallel",
    requires_explicit_expert_parallel_size=True,
)

RECORDS_BY_ADAPTER = {
    VLLM_RECORDS.adapter: VLLM_RECORDS,
    SGLANG_RECORDS.adapter: SGLANG_RECORDS,
}
