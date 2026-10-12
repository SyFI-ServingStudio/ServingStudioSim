"""CPU-only snapshots of stock SchedulerOutput's pre-increment request progress."""

from __future__ import annotations

import hashlib
import json
import math
import struct

# Same pinned stock stack as profiling.runners.neuron.vllm_forward, and below
# the same pool as profiling.runners.neuron.stock_contract. This module runs in
# the stock image's workers, where only alignment/ is mounted, so it carries
# these metadata-only values rather than importing profiler code.
STOCK_VERSIONS = {
    "vllm": "0.24.0",
    "vllm-neuron": "0.24.0.1.1.0",
    "torch": "2.11.0",
    "transformers": "5.15.0",
    "libtorch-neuronx-lite": "2.11.0.1.0.1284+f49d8626",
    "neuronx-cc": "2.27.5334.0+f702b353",
    "nki": "0.6.0+31049202112.g85070674",
}


#: Additive sub-version of the schema-v2 EngineCore request-timing record. The
#: record's own `schema_version` stays 2: the shared extractor promotes an
#: API-joined row to 3 and the analyzer accepts only 1-3, so a new top-level
#: number would collide. Records without this key predate the arrival fields.
#: Sub-version 2 adds the API-server hand-off fields (`HANDOFF_TIMING_FIELDS`);
#: sub-version 1 rows lack them.
ARRIVAL_TIMING_SCHEMA_VERSION = 2

#: Absolute stamps (one monotonic clock; the epoch value is kept for audit)
#: plus the durations derived from them. All null when admission was not seen.
ARRIVAL_TIMING_FIELDS = (
    "frontend_arrival_epoch_ns",
    "frontend_arrival_monotonic_ns",
    "engine_receive_monotonic_ns",
    "frontend_arrival_to_engine_receive_ms",
    "engine_arrival_to_queued_ms",
    "server_ttft_from_arrival_ms",
)

#: Arrival sub-version 2: the API server's send stamp (VllmNeuronHandoff, from
#: vLLM's public request log) splits frontend_arrival_to_engine_receive_ms into
#: API-side and EngineCore-side parts. EngineCore cannot see the API process, so
#: the observer emits them null; postprocessing joins the stamps by engine id.
HANDOFF_TIMING_FIELDS = (
    "api_send_monotonic_ns",
    "frontend_arrival_to_api_send_ms",
    "api_send_to_engine_receive_ms",
)


def handoff_fields(row: dict, stamp: dict) -> dict:
    """Place one VllmNeuronHandoff send stamp between arrival and receive."""
    send = stamp["send_monotonic_ns"]
    if isinstance(send, bool) or not isinstance(send, int) or send <= 0:
        raise ValueError("invalid VllmNeuronHandoff send_monotonic_ns")
    arrival, receive = row["frontend_arrival_monotonic_ns"], row["engine_receive_monotonic_ns"]
    if arrival is None or receive is None:
        raise ValueError("hand-off stamp needs the record's arrival and receive stamps")
    return {
        "api_send_monotonic_ns": send,
        "frontend_arrival_to_api_send_ms": (send - arrival) / 1e6,
        "api_send_to_engine_receive_ms": (receive - send) / 1e6,
    }


def admission_stamps(arrival_time: float, receive_epoch_ns: int, receive_monotonic_ns: int) -> dict:
    """Place vLLM's wall-clock `Request.arrival_time` on the monotonic clock.

    The API-server process stamps `arrival_time` with `time.time()`; EngineCore
    events and output batches use `time.monotonic()`. CLOCK_MONOTONIC is
    system-wide, so one (epoch, monotonic) pair sampled at EngineCore receive
    converts the frontend stamp by its epoch distance to that pair.
    """
    if (
        isinstance(arrival_time, bool)
        or not isinstance(arrival_time, (int, float))
        or not math.isfinite(arrival_time)
        or arrival_time <= 0
    ):
        raise ValueError("invalid vLLM Request.arrival_time")
    arrival_epoch_ns = round(arrival_time * 1e9)
    return {
        "frontend_arrival_epoch_ns": arrival_epoch_ns,
        "frontend_arrival_monotonic_ns": receive_monotonic_ns
        - (receive_epoch_ns - arrival_epoch_ns),
        "engine_receive_monotonic_ns": receive_monotonic_ns,
    }


class RequestTimings:
    """Observe native EngineCore events without inventing a replacement clock."""

    def __init__(self):
        self.pending: dict[str, dict] = {}
        self.completed: set[str] = set()

    def admit(self, request_id: str, stamps: dict) -> None:
        """Record the first scheduler admission; later streaming updates keep it."""
        if not request_id or request_id in self.completed:
            raise ValueError("duplicate completed or empty engine request")
        self.pending.setdefault(request_id, {"count": 0}).setdefault("admission", stamps)

    @staticmethod
    def _arrival_fields(admission: dict | None, queued: float, first: float) -> dict:
        if admission is None:
            return dict.fromkeys((*ARRIVAL_TIMING_FIELDS, *HANDOFF_TIMING_FIELDS))
        arrival = admission["frontend_arrival_monotonic_ns"]
        receive = admission["engine_receive_monotonic_ns"]
        return {
            **admission,
            "frontend_arrival_to_engine_receive_ms": (receive - arrival) / 1e6,
            "engine_arrival_to_queued_ms": (queued * 1e9 - arrival) / 1e6,
            "server_ttft_from_arrival_ms": (first * 1e9 - arrival) / 1e6,
            **dict.fromkeys(HANDOFF_TIMING_FIELDS),
        }

    def observe(self, batches: dict) -> list[dict]:
        records = []
        for batch in batches.values():
            timestamp = batch.timestamp
            if not math.isfinite(timestamp) or timestamp <= 0:
                raise ValueError("invalid native EngineCore output timestamp")
            for output in batch.outputs:
                rid = output.request_id
                if not rid or rid in self.completed:
                    raise ValueError("duplicate completed or empty engine request")
                state = self.pending.setdefault(rid, {"count": 0})
                for event in output.events or ():
                    name = event.type.name
                    if name not in {"QUEUED", "SCHEDULED", "PREEMPTED"}:
                        raise ValueError("unsupported EngineCore request event")
                    if not math.isfinite(event.timestamp) or not 0 < event.timestamp <= timestamp:
                        raise ValueError("invalid native EngineCore event timestamp")
                    if name in {"QUEUED", "SCHEDULED"}:
                        # Preserve original boundaries across preemption/rescheduling.
                        state.setdefault(name, event.timestamp)
                if output.new_token_ids:
                    if not 0 < state.get("QUEUED", 0) <= state.get("SCHEDULED", 0) <= timestamp:
                        raise ValueError("token output lacks ordered native queue/schedule events")
                    if timestamp < state.get("last", timestamp):
                        raise ValueError("out-of-order native token output")
                    state.setdefault("first", timestamp)
                    state["last"] = timestamp
                    state["count"] += len(output.new_token_ids)
                if output.finished:
                    if output.finish_reason.name not in {"STOP", "LENGTH"} or not state["count"]:
                        raise ValueError("aborted, failed, or empty request cannot supply timing")
                    queued, scheduled = state["QUEUED"], state["SCHEDULED"]
                    first, last, count = state["first"], state["last"], state["count"]
                    decode_ms = (last - first) * 1000
                    records.append(
                        {
                            "schema_version": 2,
                            "engine_request_id": rid,
                            "engine_queue_wait_ms": (scheduled - queued) * 1000,
                            "engine_first_schedule_to_first_token_ms": (first - scheduled) * 1000,
                            "engine_core_ttft_ms": (first - queued) * 1000,
                            "engine_core_decode_ms": decode_ms,
                            "engine_core_tpot_ms": decode_ms / (count - 1) if count > 1 else None,
                            "num_output_tokens": count,
                            "arrival_timing_schema_version": ARRIVAL_TIMING_SCHEMA_VERSION,
                            **self._arrival_fields(state.get("admission"), queued, first),
                        }
                    )
                    del self.pending[rid]
                    self.completed.add(rid)
            # Native control-only completion must not silently hide an abort.
            if any(rid not in self.completed for rid in batch.finished_requests or ()):
                raise ValueError("engine request finished without eligible token completion")
        return records


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def scheduler_snapshot(output) -> dict:
    """Never read mutated scheduler Request progress or device tensor contents."""
    progress = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
    cached = output.scheduled_cached_reqs
    if cached is None:
        cached_ids, cached_progress = [], []
    else:
        cached_ids, cached_progress = cached.req_ids, cached.num_computed_tokens
    if len(cached_ids) != len(cached_progress):
        raise ValueError("cached request progress length mismatch")
    for rid, computed in zip(cached_ids, cached_progress, strict=True):
        if rid in progress:
            raise ValueError("duplicate scheduled request")
        progress[rid] = computed
    padded = getattr(output, "num_scheduled_tokens_padded", output.num_scheduled_tokens)
    requests = []
    for rid, q in output.num_scheduled_tokens.items():
        if rid not in progress:
            raise ValueError("scheduled request lacks pre-increment progress")
        kv, pq = progress[rid], padded.get(rid, q)
        if (
            not isinstance(rid, str)
            or not rid
            or type(kv) is not int
            or kv < 0
            or type(q) is not int
            or q < 1
            or type(pq) is not int
            or pq < q
        ):
            raise ValueError("invalid scheduled request geometry")
        requests.append(
            {"request_id": rid, "kv_len_before": kv, "q_tokens": q, "padded_q_tokens": pq}
        )
    fingerprint = hashlib.sha256(
        json.dumps(requests, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"requests": requests, "fingerprint": fingerprint}


def decode_bucket(count: int, buckets: tuple[int, ...]) -> int:
    """The stock scheduler pads a decode batch to its smallest configured bucket."""
    for bucket in buckets:
        if bucket >= count:
            return bucket
    raise ValueError("stock Neuron decode batch exceeds the largest configured bucket")


def iteration_metric(
    snapshot: dict,
    *,
    prompt_lengths: dict[str, int],
    iteration: int,
    start_ns: int,
    stop_ns: int,
    context: int,
    decode_buckets: tuple[int, ...],
) -> dict:
    """Classify one stock scheduler step against the server's compiled geometry.

    `context` is the single full prefill bucket (= max_model_len) and
    `decode_buckets` the resolved stock `num_seqs_buckets`; the observer reads
    both from the running scheduler rather than trusting a host-side copy.
    """
    buckets = tuple(decode_buckets)
    if (
        type(context) is not int
        or context < 1
        or not buckets
        or any(type(b) is not int or b < 1 for b in buckets)
        or list(buckets) != sorted(set(buckets))
    ):
        raise ValueError("invalid stock compiled context or decode buckets")
    prefills, decode, details = [], [], []
    for row in snapshot["requests"]:
        rid, kv, q = row["request_id"], row["kv_len_before"], row["q_tokens"]
        prompt = prompt_lengths[rid]
        if kv < prompt:
            if kv != 0 or q != prompt or row["padded_q_tokens"] != context:
                raise ValueError(f"stock capture requires full padded C{context} prefills")
            prefills.append([kv, q])
        else:
            if q != 1 or row["padded_q_tokens"] != 1 or not 1 <= kv < context:
                raise ValueError(f"stock capture requires one-token decode within C{context}")
            decode.append(kv + q)
        details.append({**row, "prompt_tokens": prompt})
    if not details or (prefills and decode) or len(prefills) > 1 or len(decode) > buckets[-1]:
        raise ValueError("stock Neuron forbids mixed phases or oversized batches")
    if stop_ns <= start_ns:
        raise ValueError("invalid scheduler monotonic observation")
    phase = "prefill" if prefills else "decode"
    return {
        "schema_version": 2,
        "input_adapter": "vllm_neuron_text",
        "dp_rank": 0,
        "iteration_index": iteration,
        "prefill_tokens": sum(q for _, q in prefills),
        "decode_requests": len(decode),
        "decode_tokens_scheduled": len(decode),
        "prefill_chunk_pairs": prefills,
        "decode_kv_lens": decode,
        "observed_start_monotonic_ns": start_ns,
        "observed_end_monotonic_ns": stop_ns,
        "observed_elapsed_ms": (stop_ns - start_ns) / 1e6,
        "fingerprint": snapshot["fingerprint"],
        "requests": details,
        "phase": phase,
        "compiled_shapes": {
            "context": context,
            "token_bucket": context if prefills else decode_bucket(len(decode), buckets),
            "tp_size": 4,
            "kv_blocks": 6782,
            "block_size": 32,
        },
    }
