"""Geometry and evidence contract for one stock prefill plus one decode guard.

The controller never builds model inputs or changes scheduler output. The worker
adapter supplies the stock profiler and public asynchronous output completion.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

METHOD = "geometry_drained_prefill_decode_v1"
WORKER_CLASS = "profiling.runners.neuron.vllm_ops_worker.OpsNeuronWorker"
SOURCE_PINS = {
    "vllm_neuron/vllm/worker/neuron_worker.py": (
        "d508ea279f993d658a71ca9179d7ddb1c5e7a43ab2fcc13c4e8ada0472ecfeed"
    ),
    "vllm_neuron/vllm/worker/neuron_model_runner.py": (
        "aa4f45c977bb6cc3e3a3e838c1b06fed643eb36ebfda4e04942df0a30a3b2562"
    ),
    "vllm_neuron/vllm/worker/neuron_profiler.py": (
        "90e53f1278c8fe8de13c77906769f36ae037720c0a96277b1b84190ddb3f6317"
    ),
    "vllm/profiler/wrapper.py": "31f8a33afd23e67a2ade5a95d0d837a03b0a9164d5b18acf561d948c9ae6c06c",
    "vllm/v1/worker/worker_base.py": (
        "7da44338c2645ebf03d23394e452b31a8e3da1011fd1b42fcfcccfe99551b3fe"
    ),
    "vllm/v1/executor/multiproc_executor.py": (
        "80bde58c8336427848f87df053d756dafc88e5df6ef80df661ecf6f85d8de7e2"
    ),
    "vllm/config/parallel.py": "aba37b261236a7dceb11f9dca5ce09ddcc818f06bce13f487580dd958e94cf53",
    "vllm_neuron/vllm/platform.py": (
        "f541ae400fe95fff9a682d3ec2a9882b83cf6ac036638f2dc56eafb8e2e8ef66"
    ),
    "vllm_neuron/vllm/core/scheduler.py": (
        "11f0b36d1ef1811a1243579f99a82860cd2a5fc3d3ea8137aeba868ae0a08eb5"
    ),
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_runtime_sources(site: Path) -> dict:
    observed = {name: file_hash(site / name) for name in SOURCE_PINS}
    if observed != SOURCE_PINS:
        changed = [name for name in SOURCE_PINS if observed[name] != SOURCE_PINS[name]]
        raise ValueError(f"unsupported Neuron capture control source: {changed}")
    return observed


def prompt_hash(tokens: list[int]) -> str:
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("capture control requires concrete integer prompt tokens")
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def validate_control_receipt(receipt: dict, shape: dict, ranks: list[int]) -> None:
    """Validate the hash-bound host control evidence; native trace checks are separate."""
    if (
        receipt.get("schema_version") != 1
        or receipt.get("method") != METHOD
        or receipt.get("passed") is not True
        or shape["phase"] != "prefill"
        or receipt.get("selected_ranks") != sorted(ranks)
    ):
        raise ValueError("invalid capture control receipt identity or status")
    workers = receipt.get("workers", [])
    if [row.get("rank") for row in workers] != list(range(4)):
        raise ValueError("capture control requires every TP4 worker receipt")
    expected_multiset = None
    expected_geometries = None
    for row in workers:
        if (
            row.get("method") != METHOD
            or row.get("passed") is not True
            or row.get("state") != "complete"
            or row.get("local_rank") != row["rank"]
            or row.get("selected_ranks") != sorted(ranks)
            or row.get("profiled") is not (row["rank"] in ranks)
            or row.get("source_hashes") != SOURCE_PINS
            or row.get("canonical_requests") != 16
            or row.get("admitted_requests") != 16
            or row.get("selected_geometry")
            != {"phase": "prefill", "token_bucket": shape["token_bucket"]}
            or row.get("guard_geometry") != {"phase": "decode", "token_bucket": 16}
            or row.get("synchronization")
            != "previous_output_before_start; decode_output_before_stop"
        ):
            raise ValueError("capture control worker identity, geometry or source mismatch")
        hooks = row.get("hook_source_hashes", {})
        if set(hooks) != {"vllm_ops_control.py", "vllm_ops_worker.py"} or any(
            not re.fullmatch(r"[0-9a-f]{64}", value) for value in hooks.values()
        ):
            raise ValueError("capture control lacks hook source hashes")
        multiset = row.get("expected_prompt_multiset", {})
        if (
            sum(multiset.values()) != 16
            or any(type(n) is not int or n < 1 for n in multiset.values())
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in multiset)
            or row.get("observed_prompt_multiset") != multiset
            or (expected_multiset is not None and expected_multiset != multiset)
        ):
            raise ValueError("capture control prompt multiset differs between workers")
        expected_multiset = multiset
        events = row.get("events", [])
        actions = [event.get("action") for event in events]
        boundary = [
            action
            for action in actions
            if action not in ("geometry", "sample_returned", "empty_step")
        ]
        if boundary != [
            "armed",
            "previous_output_drain_begin",
            "previous_output_drain_complete",
            "profile_started",
            "guard_submit",
            "guard_output_drain_begin",
            "guard_output_drain_complete",
            "profile_stopped",
            "external_stop_verified",
        ]:
            raise ValueError("capture control start/drain/stop event order is invalid")
        times = [event.get("monotonic_ns") for event in events]
        if any(type(t) is not int for t in times) or times != sorted(times):
            raise ValueError("capture control event timestamps are invalid")
        geometries = [event for event in events if event["action"] == "geometry"]
        sampled = [event for event in events if event["action"] == "sample_returned"]
        keys = ("phase", "token_bucket", "request_ids", "prompt_sha256", "prefill_ordinal",
                "scope", "scheduled_batch_size", "computed_tokens", "scheduled_tokens",
                "padded_tokens", "cleanup_finished_ids")
        observed_geometries = [{k: e.get(k) for k in keys} for e in geometries]
        if len(geometries) < 17 or observed_geometries != [
            {k: e.get(k) for k in keys} for e in sampled
        ]:
            raise ValueError("capture control geometry/sample events differ")
        if expected_geometries is not None and observed_geometries != expected_geometries:
            raise ValueError("capture control worker geometry sequences differ")
        expected_geometries = observed_geometries
        for index, (geometry, sample) in enumerate(zip(geometries, sampled)):
            following = (events.index(geometries[index + 1])
                         if index + 1 < len(geometries) else len(events))
            if not events.index(geometry) < events.index(sample) < following:
                raise ValueError("capture control sampled-output ordering is invalid")
        prefills = [event for event in geometries if event.get("phase") == "prefill"]
        decode = geometries[-1]
        ids = [event.get("request_ids") for event in prefills]
        if (
            len(prefills) != 16 or any(
                event.get("phase") != "prefill"
                or event.get("token_bucket") != shape["token_bucket"]
                or event.get("prefill_ordinal") != index + 1
                for index, event in enumerate(prefills)
            )
            or any(not isinstance(values, list) or len(values) != 1 for values in ids)
            or len({values[0] for values in ids}) != 16
            or dict(Counter(event.get("prompt_sha256") for event in prefills)) != multiset
            or decode.get("phase") != "decode"
            or decode.get("token_bucket") != 16
            or decode.get("request_ids") != sorted(values[0] for values in ids)
        ):
            raise ValueError("capture control does not prove canonical P16 then D16")
        canonical_ids = {values[0] for values in ids}
        if any(set(event.get("cleanup_finished_ids", ())) & canonical_ids for event in events):
            raise ValueError("capture control contains a finished canonical request")
        # Earlier receipts contain only P*16,D. New receipts additionally prove
        # any naturally interleaved decode work before the profiler starts.
        if "decode_buckets" in row:
            _validate_progression(row, geometries, shape["token_bucket"])
        elif len(geometries) != 17:
            raise ValueError("legacy capture control cannot contain pre-capture decode steps")
        positions = {action: actions.index(action) for action in boundary}
        last_p, first_d = events.index(prefills[-1]), events.index(decode)
        p_sample, d_sample = events.index(sampled[-2]), events.index(sampled[-1])
        if not (
            last_p
            < positions["previous_output_drain_begin"]
            < positions["previous_output_drain_complete"]
            < positions["profile_started"]
            < p_sample
            < first_d
            < positions["guard_submit"]
            < d_sample
            < positions["guard_output_drain_begin"]
        ):
            raise ValueError("capture control boundaries do not enclose exactly P,D")


def _validate_progression(row, geometries, bucket):
    buckets = row["decode_buckets"]
    if (not buckets or any(type(value) is not int or not 1 <= value <= 16 for value in buckets)
            or buckets != sorted(set(buckets)) or buckets[-1] != 16):
        raise ValueError("capture control decode bucket configuration is invalid")
    computed, decode_counts, early_steps = {}, Counter(), 0
    for index, event in enumerate(geometries):
        ids = event["request_ids"]
        actual = event.get("scheduled_tokens", {})
        padded = event.get("padded_tokens", {})
        before = event.get("computed_tokens", {})
        if event.get("scheduled_batch_size") != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("capture control scheduled batch size is invalid")
        if event["phase"] == "prefill":
            request_id = ids[0]
            if (request_id in computed or actual != {request_id: bucket - 8}
                    or padded != {request_id: bucket} or before != {request_id: 0}):
                raise ValueError("capture control prefill token progression is invalid")
            computed[request_id] = bucket - 8
        else:
            if (event["phase"] != "decode" or not ids or not set(ids) <= computed.keys()
                    or actual != dict.fromkeys(ids, 1) or padded != actual
                    or before != {request_id: computed[request_id] for request_id in ids}
                    or event["token_bucket"] != next(
                        value for value in buckets if value >= len(ids))):
                raise ValueError("capture control decode token progression is invalid")
            if event.get("scope") == "pre_capture":
                if not 0 < len(computed) < 16:
                    raise ValueError("pre-capture decode occurred inside the captured window")
                early_steps += 1
                for request_id in ids:
                    decode_counts[request_id] += 1
                    if decode_counts[request_id] > 7:
                        raise ValueError("pre-capture decode exceeds the canonical output budget")
            elif event.get("scope") != "guard" or index != len(geometries) - 1:
                raise ValueError("capture control has an undeclared decode execution")
            for request_id in ids:
                computed[request_id] += 1
    if row.get("pre_capture_decode_steps") != early_steps:
        raise ValueError("capture control pre-capture decode count differs from its events")


def collect_control_receipt(root: Path, shape: dict, ranks: list[int]) -> Path:
    workers = [
        json.loads((root / "capture-control" / f"rank-{rank}.json").read_text())
        for rank in range(4)
    ]
    receipt = {
        "schema_version": 1,
        "method": METHOD,
        "passed": True,
        "selected_ranks": sorted(ranks),
        "workers": workers,
    }
    validate_control_receipt(receipt, shape, ranks)
    plan = json.loads((root / "ops-plan.json").read_text())
    expected = dict(Counter(prompt_hash(tokens) for tokens in plan["case"]["prompts"]))
    hooks = {
        name: file_hash(Path(__file__).with_name(name))
        for name in ("vllm_ops_control.py", "vllm_ops_worker.py")
    }
    expected_buckets = plan["profile_config"]["additional_config"]["neuron_config"][
        "num_seqs_buckets"
    ]
    if any(
        row["expected_prompt_multiset"] != expected or row["hook_source_hashes"] != hooks
        or row["decode_buckets"] != expected_buckets
        for row in workers
    ):
        raise ValueError("capture control differs from the launched plan or hook source")
    path = root / "capture-control-receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


class GeometryCapture:
    """A single armed canonical generation, observed on every TP worker rank."""

    def __init__(
        self,
        *,
        prompts,
        bucket,
        rank,
        local_rank,
        selected_ranks,
        decode_buckets,
        receipt_path,
        source_hashes,
        hook_source_hashes,
    ):
        if len(prompts) != 16 or any(len(tokens) != bucket - 8 for tokens in prompts):
            raise ValueError("capture control requires 16 full canonical C-8 prompts")
        if source_hashes != SOURCE_PINS:
            raise ValueError("capture control requires verified pinned runtime sources")
        self.expected = Counter(prompt_hash(tokens) for tokens in prompts)
        self.remaining = self.expected.copy()
        self.bucket = bucket
        self.rank, self.local_rank = rank, local_rank
        self.selected_ranks = sorted(selected_ranks)
        self.decode_buckets = list(decode_buckets)
        if (not self.decode_buckets or self.decode_buckets != sorted(set(self.decode_buckets))
                or any(type(value) is not int or not 1 <= value <= 16
                       for value in self.decode_buckets) or self.decode_buckets[-1] != 16):
            raise ValueError("capture control requires canonical decode buckets ending at 16")
        self.path = Path(receipt_path)
        self.source_hashes = source_hashes
        self.hook_source_hashes = hook_source_hashes
        self.state = "idle"
        self.pending = None
        self.admitted = {}
        self.computed_after = {}
        self.pre_capture_decode_counts = Counter()
        self.pre_capture_decode_steps = 0
        self.events = []
        self.native_started = False
        self.passed = False

    @property
    def active(self):
        return self.state in ("armed", "prefill", "decode")

    def _event(self, action, **fields):
        self.events.append(
            {
                "action": action,
                "monotonic_ns": time.monotonic_ns(),
                "epoch_ns": time.time_ns(),
                **fields,
            }
        )
        self._save()

    def _save(self):
        value = {
            "schema_version": 1,
            "method": METHOD,
            "passed": self.passed,
            "state": self.state,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "profiled": self.local_rank in self.selected_ranks,
            "selected_ranks": self.selected_ranks,
            "expected_prompt_multiset": dict(self.expected),
            "observed_prompt_multiset": dict(self.expected - self.remaining),
            "canonical_requests": 16,
            "admitted_requests": len(self.admitted),
            "decode_buckets": self.decode_buckets,
            "pre_capture_decode_steps": self.pre_capture_decode_steps,
            "selected_geometry": {"phase": "prefill", "token_bucket": self.bucket},
            "guard_geometry": {"phase": "decode", "token_bucket": 16},
            "synchronization": "previous_output_before_start; decode_output_before_stop",
            "source_hashes": self.source_hashes,
            "hook_source_hashes": self.hook_source_hashes,
            "events": self.events,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(self.path)

    def arm(self):
        if self.state != "idle":
            raise ValueError("capture control can only be armed once")
        self.state = "armed"
        self._event("armed")

    def _geometry(self, output):
        counts = dict(output.num_scheduled_tokens)
        if (
            output.total_num_scheduled_tokens != sum(counts.values())
            or getattr(output, "scheduled_spec_decode_tokens", None)
            or getattr(output, "scheduled_encoder_inputs", None)
            or getattr(output, "preempted_req_ids", None)
            # Finishes from the prior warmup may arrive as bookkeeping. Any
            # finish within this cohort makes the required final D16 impossible.
            or set(getattr(output, "finished_req_ids", ()))
            & (self.admitted.keys() | counts.keys())
        ):
            raise ValueError("capture control received unsupported scheduler geometry")
        if not counts:
            if output.scheduled_new_reqs:
                raise ValueError("empty scheduler step contains a new request")
            return None
        padded = getattr(output, "num_scheduled_tokens_padded", {})
        new = output.scheduled_new_reqs
        if new:
            if self.state != "armed" or len(new) != 1:
                raise ValueError("capture requires serial complete prefills before decode")
            request = new[0]
            tokens = request.prompt_token_ids
            identity = prompt_hash(tokens)
            if (
                set(counts) != {request.req_id}
                or request.req_id in self.admitted
                or request.num_computed_tokens != 0
                or counts[request.req_id] != len(tokens)
                or padded.get(request.req_id) != self.bucket
                or self.remaining[identity] <= 0
                or getattr(request, "mm_features", None)
                or getattr(request, "lora_request", None)
            ):
                raise ValueError("prefill differs from the canonical prompt multiset or bucket")
            self.remaining[identity] -= 1
            self.admitted[request.req_id] = identity
            self.computed_after[request.req_id] = len(tokens)
            return {
                "phase": "prefill",
                "token_bucket": self.bucket,
                "request_ids": [request.req_id],
                "prompt_sha256": identity,
                "prefill_ordinal": len(self.admitted),
                "scheduled_batch_size": 1,
                "scheduled_tokens": counts,
                "padded_tokens": {request.req_id: padded[request.req_id]},
                "computed_tokens": {request.req_id: 0},
            }
        cached = output.scheduled_cached_reqs
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
        if (
            not set(counts) <= self.admitted.keys()
            or not set(cached.req_ids) <= self.admitted.keys()
            or any(type(n) is not int or n != 1 for n in counts.values())
            or any(padded.get(req_id) != 1 for req_id in counts)
            or len(set(cached.req_ids)) != len(cached.req_ids)
            or any(type(computed.get(req_id)) is not int
                   or computed[req_id] != self.computed_after[req_id] for req_id in counts)
        ):
            raise ValueError("decode differs from the admitted canonical request progression")
        if self.state == "armed":
            if not 0 < len(self.admitted) < 16 or any(
                self.pre_capture_decode_counts[req_id] >= 7 for req_id in counts
            ):
                raise ValueError("pre-capture decode exceeds the canonical output budget")
            self.pre_capture_decode_counts.update(counts)
            self.pre_capture_decode_steps += 1
            scope = "pre_capture"
        elif (self.state != "prefill" or len(self.admitted) != 16
              or any(self.remaining.values()) or set(counts) != self.admitted.keys()):
            raise ValueError("capture requires the first complete canonical decode16")
        else:
            scope = "guard"
        before = {req_id: computed[req_id] for req_id in counts}
        for req_id in counts:
            self.computed_after[req_id] += 1
        compiled_bucket = next(value for value in self.decode_buckets if value >= len(counts))
        return {"phase": "decode", "scope": scope, "request_ids": sorted(counts),
                "token_bucket": compiled_bucket,
                "scheduled_batch_size": len(counts), "scheduled_tokens": counts,
                "padded_tokens": {req_id: padded[req_id] for req_id in counts},
                "computed_tokens": before}

    def before_execute(self, output, previous, *, start, drain):
        if not self.active:
            return
        if self.pending is not None:
            raise ValueError("sample_tokens did not complete the previous worker call")
        geometry = self._geometry(output)
        cleanup = sorted(getattr(output, "finished_req_ids", ()))
        if geometry is None:
            self._event("empty_step", cleanup_finished_ids=cleanup)
            return
        geometry["cleanup_finished_ids"] = cleanup
        self.pending = geometry
        self._event("geometry", **geometry)
        if geometry.get("prefill_ordinal") == 16:
            self._event("previous_output_drain_begin")
            drain(previous)
            self._event("previous_output_drain_complete")
            # Mark before calling start so partial/failed starts still get cleanup.
            self.native_started = True
            start()
            self.state = "prefill"
            self._event("profile_started")
        elif geometry.get("scope") == "guard":
            self.state = "decode"
            self._event("guard_submit")

    def after_sample(self, output, *, check_output, drain, stop):
        if not self.active or self.pending is None:
            return
        check_output(output)
        geometry = self.pending
        self.pending = None
        self._event("sample_returned", **geometry)
        if geometry.get("scope") == "guard":
            self._event("guard_output_drain_begin")
            drain(output)
            self._event("guard_output_drain_complete")
            stop()
            self.native_started = False
            self.state = "complete"
            self._event("profile_stopped")

    def finish(self):
        if self.state != "complete":
            raise ValueError(f"capture control stopped without a complete P,D window: {self.state}")
        self.passed = True
        self._event("external_stop_verified")

    def abort(self, error, *, stop):
        self.state, self.passed = "failed", False
        if self.native_started:
            try:
                stop()
            except Exception as cleanup:
                self.events.append({"action": "cleanup_failed", "error": str(cleanup)})
            self.native_started = False
        self._event("failed", error=str(error))
