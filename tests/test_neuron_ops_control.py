"""CPU checks for geometry-selected stock worker capture and its evidence gates."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from collections import Counter
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from profiling.runners.neuron import vllm_ops_control as control


def scheduled(index=None, *, bucket=128, ids=None, computed=None):
    new = (
        []
        if index is None
        else [
            NS(
                req_id=f"r{index}",
                prompt_token_ids=[index + 1] * (bucket - 8),
                num_computed_tokens=0,
                mm_features=[],
                lora_request=None,
            )
        ]
    )
    ids = list(range(16)) if ids is None else ids
    counts = {f"r{i}": 1 for i in ids} if index is None else {f"r{index}": bucket - 8}
    padded = dict(counts) if index is None else {f"r{index}": bucket}
    return NS(
        scheduled_new_reqs=new,
        scheduled_cached_reqs=NS(
            req_ids=list(counts) if index is None else [],
            num_computed_tokens=[(computed or {}).get(req_id, bucket - 8) for req_id in counts]
            if index is None else [],
        ),
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        num_scheduled_tokens_padded=padded,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        preempted_req_ids=set(),
        finished_req_ids=set(),
    )


class AsyncOutput:
    def __init__(self, calls, identity, *, materialized=False):
        self.calls, self.identity = calls, identity
        self.materialized = materialized
        self.materializations = int(materialized)

    def is_all_partial_prefill(self):
        return False

    def get_output(self):
        self.calls.append(("drain", self.identity))
        if not self.materialized:
            self.materialized = True
            self.materializations += 1
        return self


@pytest.fixture
def worker_factory(tmp_path, monkeypatch):
    calls = []

    class StockWorker:
        def __init__(self, vllm_config, local_rank=0, rank=0, **kwargs):
            self.vllm_config = vllm_config
            self.local_rank, self.rank = local_rank, rank
            self.use_async_scheduling = True
            self._profiler = None
            self.fail = None
            self.materialized = False

        def profile(self, is_start=True):
            ranks = self.vllm_config.additional_config["neuron_profiler"]["neuron_cores"]
            if self.local_rank not in ranks:
                return
            calls.append(("start" if is_start else "stop", self.rank))
            state = is_start and self.fail != "start"
            self._profiler = NS(_running=state, _profiling=state)
            if not is_start and self.fail == "stop":
                self._profiler._profiling = True

        def execute_model(self, output):
            self.last_scheduler = output
            calls.append(("execute", tuple(output.num_scheduled_tokens)))
            if self.fail == "execute":
                raise RuntimeError("stock execution failed")
            return None

        def sample_tokens(self, grammar_output):
            calls.append(("sample", grammar_output))
            self.last_output = (
                None
                if self.fail == "sample"
                else AsyncOutput(
                    calls,
                    tuple(self.last_scheduler.num_scheduled_tokens),
                    materialized=self.materialized,
                )
            )
            if self.fail == "plain":
                self.last_output = object()
            if self.fail == "partial":
                self.last_output.is_all_partial_prefill = lambda: True
            return self.last_output

    modules = {
        "vllm_neuron.vllm.worker.neuron_worker": ("NeuronWorker", StockWorker),
        "vllm_neuron.vllm.worker.neuron_model_runner": (
            "AsyncNeuronModelRunnerOutput",
            AsyncOutput,
        ),
    }
    for name, (attr, value) in modules.items():
        module = ModuleType(name)
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    name = "profiling.runners.neuron.vllm_ops_worker"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setattr(
        module.importlib.util, "find_spec", lambda name: NS(origin="/fake/vllm/__init__.py")
    )
    monkeypatch.setattr(control, "verify_runtime_sources", lambda site: dict(control.SOURCE_PINS))
    plan = {
        "shape": {"phase": "prefill", "token_bucket": 128},
        "request": {"forwards": 1, "ranks": [0]},
        "case": {"prompts": [[i + 1] * 120 for i in range(16)]},
        "profile_config": {"additional_config": {
            "neuron_config": {"num_seqs_buckets": [1, 16]}}},
    }
    (tmp_path / "ops-plan.json").write_text(json.dumps(plan))

    def make(rank=0, selected=(0,), config_ranks=None):
        plan["request"]["ranks"] = list(selected)
        (tmp_path / "ops-plan.json").write_text(json.dumps(plan))
        config = NS(
            profiler_config=NS(delay_iterations=0, max_iterations=0),
            parallel_config=NS(tensor_parallel_size=4, pipeline_parallel_size=1),
            cache_config=NS(enable_prefix_caching=False),
            additional_config={
                "neuron_config": {"num_seqs_buckets": [1, 16]},
                "neuron_profiler": {
                    "output_dir": str(tmp_path / "device-profiles"),
                    "neuron_cores": list(selected) if config_ranks is None else config_ranks,
                }
            },
        )
        return module.OpsNeuronWorker(vllm_config=config, local_rank=rank, rank=rank)

    yield make, calls, tmp_path
    sys.modules.pop(name, None)


def step(worker, output):
    before = copy.deepcopy(output)
    assert worker.execute_model(output) is None
    result = worker.sample_tokens(None)
    assert result is worker.last_output
    assert vars(output) == vars(before)
    return result


def feed_prefills(worker, count=16):
    for index in range(count):
        step(worker, scheduled(index))


@pytest.mark.parametrize("rank,selected", [(0, (0,)), (1, (0,)), (2, (2,))])
@pytest.mark.parametrize("materialized", [False, True])
def test_public_worker_exact_geometry_drains_and_identical_outputs(
    worker_factory, rank, selected, materialized
):
    make, calls, root = worker_factory
    worker = make(rank, selected)
    # Warmup remains delegated, without capture state, geometry inspection or drains.
    step(worker, scheduled(0))
    assert worker._ops_control.state == "idle"
    calls.clear()
    worker.profile(True)
    assert not calls
    worker.materialized = materialized
    feed_prefills(worker)
    output = step(worker, scheduled())
    worker.profile(False)
    assert output.materializations == 1
    output.get_output()  # Executor output thread can safely read the same object again.
    assert output.materializations == 1
    assert [call[0] for call in calls].count("drain") == 3
    previous_drain = calls.index(("drain", ("r14",)))
    final_submit = calls.index(("execute", ("r15",)))
    guard_drain = calls.index(("drain", tuple(f"r{i}" for i in range(16))))
    if rank in selected:
        assert previous_drain < calls.index(("start", rank)) < final_submit
        assert guard_drain < calls.index(("stop", rank))
    else:
        assert not any(call[0] in ("start", "stop") for call in calls)
    receipt = json.loads((root / "capture-control" / f"rank-{rank}.json").read_text())
    assert receipt["passed"] and receipt["state"] == "complete"
    assert receipt["profiled"] is (rank in selected)
    assert receipt["expected_prompt_multiset"] == receipt["observed_prompt_multiset"]
    assert receipt["source_hashes"] == control.SOURCE_PINS
    # Subsequent decode keeps the normal async result; no extra drain or profiling.
    calls.clear()
    step(worker, scheduled())
    assert [call[0] for call in calls] == ["execute", "sample"]


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_prompt",
        "wrong_bucket",
        "duplicate",
        "early_decode",
        "partial_prefill",
        "wrong_decode",
        "missing_sample",
    ],
)
def test_unexpected_geometry_fails_closed_before_new_model_work(worker_factory, failure):
    make, calls, root = worker_factory
    worker = make()
    worker.profile(True)
    feed_prefills(worker, 1)
    output = scheduled(1)
    if failure == "wrong_prompt":
        output.scheduled_new_reqs[0].prompt_token_ids[0] = 999
    elif failure == "wrong_bucket":
        output.num_scheduled_tokens_padded["r1"] = 512
    elif failure == "duplicate":
        output = scheduled(0)
    elif failure == "early_decode":
        output = scheduled()
    elif failure == "partial_prefill":
        output.scheduled_new_reqs[0].num_computed_tokens = 1
    elif failure == "wrong_decode":
        for i in range(1, 16):
            step(worker, scheduled(i))
        output = scheduled()
        output.num_scheduled_tokens.pop("r15")
        output.total_num_scheduled_tokens -= 1
    elif failure == "missing_sample":
        worker.execute_model(output)
        output = scheduled(2)
    prior = len([call for call in calls if call[0] == "execute"])
    with pytest.raises(ValueError):
        worker.execute_model(output)
    assert len([call for call in calls if call[0] == "execute"]) == prior
    assert worker._ops_control.state == "failed"
    assert not json.loads((root / "capture-control/rank-0.json").read_text())["passed"]


@pytest.mark.parametrize("failure", ["start", "execute", "sample", "plain", "partial", "stop"])
def test_profiler_and_model_failures_cleanup_and_propagate(worker_factory, failure):
    make, calls, root = worker_factory
    worker = make()
    worker.profile(True)
    feed_prefills(worker, 15)
    if failure != "stop":
        worker.fail = failure
        with pytest.raises((ValueError, RuntimeError)):
            step(worker, scheduled(15))
    else:
        step(worker, scheduled(15))
        worker.fail = failure
        with pytest.raises(RuntimeError):
            step(worker, scheduled())
    assert any(call[0] == "stop" for call in calls)
    assert worker._ops_control.state == "failed"
    assert not json.loads((root / "capture-control/rank-0.json").read_text())["passed"]


def test_incomplete_external_stop_fails_and_cleans_up(worker_factory):
    make, calls, _ = worker_factory
    worker = make()
    worker.profile(True)
    feed_prefills(worker)
    with pytest.raises(ValueError, match="complete P,D"):
        worker.profile(False)
    assert calls[-1] == ("stop", 0)


def test_worker_rejects_native_rank_selection_different_from_plan(worker_factory):
    make, calls, _ = worker_factory
    with pytest.raises(ValueError, match="rank selection"):
        make(selected=(0,), config_ranks=[0, 1])
    assert calls == []


def interleaved_generation(worker, *, decode_immediately_before_last_prefill=False):
    worker.profile(True)
    step(worker, scheduled(0))
    step(worker, scheduled(ids=[0]))
    step(worker, scheduled(1))
    step(worker, scheduled(ids=[0, 1], computed={"r0": 121}))
    for index in range(2, 15):
        step(worker, scheduled(index))
    computed = {"r0": 122, "r1": 121}
    if decode_immediately_before_last_prefill:
        step(worker, scheduled(ids=list(range(15)), computed=computed))
        computed = {f"r{i}": computed.get(f"r{i}", 120) + 1 for i in range(15)}
    step(worker, scheduled(15))
    step(worker, scheduled(computed=computed))
    worker.profile(False)


@pytest.mark.parametrize("previous_decode", [False, True])
def test_interleaved_unprofiled_decode_preserves_exact_captured_window(worker_factory,
                                                                     previous_decode):
    make, calls, root = worker_factory
    for rank in range(4):
        interleaved_generation(make(rank), decode_immediately_before_last_prefill=previous_decode)
    receipt_path = control.collect_control_receipt(
        root, {"phase": "prefill", "token_bucket": 128}, [0]
    )
    receipt = json.loads(receipt_path.read_text())
    for row in receipt["workers"]:
        assert row["pre_capture_decode_steps"] == (3 if previous_decode else 2)
        prefix = [event for event in row["events"]
                  if event["action"] == "geometry" and event.get("scope") == "pre_capture"]
        assert [(event["scheduled_batch_size"], event["token_bucket"]) for event in prefix[:2]] == [
            (1, 1), (2, 16),
        ]
        first_start = next(i for i, event in enumerate(row["events"])
                           if event["action"] == "profile_started")
        assert all(row["events"].index(event) < first_start for event in prefix)
    start = calls.index(("start", 0))
    expected_previous = tuple(f"r{i}" for i in range(15)) if previous_decode else ("r14",)
    assert calls[start - 1] == ("drain", expected_previous)
    stop = calls.index(("stop", 0))
    assert [call for call in calls[start:stop] if call[0] == "execute"] == [
        ("execute", ("r15",)), ("execute", tuple(f"r{i}" for i in range(16))),
    ]


@pytest.mark.parametrize("failure", ["foreign", "stale", "skipped", "padding", "tokens",
                                     "finished", "preempted", "budget", "final_position"])
def test_precapture_decode_requires_exact_live_canonical_progression(worker_factory, failure):
    make, calls, _ = worker_factory
    worker = make()
    worker.profile(True)
    step(worker, scheduled(0))
    step(worker, scheduled(ids=[0]))
    output = scheduled(ids=[0], computed={"r0": 121})
    if failure == "foreign":
        output = scheduled(ids=[1])
    elif failure == "stale":
        output.scheduled_cached_reqs.num_computed_tokens = [120]
    elif failure == "skipped":
        output.scheduled_cached_reqs.num_computed_tokens = [122]
    elif failure == "padding":
        output.num_scheduled_tokens_padded["r0"] = 2
    elif failure == "tokens":
        output.num_scheduled_tokens["r0"] = 2
        output.total_num_scheduled_tokens = 2
    elif failure == "finished":
        output.finished_req_ids = {"r0"}
    elif failure == "preempted":
        output.preempted_req_ids = {"r0"}
    elif failure == "budget":
        for computed in range(121, 127):
            step(worker, scheduled(ids=[0], computed={"r0": computed}))
        output = scheduled(ids=[0], computed={"r0": 127})
    elif failure == "final_position":
        for index in range(1, 16):
            step(worker, scheduled(index))
        output = scheduled()  # r0 must now start at 121, not 120.
    prior = len([call for call in calls if call[0] == "execute"])
    with pytest.raises(ValueError):
        worker.execute_model(output)
    assert len([call for call in calls if call[0] == "execute"]) == prior
    assert worker._ops_control.state == "failed"


def test_finished_warmup_bookkeeping_is_not_foreign_scheduled_work(worker_factory):
    make, _, _ = worker_factory
    worker = make()
    worker.profile(True)
    output = scheduled(0)
    output.finished_req_ids = {"old-warmup-request"}
    step(worker, output)
    assert worker._ops_control.state == "armed"
    assert worker._ops_control.events[1]["cleanup_finished_ids"] == ["old-warmup-request"]
    empty = scheduled(ids=[])
    empty.finished_req_ids = {"r0"}
    with pytest.raises(ValueError):
        worker.execute_model(empty)


def test_new_request_cannot_also_be_finished_in_the_same_step(worker_factory):
    make, _, _ = worker_factory
    worker = make()
    worker.profile(True)
    output = scheduled(0)
    output.finished_req_ids = {"r0"}
    with pytest.raises(ValueError):
        worker.execute_model(output)


def test_accepted_legacy_receipt_without_prefix_fields_remains_replayable(worker_factory):
    make, _, root = worker_factory
    for rank in range(4):
        worker = make(rank)
        worker.profile(True)
        feed_prefills(worker)
        step(worker, scheduled())
        worker.profile(False)
    shape = {"phase": "prefill", "token_bucket": 128}
    receipt = json.loads(control.collect_control_receipt(root, shape, [0]).read_text())
    for row in receipt["workers"]:
        row.pop("decode_buckets")
        row.pop("pre_capture_decode_steps")
        for event in row["events"]:
            for key in ("scope", "scheduled_batch_size", "computed_tokens", "scheduled_tokens",
                        "padded_tokens", "cleanup_finished_ids"):
                event.pop(key, None)
    control.validate_control_receipt(receipt, shape, [0])


@pytest.mark.parametrize("failure", ["count", "position", "scope", "bucket", "rank_sequence"])
def test_receipt_replays_prefix_decode_progression(worker_factory, failure):
    make, _, root = worker_factory
    for rank in range(4):
        interleaved_generation(make(rank))
    shape = {"phase": "prefill", "token_bucket": 128}
    receipt = json.loads(control.collect_control_receipt(root, shape, [0]).read_text())
    if failure == "count":
        receipt["workers"][0]["pre_capture_decode_steps"] = 0
    else:
        workers = receipt["workers"][:1] if failure == "rank_sequence" else receipt["workers"]
        for row in workers:
            for event in row["events"]:
                if event.get("scope") != "pre_capture":
                    continue
                if failure in ("position", "rank_sequence"):
                    event["computed_tokens"]["r0"] += 1
                elif failure == "scope":
                    event["scope"] = "guard"
                elif failure == "bucket":
                    event["token_bucket"] = event["scheduled_batch_size"]
    with pytest.raises(ValueError):
        control.validate_control_receipt(receipt, shape, [0])


def test_source_pin_rejects_changed_and_missing_files(tmp_path, monkeypatch):
    path = tmp_path / "vendor.py"
    path.write_text("pinned")
    monkeypatch.setattr(control, "SOURCE_PINS", {"vendor.py": control.file_hash(path)})
    assert control.verify_runtime_sources(tmp_path) == control.SOURCE_PINS
    path.write_text("changed")
    with pytest.raises(ValueError, match="unsupported"):
        control.verify_runtime_sources(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        control.verify_runtime_sources(tmp_path)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "rank",
        "source",
        "passed",
        "order",
        "geometry",
        "multiset",
        "hook",
        "sample",
        "early_sample",
    ],
)
def test_collected_receipt_requires_complete_consistent_tp4_control(worker_factory, failure):
    make, _, root = worker_factory
    for rank in range(4):
        worker = make(rank)
        worker.profile(True)
        feed_prefills(worker)
        step(worker, scheduled())
        worker.profile(False)
    shape = {"phase": "prefill", "token_bucket": 128}
    path = control.collect_control_receipt(root, shape, [0])
    receipt = json.loads(path.read_text())
    row = receipt["workers"][0]
    if failure == "rank":
        row["rank"] = 1
    elif failure == "source":
        row["source_hashes"] = {}
    elif failure == "passed":
        row["passed"] = False
    elif failure == "order":
        row["events"][-1]["action"] = "profile_started"
    elif failure == "geometry":
        row["guard_geometry"]["token_bucket"] = 1
    elif failure == "multiset":
        row["observed_prompt_multiset"] = dict(Counter({"a" * 64: 16}))
    elif failure == "hook":
        row["hook_source_hashes"] = {}
    elif failure == "sample":
        row["events"] = [e for e in row["events"] if e["action"] != "sample_returned"]
    elif failure == "early_sample":
        event = next(e for e in row["events"] if e["action"] == "sample_returned")
        row["events"].remove(event)
        position = next(i for i, e in enumerate(row["events"]) if e["action"] == "profile_started")
        event["monotonic_ns"] = row["events"][position]["monotonic_ns"]
        row["events"].insert(position + 1, event)
    if failure:
        with pytest.raises(ValueError):
            control.validate_control_receipt(receipt, shape, [0])
    else:
        control.validate_control_receipt(receipt, shape, [0])


@pytest.mark.parametrize("failure", [None, "missing", "hash", "status", "method",
                                     "ranks", "policy"])
def test_offline_facade_checks_control_receipt_hash_and_semantics(worker_factory, failure):
    from profiling import ops

    make, _, root = worker_factory
    for rank in range(4):
        worker = make(rank)
        worker.profile(True)
        feed_prefills(worker)
        step(worker, scheduled())
        worker.profile(False)
    shape = {"phase": "prefill", "token_bucket": 128}
    path = control.collect_control_receipt(root, shape, [0])
    receipt = json.loads(path.read_text())
    if failure == "status":
        receipt["workers"][0]["passed"] = False
        path.write_text(json.dumps(receipt))
    capture = {
        "shape": shape, "sampling": {"window": control.METHOD},
        "sessions": [{"rank": 0, "neffs": [], "execution_selection": {
            "policy": "prefill_with_trailing_decode_guard_v1",
            "selected_indices": [0], "guard_indices": [1],
        }}],
        "execution": {"capture_control_receipt": path.name,
                      "capture_control_receipt_sha256": control.file_hash(path)},
    }
    if failure == "missing":
        capture["execution"] = {}
    elif failure == "hash":
        path.write_text(path.read_text() + "\n")
    elif failure == "method":
        capture["sampling"]["window"] = "old_window"
    elif failure == "ranks":
        capture["sessions"][0]["rank"] = 1
    elif failure == "policy":
        capture["sessions"][0]["execution_selection"]["policy"] = "prefill_with_decode_tail_v3"
    capture = ops.normalize_manifest_paths(capture, root)
    if failure:
        with pytest.raises(ValueError):
            ops.validate_execution_receipts(capture)
    else:
        assert capture["execution"]["capture_control_receipt"] == str(path)
        evidence = ops.validate_execution_receipts(capture)
        assert evidence["validated"] is False  # Host control alone cannot prove numerics.
        assert evidence["receipt_sha256"]["capture_control_receipt"] == control.file_hash(path)
