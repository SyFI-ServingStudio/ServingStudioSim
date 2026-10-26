"""Frontend arrival and EngineCore receive stamps on the observer's monotonic clock.

No vLLM/Neuron install or device: the stock base classes are faked so the real
observer subclass runs its `add_request`/`schedule`/`update_from_output` path.
"""

import importlib
import json
import sys
import types
from types import SimpleNamespace as NS

import pytest

from alignment.neuron.vllm_records import (
    ARRIVAL_TIMING_FIELDS,
    RequestTimings,
    admission_stamps,
)
from alignment.neuron.vllm_runner import extract_stock_request_timings
from alignment.profiler import record_extraction

RID = "cmpl-case-0-1234abcd"


def event(name, timestamp):
    return NS(type=NS(name=name), timestamp=timestamp)


def batch(timestamp, tokens=(), *, events=(), finish=None, rid=RID):
    return {
        0: NS(
            timestamp=timestamp,
            outputs=[
                NS(
                    request_id=rid,
                    new_token_ids=list(tokens),
                    events=list(events),
                    finished=finish is not None,
                    finish_reason=NS(name=finish),
                )
            ],
            finished_requests=set(),
        )
    }


class FakeStockScheduler:
    """Stands in for NeuronScheduler: native QUEUED inside add_request."""

    def __init__(self, *args, clock, **kwargs):
        self.log_stats = True
        self.requests = {}
        self.clock = clock
        self.queued = {}

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.queued[request.request_id] = self.clock["mono_ns"] / 1e9

    def schedule(self):
        self.clock["mono_ns"] += 1_000  # planning takes time on the same clock
        return self.next_output

    def update_from_output(self, scheduler_output, model_output):
        return model_output


@pytest.fixture
def observer_module(monkeypatch):
    base = types.ModuleType("vllm_neuron.vllm.core.scheduler")
    base.NeuronScheduler = FakeStockScheduler
    base.NeuronAsyncScheduler = FakeStockScheduler
    worker = types.ModuleType("vllm_neuron.vllm.worker.neuron_worker")
    worker.NeuronWorker = object
    for name in ("vllm_neuron", "vllm_neuron.vllm", "vllm_neuron.vllm.core"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm_neuron.vllm.worker", types.ModuleType("w"))
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    monkeypatch.delitem(sys.modules, "alignment.neuron.vllm_observer", raising=False)
    module = importlib.import_module("alignment.neuron.vllm_observer")
    monkeypatch.setattr(module, "install_cleanup_adapter", lambda: None)
    yield module
    sys.modules.pop("alignment.neuron.vllm_observer", None)


@pytest.fixture
def observer(observer_module, monkeypatch):
    module = observer_module
    clock = {"epoch_ns": 1_700_000_000_000_000_000, "mono_ns": 5_000_000_000}
    monkeypatch.setattr(module.time, "time_ns", lambda: clock["epoch_ns"])
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: clock["mono_ns"])
    printed = []
    monkeypatch.setattr(module, "_emit", lambda tag, row: printed.append((tag, row)))
    yield module, clock, printed


def test_admission_converts_wall_clock_arrival_with_the_receive_pair():
    # Arrived 2.5 ms (epoch) before the receive pair was sampled.
    stamps = admission_stamps(1_700_000_000.0, 1_700_000_000_002_500_000, 9_000_000_000)
    assert stamps == {
        "frontend_arrival_epoch_ns": 1_700_000_000_000_000_000,
        "frontend_arrival_monotonic_ns": 8_997_500_000,
        "engine_receive_monotonic_ns": 9_000_000_000,
    }
    for bad in (float("nan"), 0.0, -1.0, True):
        with pytest.raises(ValueError):
            admission_stamps(bad, 1, 1)


def test_observed_add_request_stamps_receive_and_derives_arrival_durations(observer):
    module, clock, printed = observer
    scheduler = module.ObservedAsyncScheduler(clock=clock)
    # Frontend stamped time.time() 3 ms (wall) before EngineCore drained it.
    arrival_epoch_s = (clock["epoch_ns"] - 3_000_000) / 1e9
    scheduler.add_request(NS(request_id=RID, arrival_time=arrival_epoch_s))
    # A streaming update of an existing id must not move the first receive.
    clock["mono_ns"] += 7_000_000
    scheduler.add_request(NS(request_id=RID, arrival_time=arrival_epoch_s + 1))
    admission = scheduler._alignment_request_timings.pending[RID]["admission"]
    assert admission["engine_receive_monotonic_ns"] == 5_000_000_000
    assert admission["frontend_arrival_monotonic_ns"] == pytest.approx(4_997_000_000, abs=300)

    queued = 5.0  # native QUEUED (s) recorded inside the stock add_request
    scheduler.update_from_output(
        None, batch(5.004, [1], events=[event("QUEUED", queued), event("SCHEDULED", 5.001)])
    )
    scheduler.update_from_output(None, batch(5.020, range(7), finish="LENGTH"))
    ((tag, row),) = printed
    assert tag == "VibeSimAlignmentRequestTiming"
    assert row["schema_version"] == 2 and row["arrival_timing_schema_version"] == 2
    # Existing fields keep their meaning: QUEUED -> first output.
    assert row["engine_core_ttft_ms"] == pytest.approx(4.0)
    assert row["frontend_arrival_to_engine_receive_ms"] == pytest.approx(3.0, abs=1e-3)
    assert row["engine_arrival_to_queued_ms"] == pytest.approx(3.0, abs=1e-3)
    assert row["server_ttft_from_arrival_ms"] == pytest.approx(7.0, abs=1e-3)
    assert row["server_ttft_from_arrival_ms"] == pytest.approx(
        row["engine_arrival_to_queued_ms"] + row["engine_core_ttft_ms"], abs=1e-6
    )


def test_prompt_observation_carries_admission_stamps(observer):
    module, clock, printed = observer
    scheduler = module.ObservedAsyncScheduler(clock=clock)
    scheduler.max_model_len, scheduler.num_seqs_buckets = 512, [1, 16]
    scheduler.add_request(NS(request_id=RID, arrival_time=clock["epoch_ns"] / 1e9))
    scheduler.requests[RID].num_prompt_tokens = 504
    scheduler.next_output = NS(
        scheduled_new_reqs=[
            NS(req_id=RID, prompt_token_ids=list(range(504)), num_computed_tokens=0)
        ],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={RID: 504},
        num_scheduled_tokens_padded={RID: 512},
    )
    scheduler.schedule()
    row = dict(printed)["VllmNeuronPrompt"]
    assert row["engine_receive_monotonic_ns"] == 5_000_000_000
    assert row["frontend_arrival_monotonic_ns"] == 5_000_000_000
    assert row["prompt_tokens"] == 504


def test_unadmitted_request_keeps_old_fields_and_nulls_arrival():
    tracker = RequestTimings()
    tracker.observe(batch(10, [1], events=[event("QUEUED", 2), event("SCHEDULED", 4)]))
    row = tracker.observe(batch(24, [2], finish="LENGTH"))[0]
    assert row["engine_core_ttft_ms"] == 8000
    assert all(row[field] is None for field in ARRIVAL_TIMING_FIELDS)


def _admitted_row(arrival_offset_ns=1_000_000):
    tracker = RequestTimings()
    receive = 2_000_000_000
    tracker.admit(RID, admission_stamps(100.0, 100_000_000_000 + arrival_offset_ns, receive))
    tracker.observe(batch(2.010, [1], events=[event("QUEUED", 2.0), event("SCHEDULED", 2.002)]))
    return tracker.observe(batch(2.030, range(7), finish="LENGTH"))[0]


def _line(row):
    return "(EngineCore pid=42) VibeSimAlignmentRequestTiming " + json.dumps(row) + "\n"


MAPPING = {"client_to_engine": {"case": RID}, "output_tokens": 8}


def test_extractors_pass_arrival_fields_and_fill_nulls_for_old_captures(tmp_path):
    log = tmp_path / "measurement.log"
    row = _admitted_row()
    assert row["engine_arrival_to_queued_ms"] == pytest.approx(1.0)
    assert row["server_ttft_from_arrival_ms"] == pytest.approx(11.0)
    log.write_text(_line(row))
    path = extract_stock_request_timings(
        log, tmp_path, MAPPING, required=True, arrival_required=True
    )
    parsed = json.loads(path.read_text())
    for field in ARRIVAL_TIMING_FIELDS:
        assert parsed[field] == row[field]
    assert parsed["engine_core_ttft_ms"] == row["engine_core_ttft_ms"]

    old = {k: v for k, v in row.items() if k not in ARRIVAL_TIMING_FIELDS}
    del old["arrival_timing_schema_version"]
    log.write_text(_line(old))
    path = extract_stock_request_timings(log, tmp_path, MAPPING, required=True)
    parsed = json.loads(path.read_text())
    assert parsed["arrival_timing_schema_version"] is None
    assert all(parsed[field] is None for field in ARRIVAL_TIMING_FIELDS)
    with pytest.raises(ValueError, match="arrival"):
        extract_stock_request_timings(log, tmp_path, MAPPING, required=True, arrival_required=True)


@pytest.mark.parametrize("mutation", ["sum", "receive", "negative", "missing", "version"])
def test_generic_extractor_rejects_inconsistent_arrival_fields(tmp_path, mutation):
    row = _admitted_row()
    if mutation == "sum":
        row["server_ttft_from_arrival_ms"] += 0.5
    elif mutation == "receive":
        row["frontend_arrival_to_engine_receive_ms"] += 0.5
    elif mutation == "negative":
        row = _admitted_row(arrival_offset_ns=-1_000_000)
    elif mutation == "missing":
        del row["engine_receive_monotonic_ns"]
    else:
        row["arrival_timing_schema_version"] = 3
    log = tmp_path / "server.log"
    log.write_text(_line(row))
    with pytest.raises(ValueError, match="arrival"):
        record_extraction.extract_request_timings_jsonl(log, tmp_path / "out.jsonl")


def test_observer_emits_each_record_in_one_write(observer_module, monkeypatch):
    import io

    vllm_observer = observer_module
    writes = []

    class Stream(io.StringIO):
        def write(self, text):
            writes.append(text)
            return len(text)

    monkeypatch.setattr(vllm_observer.sys, "stdout", Stream())
    vllm_observer._emit("VllmNeuronForward", {"a": 1})
    assert writes == ['VllmNeuronForward {"a":1}\n']
