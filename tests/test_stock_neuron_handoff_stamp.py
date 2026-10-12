"""API-server hand-off stamps from vLLM's public request log (no vLLM install).

The shipped `vllm_logging.json` is loaded with `logging.config.dictConfig`, as
vLLM does for `VLLM_LOGGING_CONFIG_PATH`, and the stock `AsyncLLM._add_request`
log call is replayed on its real logger name.
"""

import io
import json
import logging
import logging.config
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignment.neuron import vllm_handoff_log
from alignment.neuron.vllm_records import (
    HANDOFF_TIMING_FIELDS,
    RequestTimings,
    admission_stamps,
)
from alignment.neuron.vllm_runner import _tagged, extract_stock_request_timings, handoff_stamps
from alignment.profiler import record_extraction
from tests.test_stock_neuron_arrival_timings import batch, event

CONFIG = Path(vllm_handoff_log.__file__).with_name("vllm_logging.json")
RID = "cmpl-case-0-1234abcd"
MAPPING = {"client_to_engine": {"case": RID}, "output_tokens": 8}
CONFIGURED = (
    "vllm",
    "vllm.v1.engine.async_llm",
    "vllm.entrypoints.logger",
    "vllm.entrypoints.serve.utils.request_logger",
)


class Stdout(io.StringIO):
    def __init__(self):
        super().__init__()
        self.writes = []

    def write(self, text):
        self.writes.append(text)
        return len(text)


@pytest.fixture
def vllm_logging(monkeypatch):
    """Apply the shipped config to a captured stdout; restore the loggers after."""
    saved = {}
    for name in CONFIGURED:
        logger = logging.getLogger(name)
        saved[name] = (logger.level, logger.propagate, logger.handlers[:], logger.filters[:])
    stdout = Stdout()
    # The handler binds its stream at config time; the filter writes to the
    # stdout of the moment, which pytest swaps between phases.
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr(vllm_handoff_log, "sys", SimpleNamespace(stdout=stdout))
    monkeypatch.setattr(vllm_handoff_log.time, "monotonic_ns", lambda: 5_001_000_000)
    monkeypatch.setattr(vllm_handoff_log.time, "time_ns", lambda: 1_700_000_000_001_000_000)
    logging.config.dictConfig(json.loads(CONFIG.read_text()))
    yield stdout
    for name, (level, propagate, handlers, filters) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = propagate
        logger.handlers[:] = handlers
        logger.filters[:] = filters


def test_added_request_writes_exactly_one_tagged_stamp(vllm_logging):
    logger = logging.getLogger("vllm.v1.engine.async_llm")
    logger.info("Added request %s.", RID)
    assert vllm_logging.writes == [
        'VllmNeuronHandoff {"request_id":"cmpl-case-0-1234abcd",'
        '"send_monotonic_ns":5001000000,"send_epoch_ns":1700000000001000000}\n'
    ]
    # Other request-log lines of the same logger keep vLLM's format.
    logger.info("Request %s aborted.", RID)
    assert len(vllm_logging.writes) == 2
    assert vllm_logging.writes[1].endswith(f"Request {RID} aborted.\n")


def test_config_keeps_vllm_format_and_mutes_per_request_prompt_logs(vllm_logging):
    logging.getLogger("vllm.engine.arg_utils").info("first\nsecond")
    line = "".join(vllm_logging.writes)
    # vLLM 0.24 default: `INFO 10-10 12:10:18 [__init__.py:44] message`.
    prefix = r"INFO \d\d-\d\d \d\d:\d\d:\d\d \[test_stock_neuron_handoff_stamp\.py:\d+\] "
    assert re.fullmatch(prefix + "first\r\n" + prefix + "second\n", line)
    vllm_logging.writes.clear()
    for name in ("vllm.entrypoints.logger", "vllm.entrypoints.serve.utils.request_logger"):
        logging.getLogger(name).info("Received request %s: params: %s", RID, "...")
    assert vllm_logging.writes == []
    logging.getLogger("vllm.entrypoints.logger").warning("kept")
    assert vllm_logging.writes and vllm_logging.writes[0].startswith("WARNING ")
    # Loggers outside vLLM's tree (e.g. the Neuron loader receipts) are untouched.
    assert not logging.getLogger("capture_backend").handlers


def test_filter_ignores_other_messages_and_shapes():
    stamp = vllm_handoff_log.HandoffStampFilter()
    other = logging.LogRecord(
        "vllm.v1.engine.async_llm", 20, "f.py", 1, "Request %s aborted.", (RID,), None
    )
    odd = logging.LogRecord(
        "vllm.v1.engine.async_llm", 20, "f.py", 1, "Added request %s.", (RID, "extra"), None
    )
    assert stamp.filter(other) and stamp.filter(odd)


def _row(*, send_ns=None, version=2):
    tracker = RequestTimings()
    receive = 2_000_000_000
    tracker.admit(RID, admission_stamps(100.0, 100_000_000_000 + 1_000_000, receive))
    tracker.observe(batch(2.010, [1], events=[event("QUEUED", 2.0), event("SCHEDULED", 2.002)]))
    row = tracker.observe(batch(2.030, range(7), finish="LENGTH"))[0]
    assert row["arrival_timing_schema_version"] == 2
    assert all(row[field] is None for field in HANDOFF_TIMING_FIELDS)
    if send_ns is not None:
        row["api_send_monotonic_ns"] = send_ns
        row["frontend_arrival_to_api_send_ms"] = (send_ns - 1_999_000_000) / 1e6
        row["api_send_to_engine_receive_ms"] = (receive - send_ns) / 1e6
    if version == 1:
        row["arrival_timing_schema_version"] = 1
        for field in HANDOFF_TIMING_FIELDS:
            del row[field]
    return row


def _timing_line(row):
    return "(EngineCore pid=42) VibeSimAlignmentRequestTiming " + json.dumps(row) + "\n"


def _stamp_line(rid=RID, send=1_999_600_000):
    stamp = {"request_id": rid, "send_monotonic_ns": send, "send_epoch_ns": 99_999_999_999}
    return "(APIServer pid=1) VllmNeuronHandoff " + json.dumps(stamp) + "\n"


def test_postprocess_joins_send_stamp_by_engine_request_id(tmp_path):
    log = tmp_path / "measurement.log"
    # A probe outside the replay population is excluded like its timing record.
    log.write_text(_stamp_line("cmpl-probe-0-00000000") + _stamp_line() + _timing_line(_row()))
    path = extract_stock_request_timings(
        log, tmp_path, MAPPING, required=True, arrival_required=True, handoff_stamp=True
    )
    (parsed,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert parsed["arrival_timing_schema_version"] == 2
    assert parsed["api_send_monotonic_ns"] == 1_999_600_000
    assert parsed["frontend_arrival_to_api_send_ms"] == pytest.approx(0.6)
    assert parsed["api_send_to_engine_receive_ms"] == pytest.approx(0.4)
    assert parsed["frontend_arrival_to_api_send_ms"] + parsed[
        "api_send_to_engine_receive_ms"
    ] == pytest.approx(parsed["frontend_arrival_to_engine_receive_ms"])


def test_postprocess_reads_the_filter_output_through_the_tagged_parser(tmp_path, vllm_logging):
    logging.getLogger("vllm.v1.engine.async_llm").info("Added request %s.", RID)
    log = tmp_path / "measurement.log"
    log.write_text("(APIServer pid=1) " + "".join(vllm_logging.writes))
    assert _tagged(log, "VllmNeuronHandoff")[0]["send_monotonic_ns"] == 5_001_000_000
    assert set(handoff_stamps(log, MAPPING)) == {RID}


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong-engine-id", "late"])
def test_postprocess_rejects_incomplete_or_unordered_stamps(tmp_path, case):
    stamps = {
        "missing": "",
        "duplicate": _stamp_line() + _stamp_line(),
        "wrong-engine-id": _stamp_line("cmpl-case-0-deadbeef"),
        "late": _stamp_line(send=2_000_000_001),  # after EngineCore receive
    }[case]
    log = tmp_path / "measurement.log"
    log.write_text(stamps + _timing_line(_row()))
    with pytest.raises(ValueError):
        extract_stock_request_timings(
            log, tmp_path, MAPPING, required=True, arrival_required=True, handoff_stamp=True
        )


def test_arrival_v1_captures_still_read_with_null_handoff(tmp_path):
    log = tmp_path / "measurement.log"
    log.write_text(_timing_line(_row(version=1)))
    path = extract_stock_request_timings(
        log, tmp_path, MAPPING, required=True, arrival_required=True
    )
    parsed = json.loads(path.read_text())
    assert parsed["arrival_timing_schema_version"] == 1
    assert all(parsed[field] is None for field in HANDOFF_TIMING_FIELDS)
    # The hand-off stamp needs arrival sub-version 2 rows.
    log.write_text(_stamp_line() + _timing_line(_row(version=1)))
    with pytest.raises(ValueError, match="schema 2"):
        extract_stock_request_timings(log, tmp_path, MAPPING, required=True, handoff_stamp=True)


def test_unstamped_capture_writes_null_handoff_and_rejects_stray_stamps(tmp_path):
    log = tmp_path / "measurement.log"
    log.write_text(_timing_line(_row()))
    path = extract_stock_request_timings(
        log, tmp_path, MAPPING, required=True, arrival_required=True
    )
    parsed = json.loads(path.read_text())
    assert parsed["arrival_timing_schema_version"] == 2
    assert all(parsed[field] is None for field in HANDOFF_TIMING_FIELDS)
    # Even a probe's stamp is stray when the server was not configured for it.
    log.write_text(_stamp_line("cmpl-probe-0-00000000") + _timing_line(_row()))
    with pytest.raises(ValueError, match="without handoff_stamp"):
        extract_stock_request_timings(log, tmp_path, MAPPING, required=True)


def test_generic_extractor_accepts_complete_or_null_handoff(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(_timing_line(_row()))
    assert record_extraction.extract_request_timings_jsonl(log, tmp_path / "a.jsonl") == 1
    log.write_text(_timing_line(_row(send_ns=1_999_600_000)))
    assert record_extraction.extract_request_timings_jsonl(log, tmp_path / "b.jsonl") == 1
    parsed = json.loads((tmp_path / "b.jsonl").read_text())
    assert parsed["api_send_monotonic_ns"] == 1_999_600_000


@pytest.mark.parametrize(
    "mutation", ["partial", "before-arrival", "after-receive", "duration", "v1-values", "missing"]
)
def test_generic_extractor_rejects_inconsistent_handoff(tmp_path, mutation):
    row = _row(send_ns=1_999_600_000)
    if mutation == "partial":
        row["api_send_to_engine_receive_ms"] = None
    elif mutation == "before-arrival":
        row = _row(send_ns=1_998_000_000)
        row["frontend_arrival_to_api_send_ms"] = 0.0  # keep durations nonnegative
    elif mutation == "after-receive":
        row = _row(send_ns=2_000_500_000)
        row["api_send_to_engine_receive_ms"] = 0.0
    elif mutation == "duration":
        row["frontend_arrival_to_api_send_ms"] += 0.1
    elif mutation == "v1-values":
        row["arrival_timing_schema_version"] = 1
    else:
        del row["api_send_monotonic_ns"]
    log = tmp_path / "server.log"
    log.write_text(_timing_line(row))
    with pytest.raises(ValueError, match="arrival"):
        record_extraction.extract_request_timings_jsonl(log, tmp_path / "out.jsonl")
