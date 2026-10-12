"""API-server hand-off stamps from vLLM's public request log; stdlib only.

`vllm_logging.json` (loaded by vLLM through the public `VLLM_LOGGING_CONFIG_PATH`)
names the two classes here. Every vLLM process imports this module while
`vllm.logger` configures logging, so it must not import vLLM itself.

With `--enable-log-requests`, stock 0.24 `AsyncLLM._add_request` logs
`"Added request %s."` on logger `vllm.v1.engine.async_llm` right after
`await engine_core.add_request_async(request)`: the EngineCoreRequest has been
handed to the EngineCore input socket, and `%s` is the internal request id
(after `assign_request_id`) that EngineCore and the observer see. The filter
samples both clocks there and writes one `VllmNeuronHandoff` line. It drops the
plain "Added request" line, which carries nothing the tagged line lacks, so the
instrument adds a single write per request to the API event loop.
"""

import json
import logging
import sys
import time

ADDED_REQUEST_MESSAGE = "Added request %s."
HANDOFF_TAG = "VllmNeuronHandoff"


def _emit(tag: str, row: dict) -> None:
    # One write per record, as vllm_observer._emit: the API server, EngineCore
    # and workers share the container stdout.
    sys.stdout.write(tag + " " + json.dumps(row, separators=(",", ":")) + "\n")
    sys.stdout.flush()


class HandoffStampFilter(logging.Filter):
    """Stamp each `Added request` record; pass every other record unchanged."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg != ADDED_REQUEST_MESSAGE:
            return True
        args = record.args
        if not isinstance(args, tuple) or len(args) != 1:
            return True
        send_monotonic_ns, send_epoch_ns = time.monotonic_ns(), time.time_ns()
        _emit(
            HANDOFF_TAG,
            {
                "request_id": str(args[0]),
                "send_monotonic_ns": send_monotonic_ns,
                "send_epoch_ns": send_epoch_ns,
            },
        )
        return False


class VllmLineFormatter(logging.Formatter):
    """vLLM's default `NewLineFormatter` at INFO (non-color, non-DEBUG) level.

    Reproduced instead of referenced so the config also loads where vLLM is not
    installed: `fileinfo` is the bare filename and continuation lines repeat the
    record prefix, giving `INFO 10-10 12:10:18 [file.py:44] message`.
    """

    def format(self, record: logging.LogRecord) -> str:
        record.fileinfo = record.filename
        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg
