"""One JSON line per request the service answers, kept on disk so the record
outlives the container: ``<dir>/<UTC date>.jsonl``.

A line holds when (``ts``, UTC), who (``client``, the address the rate limits
count, and ``ua``), what (``method``, ``path``, ``query``), the answer
(``status``, ``ms``) and, for the routes that do work, what a handler noted on
``request.state.note``: the preset and member a prediction or simulation ran
on, its case count or workload source, the id it left. Request bodies are not
kept."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request

# A user agent is free text; keep enough to tell clients apart.
UA_CHARS = 200


class AccessLog:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        path = self.directory / f"{record['ts'][:10]}.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as out:
            out.write(line)

    def install(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def log_request(request: Request, call_next):
            start = time.perf_counter()
            status = 500
            try:
                response = await call_next(request)
                status = response.status_code
                return response
            finally:
                record = {
                    "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                    "client": request.client.host if request.client else None,
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query or None,
                    "status": status,
                    "ms": round((time.perf_counter() - start) * 1000, 1),
                    "ua": (request.headers.get("user-agent") or "")[:UA_CHARS] or None,
                }
                note = getattr(request.state, "note", None)
                if note:
                    record["note"] = note
                try:
                    self.write(record)
                except OSError:
                    # A full or read-only disk must not fail the request.
                    pass
