"""Per-client rate limits for the routes that run something.

Each limited route keeps, per client address, the times of its recent requests
in this process; a request past ``limit`` within ``window_s`` seconds is
refused with how long to wait. Only accepted work counts: a request the route
then rejects (a 4xx answer) is refunded. Nothing is shared between processes or kept
across a restart, and there are no keys: the client address is the only
identity. Behind a proxy, the address is the one uvicorn takes from
``X-Forwarded-For`` when the proxy is in ``--forwarded-allow-ips``.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """At most ``limit`` requests per ``window_s`` seconds per client."""

    def __init__(self, limit: int, window_s: float, clock=time.monotonic) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[str, deque[float]] = {}

    def admit(self, client: str) -> float | None:
        """Count a request from ``client``; None if it may proceed, else the
        seconds until it may."""
        now = self._clock()
        with self._lock:
            seen = self._seen.setdefault(client, deque())
            while seen and seen[0] <= now - self.window_s:
                seen.popleft()
            if len(seen) >= self.limit:
                return seen[0] + self.window_s - now
            seen.append(now)
            # Forget idle clients, so the table holds only recent ones.
            if len(self._seen) > 10_000:
                for key in [k for k, v in self._seen.items() if v[-1] <= now - self.window_s]:
                    del self._seen[key]
            return None

    def refund(self, client: str) -> None:
        """Uncount ``client``'s latest admitted request: the route rejected it."""
        with self._lock:
            seen = self._seen.get(client)
            if seen:
                seen.pop()
