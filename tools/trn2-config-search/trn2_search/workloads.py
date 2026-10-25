"""Seeded request traces for the Trainium2 config search.

Every workload is an independent-request CSV (`trace/README.md`): `id,
input_len,output_len,arrival_time` in milliseconds, plus `accept_rate` when the
speculative (MTP) family needs it. A workload is a name plus generator
parameters, so the same spec always writes the same bytes.

Kinds:

- `fixed`: every request the same shape, a fixed spacing
  (`trace/generate_fixed_shape.py`).
- `chat`: input and output lengths uniform in a range, Poisson arrivals at
  `rate` requests per second.
- `agentic`: long prompts and short outputs, bursty arrivals. Bursts arrive as
  a Poisson process at `burst_rate` bursts per second, and each burst holds a
  Poisson(`burst_mean`) number of requests (at least one) arriving within
  `burst_spread_ms` of each other, as an agent fanning out tool calls would.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Workload:
    name: str
    kind: str
    requests: int
    params: dict = field(default_factory=dict)

    def rows(self) -> list[tuple[int, int, float]]:
        """`(input_len, output_len, arrival_ms)` per request, arrival-sorted."""
        rng = random.Random(f"{self.name}:{self.params.get('seed', 0)}")
        p = self.params
        if self.kind == "fixed":
            gap = float(p.get("interarrival_ms", 0.0))
            return [
                (int(p["input_len"]), int(p["output_len"]), i * gap) for i in range(self.requests)
            ]
        if self.kind == "chat":
            rate = float(p["rate"])
            t, out = 0.0, []
            for _ in range(self.requests):
                out.append((_uniform(rng, p["input_len"]), _uniform(rng, p["output_len"]), t))
                t += rng.expovariate(rate) * 1000.0
            return out
        if self.kind == "agentic":
            burst_rate = float(p["burst_rate"])
            burst_mean = float(p["burst_mean"])
            spread = float(p.get("burst_spread_ms", 500.0))
            t, out = 0.0, []
            while len(out) < self.requests:
                size = max(1, _poisson(rng, burst_mean))
                for _ in range(min(size, self.requests - len(out))):
                    out.append(
                        (
                            _uniform(rng, p["input_len"]),
                            _uniform(rng, p["output_len"]),
                            t + rng.uniform(0.0, spread),
                        )
                    )
                t += rng.expovariate(burst_rate) * 1000.0
            out.sort(key=lambda r: r[2])
            return out
        raise ValueError(f"workload {self.name}: unknown kind {self.kind!r}")

    def write(self, path: Path, accept_rate: float | None = None) -> Path:
        """Write the CSV; with `accept_rate`, add the speculative column."""
        if accept_rate is not None and not 0.0 <= accept_rate <= 1.0:
            raise ValueError("accept_rate must be in [0, 1]")
        header = "id,input_len,output_len,arrival_time"
        tail = ""
        if accept_rate is not None:
            header += ",accept_rate"
            tail = f",{accept_rate}"
        lines = [header]
        lines += [
            f"{i},{inp},{out},{arr:.6f}{tail}" for i, (inp, out, arr) in enumerate(self.rows())
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path


def _uniform(rng: random.Random, bounds) -> int:
    lo, hi = (int(bounds), int(bounds)) if isinstance(bounds, (int, float)) else map(int, bounds)
    if lo <= 0 or hi < lo:
        raise ValueError(f"bad length range {bounds!r}")
    return rng.randint(lo, hi)


def _poisson(rng: random.Random, mean: float) -> int:
    # Knuth; the means here are small.
    limit, k, prod = pow(2.718281828459045, -mean), 0, rng.random()
    while prod > limit:
        k += 1
        prod *= rng.random()
    return k


def from_spec(spec: dict) -> Workload:
    """A `Workload` from one stage-file entry: `{name, kind, requests, ...params}`
    (or `{name, file}` for an existing trace, which `trace_path` returns as is)."""
    spec = dict(spec)
    name = spec.pop("name")
    if "file" in spec:
        return Workload(name, "file", 0, {"file": spec["file"]})
    return Workload(name, spec.pop("kind"), int(spec.pop("requests")), spec)


def trace_path(w: Workload, out_dir: Path, accept_rate: float | None = None) -> Path:
    """The CSV this workload runs from, writing it under `out_dir` if generated.
    An existing `file` trace with an accept rate is re-written with the column."""
    if w.kind == "file":
        src = Path(w.params["file"])
        if accept_rate is None:
            return src
        lines = src.read_text().splitlines()
        cols = lines[0].split(",")
        if "accept_rate" in cols:
            idx = cols.index("accept_rate")
            body = [
                ",".join(c if j != idx else str(accept_rate) for j, c in enumerate(ln.split(",")))
                for ln in lines[1:]
            ]
            head = lines[0]
        else:
            head = lines[0] + ",accept_rate"
            body = [ln + f",{accept_rate}" for ln in lines[1:] if ln]
        dst = out_dir / f"{w.name}_a{accept_rate}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join([head, *body]) + "\n")
        return dst
    suffix = f"_a{accept_rate}" if accept_rate is not None else ""
    return w.write(out_dir / f"{w.name}{suffix}.csv", accept_rate)
