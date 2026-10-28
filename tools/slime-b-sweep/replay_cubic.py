"""Replay slime's online threshold tuners (CUBIC, linear) against a simulated B grid.

    uv run python tools/slime-b-sweep/replay_cubic.py logs/<grid_dir> [--validate]

The tuner's only state that crosses rollouts is B (plus its own W_max / t /
slow-start flags), and every B it can pick lands on the grid's 8-step lattice
inside [16, 128]. A simulated rollout at a given B does not depend on the
rollouts before it. So reading cell (r, B_r), feeding its observation to the
tuner and moving to (r+1, B_{r+1}) IS the closed loop, not an approximation of it.

Control law: a port of `CubicTuner` from coriander
`mjacob2/slime/slime/router/threshold_tuner.py`, with the arm's own flags
(`run_cubic_B16.sh`: B0=16, `--tuner-b-min 16 --tuner-b-max 128`, the rest
default: C=1.0, beta=0.7, gamma=2.0, eps=0.005, no fast convergence,
snap_quantum = n_samples_per_prompt = 8, rollout 0 skipped).

Signal: `interior_idle_ratio`, as `collect_observation` computes it per GPU and
sums over the 8. A GPU's `training` span runs from its block's flip to the ONE
global training end; busy = its block's chunks; trailing = span end - last
chunk end; interior = span - busy - trailing, and a block that got no chunk at
all is charged its whole span. The ratio is dimensionless, so K drops out.

The linear arm is slime's `InteriorIdleTuner`: the same signal and epsilon, a
bang-bang law that moves B by exactly `step` = 16 every rollout (down when the
ratio exceeds eps, up otherwise), the base class's +-step rail, rollout 0 skipped,
no lattice snap -- starting at 16 it stays on multiples of 16. A third arm is
the same law made asymmetric: +16 on headroom, -32 on starvation.

`KvCappedTuner` is the linear law under a KV ceiling. Its second signal is the
peak number of migrated-in requests waiting on one engine (`engine_queues`'
migrated column): a handed-over request waits there when the destination has
no KV to re-prefill it, which is what an engine's waiting queue shows the
router. Above `queue_max` the tuner backs off one step and pins that B as the
ceiling for the rest of the run; below it, the +-16 idle law runs under the
ceiling. The ceiling never rises again -- right for one workload replayed, not
for responses that lengthen over a long run.

`window_best` is not an online tuner but a hindsight one: before rollout r it
takes the B in [16, 128] with the best mean speedup over rollouts r-5..r-1
(fewer at the start; B0=16 for rollout 0). It needs every B's outcome on the
past rollouts, which only a simulator has -- the bound for "pick B from recent
history", not a law slime could run. `window=None` widens it to every past
rollout (the running best fixed B).

`--validate` replays against the measured arm's own trajectory
(`data/cubic_B16.csv` + the per-step signal in `data/cubic_B16_steps.csv`).
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import analyze_sweep as sw

EPS = 0.005


class CubicTuner:
    """`slime.router.threshold_tuner.CubicTuner` with the base-class rails inlined."""

    def __init__(self, initial=16, b_min=16, b_max=128, C=1.0, beta=0.7, gamma=2.0,
                 epsilon=EPS, snap_quantum=8, skip_first=True):
        self.b_min, self.b_max = b_min, b_max
        self.C, self.beta, self.gamma, self.epsilon = C, beta, gamma, epsilon
        self.q, self.skip_first = snap_quantum, skip_first
        self.current = self._clamp(initial)
        self.W_max = None
        self.ssthresh = float(b_max)
        self.t = 0
        self.in_slow_start = True
        self.reason = "initial"

    def _clamp(self, b):
        return max(self.b_min, min(self.b_max, int(b)))

    def _snap(self, b):
        q = self.q
        snapped = max(self.b_min, int(round(b / q)) * q) if q > 1 else int(round(b))
        if q > 1 and snapped == self.current and abs(b - self.current) > 1e-9:
            snapped = self.current + (q if b > self.current else -q)
        return snapped

    def K(self):
        return 0.0 if self.W_max is None else (self.W_max * (1 - self.beta) / self.C) ** (1 / 3)

    def update(self, rollout_id, r, migrated_queue=None):
        proposed = self._propose(rollout_id, r)
        # Rails are widened to the full range for CUBIC, so only the clamp bites.
        self.current = self._clamp(self.current if proposed is None else proposed)
        return self.current

    def _propose(self, rollout_id, r):
        if self.skip_first and rollout_id == 0:
            self.reason = "rollout 0 skipped"
            return None
        if r is None or not math.isfinite(r):
            self.reason = "no signal; hold"
            return None
        if r > self.epsilon:
            W = float(self.current)
            self.W_max = W
            self.ssthresh = max(float(self.b_min), self.beta * W)
            self.in_slow_start = False
            self.t = 0
            nxt = max(float(self.b_min), self.beta * W)
            self.reason = f"STARVED, W_max={W:.0f} -> {nxt:.1f}"
            return self._snap(nxt)
        self.t += 1
        if self.in_slow_start:
            nxt = min(self.ssthresh, self.current * self.gamma)
            self.reason = f"slow start x{self.gamma:g} -> {nxt:.1f}"
            return self._snap(nxt)
        k = self.K()
        nxt = self.C * (self.t - k) ** 3 + self.W_max
        region = "concave" if self.t < k else ("plateau" if self.t < k + 1.5 else "convex")
        self.reason = f"cubic t={self.t} K={k:.1f} W_max={self.W_max:.0f} -> {nxt:.1f} ({region})"
        return self._snap(nxt)


class LinearTuner:
    """`slime.router.threshold_tuner.InteriorIdleTuner`: +-step on the same signal."""

    def __init__(self, initial=16, b_min=16, b_max=128, step=16, down=None, epsilon=EPS,
                 skip_first=True):
        # `down` makes the law asymmetric (back off faster than it climbs); None = `step`.
        self.b_min, self.b_max, self.step = b_min, b_max, step
        self.down = step if down is None else down
        self.epsilon, self.skip_first = epsilon, skip_first
        self.current = max(b_min, min(b_max, int(initial)))
        self.reason = "initial"

    def update(self, rollout_id, r, migrated_queue=None):
        if self.skip_first and rollout_id == 0:
            self.reason = "rollout 0 skipped"
        elif r is None or not math.isfinite(r):
            self.reason = "no signal; hold"
        elif r > self.epsilon:
            self.reason = f"STARVED -> -{self.down}"
            self.current = max(self.b_min, self.current - self.down)
        else:
            self.reason = f"headroom -> +{self.step}"
            self.current = min(self.b_max, self.current + self.step)
        return self.current


class KvCappedTuner(LinearTuner):
    """+-step on the idle signal, under a ceiling set by the migrated-in queue."""

    needs_queue = True

    def __init__(self, queue_max=20, **kw):
        super().__init__(**kw)
        self.queue_max = queue_max
        self.ceiling = self.b_max

    def update(self, rollout_id, r, migrated_queue=None):
        if not (self.skip_first and rollout_id == 0) and migrated_queue is not None \
                and migrated_queue > self.queue_max:
            self.ceiling = max(self.b_min, self.current - self.down)
            self.current = self.ceiling
            self.reason = f"KV queue {migrated_queue} > {self.queue_max} -> ceiling {self.ceiling}"
            return self.current
        super().update(rollout_id, r)
        self.current = min(self.current, self.ceiling)
        return self.current


TUNERS = {
    "CUBIC": CubicTuner,
    "linear +-16": LinearTuner,
    "linear +16/-32": lambda: LinearTuner(step=16, down=32),
    "KV-capped +-16": KvCappedTuner,
}


def observation(cell: Path) -> dict[str, float]:
    """`collect_observation`'s idle split for one simulated cell, in sim GPU-seconds."""
    log = cell / "raw" / "cost_log"
    inference = [sw._intervals(pq.read_table(log / f"worker_main_{gpu}.parquet",
                                             columns=["wall_start_ms", "total_time_ms"]))
                 for gpu in range(sw.GPUS)]
    blocks = sw.GPUS // sw.GPUS_PER_BLOCK
    chunks = [(np.zeros(0), np.zeros(0))] * blocks
    for path in log.glob("worker_main.train_*.parquet"):
        chunks[int(path.stem.rsplit("_", 1)[1])] = sw._intervals(
            pq.read_table(path, columns=["wall_start_ms", "total_time_ms"]))
    training_end = max(end.max() for _, end in chunks if len(end))
    span = busy = interior = trailing = 0.0
    for block in range(blocks):
        gpus = range(block * sw.GPUS_PER_BLOCK, (block + 1) * sw.GPUS_PER_BLOCK)
        flip = max(inference[gpu][1].max() for gpu in gpus)
        start, end = chunks[block]
        length = training_end - flip
        if not len(start):
            span += length * len(gpus)
            interior += length * len(gpus)
            continue
        start = np.maximum(start, flip)
        trained = sw._union_length(start, end)
        tail = max(0.0, training_end - end.max())
        span += length * len(gpus)
        busy += trained * len(gpus)
        trailing += tail * len(gpus)
        interior += max(0.0, length - trained - tail) * len(gpus)
    return {"span": span / 1e3, "busy": busy / 1e3, "interior": interior / 1e3,
            "trailing": trailing / 1e3, "ratio": interior / span, "end": training_end / 1e3 * sw.K}


def grid_bs(grid: Path) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in grid.glob("r0_*") if p.is_dir())


def measured_steps() -> dict[int, dict[str, float]] | None:
    path = sw.HERE / "data" / "cubic_B16_steps.csv"
    if not path.exists():
        return None
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    head, body = rows[0], rows[1:]
    return {int(r[0]): dict(zip(head, r)) for r in body}


def replay(grid: Path, make=CubicTuner, forced: list[int] | None = None):
    """Closed loop over the grid. With `forced`, B follows that trajectory instead."""
    bs = grid_bs(grid)
    tuner = make()
    out = []
    for r in range(sw.ROLLOUTS):
        b = forced[r] if forced else tuner.current
        assert b in bs, f"B={b} is not a column of {grid}"
        cell = grid / f"r{r}_{b}"
        obs = observation(cell)
        if getattr(tuner, "needs_queue", False):
            obs["migrated_queue"] = int(sw.engine_queues(cell)[1])
        tuner.current = b
        nxt = tuner.update(r, obs["ratio"], obs.get("migrated_queue"))
        out.append((r, b, obs, nxt, tuner.reason))
    return out


def fixed_ends(grid: Path) -> tuple[list[int], np.ndarray]:
    bs = [b for b in grid_bs(grid) if 16 <= b <= 128]
    end = np.array([[sw.training_window(grid / f"r{r}_{b}")[1] for b in bs]
                    for r in range(sw.ROLLOUTS)])
    return bs, end


def window_best(grid: Path, window: int | None = 5, initial: int = 16) -> tuple[np.ndarray, list[int]]:
    """B per rollout: the column with the best mean speedup over the last `window` rollouts.

    `window=None` means all of them.
    """
    bs, end = fixed_ends(grid)
    speed = sw.measured_colocate()[:, None] / end
    picks = [initial]
    for r in range(1, sw.ROLLOUTS):
        lo = 0 if window is None else max(0, r - window)
        picks.append(bs[int(speed[lo:r].mean(0).argmax())])
    return np.array(picks), bs


def trajectories(grid: Path) -> dict[str, np.ndarray]:
    """B per rollout for every closed-loop tuner plus the trailing-window pick."""
    out = {name: np.array([b for _, b, *_ in replay(grid, make)]) for name, make in TUNERS.items()}
    out["best of last 5"] = window_best(grid)[0]
    out["best of all past"] = window_best(grid, window=None)[0]
    return out


def main() -> None:
    grid = Path(sys.argv[1])
    validate = "--validate" in sys.argv[2:]
    colocate = sw.measured_colocate()

    results = {}
    for name, make in TUNERS.items():
        rows = replay(grid, make)
        print(f"\nclosed-loop {name} on {grid.name} (B0=16, [16,128], eps={EPS})")
        print(f"{'r':>3} {'B':>4} {'interior':>9} {'ratio':>8} {'end s':>7} {'speedup':>8} {'next':>5}  reason")
        for r, b, o, nxt, why in rows:
            flag = " *" if o["ratio"] > EPS else ""
            print(f"{r:>3} {b:>4} {o['interior']:9.2f} {o['ratio']:8.5f} {o['end']:7.1f} "
                  f"{colocate[r] / o['end']:8.3f} {nxt:>5}  {why}{flag}")
        results[name] = rows

    bs, end = fixed_ends(grid)
    speed = colocate[:, None] / end
    best = speed.mean(0).argmax()
    print(f"\n{'policy':<32} {'all 20':>7} {'r4-19':>7} {'mean B':>7} {'starved':>8}   (speedup vs measured colocate)")
    for name, rows in results.items():
        tuned = np.array([colocate[r] / o["end"] for r, _, o, _, _ in rows])
        starved = sum(o["ratio"] > EPS for _, _, o, _, _ in rows)
        print(f"{name + ' B0=16 (closed loop)':<32} {tuned.mean():7.3f} {tuned[4:].mean():7.3f} "
              f"{np.mean([b for _, b, *_ in rows]):7.0f} {starved:>8}")
    for label, window in (("best of last 5 rollouts", 5), ("best of all past rollouts", None)):
        picks, _ = window_best(grid, window=window)
        held = np.array([speed[r, bs.index(b)] for r, b in enumerate(picks)])
        print(f"{label:<32} {held.mean():7.3f} {held[4:].mean():7.3f} "
              f"{picks.mean():7.0f} {'-':>8}   B = {picks.tolist()}")
    print(f"{f'best fixed B={bs[best]} (in [16,128])':<32} {speed[:, best].mean():7.3f} "
          f"{speed[4:, best].mean():7.3f} {bs[best]:>7}")
    print(f"{'per-rollout oracle in [16,128]':<32} {speed.max(1).mean():7.3f} {speed[4:].max(1).mean():7.3f} "
          f"{np.mean([bs[j] for j in speed.argmax(1)]):7.0f}")

    if validate:
        steps = measured_steps()
        meas_b = sw.measured_cubic()[0].tolist()
        forced = replay(grid, CubicTuner, forced=meas_b)
        print("\nmeasured CUBIC trajectory: sim signal at the same cells vs measured signal")
        print(f"{'r':>3} {'B':>4} {'sim ratio':>10} {'meas ratio':>11}  next B sim / measured")
        for r, b, o, nxt, _ in forced:
            m = steps[r]
            print(f"{r:>3} {b:>4} {o['ratio']:10.5f} {float(m['interior_idle_ratio']):11.5f}  "
                  f"{nxt:>4} / {m['B_next']}")


if __name__ == "__main__":
    main()
