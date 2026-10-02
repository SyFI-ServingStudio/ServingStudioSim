"""Closed-form analytic memory-roofline floor for the GLM-5.3-Flash MI300X port.

These are pure-CPU tests: the ``elementwise_floor`` backend is now arithmetic
(decision #49), so it must resolve a time for any footprint — including the DSA
prefill indexer cells whose naive output materialization exceeds 192 GB HBM —
without allocating a tensor, launching a kernel, or importing torch.
"""

from __future__ import annotations

import subprocess
import sys

from profiling.runners.elementwise import floor


def test_module_import_does_not_pull_in_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import profiling.runners.elementwise.floor as f; "
            "print('torch' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_roofline_is_physical():
    roofline = floor._hbm_roofline()
    peak_bytes_per_ms = floor._peak_hbm_gbps() * 1e6
    assert roofline.latency_ms >= 0.0
    # The effective bandwidth must never exceed the physical HBM peak: the
    # empirical fit over the (small, latency-bound) measured cells is clamped to
    # the achievable fraction of peak.
    assert 0.0 < roofline.bw_bytes_per_ms <= peak_bytes_per_ms
    assert roofline.source in {"empirical_fit", "spec_roofline"}


def test_closed_form_matches_latency_plus_traffic_over_bandwidth():
    roofline = floor._hbm_roofline()
    in_bytes, out_bytes = 3_000_000, 5_000_000
    expected = roofline.latency_ms + (in_bytes + out_bytes) / roofline.bw_bytes_per_ms
    metrics = floor._floor(in_bytes, out_bytes)
    assert metrics.time_ms == expected
    assert metrics.tflops == 0.0
    assert metrics.energy_j == 0.0


def test_cost_driver_is_total_footprint_not_fan_in():
    # A map (260 MB in / 130 MB out) and its byte-swapped twin move the same total
    # bytes, so the roofline must price them identically — total footprint, not
    # fan-in, is the cost driver (confirmed by measurement).
    a = floor._floor(260_000_000, 130_000_000)
    b = floor._floor(130_000_000, 260_000_000)
    assert a.time_ms == b.time_ms


def test_time_is_monotonic_in_traffic():
    small = floor._floor(1_000, 1_000)
    large = floor._floor(1_000_000_000, 1_000_000_000)
    assert 0.0 < small.time_ms < large.time_ms


def test_oversized_prefill_cell_is_finite_without_allocation():
    # num_queries * num_keys * 4 bytes = ~268 GB of output, which OOMs a real
    # torch.empty on a 192 GB MI300X. The closed form must still return a finite,
    # positive, plausible (sub-second) time.
    metrics = floor.profile_dsa_mqa_logits_prefill_floor(
        num_queries=8192,
        num_keys=8_192_000,
        num_heads=64,
        head_dim=128,
        q_dtype="fp8_e4m3",
        k_dtype="fp8_e4m3",
        output_dtype="fp32",
    )
    assert 0.0 < metrics.time_ms < 1000.0


def test_output_has_a_write_pass_floor():
    # A degenerate zero-output footprint is still charged one write byte so the
    # memory movement is never zero.
    metrics = floor._floor(0, 0)
    assert metrics.time_ms > 0.0
