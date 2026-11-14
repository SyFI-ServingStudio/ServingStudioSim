"""CPU tests for the DSA fp32-split state update kind: vendored bytes, geometry gate, acceptance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from profiling.runners.attention.dsa_fp32_split_state_update_triton import (
    _Shape,
    assert_accepted,
    logical_bytes,
    validate_args,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_VENDORED = (
    Path(__file__).resolve().parents[1]
    / "profiling/runners/attention/state_writer_vendor/triton_post_plus.py"
)
# md5 of the origin wt-glm53-decode-investigate/tmp/dsa-fusion/small/kern.py (job 22204).
_ORIGIN_MD5 = "e478822ec89ea169fe20fe1ce67e875f"

_GEOMETRY = dict(q_lora_cols=1536, kv_lora_cols=512, index_dim=128, fp32_tail_cols=32, pool_size=4)


def test_vendored_kernel_is_byte_identical_to_its_origin():
    # Catches a reformat or edit of the vendored kernel: its rows would no
    # longer be the measured (and bit-checked) kernel's.
    assert hashlib.md5(_VENDORED.read_bytes()).hexdigest() == _ORIGIN_MD5


def test_other_row_geometry_is_refused_not_timed():
    # The kernel hard-codes its column offsets; another geometry would time the wrong work.
    validate_args(16, 8, 1536, 512, 128, 32, 4, 4, 64)
    with pytest.raises(ProfilerNotImplemented):
        validate_args(16, 8, 2048, 512, 128, 32, 4, 4, 64)
    with pytest.raises(ProfilerNotImplemented):
        validate_args(16, 8, 1536, 512, 128, 32, 8, 8, 64)
    with pytest.raises(ValueError):
        validate_args(16, 8, 1536, 512, 128, 32, 4, 6, 64)


def test_acceptance_is_exact_except_one_ulp_in_k():
    clean = dict(
        q=0,
        gate=0,
        mla_cache=0,
        index_cache=0,
        tail=0,
        k_bytes=3,
        k_ulp_gt1=0,
        y_modified=0,
        index_rows_written=4,
    )
    assert_accepted(clean)
    for field in ("q", "gate", "mla_cache", "index_cache", "tail", "k_ulp_gt1", "y_modified"):
        with pytest.raises(KernelLaunchFailed, match=field):
            assert_accepted({**clean, field: 1})


def test_logical_bytes_count_rows_and_completed_pools():
    # 2 requests x 4 rows at pool 4: each request completes exactly one pool.
    shape = _Shape(num_requests=2, next_n=4, tail_ring=4, block_size=64, **_GEOMETRY)
    per_row = 4 * 2304 + 2 * 1536 + 512 + 4 * 128 + 4 * 128 + 20
    assert logical_bytes(shape) == 8 * per_row + 2 * (3 * 4 * 128 + 132)
