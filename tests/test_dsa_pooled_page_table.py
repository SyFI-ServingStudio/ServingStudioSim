"""CPU checks for the pooled DSA page-table kinds (vendored indexer-back kernels).

The kernels need an SM100 GPU; these tests pin the argument contracts, the shapes the runners
refuse, the vendored source, and the Torch reference against a per-slot Python statement of the
fork chain (expand pools + tail, remap, compact to front).
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

from profiling.kernels.dsa_pooled_logits_page_table import DsaPooledLogitsPageTableArgs
from profiling.kernels.dsa_pooled_topk_page_table import DsaPooledTopkPageTableArgs
from profiling.runners.attention import dsa_pooled_page_table as runner
from profiling.runners.attention.dsa_pooled_page_table_reference import pooled_page_table_reference

_PT = runner.profile_dsa_pooled_topk_page_table_cuda_row_scan_compact
_LG = runner.profile_dsa_pooled_logits_page_table_cuda_threshold_filter_radix
_PT_ARGS = dict(batch_size=2, context_len=600, next_n=8, max_model_len=1 << 20, top_k=512, kpool=4,
                page_table_width=2176, block_size=64, index_dtype="int32")
_LG_ARGS = dict(_PT_ARGS, logits_row_stride=1 << 18, logits_dtype="fp32", context_mode="uniform_measured")


def test_args_fields_are_the_runner_kwargs() -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    assert list(inspect.signature(_PT).parameters) == [f.name for f in fields(DsaPooledTopkPageTableArgs)]
    assert list(inspect.signature(_LG).parameters) == [f.name for f in fields(DsaPooledLogitsPageTableArgs)]


def test_vendored_source_is_present_and_unchanged_header() -> None:
    # Defect: the runner shipped without the kernel it builds, or without its provenance.
    text = runner._SOURCE.read_text()
    assert text.startswith("// VENDORED")
    assert "Verbatim below this header; no changes." in text.splitlines()[2]
    for entry in ('m.def("e2a", &e2a);', 'm.def("topk6", &topk6);'):
        assert entry in text


@pytest.mark.parametrize(
    "override",
    [{"top_k": 256}, {"kpool": 8}, {"block_size": 32}, {"page_table_width": 2048}, {"index_dtype": "int64"},
     {"context_len": 1}, {"context_len": 1 << 19}],
)
def test_rejects_shapes_the_kernels_are_not_built_for(override: dict) -> None:
    # Defect: timing a kernel compiled for 512 pools x 4 tokens / 64-token blocks on another shape.
    with pytest.raises(ValueError):
        _PT(**{**_PT_ARGS, **override})
    with pytest.raises(ValueError):
        _LG(**{**_LG_ARGS, **override})


@pytest.mark.parametrize("override", [{"context_mode": "uniform"}, {"logits_row_stride": 602}, {"logits_dtype": "bf16"}])
def test_logits_runner_rejects_bad_logits_layouts(override: dict) -> None:
    with pytest.raises(ValueError):
        _LG(**{**_LG_ARGS, **override})


def _slow_reference(pool, seq, req, bt, *, kpool, block_size, stride, width):
    rows, top_k = pool.shape
    out = torch.full((rows, width), -1, dtype=torch.int64)
    counts = torch.zeros(rows, dtype=torch.int64)
    for r in range(rows):
        toks = []
        for c in range(top_k * kpool):
            p = int(pool[r, c // kpool])
            t = p * kpool + c % kpool
            t = (t + 2**31) % 2**32 - 2**31
            toks.append(t if p >= 0 else -1)
        s = int(seq[r])
        start = (s // kpool) * kpool
        toks += [start + j if j < s - start else -1 for j in range(kpool - 1)]
        d = 0
        for t in toks:
            if t >= 0 and t // block_size < bt.shape[1]:
                out[r, d] = int(bt[int(req[r]), t // block_size]) * stride + t % block_size
                d += 1
        counts[r] = d
    return out.to(torch.int32), counts.to(torch.int32)


def test_reference_matches_the_fork_chain_statement() -> None:
    # Defect: an oracle that misplaces the tail, keeps out-of-table blocks, or reorders compaction.
    g = torch.Generator().manual_seed(3)
    rows, top_k, kpool, width = 6, 8, 4, 40
    bt = torch.randperm(4 * 12, generator=g).to(torch.int32).view(4, 12)
    pool = torch.randint(0, 40, (rows, top_k), generator=g, dtype=torch.int32)
    pool[0, 5:] = -1
    pool[1, 2] = 200  # block beyond the table -> invalid
    pool[2, 0] = 2**29 + 7  # int32 wrap of pid * kpool
    seq = torch.tensor([0, 1, 2, 3, 161, 170], dtype=torch.int32)
    req = torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.int32)
    for stride in (16, 48):
        got = pooled_page_table_reference(pool, seq, req, bt, kpool=kpool, block_size=16, block_stride=stride,
                                          page_table_width=width)
        want = _slow_reference(pool, seq, req, bt, kpool=kpool, block_size=16, stride=stride, width=width)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
