import pytest
import torch

from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.attention import paged_attn_batch_trtllm_gen as runner
from profiling.runners.attention.fa4_varlen_attention_reference import varlen_attention_reference
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_GEOMETRY = dict(
    num_qo_heads=6,
    num_kv_heads=2,
    head_dim=64,
    q_dtype="fp8_e4m3",
    kv_dtype="fp8_e4m3",
    o_dtype="bf16",
)


def _decode(**overrides):
    args = dict(_GEOMETRY, batch_size=4, max_kv_len=100, total_kv_len=250)
    args.update(overrides)
    return runner.validate_decode_args(**args)


def _prefill(**overrides):
    args = dict(_GEOMETRY, requests=((40, 4), (0, 3), (7, 3)))
    args.update(overrides)
    return runner.validate_prefill_args(**args)


def test_both_kinds_register_the_trtllm_gen_backend_on_vllm_env():
    # Defect caught: a kind missing from the barrel, or run on an env without
    # FlashInfer's trtllm-gen cubins.
    for kind in ("paged_attn_batch_decode", "paged_attn_batch_prefill"):
        spec = find_kernel_profiler_spec(kind, "flashinfer_trtllm_gen")
        assert spec.backend == "flashinfer_trtllm_gen"
        assert spec.subprocess_env == "vllm_env"
        assert {str(d) for d in spec.supports.compute} == {"fp8_e4m3"}


def test_canonical_decode_batch_holds_the_key():
    # Defect caught: a canonical batch whose longest or total context drifts
    # from the row's key.
    for batch, longest, total in [(1, 7, 7), (4, 100, 250), (5, 1000, 1004), (3, 9, 27)]:
        lengths = _decode(batch_size=batch, max_kv_len=longest, total_kv_len=total).lengths
        assert len(lengths) == batch
        assert lengths[0] == longest == max(lengths)
        assert sum(lengths) == total
        assert max(lengths[1:], default=0) - min(lengths[1:], default=0) <= 1


def test_prefill_batch_reads_each_request_after_its_prefix():
    # Defect caught: a context length or query-key count that ignores one
    # request's own prefix.
    shape = _prefill()
    assert shape.appends == [4, 3, 3]
    assert shape.lengths == [44, 3, 10]
    assert shape.interactions == (40 * 4 + 10) + 6 + (7 * 3 + 6)


@pytest.mark.parametrize(
    "overrides, error",
    [
        (dict(total_kv_len=102), ValueError),
        (dict(total_kv_len=401), ValueError),
        (dict(q_dtype="bf16"), ProfilerNotImplemented),
        (dict(num_qo_heads=5), ValueError),
        (
            dict(batch_size=4, max_kv_len=1 << 30, total_kv_len=(1 << 30) + 3),
            ProfilerNotImplemented,
        ),
    ],
)
def test_decode_validation_rejects_unprofilable_args(overrides, error):
    with pytest.raises(error):
        _decode(**overrides)


@pytest.mark.parametrize(
    "requests, error",
    [((), ValueError), (((3, 0),), ValueError), (((-1, 4),), ValueError), (([0, 4],), TypeError)],
)
def test_prefill_validation_needs_new_tokens_per_request(requests, error):
    with pytest.raises(error):
        _prefill(requests=requests)


def _workload(lengths, tokens):
    geometry = _decode().geometry
    return geometry, runner._build_workload(torch, geometry, lengths, tokens, torch.device("cpu"))


def _reference_out(geometry, work, lengths, positions_per_request, row_starts):
    """Fill ``work.out`` from a direct read of the HND cache through the page table."""
    page = runner._PAGE_SIZE
    for request, (positions, start) in enumerate(zip(positions_per_request, row_starts)):
        context = lengths[request]
        pages = work.block_table[request].long()[torch.arange(context) // page]
        offsets = torch.arange(context) % page
        dim = geometry.head_dim
        keys = work.raw_cache[pages, :, offsets, :dim].float()
        values = work.raw_cache[pages, :, offsets, dim:].float()
        rows = torch.arange(start, start + len(positions))
        out = varlen_attention_reference(
            work.q[rows].float(),
            torch.tensor(positions),
            keys,
            values,
            softmax_scale=dim**-0.5,
            window=0,
        )
        work.out[rows] = out.to(work.out.dtype)


def test_decode_check_reads_the_hnd_cache_through_the_page_table():
    # Defect caught: the check reading K/V with the wrong layout (HND pages
    # viewed as NHD) or the wrong page, which would pass a broken kernel or
    # fail a correct one.
    shape = _decode(batch_size=3, max_kv_len=70, total_kv_len=110)
    geometry, work = _workload(shape.lengths, shape.batch_size)
    positions = [[length - 1] for length in shape.lengths]
    _reference_out(geometry, work, shape.lengths, positions, range(shape.batch_size))
    report = runner.check_decode(torch, shape, work)
    assert report["rows_checked"] == 3 and report["max_rel_l2"] < 0.01
    work.out[1] = work.out[1].flip(-1)
    with pytest.raises(KernelLaunchFailed):
        runner.check_decode(torch, shape, work)


def test_prefill_check_places_rows_after_the_prefix():
    # Defect caught: query rows checked at positions that ignore the cached
    # prefix or the request's start in the packed query.
    shape = _prefill(requests=((33, 5), (2, 4)))
    geometry, work = _workload(shape.lengths, shape.query_tokens)
    positions = [list(range(p, p + a)) for p, a in shape.requests]
    _reference_out(geometry, work, shape.lengths, positions, [0, shape.appends[0]])
    report = runner.check_prefill(torch, shape, work)
    assert report["rows_checked"] == 9 and report["max_rel_l2"] < 0.01
    work.out[shape.appends[0] + 1] = 0
    with pytest.raises(KernelLaunchFailed):
        runner.check_prefill(torch, shape, work)


# --- flashinfer_fa2 (vLLM's native FlashInfer path off TRTLLM, e.g. L40S) ---

from profiling.runners.attention import paged_attn_batch_fa2 as fa2  # noqa: E402

_FA2_GEOMETRY = dict(_GEOMETRY, q_dtype="bf16")


def test_fa2_backend_runs_bf16_queries_over_fp8_kv_on_sm80_and_newer():
    # Defect caught: the fa2 rows gated to the trtllm-gen dtypes/architecture,
    # or run outside the env that pins vLLM's FlashInfer.
    for kind in ("paged_attn_batch_decode", "paged_attn_batch_prefill"):
        spec = find_kernel_profiler_spec(kind, "flashinfer_fa2")
        assert spec.subprocess_env == "vllm_env"
        assert spec.runner_ref.module_name == "profiling.runners.attention.paged_attn_batch_fa2"
        assert spec.supports.allows("bf16", "fp8_e4m3", gpu="NVIDIA L40S")
        assert not spec.supports.allows("fp8_e4m3", "fp8_e4m3")


def test_fa2_validation_rejects_the_fp8_query_vllm_never_sends_it():
    with pytest.raises(ProfilerNotImplemented):
        fa2.validate_decode_args(**_GEOMETRY, batch_size=2, max_kv_len=10, total_kv_len=12)
    with pytest.raises(ProfilerNotImplemented):
        fa2.validate_prefill_args(**_GEOMETRY, requests=((0, 4),))
    shape = fa2.validate_decode_args(
        **_FA2_GEOMETRY, batch_size=4, max_kv_len=100, total_kv_len=250
    )
    assert shape.lengths == [100, 50, 50, 50]


def test_fa2_page_table_keeps_each_cache_block_whole():
    # Defect caught: kernel pages scattered one by one, unlike vLLM's view of a
    # 1568-token block as 49 consecutive 32-token pages, or a last-page length
    # / indptr that disagrees with the request lengths.
    generator = torch.Generator().manual_seed(1)
    lengths = [1, 32, 33, 49 * 32 + 5]
    table = fa2.page_table(torch, lengths, generator)
    assert table.page_counts == [1, 1, 2, 50]
    assert table.indptr.tolist() == [0, 1, 2, 4, 54]
    assert table.last_page_len.tolist() == [1, 32, 1, 5]
    pages = table.indices.tolist()
    assert min(pages) >= fa2._BLOCK_PAGES  # block 0 is the null block
    long_request = pages[4:]
    assert long_request[:49] == list(range(long_request[0], long_request[0] + 49))
    assert long_request[0] % 49 == 0 and long_request[49] % 49 == 0
    assert len(set(pages)) == len(pages) and max(pages) < table.num_pages


def _fa2_reference_out(work, lengths, positions_per_request, row_starts, dim):
    """Fill ``work.out`` from a direct read of the NHD cache through the CSR table."""
    page = fa2._PAGE_SIZE
    indptr = work.table.indptr.tolist()
    for request, (positions, start) in enumerate(zip(positions_per_request, row_starts)):
        context = lengths[request]
        own = work.table.indices[indptr[request] : indptr[request + 1]].long()
        pages = own[torch.arange(context) // page]
        offsets = torch.arange(context) % page
        keys = work.raw_cache[pages, offsets, :, :dim].float()
        values = work.raw_cache[pages, offsets, :, dim:].float()
        rows = torch.arange(start, start + len(positions))
        out = varlen_attention_reference(
            work.q[rows].float(),
            torch.tensor(positions),
            keys,
            values,
            softmax_scale=dim**-0.5,
            window=0,
        )
        work.out[rows] = out.to(work.out.dtype)


def test_fa2_checks_read_the_nhd_cache_after_each_prefix():
    # Defect caught: the check reading the packed NHD cache as HND, the wrong
    # half for V, or query rows placed without their request's prefix.
    shape = fa2.validate_prefill_args(**_FA2_GEOMETRY, requests=((70, 5), (2, 4)))
    work = fa2._build_workload(
        torch, shape.geometry, shape.lengths, shape.query_tokens, torch.device("cpu")
    )
    positions = [list(range(p, p + a)) for p, a in shape.requests]
    _fa2_reference_out(work, shape.lengths, positions, [0, shape.appends[0]], 64)
    report = fa2.check_prefill(torch, shape, work)
    assert report["rows_checked"] == 9 and report["max_rel_l2"] < 0.01
    work.out[shape.appends[0] + 1] = work.out[shape.appends[0] + 1].flip(-1)
    with pytest.raises(KernelLaunchFailed):
        fa2.check_prefill(torch, shape, work)
