"""CPU checks of the DeepInfer prefill-indexer kinds' shape math and argument gates."""

import pytest

from profiling.runners.attention import (
    candidate_block_max_tilelang,
    deep_select_topk_cuda,
    mqa_logits_prefill_chunk_deepgemm,
    sparse_mqa_logits_prefill_deepgemm,
)
from profiling.runners.attention import deepinfer_indexer_common as common
from profiling.runners.attention import deepinfer_paged_indexer_common as paged
from profiling.runners.attention import dsa_paged_mqa_logits_decode_deepinfer as dense_decode
from profiling.runners.attention import sparse_mqa_logits_decode_deepgemm as sparse_decode


def test_tail_row_keys_matches_deepinfer_ke():
    # prepare_cp_mirror _write_ks_ke: ke - ks = (sk - sq + i + 1) // ratio.
    assert common.tail_row_keys(4, 10, 1) == [7, 8, 9, 10]
    assert common.tail_row_keys(4, 10, 2) == [3, 4, 4, 5]
    assert common.tail_row_keys(1, 131072, 2) == [65536]
    with pytest.raises(ValueError):
        common.tail_row_keys(11, 10, 1)


def test_sparse_candidate_math():
    sparse = sparse_mqa_logits_prefill_deepgemm
    assert sparse.row_valid_blocks(3, 17, 1, 8) == [2, 2, 3]
    # Rows past candidate_blocks * block_size keys score only the candidates.
    assert sparse.candidate_tokens(2, 100, 1, 4, 8) == 32 + 32
    assert sparse.candidate_tokens(2, 20, 1, 4, 8) == 19 + 20


@pytest.mark.parametrize(
    "call",
    [
        lambda: mqa_logits_prefill_chunk_deepgemm.validate(8, 4, 1, 32, 128, "mxfp4_e2m1"),
        lambda: mqa_logits_prefill_chunk_deepgemm.validate(8, 64, 1, 32, 64, "mxfp4_e2m1"),
        lambda: mqa_logits_prefill_chunk_deepgemm.validate(8, 64, 1, 32, 128, "fp8_e4m3"),
        lambda: sparse_mqa_logits_prefill_deepgemm.validate(8, 64, 1, 2047, 8, "mxfp4_e2m1"),
        lambda: sparse_mqa_logits_prefill_deepgemm.validate(8, 64, 1, 2048, 32, "mxfp4_e2m1"),
        lambda: candidate_block_max_tilelang.validate(8, 64, 1, 4, "bf16"),
        lambda: deep_select_topk_cuda.validate(8, 64, 8192, "bf16"),
        lambda: deep_select_topk_cuda.validate(8, 64, 512, "fp32"),
    ],
)
def test_validate_rejects_unsupported(call):
    with pytest.raises(ValueError):
        call()


def test_validate_accepts_deepinfer_shapes():
    assert mqa_logits_prefill_chunk_deepgemm.validate(8192, 131072, 2, 32, 128, "mxfp4_e2m1")
    assert sparse_mqa_logits_prefill_deepgemm.validate(140, 131072, 1, 2048, 8, "mxfp4_e2m1")
    assert candidate_block_max_tilelang.validate(140, 131072, 1, 8, "bf16")
    assert deep_select_topk_cuda.validate(140, 16384, 2048, "bf16")


# ---------------------------------------------------------------- decode (paged)


_FP4_DECODE = dict(
    q_dtype="mxfp4_e2m1",
    cache_dtype="mxfp4_e2m1",
    scale_dtype="mxfp4_e2m1",
    weight_dtype="bf16",
    output_dtype="bf16",
    context_mode="uniform",
    page_mapping="unique_scattered",
    cache_format="page_planar_fp4_ue8m0_scale",
    clean_logits=False,
)


def test_decode_rows_group_into_requests_consecutively():
    # DeepGEMM's SM100 paged scheduler groups consecutive equal indices into one
    # request; interleaved rows would split every request into next_n requests.
    assert paged.request_indices(3, 2) == [0, 0, 1, 1, 2, 2]
    # A 64-key FP4 page: 64 x 64 packed bytes + 64 int32 scale words.
    assert paged.page_bytes(64, 128) == 64 * 68


def test_dense_decode_reads_each_requests_keys_once():
    # Keys are page-rounded and shared by the request's next_n rows.
    assert dense_decode.scheduled_keys(2, 65, 64) == 2 * 128


def test_sparse_decode_work_saturates_at_the_candidate_width():
    assert sparse_decode.candidate_tokens(1, 1 << 20, 6, 2048, 8) == 6 * 16384
    assert sparse_decode.candidate_tokens(1, 100, 6, 2048, 8) == 6 * 100


@pytest.mark.parametrize(
    "call",
    [
        # FP8 identity belongs to deepgemm_fp8, not this backend.
        lambda: dense_decode.validate(
            64, 4096, 6, 8192, 32, 128, 64, **dict(_FP4_DECODE, q_dtype="fp8_e4m3")
        ),
        lambda: dense_decode.validate(64, 4096, 6, 8192, 32, 128, 96, **_FP4_DECODE),
        lambda: dense_decode.validate(64, 9000, 6, 8192, 32, 128, 64, **_FP4_DECODE),
        lambda: dense_decode.validate_metadata(64, 6, 256),
        lambda: sparse_decode.validate(64, 4096, 6, 128, 2047, 8, "mxfp4_e2m1"),
        lambda: sparse_decode.validate(64, 4096, 6, 4, 2048, 8, "mxfp4_e2m1"),
    ],
)
def test_decode_validate_rejects_unsupported(call):
    with pytest.raises(ValueError):
        call()


def test_decode_validate_accepts_deepinfer_shapes():
    assert dense_decode.validate(64, 65890, 6, 524288, 32, 128, 64, **_FP4_DECODE)
    assert dense_decode.validate_metadata(64, 6, 128)
    assert sparse_decode.validate(64, 131780, 6, 128, 2048, 8, "mxfp4_e2m1")
