"""CPU checks of the DeepInfer prefill-indexer kinds' shape math and argument gates."""

import pytest

from profiling.runners.attention import deepinfer_indexer_common as common
from profiling.runners.attention import (
    candidate_block_max_tilelang,
    deep_select_topk_cuda,
    mqa_logits_prefill_chunk_deepgemm,
    sparse_mqa_logits_prefill_deepgemm,
)


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
