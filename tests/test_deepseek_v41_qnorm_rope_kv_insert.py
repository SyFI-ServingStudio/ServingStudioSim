import pytest
import torch

from profiling.runners.attention import deepseek_v41_qnorm_rope_kv_insert as runner
from profiling.runners.exceptions import ProfilerNotImplemented

_ARGS = dict(
    num_tokens=48,
    num_insert_tokens=48,
    num_heads=16,
    padded_heads=64,
    block_size=32,
    input_dtype="bf16",
    swa_cache_format="mxfp8",
)


def test_padded_q_puts_pad_heads_at_every_chunk_tail():
    # Defect caught: padding after the live heads (head-major) instead of at the
    # tail of each 16-dim chunk, which the mega-attention kernel reads.
    q = torch.arange(2 * 512, dtype=torch.float32).view(1, 2, 512).to(torch.bfloat16)
    out = runner.reference_q(torch, q, 4).view(-1)
    chunk = out[: 4 * 16].view(4, 16)
    assert torch.equal(chunk[0], q[0, 0, :16])
    assert torch.equal(chunk[1], q[0, 1, :16])
    assert torch.all(chunk[2:] == 0)
    assert torch.equal(out[4 * 16 : 4 * 16 + 16], q[0, 0, 16:32])


def test_kv_rope_rotates_gptj_pairs_of_the_last_64_dims_only():
    # Defect caught: NeoX halves instead of GPT-J pairs, or rotating NoPE dims.
    kv = torch.randn(1, 512).to(torch.bfloat16)
    angle = torch.full((1, 32), 0.5)
    cos_sin = torch.cat((angle.cos(), angle.sin()), dim=-1)
    rows = runner.reference_kv_rows(torch, kv, torch.tensor([0]), cos_sin).float()
    assert torch.equal(rows[0, :448], kv[0, :448].float())
    x, y = kv[0, 448].float(), kv[0, 449].float()
    c, s = angle[0, 0].cos(), angle[0, 0].sin()
    torch.testing.assert_close(rows[0, 448], (x * c - y * s).to(torch.bfloat16).float())
    torch.testing.assert_close(rows[0, 449], (x * s + y * c).to(torch.bfloat16).float())


def test_padded_heads_follow_the_layer_rule():
    # Defect caught: accepting a head count the layer never launches, e.g. a
    # KV-only launch while the shard still needs padding.
    runner._validate_args(**_ARGS)
    runner._validate_args(**dict(_ARGS, num_heads=64, padded_heads=0))
    with pytest.raises(ValueError, match="padded_heads"):
        runner._validate_args(**dict(_ARGS, padded_heads=0))
    with pytest.raises(ValueError, match="padded_heads"):
        runner._validate_args(**dict(_ARGS, padded_heads=128))
    with pytest.raises(ValueError, match="num_insert_tokens"):
        runner._validate_args(**dict(_ARGS, num_insert_tokens=49))
    with pytest.raises(ProfilerNotImplemented, match="mxfp8"):
        runner._validate_args(**dict(_ARGS, swa_cache_format="fp8_ds_mla"))
