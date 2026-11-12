"""CPU checks of the gated hyper-connection references against per-stream loops.

The GPU runners trust these references to reject a wrong production result, so a
reference that mixed up the stream-outer layout (``[tokens, hc_mult * hidden]``)
or the injection coefficient would let a wrong kernel through.
"""

import torch

from profiling.runners.mhc import hc_reference as ref

TOKENS, HIDDEN, STREAMS, EPS = 3, 8, 4, 1e-6


def _streams(x: torch.Tensor) -> list[torch.Tensor]:
    return [x[:, j * HIDDEN : (j + 1) * HIDDEN].float() for j in range(STREAMS)]


def _gemma_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * (1.0 + w.float())


def test_combine_norm_matches_a_per_stream_loop() -> None:
    gen = torch.Generator().manual_seed(0)
    residual = torch.randn(TOKENS, STREAMS * HIDDEN, generator=gen).bfloat16()
    block = torch.randn(TOKENS, HIDDEN, generator=gen).bfloat16()
    logits = torch.randn(TOKENS, STREAMS, generator=gen).bfloat16()
    weight = (0.1 * torch.randn(STREAMS * HIDDEN, generator=gen)).bfloat16()

    combined, normed = ref.hc_combine_norm_reference(
        torch, residual, block, logits, weight, EPS, STREAMS
    )
    for j, stream in enumerate(_streams(residual)):
        coeff = 2.0 * torch.sigmoid(logits[:, j : j + 1].float() / STREAMS)
        expected = (stream + coeff * block.float()).bfloat16()
        torch.testing.assert_close(combined[:, j * HIDDEN : (j + 1) * HIDDEN], expected)
        w = weight[j * HIDDEN : (j + 1) * HIDDEN]
        torch.testing.assert_close(
            normed[:, j * HIDDEN : (j + 1) * HIDDEN].float(),
            _gemma_norm(expected.float(), w),
            rtol=2e-2,
            atol=2e-2,
        )

    unit = ref.hc_combine_reference(torch, residual, block, None, STREAMS)
    for j, stream in enumerate(_streams(residual)):
        torch.testing.assert_close(
            unit[:, j * HIDDEN : (j + 1) * HIDDEN], (stream + block.float()).bfloat16()
        )


def test_gate_mix_averages_the_gated_streams() -> None:
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(TOKENS, STREAMS * HIDDEN, generator=gen).bfloat16()
    gate = torch.randn(TOKENS, STREAMS * HIDDEN, generator=gen).bfloat16()
    expected = (
        sum(torch.sigmoid(g) * s for g, s in zip(_streams(gate), _streams(x), strict=True))
        / STREAMS
    )
    torch.testing.assert_close(
        ref.hc_gate_mix_reference(torch, x, gate, STREAMS).float(),
        expected,
        rtol=1e-2,
        atol=1e-2,
    )
