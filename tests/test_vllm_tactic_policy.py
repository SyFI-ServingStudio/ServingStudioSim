import pytest

from profiling.runners import tactic_labels, vllm_tactic_policy as p

KIMI = dict(
    num_experts=896, num_local_experts=56, hidden_size=3584, intermediate_size=3072,
    routing_method="renormalize",
)


def test_hybrid_buckets_match_vllm_warmup_log():
    # kernel_warmup.py log of the Kimi K3.1 B200 TP8/DP2 server (budget 4096).
    assert p.hybrid_buckets(4096) == (
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048,
        2560, 3072, 3584, 4096,
    )
    assert p.hybrid_buckets(3000)[-1] == 3000
    assert p.hybrid_buckets(16384)[-3:] == (4096, 8192, 16384)


@pytest.mark.parametrize(
    "x,tune_max,bucket",
    [(1, 8192, 1), (3, 8192, 4), (300, 8192, 512), (2100, 8192, 2560),
     (4097, 8192, 8192), (8192, 8192, 8192), (65536, 8192, 8192), (20000, 16384, 16384)],
)
def test_map_to_hybrid_bucket(x, tune_max, bucket):
    assert p.map_to_hybrid_bucket(x, tune_max) == bucket


def test_kimi_dp2_prefill_misses():
    chunk = p.trtllm_routed_chunk_size(16, 896)
    assert chunk == 300000
    for m, tuned in [(1, True), (4096, True), (4097, False), (8192, False), (65536, False)]:
        d = p.vllm_tactic(m, batch_tokens=4096, dp_size=2, routed_chunk=chunk)
        assert d.tune_max == 8192
        assert d.tuned is tuned, m


def test_tune_max_clamp_hits_largest_bucket():
    # Budget 16384, DP1: tune_max 16384, so M beyond it clamps into the tuned
    # 16384 bucket instead of missing.
    d = p.vllm_tactic(40000, batch_tokens=16384, dp_size=1)
    assert d.bucket == 16384 and d.tuned


def test_vllm_default_budget():
    gib = 1 << 30
    assert p.vllm_default_batch_tokens(179 * gib, "NVIDIA B200") == 16384
    assert p.vllm_default_batch_tokens(80 * gib, "NVIDIA H100 80GB HBM3") == 8192
    assert p.vllm_default_batch_tokens(80 * gib, "NVIDIA A100-SXM4-80GB") == 2048


def test_policy_env_and_labels(monkeypatch):
    monkeypatch.delenv(p.POLICY_ENV, raising=False)
    monkeypatch.delenv(p.BATCH_TOKENS_ENV, raising=False)
    # Unset keeps the runner's own autotune and the existing tactic file name.
    assert p.tactic_policy() == p.POLICY_RUNNER_AUTOTUNE
    base = "nvfp4_fused_moe.vllm.routed.e896.l56.h3584.i3072"
    assert tactic_labels.nvfp4_fused_moe_vllm_routed(**KIMI) == base
    monkeypatch.setenv(p.BATCH_TOKENS_ENV, "4096")
    assert tactic_labels.nvfp4_fused_moe_vllm_routed(**KIMI) == base
    monkeypatch.setenv(p.POLICY_ENV, p.POLICY_VLLM)
    assert p.tactic_policy() == p.POLICY_VLLM
    assert tactic_labels.nvfp4_fused_moe_vllm_routed(**KIMI) == base + ".vllmtac-n4096"
    monkeypatch.delenv(p.BATCH_TOKENS_ENV)
    assert tactic_labels.nvfp4_fused_moe_vllm_routed(**KIMI) == base + ".vllmtac-nauto"
    monkeypatch.setenv(p.POLICY_ENV, "bogus")
    with pytest.raises(ValueError):
        p.tactic_policy()
