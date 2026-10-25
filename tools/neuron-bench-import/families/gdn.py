"""Gated DeltaNet rows: logs/bench/gdn.jsonl (bench_gdn.py; device_ms = runtime.device_bench).

Per-rank geometry by the row's tp (L1_CONTRACT.md): num_key_heads = 16 / tp, num_value_heads = 48 /
tp. Bench `gdn_decode_mixer` rows (one token per request) and `gdn_verify_mixer` rows
(`spec_tokens` = L tokens per request, MTP verify) are both the kind `gdn_decode_mixer`, with
query_len 1 and L. The bench's per-op decode rows time one token per request (query_len 1).
`gdn_in_proj` and `gdn_output` are token-count kinds: a decode row of B requests is the same GEMM
as a prefill row of B tokens, so a decode row is kept only where no prefill row has its token
count. `nki_fused` rows (LNC=1 only) have no backend and are dropped.
"""

from __future__ import annotations

from pathlib import Path

from import_bench import ImportRow, device_ms, jsonl

SOURCE = "gdn.jsonl"
HIDDEN, KEY_HEADS, VALUE_HEADS, HEAD_DIM, CONV = 2560, 16, 48, 128, 4
VALID_TP = (1, 2, 4, 8, 16)

# bench kind -> (table, impls the table registers, args from the row's shape)
KINDS = {
    "gdn_prefill_mixer": (
        "gdn_prefill_mixer",
        ("torch", "nki"),
        lambda r: dict(num_tokens=r["tokens"]),
    ),
    "gdn_decode_mixer": (
        "gdn_decode_mixer",
        ("torch", "nki"),
        lambda r: dict(batch_size=r["batch"], query_len=1),
    ),
    "gdn_verify_mixer": (
        "gdn_decode_mixer",
        ("torch", "nki"),
        lambda r: dict(batch_size=r["batch"], query_len=r["spec_tokens"]),
    ),
    "gdn_conv1d_prefill": (
        "gdn_conv1d_prefill",
        ("torch", "nki"),
        lambda r: dict(num_tokens=r["tokens"]),
    ),
    "gdn_conv1d_decode": (
        "gdn_conv1d_decode",
        ("torch", "nki"),
        lambda r: dict(batch_size=r["batch"], query_len=1),
    ),
    "gdn_split_gating": ("gdn_split_gating", ("torch",), lambda r: dict(num_tokens=r["tokens"])),
    "gdn_prefill_core": (
        "gdn_prefill_core",
        ("torch", "nki"),
        lambda r: dict(num_tokens=r["tokens"]),
    ),
    "gdn_decode_core": (
        "gdn_decode_core",
        ("torch", "nki"),
        lambda r: dict(batch_size=r["batch"], query_len=1),
    ),
    "gdn_state_io": ("gdn_state_io", ("torch",), lambda r: dict(batch_size=r["batch"])),
    # token-count kinds: a prefill row has `tokens` tokens, a decode row `batch` requests x 1 token
    "gdn_in_proj": ("gdn_in_proj", ("torch",), None),
    "gdn_output": ("gdn_output", ("torch",), None),
}


def geometry(tp: int) -> dict:
    """The common GDN args of a rank at TP size tp."""
    if tp not in VALID_TP:
        raise ValueError(f"tp {tp} does not split 16 key / 48 value heads")
    return dict(
        hidden_size=HIDDEN,
        num_key_heads=KEY_HEADS // tp,
        num_value_heads=VALUE_HEADS // tp,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV,
        dtype="bf16",
    )


# Analytic per-rank cost. flashnext/neuron/sim_l1/gdn.py sets Region.flops/bytes from the
# same formulas.
CHUNK_FLOP_WIDTH = 64  # intra-chunk work of the chunked delta rule, per token and head dim


def cost(table: str, a: dict) -> tuple[float, float]:
    """(flops, bytes) of one call of `table` at args `a`: minimum logical work and HBM traffic.

    bf16 activations and weights (2 B), fp32 gates and recurrent state (4 B). A whole mixer is the
    sum of its ops plus its state-pool gather and scatter.
    """
    H, hk, hv, d, K = (
        a[k]
        for k in ("hidden_size", "num_key_heads", "num_value_heads", "head_dim", "conv_kernel_size")
    )
    C = (2 * hk + hv) * d  # conv channels
    vd = hv * d
    n_in = C + vd + 2 * hv  # in_proj outputs [qkv; z; b; a]
    state = 2 * C * (K - 1) + 4 * hv * d * d  # one request's conv + ssm state, bytes

    def in_proj(t):
        return 2.0 * t * H * n_in, 2.0 * (t * H + n_in * H + t * n_in)

    def conv(t, n):  # t tokens; n conv states read or written
        return 2.0 * t * C * K, 2.0 * 2 * t * C + 4.0 * C * K + 2.0 * n * C * (K - 1)

    def split(t):
        return 4.0 * t * 2 * hk * d + 10.0 * t * hv, (
            2.0 * t * C + 4.0 * 2 * t * hk * d + 2.0 * t * vd + 2.0 * 2 * t * hv + 4.0 * 2 * t * hv
        )

    def core(t, b, states_out, chunked):  # recurrence over t tokens of b requests
        flops = t * hv * (6.0 * d * d + (4.0 * CHUNK_FLOP_WIDTH * d if chunked else 0.0))
        io = 2.0 * t * (2 * hk * d + 2 * vd) + 4.0 * 2 * t * hv
        return flops, io + 4.0 * hv * d * d * (b + states_out)

    def output(t):
        return 2.0 * t * vd * H, 2.0 * (2 * t * vd + vd * H + t * H)

    def total(*parts):
        return sum(p[0] for p in parts), sum(p[1] for p in parts)

    if table == "gdn_in_proj":
        return in_proj(a["num_tokens"])
    if table == "gdn_conv1d_prefill":
        return conv(a["num_tokens"], 2)
    if table == "gdn_split_gating":
        return split(a["num_tokens"])
    if table == "gdn_prefill_core":
        return core(a["num_tokens"], 1, 1, True)
    if table == "gdn_output":
        return output(a["num_tokens"])
    if table == "gdn_state_io":
        return 0.0, 2.0 * a["batch_size"] * state
    b = a.get("batch_size", 1)
    if table in ("gdn_conv1d_decode", "gdn_decode_core", "gdn_decode_mixer"):
        L = a["query_len"]
        if table == "gdn_conv1d_decode":
            return conv(b * L, b + b * L)
        if table == "gdn_decode_core":
            return total(split(b * L), core(b * L, b, b * L, False))
        t = b * L
        return total(
            in_proj(t),
            conv(t, b + t),
            split(t),
            core(t, b, t, False),
            output(t),
            (0.0, (b + t) * state),
        )
    if table == "gdn_prefill_mixer":
        t = a["num_tokens"]
        return total(
            in_proj(t), conv(t, 2), split(t), core(t, 1, 1, True), output(t), (0.0, 2.0 * state)
        )
    raise KeyError(table)


def shape_args(r: dict) -> tuple[str, dict] | None:
    """(table, shape args) of one bench row, or None when the row maps to no Neuron table."""
    entry = KINDS.get(r.get("kind"))
    if entry is None or r.get("impl") not in entry[1] or r.get("lnc", 2) != 2:
        return None
    table, _, shape = entry
    if shape is None:
        n = r["tokens"] if r.get("phase") == "prefill" else r["batch"]
        return table, dict(num_tokens=int(n))
    return table, {k: int(v) for k, v in shape(r).items()}


def rows(bench_dir: Path):
    decode_token_rows, out = [], []
    for line, r in jsonl(bench_dir / SOURCE):
        ms = device_ms(r)
        if ms is None or r.get("tp", 1) not in VALID_TP:
            continue
        mapped = shape_args(r)
        if mapped is None:
            continue
        table, shape = mapped
        args = shape | geometry(int(r.get("tp", 1)))
        flops, nbytes = cost(table, args)
        row = ImportRow(table, f"neuron_{r['impl']}", args, ms, f"{SOURCE}:{line}", flops, nbytes)
        is_decode_tokens = KINDS[r["kind"]][2] is None and r.get("phase") != "prefill"
        (decode_token_rows if is_decode_tokens else out).append(row)
    # import_bench.latest keeps the last row per key: prefill rows win over decode rows
    # of the same shape.
    yield from decode_token_rows
    yield from out
