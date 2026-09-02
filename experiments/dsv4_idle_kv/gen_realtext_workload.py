#!/usr/bin/env python3
"""Build deterministic real-text token-ID workloads (no random filler).

Token IDs are taken from a frozen corpus of public-domain prose + real code.
When the DeepSeek-V4 tokenizer is installed it is used; otherwise UTF-8 bytes
are packed into 16-bit ids (still real text, not a cycling vocab).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
OUT_DIR = HERE / "workloads"


def load_corpus_bytes() -> bytes:
    parts = []
    for name in ("prose.txt", "code.py"):
        p = CORPUS_DIR / name
        parts.append(p.read_bytes())
        parts.append(b"\n")
    data = b"".join(parts)
    if len(data) < 2048:
        raise SystemExit(f"corpus too small: {len(data)} bytes")
    return data


def tokenize(raw: bytes) -> list[int]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V4-Flash-0731", trust_remote_code=True
        )
        ids = tok.encode(raw.decode("utf-8", errors="replace"), add_special_tokens=False)
        if ids:
            return [int(x) for x in ids]
    except Exception as exc:  # noqa: BLE001
        print(f"V4 tokenizer unavailable ({exc}); using utf-8 packed ids", file=sys.stderr)
    # Real bytes of the corpus, packed two-at-a-time. Not random, not "alpha ".
    ids = []
    blob = raw if len(raw) % 2 == 0 else raw + b"\n"
    for i in range(0, len(blob), 2):
        ids.append(blob[i] | (blob[i + 1] << 8))
    return ids


def offset_for(session_key: str, pool_len: int, span: int) -> int:
    digest = hashlib.sha256(f"dsv4-idle-kv-v1:{session_key}".encode()).hexdigest()
    return int(digest[:16], 16) % max(1, pool_len - span)


def take(pool: list[int], start: int, n: int) -> list[int]:
    out = []
    i = start
    for _ in range(n):
        out.append(pool[i % len(pool)])
        i += 1
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def idle_rows(pool: list[int], n_sess: int, n_rounds: int, wait_h_ms: int, wait_l_ms: int, prefix0: int = 8192) -> list[dict]:
    rows = []
    prefix0, append = prefix0, 512
    out_tok = 128
    for s in range(n_sess):
        key = f"idle-{s:04d}"
        off = offset_for(key, len(pool), prefix0 + n_rounds * append + 8)
        ctx: list[int] = []
        for r in range(n_rounds):
            need = prefix0 + (r + 1) * append
            if len(ctx) < need:
                ctx.extend(take(pool, off + len(ctx), need - len(ctx)))
            wait = wait_h_ms if (s + r) % 2 == 0 else wait_l_ms
            rows.append(
                {
                    "session_id": key,
                    "round": r,
                    "token_ids": ctx[:],
                    "output_tokens": out_tok,
                    "wait_ms": wait,
                    "gpu_hint": 0 if s < (n_sess * 3) // 4 else 1,
                }
            )
    return rows


def fork_rows(pool: list[int], n_stems: int, n_forks: int, stem_tokens: int) -> list[dict]:
    rows = []
    tail = 400
    for stem_i in range(n_stems):
        stem_key = f"stem-{stem_i:02d}"
        stem_off = offset_for(stem_key, len(pool), stem_tokens + n_forks * tail)
        stem = take(pool, stem_off, stem_tokens)
        for f in range(n_forks):
            sid = f"{stem_key}-fork-{f}"
            tail_ids = take(pool, stem_off + stem_tokens + f * tail, tail)
            rows.append(
                {
                    "session_id": sid,
                    "stem_id": stem_key,
                    "round": 0,
                    "token_ids": stem + tail_ids,
                    "stem_len": stem_tokens,
                    "output_tokens": 128,
                    "wait_ms": 8000,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-h-ms", type=int, default=8000)
    ap.add_argument("--wait-l-ms", type=int, default=200)
    args = ap.parse_args()
    pool = tokenize(load_corpus_bytes())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "idle_24x8.jsonl", idle_rows(pool, 24, 8, args.wait_h_ms, args.wait_l_ms, prefix0=8192))
    write_jsonl(OUT_DIR / "idle_64x8.jsonl", idle_rows(pool, 64, 8, args.wait_h_ms, args.wait_l_ms, prefix0=32768))
    write_jsonl(OUT_DIR / "fork_s8_f4.jsonl", fork_rows(pool, 8, 4, 3000))
    manifest = {
        "corpus_files": ["corpus/prose.txt", "corpus/code.py"],
        "pool_tokens": len(pool),
        "tokenizer": "deepseek_v4_or_utf8_packed",
        "note": "token_ids are real corpus text, not random/filler",
    }
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_DIR} pool_tokens={len(pool)}")


if __name__ == "__main__":
    main()
