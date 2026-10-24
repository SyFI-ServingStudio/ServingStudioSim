"""Closed-loop replay of the RAG-agent conversations against an OpenAI-compatible server.

N lanes each replay one conversation's requests in order, back to back (no think time), so the server holds N
concurrent requests; a finished conversation hands its lane the next one. A lane's first conversation starts at a
seeded random request (its history is then a cold prefill), so the mix reaches mid-conversation contexts at once.
Model-default sampling (no temperature sent, as recorded); exactly the recorded completion length (ignore_eos).
Runs until --seconds elapse, then lets in-flight requests finish. One JSON line per request on stdout.
Usage: replay.py URL REPLAY.jsonl --lanes N --seconds S [--seed K]"""
import argparse, json, random, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("replay")
ap.add_argument("--lanes", type=int, required=True); ap.add_argument("--seconds", type=float, required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--model", default="Qwen3.8-27B-NVFP4")
ap.add_argument("--max-completion", type=int, default=0, help="cap a request's output (0: as recorded)")
a = ap.parse_args()
convs = [json.loads(l) for l in open(a.replay)]
rng = random.Random(a.seed); rng.shuffle(convs)
lock = threading.Lock(); queue = list(convs); deadline = time.monotonic() + a.seconds


def post(req):
    n = req["max_tokens"] if not a.max_completion else min(req["max_tokens"], a.max_completion)
    body = {"model": a.model, "messages": req["messages"], "max_tokens": n, "min_tokens": n, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); first = None; usage = None
    with urllib.request.urlopen(r, timeout=7200) as s:
        for raw in s:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            if first is None and any((c.get("delta") or {}) for c in ev.get("choices") or []):
                first = time.monotonic()
    return t0, first, time.monotonic(), usage or {}


def lane(i):
    start_mid = True
    while time.monotonic() < deadline:
        with lock:
            if not queue: return
            conv = queue.pop()
        reqs = conv["requests"]
        k = rng.randrange(len(reqs)) if start_mid else 0
        start_mid = False
        for j in range(k, len(reqs)):
            if time.monotonic() >= deadline: return
            t0, first, end, u = post(reqs[j])
            line = {"lane": i, "trial": conv["trial"], "request": j, "t0": t0, "ttft_s": (first or end) - t0,
                    "e2e_s": end - t0, "prompt_tokens": u.get("prompt_tokens"),
                    "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "recorded_prompt_tokens": reqs[j]["recorded_prompt_tokens"], "max_tokens": reqs[j]["max_tokens"]}
            with lock:
                print(json.dumps(line), flush=True)


threads = [threading.Thread(target=lane, args=(i,)) for i in range(a.lanes)]
for t in threads: t.start()
for t in threads: t.join()
