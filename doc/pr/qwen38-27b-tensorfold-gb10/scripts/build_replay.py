"""Turn the RAG-agent trajectories into replayable chat requests (read-only over the extracted archive).

One conversation -> its LLM calls in order. Call k's prompt is the conversation's messages before its k-th
assistant turn (the last call is the wrap-up turn, from wrapup_messages). Assistant turns keep their visible
content and tool_calls; tool results and user turns are verbatim. The original system prompt and MCP tool schemas
were not recorded, so a fixed stand-in system message (`--system-tokens`, about the recorded first prompt minus its
user turn) takes their place; each request keeps the recorded prompt/completion token counts so the replay can
report how close it lands. Usage: build_replay.py ARCHIVE_DIR OUT.jsonl"""
import argparse, csv, collections, glob, json

ap = argparse.ArgumentParser()
ap.add_argument("archive"); ap.add_argument("out")
ap.add_argument("--system-chars", type=int, default=5200)
a = ap.parse_args()
rows = collections.defaultdict(list)
for r in csv.DictReader(open(f"{a.archive}/requests.csv")):
    rows[r["trial"]].append(r)
filler = ("You are the eMerge reporting assistant. Answer questions about the HR and finance data using the "
          "run_sql, describe_view, kb_search and kb_fetch tools. ")
system = {"role": "system", "content": (filler * (a.system_chars // len(filler) + 1))[:a.system_chars]}

def clean(m):
    out = {"role": m["role"]}
    if m.get("content") is not None: out["content"] = m["content"]
    if m.get("tool_calls"): out["tool_calls"] = m["tool_calls"]
    if m.get("tool_call_id"): out["tool_call_id"] = m["tool_call_id"]
    if m["role"] == "assistant" and "content" not in out: out["content"] = ""
    return out

n_conv = n_req = 0
with open(a.out, "w") as out:
    for path in sorted(glob.glob(f"{a.archive}/trials/*/*/agent/conversation.json")):
        trial = path.split("/")[-3]
        c = json.load(open(path))
        recorded = sorted(rows[trial], key=lambda r: int(r["request_index"]))
        msgs = [clean(m) for m in c["messages"]]
        turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][1:]   # [0]: scripted greeting
        calls = [msgs[:i] for i in turns]
        wrap = [clean(m) for m in c.get("wrapup_messages", [])]
        for j, m in enumerate(wrap):
            if m["role"] == "assistant":
                calls.append(msgs + wrap[:j])
        if len(calls) != len(recorded):
            continue
        reqs = [{"messages": [system] + p, "max_tokens": int(r["completion_tokens"]),
                 "recorded_prompt_tokens": int(r["prompt_tokens"])} for p, r in zip(calls, recorded)]
        out.write(json.dumps({"trial": trial, "requests": reqs}) + "\n")
        n_conv += 1; n_req += len(reqs)
print(f"{n_conv} conversations, {n_req} requests -> {a.out}")
