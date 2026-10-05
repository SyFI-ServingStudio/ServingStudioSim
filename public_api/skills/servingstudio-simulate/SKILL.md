---
name: servingstudio-simulate
description: >-
  Simulate an LLM serving deployment (model, GPU, parallelism, worker, replicas)
  on a request workload through the ServingStudio public API: a recorded
  capture, a generated trace or an uploaded CSV, with TTFT, TPOT, end-to-end
  latency and throughput. Not for the time of one batch (that is a timing
  prediction).
---

# ServingStudio Simulation

A ServingStudio simulation replays a stream of requests through one deployment
and reports what a client would see: time to first token (TTFT), time per
output token (TPOT), end-to-end latency, and throughput. A deployment is one
member of a sim preset: a checkpoint, a deployment shape (`unified` or `pd`),
its pools of GPUs, the worker each pool runs (the scheduler: batching, chunked
prefill, speculative decoding) and one value of each parameter the preset
sweeps, such as `replicas` or `tp_size`. Every kernel time comes from
measurements. Use this skill to answer questions such as "what TTFT and TPOT
does GLM-5.2 NVFP4 on 4 B200s give at 2 requests/s with 4k prompts?"

The API is JSON over HTTPS. Set the base URL once:

```bash
API=${SERVINGSTUDIO_API:-https://servingstudio.cs.washington.edu/api/public/v1}
curl -s "$API/health"
```

If `/health` does not return `{"status": "ok", ...}`, ask the user for the base
URL; do not guess another host.

A simulation runs in the background: `POST /simulate` queues it and answers at
once, and you read it with `GET /simulations/{id}` until it is done. Two run at
a time, each for at most 10 minutes of wall clock and 2000 requests; results
are kept 24 hours. `POST /simulate` and `POST /workloads` take 6 requests per
minute per client; a refused request does not count.

## 1. Find a deployment

`/simulations/presets` lists every sim preset with its `axes` and `members`:

```bash
curl -s "$API/simulations/presets" > presets.json
jq -r '.presets[] | .id as $id | .members[]
  | [$id, (.params | tojson), .gpus, (.unavailable | keys | join(","))] | @tsv' presets.json
```

A preset gives its `deployment`, its `pools` (per role, the arch preset, arch,
GPU and worker type) and `captures`, the recorded workloads its members can
replay. Each member gives:

- `params`: its axis values. A simulation names the member by these, exactly.
- `gpus`: the GPUs it occupies, every pool and replica included.
- `pools`: per role, `replicas`, the arch preset's `arch_params` (such as
  `max_model_len`), `gpus_per_replica` and the `worker` block.
- `unavailable`: per capture, why the member cannot run it: `{"error": ...}`
  when it does not build, `{"missing": {<kernel role>: <rows>}}` when
  profile.db lacks rows, or `{"misfit": {"reason", "max_model_len", ...}}`
  when the capture's requests do not fit its pools' `max_model_len` (that
  blocks a replay only; the capture still serves as routing). A member can run
  a capture missing from this map.

```bash
jq '.presets[] | select(.id == "GLM-5.2-NVFP4/glm52_vllm_nvfp4_dsa_moe_chunked_prefill")
  | {deployment, pools, axes, captures}' presets.json
```

A worker of type `speculative` drafts `draft_tokens` per request per step; its
member needs an `accept_rate` (step 2).

## 2. Choose the workload

`workload.source` says where the requests come from. `GET /workloads`
describes the three sources:

```bash
curl -s "$API/workloads" | jq '.sources | map_values(.summary)'
```

**Routing.** An MoE model's experts are routed as one of the member's captures
routes them, whatever the source: `workload.capture` names the capture, the
preset's first by default. The service never falls back to a synthetic
routing, and the answer states the routing it used (step 4). A dense model
routes nothing.

**A capture** (`"source": "capture"`, the default). The requests are the
capture's recorded trace, and its routing is the capture's own. Each capture's
`facts` say what it holds: `requests`, mean `prompt_tokens` and
`output_tokens`, `rate`, its recorded requests per second, and
`input_file_tags`. A capture tagged `speculative` carries the acceptance it
recorded, per request, so a speculative member needs no `accept_rate` on it:

```json
{"source": "capture", "capture": "c32_long", "load": {"rate": 8}}
```

**A generated trace** (`"source": "generated"`). req-frontend's `tracegen`
draws it from `generator`: `type` and the generator's arguments. List them:

```bash
curl -s "$API/workloads" | jq -r '.sources.generated.generators[] | .name as $g
  | .arguments[] | [$g, .name, .default, (.choices | join("|")), .help] | @tsv'
```

Each length argument (`input_len`, `output_len`, `rounds`, `tool_wait_ms`)
takes a distribution as a string: `"512"` for a constant, `"uniform:256..1024"`
for an even range, `"lognormal:2048,0.8"` for a long right tail (median 2048).
`sessions` is how many sessions to draw and `rounds` how many requests each
session sends; a later round carries the conversation so far as its prefix.
`arrival_rate` is sessions per second and `seed` fixes the draw. For
independent requests, give `"rounds": "1"`:

```json
{"source": "generated",
 "generator": {"type": "synthetic", "sessions": 200, "rounds": "1",
               "input_len": "lognormal:4096,0.6", "output_len": "lognormal:512,0.5",
               "arrival_rate": 2, "seed": 1}}
```

A session trace's rounds always chain: each round waits for the previous round
to finish and for its tool wait.

**An uploaded CSV** (`"source": "upload"`). Send the file to `POST /workloads`
with its `format` (and, optionally, comma-separated `tags`), then name the
answer's `workload_id`:

```bash
curl -s "$API/workloads" | jq '.sources.upload | {formats, tags, max_bytes}'
cat > trace.csv <<'EOF'
id,arrival_time,input_len,output_len
r0,0,2048,256
r1,250,1024,128
r2,500,4096,512
EOF
UP=$(curl -s -X POST "$API/workloads" \
  -H 'content-type: text/csv' --data-binary @trace.csv | jq -r .workload_id)
```

A format's `columns` are the header the file must have, exactly, in any order;
each tag adds its own columns. With no `format` (and no `tags`), the service
reads the file as the format and tags its header fits; give `format` (and
`tags`) to say it yourself. In `text-generation-independent`,
`arrival_time` is in milliseconds and must not decrease down the file. The
service reads the file as a run will and answers 400 with the reason when it
cannot; the answer gives the format and tags it read, the same facts a capture
has (`requests`, `prompt_tokens`, `output_tokens`, `rate`) and `expires_at`
(24 hours). An upload
tagged `speculative` carries each request's acceptance in its `accept_rate`
column (one probability, or a JSON list of `draft_tokens` of them); a
speculative member then takes no `accept_rate` in the simulation.

**Knobs for every source.** Every request of the source runs.

| Field | Default | Meaning |
| --- | --- | --- |
| `load` | the trace's own arrival times | One of `{"rate": r}`: r requests (sessions, when rounds chain) per second, the trace's spacing scaled to that mean; or `{"concurrency": c}`: every request ready at once, at most c in flight. A trace whose requests all arrive at once (`rate` null) takes only a concurrency. |
| `run_to_end` | true | Run until every request finishes; with false, `duration_ms` of simulated time. |
| `duration_ms` | none | Simulated milliseconds to run; required when `run_to_end` is false. |
| `accept_rate` | none | Speculative workers only, and then required unless the upload carries its own column: the chance each draft token is accepted, one number for every position or a list of `draft_tokens` numbers, one per position. |

`/simulations/presets` also gives the full JSON schema of the workload under
`workload`.

## 3. Start the simulation

Post the preset id, the member's `params`, and the workload. Each `params`
value must equal the member's value; a number given as a string also matches.

```bash
SIM=$(curl -s -X POST "$API/simulate" -H 'content-type: application/json' -d '{
  "preset": "GLM-5.2-NVFP4/glm52_vllm_nvfp4_dsa_moe_chunked_prefill",
  "params": {"replicas": 1, "server": "ep4_ctx128k"},
  "workload": {"source": "generated",
               "generator": {"type": "synthetic", "sessions": 100, "rounds": "1",
                             "input_len": "lognormal:4096,0.6", "output_len": "256",
                             "arrival_rate": 2, "seed": 1}}
}')
echo "$SIM" | jq .
ID=$(echo "$SIM" | jq -r .simulation_id)
```

The answer is 202 with `simulation_id`, `status` (`queued`) and `routing`:
`capture` (null for a dense model) and `label`, such as "requests from your
workload, routing from capture c32_long".

A speculative member, with one acceptance per draft position:

```json
{"preset": "GLM-5.2-NVFP4/glm52_vllm_nvfp4_dsa_moe_speculative",
 "params": {"replicas": 1, "server": "ctx256k"},
 "workload": {"source": "upload", "upload": "<workload_id>",
              "accept_rate": [0.9, 0.8, 0.7, 0.6, 0.5]}}
```

The request is checked before it queues:

| Status | Cause | `detail` |
| --- | --- | --- |
| 400 | `params` miss an axis, name an unknown one, or match no member | `{message, choices}`; `choices` lists every member's `params` |
| 400 | A bad workload: an unknown capture or upload, a generator argument tracegen rejects, a trace the simulator cannot read, more than 2000 requests, a request longer than a pool's `max_model_len` (prefix + input + output, plus the draft tokens of a speculative worker; a capture's own requests included, which `/simulations/presets` lists as its `misfit`), a missing or wrong `accept_rate` (both the request's and an upload's column, or a list that is not `draft_tokens` long) | the reason, naming the request or argument; for a request too long, `{message, too_long: {max_model_len, requests, total}}` |
| 404 | Unknown preset id | `no public preset '...'` |
| 409 | The member does not build or lacks profile.db rows for that capture | the reason; pick another member or capture |
| 413 | An upload over `max_bytes` | the limit |
| 429 | The client's rate limit (`Retry-After` header) or 32 simulations already waiting | the limit |

## 4. Read the result

Poll every few seconds; most runs take seconds to a minute:

```bash
until S=$(curl -s "$API/simulations/$ID"); echo "$S" | jq -e '.status | test("done|failed|timed_out")' >/dev/null; do
  echo "$S" | jq -c '{status, queue_position}'; sleep 5; done
echo "$S" | jq '{status, error, routing, workload, gpus, summary}'
```

`status` is `queued` (with `queue_position`), `running`, `done`, `failed`
(with `error`) or `timed_out`. The record repeats the request: `preset`,
`params`, `workload` (with the `capture` used and `trace`, its facts), `routing`
and `gpus`. Once done, `summary` has:

- `requests`: `total` and `finished`; `cause` says why the run stopped;
- `sim_ms`: simulated time;
- `throughput`: `total_tok_s`, `prefill_tok_s`, `decode_tok_s`,
  `total_tok_s_per_gpu`, `completed_req_s`;
- `ttft_ms`, `tpot_ms`, `e2e_ms`: `mean`, `p50`, `p90`, `p99`, `max` and `n`, in
  milliseconds.

```bash
echo "$S" | jq '.summary | {requests, sim_ms, throughput, ttft: .ttft_ms, tpot: .tpot_ms, e2e: .e2e_ms}'
```

For more, the Analyzer serves the run under `$API/analyzer/runs/{run_id}/`,
with the record's `run_id`:

```bash
R="$API/analyzer/runs/$(echo "$S" | jq -r .run_id)"
curl -s "$R/descriptor" | jq .
curl -s "$R/subjects/slo-general/report" | jq .
```

Delete a simulation you no longer need, or stop one still running, with
`curl -s -X DELETE "$API/simulations/$ID"`.

## 5. Interpret and report

The simulation is an event-driven model of the serving engine's scheduler
over measured kernel times: queueing, batching, chunked prefill, KV-cache
admission and, for speculative workers, drafting with acceptance drawn from
`accept_rate`. It models no tokenizer, network or client. A generated trace is
not a model of real traffic; it tells how the deployment responds to the shape
you asked for, so say which shape you simulated.

When you answer, name the preset, the member's `params`, the GPUs, the
workload (its source, the generator arguments or upload, its request count,
the `load`, `accept_rate`), the
`routing.label`, and the numbers from `summary` you quote, with `requests`
finished out of total. Quote only numbers the API returns; label any
arithmetic on them as your own.
