---
name: operate-use-analyzer
description: Use when selecting, reading, comparing, interpreting, or citing an existing VibeSim Analyzer result, including simulation sweeps/singletons, timing predictions, kernel profiles, and kernel measurements. Makes Analyzer the numerical authority, selects an exact resource from explicit UI context, a just-created managed result, or the recent catalog, and copies exact citation tokens returned beside sweep values. Not for producing the result itself; use the matching operate-run-* or operate-profile-* skill first.
---

# Use VibeSim Analyzer

Use Analyzer whenever a user-visible claim depends on an existing VibeSim
result. The conversation backend owns lifecycle and ownership links; Analyzer
owns result values, descriptors, curves, plots, and hardware limits.

## Select the exact resource

Prefer evidence in this order:

1. the explicit Analyzer selection supplied by the UI;
2. the stable Analyzer resource ID returned by the managed workflow just run;
3. a resource ID the user named;
4. recent ready candidates from
   `/api/v1/sweeps?status=ready&limit=5`, or
   `/api/v1/sweeps/latest` when one newest candidate is useful.

Catalogs are discovery evidence, not result evidence. They intentionally expose
opaque IDs. Before choosing a candidate, compare its `display_name`, ordered
`axes`, `deployments`, `traces`, `status`, and time with the user's request.
“Latest” means newest ready result, not automatically the semantically correct
result. If multiple candidates remain plausible, show the concise candidates
and ask instead of silently choosing.

A ready result produced by another conversation in the same workspace is valid.
The producing conversation and turn are provenance, not read authorization.
Never cross a workspace boundary unless the product supplies an explicit
attachment/import mechanism.

## Read through Analyzer MCP

Use the `read_analyzer_resource` MCP tool. Set:

- `source="host"` for the shared Analyzer service, including a result selected
  in the Analyzer UI;
- `source="workspace"` for a result that exists only in the current Agent
  workspace.

`source` selects the data location; it does not change workspace ownership.
After selecting a sweep ID, read exactly:

```text
/api/v1/sweeps/{sweep_id}/payload
```

In a managed UI turn this returns one compact evidence block:

```json
{
  "resource": {"id": "e_...", "name": "20260801_3_llama3_8b_tp"},
  "axes": ["tensor_parallel", "request_rate"],
  "metrics": {
    "total_tps": {
      "label": "Total throughput",
      "unit": "tok/s",
      "objective": "maximize",
      "citation": "exp.throughput"
    }
  },
  "rows": [
    {
      "coordinates": {"tensor_parallel": 2, "request_rate": 20},
      "values": {
        "total_tps": {
          "raw": 30124.2,
          "citation": "exp.tp2.rate20.throughput"
        }
      }
    }
  ]
}
```

The compact block is the user-visible numerical authority. Do not replace it
with `/api/jobs`, launcher status text, an implementer summary, or direct report
file parsing when the Analyzer resource is ready.

## Write the answer

- Report the exact `raw` value and its declared unit.
- Copy the adjacent `citation` token unchanged as Markdown inline code.
- Never assemble a token from axes or metric names, and never invent a token.
- Cite a panel-wide qualitative claim with the metric citation when suitable;
  cite coordinate-specific numerical claims with the matching row citation.
- A derived value may be calculated from returned raw values, but cite every
  input coordinate used and state that the result is derived.
- Label provenance accurately: **simulated prediction**, **measured result**,
  **catalog fact**, or **derived from Analyzer values**.
- Do not emit Analyzer URLs, navigation target JSON, percent-encoded strings,
  or workspace file links as substitutes for evidence citations.
- Writing a token never navigates automatically. The user chooses whether to
  click the rendered citation.

Example:

```markdown
At TP2 and 20 req/s, simulated throughput is 30,124.2 tok/s.
`exp.tp2.rate20.throughput`
```

## Follow-up turns and unsupported resources

Re-read the exact Analyzer resource in every turn that reports its numbers.
This registers a current citation dictionary while historical messages keep
their already-frozen targets.

Timing predictions, kernel profiles, and kernel measurements are also
Analyzer-owned, but do not invent `exp.*` tokens for a resource family whose
citation contract is not yet defined. Read its documented descriptor, cases,
curve, summary, plot, and hardware endpoints; rely on its typed result card for
navigation and report the resource ID plus provenance.

If a resource is missing, pending, failed, ambiguous, or lacks the requested
metric/coordinate, say exactly that. Do not estimate a replacement value.
