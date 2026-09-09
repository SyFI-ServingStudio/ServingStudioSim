---
name: operate-gpu-spec
description: Use when reading, querying, verifying, or extending the GPU spec catalog at `gpu/spec.json` — the per-GPU datacenter hardware table (mem/BW, dense TFLOPS by dtype, interconnect, nvl_domain_size, dollar_per_hour). Covers how to INTERPRET each field (TFLOPS are DENSE not sparse, interconnect BW is BIDIRECTIONAL, fp32 is CUDA-core not tensor, dollar_per_hour is the getdeploying on-demand "medium" rate, nvl_domain_size is a deployment param), how to SEARCH for a value (jq/python), how to VERIFY a value is right (sparse-doubling trap, one-way vs bidir, SXM vs PCIe vs NVL form-factor distinction, GB200 combined-for-2-GPU), what to do when a GPU/field is NOT in the table (web-search the fallback and clearly tell the user it is a searched result, not catalog-backed), and how to ADD/refresh a GPU. NOT the simulator timing path — modeled timings come from measured kernels in profile.db, not from these peaks.
---

# The GPU spec catalog — `gpu/spec.json`

`gpu/spec.json` is a hand-maintained catalog of datacenter GPU hardware specs:
a top-level `{"gpus": [ … ]}` array, one JSON object per GPU (NVIDIA / AMD /
Intel). It is a **reference / analysis** table — roofline sanity checks, cost
($/token) estimates, memory-fit checks, and picking which GPU a study targets.

## Read this first: it is NOT on the timing path

Editing a number here does **not** change any simulated run. The simulator's
kernel timings come from **measured rows in `profiling/profile.db`**, keyed by
the GPU **name string** a preset carries (`gpu: "NVIDIA H200"`), not from these
peak-TFLOPS figures. So:

- The `name` field here (e.g. `H200-SXM-141GB`) is a catalog label; the preset's
  `gpu:` string (e.g. `"NVIDIA H200"`) is a separate profile.db cache identity.
  They are not auto-linked — do not assume renaming one updates the other.
- Use this catalog for **human/analysis** math (peak FLOPs, HBM budget, $/hr),
  never as the source of a modeled kernel latency.

## How to interpret each field

| Field | Meaning | Gotcha |
|---|---|---|
| `name` | Catalog label incl. form factor + memory (`H100-SXM5-80GB`) | Distinct from a preset's `gpu:` string |
| `vendor` / `architecture` / `year` | NVIDIA/AMD/Intel; uarch; launch year | — |
| `mem_size_gb` | HBM/GDDR capacity, GB | Per **GPU** (except GB200, see below) |
| `mem_type` | HBM2/HBM3/HBM3e/GDDR6 | — |
| `mem_bandwidth_gbps` | Memory bandwidth, **GB/s** (bytes) | GB/s not Gb/s; H200 = 4.8 TB/s = `4800` |
| `fp16_tflops` `bf16_tflops` `fp8_tflops` `fp4_tflops` `int8_tops` | **DENSE** Tensor-Core peaks | **Sparse (2:4) is ~2×** — never store/use the sparse number |
| `fp32_tflops` | Non-Tensor FP32 (**CUDA-core**) peak | ~1/30 of the tensor peak — **not** the GEMM roofline number |
| `interconnect` | NVLink x.y / PCIe / Infinity Fabric / RoCE | — |
| `interconnect_bandwidth_gbps` | **BIDIRECTIONAL** aggregate per GPU, GB/s | One-way ≈ **half**. NVLink 4.0 = 900 bidir = 450 each way; NVLink 5.0 = 1800 bidir |
| `nvl_domain_size` | Typical GPUs reachable at full NVLink BW in one node/rack | **Deployment param**, overridable per run via preset `nvl_num_gpu`; see below |
| `dollar_per_hour` | Typical **on-demand** rental, USD per **GPU**·hr | getdeploying "medium" (see Pricing); volatile; `null` = unlisted |
| `notes` | Free-text caveats (export variant, Superchip, etc.) | Present only where needed |
| `null` anywhere | Not published or N/A for that part | e.g. no FP8 tensor core, non-NVLink fabric, price unlisted |

### The interpretation rules that bite (memorize these)

1. **TFLOPS are DENSE, no 2:4 sparsity.** Vendor slides love the sparse number
   (2× the dense). If a datasheet value is *exactly double* a catalog value,
   you are looking at the sparse figure — do not overwrite with it.
2. **`fp32_tflops` is CUDA-core FP32**, ~1/30 of the tensor peak. It is not the
   number for a bf16/fp8 GEMM roofline — use `bf16_tflops` / `fp8_tflops`.
3. **`interconnect_bandwidth_gbps` is bidirectional** (both directions summed).
   For a one-way transfer estimate, halve it.
4. **Bandwidths are byte/s (GB/s), not bit/s.**
5. **`fp8` needs FP8 Tensor Cores (Hopper+ / Ada Lovelace+)**; `null` on older
   parts. `fp4_tflops` is **Blackwell-only** and absent elsewhere.
6. **China-export variants (H800, H20)** carry export-restricted specs (reduced
   NVLink BW and/or compute) — expected, not a typo.
7. **GB200-Superchip compute/memory fields are COMBINED for its 2 B200 GPUs**
   (+1 Grace CPU). Divide by 2 for a per-GPU number. Its `dollar_per_hour`,
   however, is already **per-GPU**.

### `dollar_per_hour` — what the number means (Pricing)

- **Source:** `https://getdeploying.com/gpus`, per-model detail page
  `/gpus/<vendor>-<model>` (e.g. `/gpus/nvidia-h100`), the **On-demand average**
  from that page's billing-type table.
- **It is the "medium"/mid-market rate**, USD per **GPU** per hour — *not* the
  spot floor on the index page (often ~3–5× lower), and *not* a
  reserved/contract rate.
- **Form factors are kept distinct.** Where a chip sells as SXM *and* PCIe (and
  NVL), each row is priced from that form factor's own on-demand listing median
  — SXM runs pricier than PCIe. Do not blur one price across form factors.
- **Volatile.** Prices move; re-fetch at use time. `as_of` this catalog's last
  refresh: **2026-07-02**. `null` = not listed at that date (H800, H20, Gaudi3).

### `nvl_domain_size` — a deployment param, not pure hardware

Typical count of GPUs reachable at full NVLink bandwidth within one node/rack
(the intra-node high-BW domain): **8** for mainstream HGX SXM nodes, **72** for
a GB200 NVL72 rack, **2** for PCIe NVLink-bridge pairs, **1** for cards with no
NVLink. It is a **topology choice** and is overridable per run via the preset
field `nvl_num_gpu`. It is `null` for non-NVLink fabrics (AMD Infinity Fabric,
Intel RoCE), which have their own (typically 8-GPU) intra-node domain.

## Search / query a value

The file is JSON — use `jq` (from `VibeSim/`):

```bash
# All fields for one GPU
jq '.gpus[] | select(.name=="H200-SXM-141GB")' gpu/spec.json

# One field across every GPU, as name: value
jq -r '.gpus[] | "\(.name): \(.fp8_tflops)"' gpu/spec.json

# Filter: Blackwell parts with >5 TB/s HBM, just their names
jq -r '.gpus[] | select(.architecture=="Blackwell" and .mem_bandwidth_gbps>5000) | .name' gpu/spec.json

# Cheapest priced GPU by $/hr (skip nulls)
jq -r '[.gpus[] | select(.dollar_per_hour!=null)] | min_by(.dollar_per_hour) | "\(.name) $\(.dollar_per_hour)/hr"' gpu/spec.json
```

Python:

```python
import json, pathlib
cat = json.loads(pathlib.Path("gpu/spec.json").read_text())
by_name = {g["name"]: g for g in cat["gpus"]}
h200 = by_name["H200-SXM-141GB"]
# dense bf16 roofline, one-way NVLink BW:
tflops, nvlink_one_way = h200["bf16_tflops"], h200["interconnect_bandwidth_gbps"] / 2
```

## Not in the catalog? Search — and label the answer as searched

The catalog is finite (~21 GPUs) and some fields are `null`. If a user asks
about a GPU with no row, or a field that is `null`/absent:

1. **Confirm it's really absent** first. A user's loose name may map to an
   existing row — "H100" → `H100-SXM5-80GB` / `H100-PCIe-80GB`; "A100" → the
   four A100 rows. Match loosely before concluding it's missing. If it maps to a
   row, answer from the catalog (not a search).
2. **If truly absent** (e.g. RTX 4090, GH200, MI355X — no row), fall back to a
   **web search**: vendor datasheet for specs; `getdeploying.com/gpus/<vendor>-<model>`
   **On-demand average** for price (same semantics as the catalog). Apply the
   same interpretation rules — dense TFLOPS, bidirectional interconnect,
   on-demand "medium" price.
3. **Clearly label it as a searched result**, so the user knows it is NOT a
   vetted catalog number. Prefix the answer, e.g.:
   > ⚠ Not in the VibeSim GPU catalog — this is a web-searched result (source: …),
   > not a catalog-backed value.
   Cite the source and, for prices, the fetch date (they move).
4. **Never present a searched or from-memory value as catalog-backed.**
   Catalog-sourced vs searched must be distinguishable in every answer. If you
   answer a spec/price without opening `gpu/spec.json`, say where it came from.
5. Optionally **offer to add it** to `gpu/spec.json` (see "Add or refresh a GPU")
   so the next lookup is catalog-backed.

## Verify a value is correct

Run these checks before trusting or editing a number:

1. **Dense vs sparse (the #1 error).** If a vendor datasheet number is *exactly
   2×* the catalog value, it's the **sparse** (2:4) figure — the catalog stores
   dense. Keep dense.
2. **fp8 ≈ 2× fp16 (dense)** on Hopper/Blackwell. If `fp8_tflops == fp16_tflops`
   something is off — *except by design* (Gaudi3: BF16 and FP8 share a peak; its
   `notes` say so).
3. **Interconnect one-way vs bidir.** Catalog stores **bidir**. NVLink 4.0 = 900
   GB/s bidir (450 one-way); NVLink 5.0 = 1800. If a source quotes ~450 for a
   Hopper part, that's one-way — double it before storing.
4. **Right form factor / SKU.** `H100-SXM5-80GB` ≠ `H100-PCIe-80GB` ≠ H100 NVL:
   they differ in `mem_bandwidth_gbps`, `tdp_watts`, tensor peaks,
   `nvl_domain_size`, **and price**. Confirm you're reading/writing the intended
   row. (H100 NVL — 94 GB/GPU, 188 GB for the bridged pair — is a *separate SKU*
   not currently in this catalog; add it as its own row rather than folding its
   spec/price into the SXM row.)
5. **GB200 combined-for-2.** Its compute/memory are for **2 GPUs**; halve for
   per-GPU. Price is already per-GPU.
6. **Cross-check the source.** Specs → the vendor datasheet (NVIDIA/AMD/Intel
   docs). For a GPU you actually run on, `nvidia-smi -q` (memory, BW) and the
   `profile.db` are ground truth. Price → re-fetch the getdeploying detail page
   and read the **On-demand average**, not the index-page floor.

## Add or refresh a GPU

1. **Append** a `gpus[]` object carrying the full field set (copy the shape of a
   same-vendor neighbor; keep the field order). Use `null` for
   unpublished/N-A fields rather than omitting them.
2. **TFLOPS:** enter **dense** tensor peaks per dtype (see check #1). `fp4_tflops`
   only for Blackwell; `fp8_tflops` only Hopper/Ada+.
3. **Bandwidths:** GB/s, bytes; interconnect is **bidirectional**.
4. **`nvl_domain_size`:** 8 (HGX SXM) / 72 (GB200 NVL72) / 2 (PCIe bridge) / 1
   (no NVLink) / `null` (AMD/Intel).
5. **`dollar_per_hour`:** fetch `/gpus/<vendor>-<model>` on getdeploying, use the
   **On-demand average** (per form factor if SXM/PCIe/NVL differ). `null` if
   unlisted. When you refresh prices in bulk, note the new `as_of` date wherever
   you record it (commit message / this skill's Pricing section).
6. **Validate** the file parses and every row has the required keys:

```bash
python3 -c "import json; d=json.load(open('gpu/spec.json')); \
print('ok', len(d['gpus']), 'gpus'); \
req={'name','vendor','architecture','year','mem_size_gb','mem_type','mem_bandwidth_gbps','fp16_tflops','fp8_tflops','bf16_tflops','fp32_tflops','int8_tops','tdp_watts','interconnect','interconnect_bandwidth_gbps','nvl_domain_size','dollar_per_hour'}; \
miss={g['name']:sorted(req-set(g)) for g in d['gpus'] if req-set(g)}; \
print('missing keys:', miss or 'none')"
```

## Provenance

- **Specs:** NVIDIA datacenter docs, AMD Instinct docs, Intel Gaudi docs
  (seeded from `ref/GPU/datacenter_gpus.json`).
- **Pricing:** `getdeploying.com/gpus` per-model On-demand average, `as_of`
  2026-07-02.
