# Server Latency: Partial Warm Matrix

Generated from the Analyzer HTTP E2E reports retained in `server_latency.json`.
Error is `(simulated / measured - 1) * 100` for the named distribution percentile.
These are server-side TTFT/TPOT, not client TTFT, request E2E, or means.
Snapshot: completed cases01-10 and13-15; all17 cases remain visible below.

| Case | TTFT P50 | TTFT P90 | TTFT P99 | TPOT P50 | TPOT P90 | TPOT P99 |
|---|---:|---:|---:|---:|---:|---:|
| 01_micro_throughput_c256 | +3.94% | +23.42% | +19.44% | -2.77% | -0.58% | -0.92% |
| 02_graph_boundaries_c128 | -3.50% | +3.97% | +19.95% | -4.71% | -2.44% | -1.24% |
| 03_balanced_anchor_c32 | -0.72% | -6.33% | -0.56% | -1.39% | -0.26% | -0.04% |
| 04_prefill_spectrum_c32 | -6.37% | -6.50% | -3.65% | +7.28% | -1.56% | +9.41% |
| 05_decode_spectrum_c64 | +241.60% | +15.79% | +14.53% | -17.37% | -30.23% | -7.25% |
| 06_long_long_c8 | -3.07% | -3.56% | +8.97% | +10.60% | +12.12% | +11.47% |
| 07_quadrant_interference_c48 | +3.34% | +8.20% | +13.04% | -4.22% | -14.60% | +2.70% |
| 08_completion_churn_c96 | +4.53% | +3.84% | -8.75% | -3.40% | -2.14% | -11.45% |
| 09_rate_knee | -5.39% | -6.53% | -1.98% | +4.27% | +6.59% | +10.56% |
| 10_full_saturation_rate | +2.94% | -5.53% | -5.46% | -6.92% | +1.13% | -0.76% |
| 11 | capacity failure | capacity failure | capacity failure | capacity failure | capacity failure | capacity failure |
| 12 | capacity failure | capacity failure | capacity failure | capacity failure | capacity failure | capacity failure |
| 13_generation_length_matrix_c8 | -36.46% | +1.22% | +1.12% | +9.99% | +10.16% | +10.67% |
| 14_near_full_long_decode_c21 | +5.18% | +37.95% | +40.36% | +0.33% | +1.48% | +4.82% |
| 15_kv_overcommit_mixed_churn_c14 | +23.30% | +8.85% | +8.98% | -5.37% | -5.40% | -32.56% |
| 16 | pending | pending | pending | pending | pending | pending |
| 17 | pending | pending | pending | pending | pending | pending |

Raw measured and simulated values (milliseconds), sample counts and per-report
SHA-256 are in the JSON sidecar. Case08 has fewer TPOT samples because some requests
have no decode span. Case05's large error is retained as an unresolved discrepancy.
No acceptance threshold or golden is changed by this display.

See README.md for observed-conditioned acceptance, borrowed time multipliers and
the KV conversion difference in earlier completed simulations.
