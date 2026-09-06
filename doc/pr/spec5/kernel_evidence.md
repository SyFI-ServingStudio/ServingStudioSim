# Kernel Evidence: Warm and Historical

Generated from Analyzer iteration reports; raw values, report hashes and resource
IDs are retained in `kernel_evidence.json`. This table does not assert that historical
captures match the warm E2E configuration. No new NSYS capture was taken.

| Case | Capture | Chunk | Iterations | Kernel signed error | Critical-path coverage | Simulator workload coverage |
|---|---|---:|---:|---:|---:|---:|
| 01_micro_throughput_c256 | warm | 2052 | 120 | -2.44% | 94.05% | 99.27% |
| 02_graph_boundaries_c128 | warm | 2052 | 124 | -5.25% | 94.60% | 99.15% |
| 03_balanced_anchor_c32 | warm | 4098 | 162 | -3.44% | 96.95% | 99.42% |
| 04_prefill_spectrum_c32 | warm | 8196 | 75 | -3.43% | 97.60% | 99.42% |
| 05_decode_spectrum_c64 | warm | 4098 | 163 | -5.64% | 96.48% | 99.25% |
| 06_long_long_c8 | warm | 4098 | 414 | +7.82% | 95.97% | 98.58% |
| 07_quadrant_interference_c48 | warm | 8196 | 82 | -3.74% | 97.84% | 99.64% |
| 08_completion_churn_c96 | warm | 2052 | 117 | -5.28% | 93.38% | 99.68% |
| 09_rate_knee | historical, no warmup | 2048 | 582 | +7.22% | 93.86% | 97.62% |
| 10_full_saturation_rate | historical, no warmup | 2048 | 116 | -4.28% | 94.10% | 97.44% |
| 11 | capacity failure | - | - | - | - | - |
| 12 | capacity failure | - | - | - | - | - |
| 13_generation_length_matrix_c8 | historical, no warmup | 4096 | 140 | +2.91% | 94.87% | 98.11% |
| 14_near_full_long_decode_c21 | historical, no warmup | 4096 | 263 | -1.13% | 95.28% | 98.54% |
| 15_kv_overcommit_mixed_churn_c14 | historical, no warmup | 8192 | 252 | -2.31% | 94.99% | 98.20% |
| 16_chunk8k_mixed_pressure_c64 | historical, no warmup | 8192 | 326 | -3.53% | 94.67% | 97.85% |
| 17_concurrent_long_prefill_c16 | historical, no warmup | 4096 | 359 | +2.99% | 95.38% | 98.17% |

Critical-path coverage removes collective arrival wait and accounts for overlap.
Simulator coverage uses folded leaf workload; its denominator differs from the
measured critical path. These are not the raw residency coverage metric in the
standard campaign table. Kernel error compares simulated and measured reduced work.

Historical rows09/10/13-17 retain their original prediction/label reports and
startup effects. Their current warm E2E uses corrected chunk sizes and separate
workload measurements. Large historical errors remain visible and are not evidence
that the current warm configuration has the same error or that the model is accepted.
