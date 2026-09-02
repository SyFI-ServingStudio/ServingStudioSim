# DSV4 idle-KV policy sim (session-level, V4 KV bytes)

## idle_kv (24×8, mem 0.35, 1 GPU)

| policy | migrate | mem_util | n_gpus | sessions | ttft_p50_ms | e2e_p50_ms | re_prefills | keep_hits | migrations | hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| keep | False | 0.35 | 1 | 24 | 0.0 | 5.12 | 0 | 192 | 0 | 1.0 |
| always_handoff | False | 0.35 | 1 | 24 | 83.97 | 89.09 | 192 | 0 | 0 | 0.0 |
| jit | False | 0.35 | 1 | 24 | 34.82 | 39.94 | 96 | 96 | 0 | 0.5 |

## wait_sweep (skew via gpu_hint, 2 GPU)

| policy | migrate | mem_util | n_gpus | sessions | ttft_p50_ms | e2e_p50_ms | re_prefills | keep_hits | migrations | hit_rate | wait_h_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jit | False | 0.75 | 2 | 24 | 0.0 | 5.12 | 0 | 192 | 0 | 1.0 | 1000 |
| jit | True | 0.75 | 2 | 24 | 0.0 | 5.12 | 0 | 192 | 0 | 1.0 | 1000 |
| jit | False | 0.75 | 2 | 24 | 34.82 | 39.94 | 96 | 96 | 0 | 0.5 | 4000 |
| jit | True | 0.75 | 2 | 24 | 34.82 | 39.94 | 96 | 96 | 6 | 0.5 | 4000 |
| jit | False | 0.75 | 2 | 24 | 34.82 | 39.94 | 96 | 96 | 0 | 0.5 | 8000 |
| jit | True | 0.75 | 2 | 24 | 34.82 | 39.94 | 96 | 96 | 6 | 0.5 | 8000 |
| jit | False | 0.75 | 2 | 24 | 34.82 | 39.94 | 96 | 96 | 0 | 0.5 | 30000 |
| jit | True | 0.75 | 2 | 24 | 34.82 | 39.94 | 96 | 96 | 6 | 0.5 | 30000 |

## membound (64×8, mem 0.25, 2 GPU)

| policy | migrate | mem_util | n_gpus | sessions | ttft_p50_ms | e2e_p50_ms | re_prefills | keep_hits | migrations | hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| keep | False | 0.25 | 2 | 64 | 274.43 | 279.55 | 366 | 146 | 0 | 0.285 |
| jit | True | 0.25 | 2 | 64 | 280.58 | 285.7 | 503 | 9 | 87 | 0.018 |

## fork colocation

| policy | stems | stems_colocated | gpu_load |
|---|---|---|---|
| scatter_stems | 8 | 0 | [16, 16] |
| stem_binpack | 8 | 8 | [16, 16] |
