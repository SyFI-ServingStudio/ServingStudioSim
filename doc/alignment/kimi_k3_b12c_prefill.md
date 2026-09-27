# Kimi-K3 B12c prefill prediction comparison

The B12c rows were filled on one Slurm-allocated B200 with the branch profile
database. Times below are milliseconds. The prediction column is the total from
the regenerated `iter_breakdown.ans` report.

| Layer | Point | Measured step | Predicted total | Delta |
| --- | --- | ---: | ---: | ---: |
| KDA | 16k, prefix 0 | 9.07 | 8.912 | -1.7% |
| KDA | 16k, prefix 49,152 | 8.87 | 8.990 | +1.4% |
| KDA | 4 x 4k, prefix 0 | 8.57 | 8.555 | -0.2% |
| KDA | 16k, prefix 245,760 | 9.03 | 9.301 | +3.0% |
| KDA | 32k, prefix 229,376 | 17.94 | 18.230 | +1.6% |
| KDA | 16k, prefix 131,072 | 9.20 | 9.119 | -0.9% |
| KDA | 32k, prefix 131,072 | 17.00* | 17.761 | +4.5% |
| MLA | 16k, prefix 49,152 | 12.12 | 11.769 | -2.9% |
| MLA | 16k, prefix 0 | 8.13 | 8.010 | -1.5% |
| MLA | 4 x 4k, prefix 0 | 7.79 | 7.566 | -2.9% |
| MLA | 16k, prefix 245,760 | 27.69 | 26.303 | -5.0% |
| MLA | 32k, prefix 229,376 | 48.36 | 48.476 | +0.2% |
| MLA | 16k, prefix 131,072 | 18.37 | 18.056 | -1.7% |
| MLA | 32k, prefix 131,072 | 35.09 | 35.027 | -0.2% |

\* The supplied B12c measured table did not include the KDA 32k/131,072
point. Its measured value here is the 17.00 ms prior-campaign reference; the
other rows use the supplied judge medians.

The MLA prefix runner now profiles one physical FlashInfer launch per prefix
chunk and carries both `prefix_len` and `num_prefix_chunks` in the cache key.
Its measured rows are 9.38 ms at 16k/131,072, 17.63 ms at 16k/245,760, and
31.36 ms at 32k/229,376. Prefix attention is the first cost family for all
four long-context MLA points. For KDA, the GEMM family remains first when the
projection leaves are aggregated, followed by routed MoE and the chunk leaf.

The KDA chunk row is intentionally unchanged at 1.445 ms for 16k long
context. The runner times the fused `chunk_kda` boundary with
`use_qk_l2norm_in_kernel=True`; it is not retuned to the optimized-tree
h-scan/intra/conv launch sum of about 0.72 ms.

Decode rows and decode branches were not changed.
