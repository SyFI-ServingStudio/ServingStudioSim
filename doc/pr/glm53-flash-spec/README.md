# GLM-5.3 and GLM-5.3-Flash speculative decoding: branch `glm53-flash-spec`

This branch carries the simulator changes behind the speculative-decoding study
in the companion repository
[`understanding-speculative-decoding`](https://github.com/mjacob1002/understanding-speculative-decoding)
(its `specbench/` tree: measured vLLM runs, alignment captures, simulation
sweeps, analysis and plots). It is review evidence and a reproducibility pin,
not an accepted baseline: no golden is recorded, no public preset is added, and
the open items below stay open.

Base: `main` at `d2e31e78` (PR #102). Hardware: 4x B200, TP4 + EP4.

## What the branch adds

| Area | Change |
|---|---|
| Arch | `glm53_flash_vllm_fp8_kda_dsa_moe_speculative` (MTP, `--speculative-config {"method":"mtp"}`) and `glm53_flash_vllm_fp8_kda_dsa_moe_dflash2` (`incoai/GLM-5.3-Flash-DFlash2`), built on the no-spec Flash arch, whose target forward, MoE block and TP all-reduce are split out so the no-spec replay stays bit-identical (AIME: 791,899.2 ms before and after). |
| Worklets | KDA verify path: `q = draft_tokens + 1` tokens per request with `q` state slots (`resolve_config_with_verify_width`); DSA verify and `index_share_reuse` modes (`Glm53DsaSpecMode`) for the MTP layer's K−1 recurrent passes; the DFlash2 proposer takes its dimensions from the draft config instead of pinning GLM-5.3's, so one proposer serves both checkpoints. |
| L1 kernel | `kda_recurrent_decode` gains `num_tokens = q · batch_size` for the spec-verify call (`[B, q]` state table, `num_accepted_tokens`), with a verify oracle and tests. The one-token rows keep their keys. |
| KV | vLLM main / 0.32's hybrid allocator as `kv_allocator: main` (`FlashKvAllocator::MAIN`): a KDA prefill-checkpoint page, block ids charged at the widest group's page rounded to the kernels' stride (`padded_per_block`), a kpool tail ring sized for the drafts. Reproduces every measured server's logged capacity. |
| Profiling | Container workers under a rootless engine (Podman) run as container root and pin their GPUs through `CUDA_VISIBLE_DEVICES` by UUID, since `--gpus device=` is advisory there (job 22664 had four workers on one card). |
| Packs | `glm53_nvfp4_b200_aime`, `glm53_nvfp4_b200_mtp`, `glm53_flash_fp8_b200_aime`, `glm53_flash_fp8_b200_spec`: AIME-shaped and TB-shaped alignment cases for both models and all three drafters. |
| `profile.db` | Every B200 row the study's sweeps needed: GLM-5.3 MTP / DFlash at K = 3, 5, 7, Flash no-spec, MTP K = 1, 3, 5, 7 and DFlash2 K = 3, 5, 7, at batch 1 to 256. |

## Evidence

All numbers are in the companion repository's `ServingStudio/progress.md` and
`specbench/runs/*/metrics_*.json`; the Analyzer reports live in the untracked
experiment trees listed in `specbench/scripts/PIPELINE.md`.

- **GLM-5.3 (NVFP4), AIME-shaped pack, 4 cases (none, MTP K=3, MTP K=5, DFlash2 K=7):** throughput within about 4% and kernel time within about 4% of the fork vLLM capture (`specbench/runs/glm53/aime_align/metrics_0*_variants.json`).
- **GLM-5.3-Flash no-spec, fork vLLM `e1423a4d`:** kernel path +4.2% per iteration (decode +3.9%), fused MoE +6.1% at 63% of GPU time, all-reduce −4.6%; AIME replay 791.9 s simulated vs 838.8 s measured (−5.6%, served by vLLM `187a0eb98`, which is about 13% slower than the fork), TB replay −5.2%.
- **Flash MTP K=3, fork `3d62fcb893` (aux hidden states):** kernel path −2.8% over 221 mixed iterations, +2.4% over decode-only; GPU cycle including idle −7.1% / −1.3%.
- **KDA verify rows (q = 2..8):** within ±8% of the fork kernel. **DSA verify top-k (next_n 4, 6):** 31 of 36 probes within 15%, medians about 1.0.
- **Flash speculative, own-length AIME replays vs measured wall:** none −2.6%; MTP K=1/3/5/7 −6.5/−9.8/−12.5/−15.1%; DFlash2 K=3/5/7 −14.4/−16.8/−14.3%. Simulated acceptance matches the measured tokens per step within 0.4%. The growing gap with K is host overhead per speculative step, which the simulator does not model; the study reports simulated speedups as they are (decision of 2026-10-21).

## Open items

- **vLLM fork pointer.** The Flash spec captures used `alignment/profiler/vllm` at `3d62fcb893` (branch `flash-dflash2-aux`: Glm5Next aux hidden states for DFlash2 / EAGLE-3), one commit above the public `e1423a4d`. That commit is not on any remote yet, so the submodule pointer on this branch stays at `e1423a4d`; the patch is a 53-line change to `vllm/models/glm5next/common/model.py`.
- **DFlash draft attention bytes.** `Dflash2DraftStage` folds every request's draft attention into one `flashinfer_attn_rect` launch with the work-weighted mean context. Right for time (one varlen launch per layer), wrong for bytes: K/V are billed for one context, so the logged intensity rises with the request count. The study corrects this in its analysis; the fork-side fix is to bill per-request K/V inside the folded call.
- **Flash attention projections** are modelled as BF16 GEMMs; the checkpoint stores them with FP8 scales. **GLM-5.3's MTP experts** run as a BF16 fused MoE; the stored datatype could not be confirmed.
- **Pack paths.** The GLM-5.3 AIME pack's routing corpora (1.5 GB each) and every `evidence:` entry point into the companion repository by relative path (`../../specbench/...` from this repository's root; the corpora pack-relative). They are not on the Hugging Face dataset. `host: cayenne` is a local host profile.
- Not done from `dev-finalize-model-addition`: public presets and catalog entries, Hugging Face uploads, goldens.

## Reproducing

The companion repository's `specbench/jobs/README.md` gives the order: measured
runs, alignment captures, sweep construction, kernel profiling, simulation.
With this branch checked out and `profiling/profile.db` in place, every sweep
in `specbench/scripts/PIPELINE.md` runs on CPU.
