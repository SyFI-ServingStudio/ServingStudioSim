# VibeSim UI prompt (paste into a conversation)

Use this after the VibeSim stack is up. Do not ask for Qwen.

---

Run the DeepSeek-V4-Flash idle KV / GPU-cache experiment in
`experiments/dsv4_idle_kv`.

Constraints:
- Model is DeepSeek-V4-Flash-0731, not Qwen or Llama.
- Prompts must be real corpus token IDs from `corpus/` (prose + code), never random filler.
- Simulator path: `python3 gen_realtext_workload.py && python3 sim_cache_policy.py --cell all`.
- Do not claim a full L4 V4 architecture run; the engine is still single-round and `deepseek_ffn_moe` is a stub.
- Real GPU: `./run_real_vllm.sh --check` first. Refuse a full serve on 4×H100. Do not download weights unless disk and GPU count pass.
- Report keep vs always-handoff vs JIT and migrate-on-idle. Preserve caveats: oracle waits, hand-set 2.5 s threshold, sim costs are A/B not vLLM wall times.

Write `experiments/dsv4_idle_kv/results/POLICY_SIM.md` and summarize the table.

---
