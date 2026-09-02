# Open a VibeSim PR from this tree

Do **not** copy this into a personal archive repo. Work on the VibeSim checkout:

```bash
cd ~/vibesim-workspace/main   # on ptc, or the submodule locally
git checkout -b exp/dsv4-idle-kv
git add experiments/dsv4_idle_kv presets/dsv4_agent
git commit -m "Add DeepSeek-V4-Flash idle-KV experiment package and DES presets."
git push -u origin exp/dsv4-idle-kv
```

Then open the PR against the VibeSim default branch.

Follow-on PRs (not this package):

1. L4 DeepSeek V4 arch via `skills/top-add-new-arch` (replace `deepseek_ffn_moe` stub).
2. Multi-turn request lifecycle so keep/handoff/migrate live in the Rust engine
   (`simulator/src/log/rows.rs` already documents the missing session columns).
