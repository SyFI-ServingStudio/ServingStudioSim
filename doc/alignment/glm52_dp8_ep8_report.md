# GLM-5.2 FP8 · DP8+EP8 · VibeSim ↔ vLLM 对齐报告

**实验目录** `logs/20260803_8_glm52_fp8_dp8_ep8_alignment_ctx8k_out1k_c64/`
**分支** `glm52-dp-alignment`(worktree `wt-glm52-dp-alignment`)
**状态** 六阶段全线跑通(profile → timing-predict → 打标 → kernel-align → sim → e2e),
**三层判读全部产出**:逐 kernel(§8.3)、duty cycle(§8.1)、TTFT/TPOT/吞吐(§9)。
待办见 §10。

这份报告随工作推进持续更新。判读顺序遵循 `top-align-with-framework`:
逐 kernel → GPU duty cycle → TTFT/TPOT,由紧到松。

---

## 0. 被测配置

| | 实测 (vLLM) | 模拟 (VibeSim) |
|---|---|---|
| 模型 | `zai-org/GLM-5.2-FP8` | `model/config/glm52.json`, `fp8: true` |
| 并行 | tp=1, dp=8, EP=8, `flashinfer_nvlink_two_sided` | `ep_size: 8`, `nvl_num_gpu: 8` |
| 硬件 | 8 × NVIDIA H200 | 同 |
| workload | `logs/20260802_5_glm52_fp8_ctx8k_c64/trace.csv`,256 请求,ctx 8k / out 1k,并发 64 | 同一条 trace |
| capture | NSYS `cuda_profiler_api`,30 s 窗口,`--cuda-graph-trace=node` | — |

产物:1.1 GB sqlite、27.4 s kernel、2286 个 forward NVTX range、256 请求全部完成,
外加一次 `profile_kind: expert_popularity` pass。

---

## 1. 起点:profile 阶段最后一步失败

```
[invalid] alignment profile: tensor-parallel kernel sequences are not symmetric:
          device 1 differs from representative device 0
```

`parsed.json` / `kernel_sequences.json` 未生成,后续 timing-predict / analyze / sim 四个阶段
一步未跑。根因是整条 alignment 流水线按「单 replica 对称 TP」设计,而本实验是 DP8 + EP8 ——
8 个 DP rank 各自调度自己的 batch,kernel 序列天然不对称。

调查确认了四个洞:

| # | 洞 | 位置 |
|---|---|---|
| 1 | 对称性硬断言 | `alignment/nsys/sequence.py:64` |
| 2 | metrics 无 rank 标签,dict 覆盖只留最后一条 | `alignment/nsys/parse.py:180` |
| 3 | timing-predict case 只产 1 个 group,arch 要 8 个 | `simulator/src/timing_predict.rs:257` |
| 4 | analyzer 用一条代表序列套所有卡,按 `phase/row_id` 跨卡归约 | `analyzer/rust/src/alignment_iteration/mod.rs:484` |

**关键判断:这四个洞都不需要重跑 GPU。** DP rank ↔ device ↔ metrics 记录三者可以从
现有产物完整恢复(见 §2)。

---

## 2. Step 1 — parse 支持 DP rank ✅

### 2.1 rank 溯源:从日志前缀恢复,而不是改 fork

计划原本要给 vLLM fork 的记录加 `dp_rank` 字段并 bump schema。实际实现改成
**完全在 extractor 侧从日志前缀恢复**:

```
(EngineCore_DP3 pid=3376110) INFO ... VibeSimAlignmentIteration {...}   → dp_rank
(Worker pid=3379947) INFO [parallel_state.py:1568] world_size=8 rank=7 …  → pid ↔ global rank
nsys PROCESSES ⋈ CUPTI_ACTIVITY_KIND_KERNEL                              → pid ↔ device
```

**偏离计划的理由**:这条路径对新旧 capture 一视同仁 —— 一条代码路径覆盖所有历史 capture,
不必 bump 记录 schema,也不必在 worktree 里动子模块。fork 侧无需任何改动。

实测验证(现有 08-03 capture):

```
worker pid -> global rank: {3379942: 6, 3379943: 5, 3379944: 0, 3379945: 1,
                            3379946: 4, 3379947: 7, 3379948: 3, 3379949: 2}
```

与 nsys 的 pid→device 映射逐一吻合(pid 3379944 = rank 0 = device 0,
3379942 = rank 6 = device 6,3379947 = rank 7 = device 7)。

### 2.2 意外发现:旧 metrics 丢掉了 7/8 的数据

重抽后:41350 条记录 = **8 个 rank × 各自约 5136–5265 个 iteration**,
`(dp_rank, iteration_index)` 键**零重复**。

原先 `load_metrics` 以 `iteration_index` 为键做 dict 赋值,同一 index 有 8 条记录时
只保留最后一条 —— 即**七个 rank 的 batch 形状被静默丢弃**。这不止影响 DP:
它意味着此前任何 dp>1 的 capture,timing-predict 的输入形状都是错的。

### 2.3 意外发现:各 rank 的 iteration 数不同

```
records per dp_rank: {0: 5138, 1: 5265, 2: 5137, 3: 5137,
                      4: 5137, 5: 5136, 6: 5265, 7: 5135}
```

rank 1 和 rank 6 比其余多跑约 130 步(+2.5%)。DP ranks 并非严格同步 —— 这是
**负载不均的直接实测证据**,与 §5 要查的 shard 放置策略问题直接相关。也意味着
「某 rank 在某 iteration 无记录」是正常的 idle/dummy step,parse 必须容忍而不是报错。

### 2.4 并集目录取代对称断言

`build_symmetric_device_kernel_sequences` → `build_device_kernel_sequences`:
每张卡各自折叠,再按 `sequence_id`(有序 kernel 名的 hash)取并集。跑出相同序列的卡
自动合并成一条目录项、一次打标决策;真正分叉的卡各留各的。每条序列记录
`occurrences: [{device_id, iterations}]`,标签只作用于它被推导出来的那些位置。

对称场景(TP)退化成今天的行为:8 卡相同 → 一条目录项、8 个 occurrence。

### 2.5 改动清单

| 文件 | 改动 |
|---|---|
| `alignment/profiler/vllm_server.py` | `_ENGINE_CORE_PREFIX_RE` / `_WORKER_RANK_RE`;`extract_worker_device_ranks()`;`extract_metrics_jsonl(..., dp_size)` 给每行盖 `dp_rank`,`dp_size>1` 缺前缀即报错 |
| `alignment/nsys/parse.py` | `load_metrics` 改 `(dp_rank, iteration)` 键 + 重复检测;新增 `aggregate_metrics_by_iteration`(replica 步 = 各 rank 局部 batch 的并集)、`resolve_dp_rank_by_device`(global rank ÷ tp_size);`build_iteration_details` 每个 range 带 `dp_rank` + 该 rank 的 metrics,iteration 级新增 `metrics_by_dp_rank`;`parse_trace` 新增 `worker_ranks`/`tp_size`;parsed schema 2 → 3 |
| `alignment/nsys/sequence.py` | 并集目录 |
| `alignment/nsys/parse.py` (CLI) | 新增 `--server-log` / `--tp-size`,可在不重跑 GPU 的前提下重 parse |
| `alignment/runner.py` | 传 rank 表与 tp_size;新增「观测到的 DP rank population 必须等于 dp_size」校验;`--resume` 通路(见 §3) |
| `launcher/alignment_config.py` | labeled inventory schema 4(`occurrences`);「一个位置只能被认领一次」的校验从 iteration 改为 (device, iteration) |
| `alignment/timing_predict_input/` | `group_assignment: per_dp_rank`:一个 case = 8 个 group,group *g* 取 rank *g* 的批形状;无记录的 rank 给空 group(它仍要跑 forward 以维持 EP 集合通信的步调);`schema_version` 放开到 1..3 |

### 2.6 测试

`tests/test_alignment_nsys_parse.py` 21 passed,`tests/test_alignment_v1.py` 22 passed + 1 skipped。

- 原 `test_tp_sequence_inventory_rejects_asymmetric_ranks`(断言抛 "not symmetric")
  重写为 `test_asymmetric_ranks_each_keep_their_own_sequence`:DP 分叉是正常现象,各卡各留各的。
- 新增:rank 标签往返、`dp_size>1` 缺前缀报错、worker banner 解析与不完整 population 拒绝、
  metrics 按 (rank, iteration) 索引与重复检测、iteration 聚合语义、rank→device 按 tp_size 折叠。

---

## 3. 意外发现:profile 阶段没有 resume 通路

`operate-run-alignment` 明写「resume from the last verified artifact root; never rerun the
expensive GPU profile because a later label or analyzer step failed」,但这条通路**在代码里
根本不存在** —— `run_profile` 是一条从「起 vLLM + NSYS 采集」到「写 profile_result.json」
的直线,中间任何一步失败都只能整个重跑(本例是 400 s 的 8 卡 replay + 6 min 的 sqlite 导出)。

补上了 `alignment profile --resume`:

- `run_profile(cfg, *, resume=False)`;capture 之后的收尾抽成 `_finalize_profile()`,
  两条路径共用。
- resume 跳过 `build_session_runner()`、fork venv 解析、server 启动、NSYS 采集;
  `.nsys-rep → .sqlite` 是纯函数,已存在就直接复用。
- `drive_summary` 从 config + prepared replay 精确重建;`reached_idle` 是对已结束 server 的
  存活观测,**留空而不是伪造**。
- `nsys_profiler` provenance 从采集时写的 `launch.json` 读回;该 capture 早于这个字段
  → 记 `null`。**没记录的事实就让它空着,不用当前主机的 nsys 冒充。**

沿途还修了两个陈旧问题:

1. `profile_nsys.yaml` / `profile_expert_popularity.yaml` 里的 `server.use_ray_nsight`
   在 8-04(capture 之后)被有意移除(ray-nsight 采集路径已被外部 nsys 取代),配置未同步
   → resume 直接报 `unexpected keyword argument`。该字段原值为 `false`、本次采集本来就走
   外部 nsys 路径,删掉是忠实的。
2. `_finalize_profile` 里 `nsys_executable.provenance()` 在 resume 下是 `None`(见上)。

**结果:profile 阶段完整跑通**,`parsed.json`(282 iteration / 7,763,831 kernel row)、
`kernel_sequences.json`、`profile_result.json` 全部产出,**GPU 一次都没有重跑**。

---

## 4. Step 2 — vLLM 真实 kernel 分解

### 4.1 层日程与 arch 声明逐项吻合

用「每层入口 RMSNorm」(`triton_red_fused_add_fused_add_rms_norm_mul_1`,全序列出现 75 次)
切分 decode 主序列的 3469 个 kernel:

| | 实测 | `glm52_dsa_moe.rs` 声明 |
|---|---|---|
| dense 层 | 3 × 34 kernel | `NUM_DENSE_LAYERS = 3` |
| IndexShare 层 | 57 × 42 kernel | 3 initial + 18×3 cycle = 57 |
| full-index 层 | 18 × 54 kernel | `NUM_SPARSE_CYCLES = 18` |
| 合计 | **78 层** | `NUM_LAYERS = 78` |

`3×34 + 57×42 + 18×54 = 3469`,与 `expanded_kernel_count` 逐位相等。
**计划里「arch 层日程可能与真实 checkpoint 不符」的风险(#4)排除。**

### 4.2 sim 的层变体与实测分类一一对应

timing-predict 用现有 `glm52_dsa_moe` arch 跑出 1074 个 leaf slot 实例 / 150 个唯一 slot 名,
分成四个层变体 —— 与实测的四类层完全对齐:

| sim 层变体 | 唯一 slot 数 | 对应实测 | 实测 kernel 数 |
|---|---|---|---|
| `dense_full_index` | 33 | dense 层 ×3 | 34 |
| `sparse_initial_index_share` | 33 | initial IndexShare ×3 | 42 |
| `sparse_cycle_full_index` | 48 | cycle full-index ×18 | 54 |
| `sparse_cycle_index_share` | 33 | cycle IndexShare ×54 | 42 |

**结论:现有 arch 的结构是对的,差的是 leaf 切分粒度。** 新的 `glm52_vllm_dsa_moe`
应当保留这套结构,只在 vLLM 的 launch 粒度与之不同的地方重切叶子。

### 4.3 粒度差异集中在三处

以一个 IndexShare 层(42 实测 kernel vs 33 sim slot)为例:

1. **dense GEMM 的 fp8 量化 kernel 没有独立 slot —— 已确认是真缺口,并已量化。**

   `profiling/runners/gemm/deepgemm.py:profile_single_gemm` 把 fp8 量化
   (`per_token_cast_to_fp8`)放在计时闭包**外面**,只有 `deep_gemm.fp8_gemm_nt` 进入
   `Timer.cupti`。所以 sim 的 `single_gemm` 叶子测的是纯 GEMM,不含量化。
   而 vLLM 每个 dense deep_gemm 之前都实发一个
   `fp8_blockscale_gemm::scale_1x128_kernel`。

   实测(device 0,iteration 200–232,forward 相 40.701 ms/iter):

   | 量化 kernel | launches/iter | ms/iter | sim 是否建模 |
   |---|---|---|---|
   | `scale_1x128<bf16, fp8_e4m3, float>` | **411** | **0.758** (1.9%) | ❌ **无对应 slot** |
   | `scale_1x128<(bool)0, …>` | 150 = 75 稀疏层 × 2 | 0.584 | ✅ `routed_experts.{gate_up,down}.input_quant` |
   | `per_token_group_quant_8bit` | 21 = 21 个 full-index 层 × 1 | 0.050 | ✅ `indexer.q_quant` |

   后两行的 launch 计数与 sim 的 slot 数**逐一相等**,这反证第一行的 411 次确实是漏的
   ——不是分类错误,是模型里根本没有这个叶子。

   **进一步的结构证据(决定性)。** 把 decode 主序列 `sequence_c3757a65a0fd` 完全展开
   (3469 kernel)后逐位置检查:

   ```
   scale_1x128_kernel<bf16, fp8_e4m3, float> 总数        561
   其中后继是 deep_gemm::fp8_gemm 的                     561      ← 561/561,无一例外
   ```

   **每一次这个量化 kernel 的紧后继都是一个 dense fp8 GEMM。** 所以「每个 dense fp8 GEMM
   前一个量化叶子」不是猜测的建模选择,是实测序列直接给出的确定关系。

   按层切分后的分布(用层入口 RMSNorm 切,75 个入口):

   | 段 | quant 次数 | 层数 | 小计 |
   |---|---|---|---|
   | 层前缀(embedding + 3 个 dense 层) | 25 | — | 25 |
   | IndexShare 层(42 kernel) | 7 | 56 | 392 |
   | full-index 层(54 kernel) | 8 | 18 | 144 |
   | | | | **561** ✓ |

   full-index 层比 IndexShare 层正好多 1 次 —— 与「full-index 层多一个 indexer 的
   dense fp8 GEMM」一致。

   注:561 是**两个模板实例之和**。按后继 GEMM 的 `(N, K)` 模板参数逐个解码后,
   两个模板各自的归属完全闭合:

   | 后继 GEMM `(N, K)` | 解码 | quant 模板 | 次数/iter | sim slot |
   |---|---|---|---|---|
   | (2624, 6144) | `attn.fused_qkv_a_proj`,N = q_lora 2048 + kv_lora 512 + rope 64 | `<bf16,…>` | 78 | ❌ |
   | (16384, 2048) | `attn.q_b_proj`,N = 64 头 × 256 | `<bf16,…>` | 78 | ❌ |
   | (6144, 16384) | `attn.o_proj`,K = 64 头 × 256 | `<bf16,…>` | 78 | ❌ |
   | (4096, 2048) | `indexer`,N = 32 头 × 128,K = q_lora | `<bf16,…>` | 21 | ❌ |
   | (24576, 6144) | `dense_ffn.gate_up`,N = 2 × 12288 | `<bf16,…>` | 3 | ❌ |
   | (6144, 12288) | `dense_ffn.down`,K = dense_inter | `<bf16,…>` | 3 | ❌ |
   | (4096, 6144) | `shared_expert.gate_up`,N = 2 × moe_inter | `<bf16,…>` | 75 | ❌ |
   | (6144, 2048) | `shared_expert.down`,K = moe_inter | `<bf16,…>` | 75 | ❌ |
   | | | | **411** | |
   | (4096, 6144) / (6144, 2048) | **routed** experts(在 `expandInputRows` 之后) | `<(bool)0,…>` | 150 | ✅ |
   | | | | **561** | |

   **411 逐位点闭合,零残差。** 这是一份可以直接照着切叶子的规格,不是估计。

   曾经的一个疑点已排除:每个稀疏层确实出现**两对** `(4096,6144)+(6144,2048)`,
   但它们不是「两个 shared expert」(checkpoint 明写 `n_shared_experts: 1`)。
   看层内有序邻居即可定案 —— 第一对在 MoE dispatch **之前**(idx 19–23,
   夹着 `triton_poi_fused_mul_silu_slice_0`),是 shared expert;第二对在
   `expandInputRows` **之后**(idx 33–37,夹着 `doActivationKernel`),
   是 routed experts,用的是另一个 quant 模板且已有 slot。

   **修法**:`glm52_vllm_dsa_moe` 在上表前 8 行的每个位点前加一个 `fp8_block_quant` 叶子。
   该 L1 kernel **已注册并在用**(routed-expert 路径就在用),不需要新增 L1 kernel。
   需要新建 `VllmGlm52{DsaAttn,DenseFfn,SharedExpert}Local` 三个兄弟 worklet
   (attention 3 处 + indexer 1 处 / dense FFN 2 处 / shared expert 2 处)。

   **可证伪的验收条件**:改完后 timing-predict 出的 quant slot 计数必须逐行等于上表
   ——总计 411,且 dense 层 6、full-index 稀疏层 6、IndexShare 稀疏层 5。
   对不上就是叶子切错了地方。
2. **MoE dispatch 的 bookkeeping 扇出。** 实测 8 个 kernel
   (`computeCountAndIndice` / `computeCumsum` / `moveIndice` / `allToAllMetadata` /
   `memsetExpertIds` / `moeAllToAll` / `fusedBuildExpertMapsSortFirstToken` /
   `expandInputRows`)对 sim 的 2 个 slot(`dispatch_inter` / `dispatch_intra`)。
   多对一是允许的,但要确认 sim 的 p2p 叶子是否覆盖了 metadata prep 的开销。
3. **`moe.router.router_fp32_cast` 没有明显的实测对应物** —— vLLM 把它融进了
   `triton_red_fused_fused_add_rms_norm_moe_forward_shared_0` 或 router GEMM 路径。
   这是一个「模拟有、实测无」的叶子,按规则应保留为未映射的 simulated slot。

另有两处待确认身份的实测 kernel:`triton_poi_fused_2`(idx 3)、
`cublasLt::splitKreduce_kernel`(idx 18,router GEMM 的 splitK 归约)。

### 4.4 待 Check 2 解释的 duty cycle

capture 的 `kernel_span_ms = 27425`,窗口内 282 个 iteration → 约 **97 ms/iteration**;
而 device 0 的 forward 相 kernel 时间合计只有 **40.7 ms/iter**。即 forward 的 kernel 占用
约占 iteration 周期的 42%(其余是另外三个相、相间空隙、launch 开销)。

这意味着 `recommended_gpu_time_multiplier` 会明显大于 1。按 `top-align-with-framework`
的 Check 2,大倍数必须给出解释而不是被静默吸收 —— 待 kernel-align 跑出来后回填。
需要留意的是本次 capture 开了 `--cuda-graph-trace=node`,graph 追踪本身会拉长 host 侧。

### 4.5 forward 相不含 lm_head

decode 主序列的最后一个 kernel(index 3468)就是 final norm,**`lm_head` 不在 forward
NVTX 相里** —— 它落在 `postprocess` / `sample` 相(各 3 个 kernel)。
sim 的 `unified.main.lm_head` 与 `unified.main.final_residual_rms_norm` 因此要跨相映射,
打标时不能只在 forward 相里找。

---

## 5. Step 6 — analyzer kernel-align 的 DP 归约 ✅

`analyzer/rust/src/alignment_iteration/mod.rs` 原来假设「一次 iteration 全体设备跑同一条
序列」,归约写死为:所有卡的同一 `row_id` 合成一条 → `occurrence_ns()` 跨卡取
max(独立)/ `max(end) − max(start)`(同步) → **把所有条目的时长直接相加**得到
`measured_ms`。

DP 下这条相加是错的:不同 rank 跑不同序列时,它们的 `row_id` 落在互不相交的组里,
而这些组是**并发**的,相加等于把 8 张卡的时间串起来算。

### 5.1 归约的新口径:时间轴求和、设备轴取 max

比计划里写的「按 sequence_id 分组」更简单也更正确 —— 因为按 sequence_id 分组会把同一张卡的
preprocess 和 forward 拆到两个组里再取 max(它们本该相加)。实际实现改成按**设备**归约:

1. 每个 occurrence 先做今天的跨卡归约(一字未改),得到一个时长;
2. 把这个时长**记到参与该 occurrence 的每一张卡**名下;
3. 沿每张卡自己的时间轴求和(preprocess + forward + postprocess 串行,该加);
4. 跨卡取 max(并发,取最慢的 rank)。

**对称场景逐位不变**:8 卡跑同一序列时每张卡累加的是同一串数、同一顺序,求和结果逐位相同,
max 取出来还是它。浮点加法顺序没有被打乱 —— 这是刻意的,现有 llama3/qwen golden 才不会动。

`measured_ops` / `iteration_mapped_ms` / `iteration_unmapped_measured_ms` 三个量都走这条路。

### 5.2 schema 4 的读入

| 位置 | 改动 |
|---|---|
| `FoldedSequence` | `iterations: Vec<u64>` → `Option`,新增 `occurrences: Option<Vec<FoldedOccurrence>>`(带 `device_id`) |
| `PhaseInventory.sequence_by_iteration` | → `sequence_by_position: BTreeMap<(Option<i64>, u64), usize>`;`None` = schema 2/3 的「全卡共用一个打标决定」 |
| 序列查表 | 先查 `(Some(device_id), iter)`,查不到回落 `(None, iter)` |
| `load_inventory` | 接受 schema 2/3/4;schema 4 要求有 `device_ids`、**不许**有 `representative_device_id`(并集目录里没有代表卡),且必须用 `occurrences` 而非 `iterations` |
| `measured_kernel_rows` | 新增 `sequence_id` 字段,便于事后审计哪张卡跑了哪条序列 |

`mod.rs:320` 的 `devices == expected_devices` 断言按计划保留。

### 5.3 验证 ✅

- [x] `cargo test -p analyzer`:**154 passed**(改动前 149,新增 5 条 inventory 用例覆盖
      「schema 4 各卡各自序列」「schema 3 保持设备无关」「schema 4 拒绝代表卡」
      「schema 4 拒绝设备无关的 iterations」「同一 (device, iteration) 重复分配报错」)。
- [x] **对称回归逐位一致**。拿 `20260803_1_qwen3_235b_fp8_tp4_ep4_alignment_ctx8k_out1k_c64`
      (TP4+EP4,4 卡对称)用改动后的 analyzer 重跑 kernel-align,与已归档的
      `analysis_kernel/` 比对:

      reports/alignment_iteration_report.json   : IDENTICAL
      payloads/alignment_iteration_series.json  : IDENTICAL

      两份产物的**唯一**差异是 (a) `analysis_log_dir` 路径本身,(b) 本次新增的
      `sequence_id` 审计字段。**没有任何数值变化**,包括
      `recommended_gpu_time_multiplier`。回归产物(98 MB)验证后已删除,
      配置留在 `analyze_kernel_dp_regression.yaml` 以便重跑。

---

## 6. Step 3 — `glm52_vllm_dsa_moe` 新 arch ✅

按 Qwen 三档的形态建了**独立静态 L4 图** `simulator/src/arch/glm52_vllm_dsa_moe.rs`
(2053 行),`glm52_dsa_moe.rs` 一行未改(只把 `parse_model_json` 从私有提到 `pub(super)`)。
**零 if 分支**:两张图是两个文件,不是一个文件加开关。

### 6.1 共享 vs 复制的分界

| | 处置 | 理由 |
|---|---|---|
| `Glm52ModelCfg` / `Glm52MtpMode` / `parse_model_json` | **共享**(从 native 图 import) | 它们描述 checkpoint,不随「用什么口径测量」而变 |
| 拓扑常量、config 展开、CostTree 构造、`IterwiseUnifiedModel` 实现 | 复制 | 这是要分道扬镳的部分 |
| checkpoint 解析的两条测试 | 只留在 native 图 | 测的是共享解析器,不该重复 |

### 6.2 接线

- `arch/mod.rs`:`pub mod` + `pub use Glm52VllmDsaMoe{Configs,Model,Parallel,Resolved}`
- `arch/config.rs`:`IterArchSel::Glm52VllmDsaMoe`,参数面与 `Glm52DsaMoe` 完全一致
  (`ep_size` / `nvl_num_gpu` / `routing` / `routing_seed` / `mtp_mode`)
- `arch/build.rs`:`pub fn glm52_vllm_dsa_moe(...)` + `build_iter_model` 分发臂;
  沿用同一条「拒绝 num_layers override」的 checkpoint 约束
- `deployment/unified.rs`:与 `hp_unified` worker 配对的臂

### 6.3 分歧落地:411 / 411 ✅

按 §4.3 那张逐位点闭合的规格切叶子。第一个兄弟 worklet 已完成:

**`simulator/src/worklet/vllm_glm52_shared_expert_local.rs`**(新建)——
在 `gate_up_proj` 和 `down_proj` **各自之前**加一个 `fp8_block_quant` 叶子:

```
SOURCE_ORDER: gate_up_input_quant → gate_up_proj → silu_and_mul
              → down_input_quant  → down_proj                     (3 → 5 叶)
```

- 两个 quant config 都是 `Option`,**仅当 `gemm_dtype` 是 FP8 时为 `Some`** ——
  BF16 GEMM 两侧都不做量化,不能凭空多出叶子。
- `num_problems: 1`(dense 投影是单问题,与 routed grouped GEMM 不同);
  `hidden_size` 两个叶子不同:gate_up 吃 hidden 6144,down 吃 shared_width 2048。
- 不新增 L1 kernel:`Fp8BlockQuantKernel` 已注册且 routed-expert 路径在用。
- `glm52_vllm_dsa_moe` 已切到这个 worklet(native 图仍用原来的)。

另外两个兄弟 worklet 同样完成:

| worklet | 加的叶子 | 覆盖次数/iter |
|---|---|---|
| `vllm_glm52_shared_expert_local` | `gate_up_input_quant` / `down_input_quant` | 150 |
| `vllm_glm52_dense_ffn_local` | 同上 | 6 |
| `vllm_glm52_dsa_attn_local` | `fused_qkv_a_proj_input_quant` / `q_b_proj_input_quant` / `o_proj_input_quant` | 234 |
| ↑ 同一文件 | `indexer_q_proj_input_quant` | 21 |
| | | **411 / 411** |

三个 worklet 共用同一条约束:quant config 是 `Option`,**仅 FP8 时为 `Some`**;
`num_problems: 1`;`hidden_size` 取该 GEMM 的 **K 轴**(被量化的行宽),所以
`fused_qkv_a_proj` 用 6144、`q_b_proj` 用 q_lora 2048、`o_proj` 用 heads×v_head_dim 16384、
dense FFN 的 down 用 12288、shared expert 的 down 用 2048 —— 与 §4.3 解码出的
`(N, K)` 逐一对应。

indexer 的那 21 次没有 fork L2 的 15 叶 `DsaIndexerOp`,而是把 quant 叶子放在
**组合 indexer 的那个 worklet**里(`indexer_q_proj_input_quant`,仅
`include_indexer && fp8` 时存在,`hidden_size = q_lora_rank`)。
`indexer.q_proj` 的 N = 32×128 = 4096、K = 2048,与实测形状逐位相符;
为加一个叶子复制那个 op 的另外十四个叶子,代价远大于收益。

### 6.4 slot 公式已同步,并且自校验

`expected_slot_count` 里已有 `usize::from(fp8)` 的先例,照它加:

```
attn_full / attn_shared  +3 (fp8)      dense_ffn / shared_expert  +2 (fp8)
```

四个层变体各 +5,带 indexer 的两个(dense、sparse_full)再各 +1 →
**每 EP rank 22 个新叶子**,ep=8 时 1074 → 1250。
BF16 计数**一个都没变**(没有 FP8 GEMM 就没有要量化的东西),
与 native 图逐位相同 —— 这条是新图「只加不改」的证据。

注意 `build` 会拿真实编译出的树核对这个公式,所以公式错了会在建图时炸,
不会被一个陈旧的手写期望值掩盖。

### 6.5 把 §4.3 的验收条件钉成测试

新增 `fp8_quant_leaves_reproduce_the_measured_launch_census`,直接断言 §4.3 那张表:

```
3 dense 层 × 6  +  18 full-index 稀疏层 × 6  +  54 IndexShare 稀疏层 × 5  =  411
```

再从 slot 公式反读同一普查。这里落出一个意外的交叉验证:FP8 − BF16 的 slot 差是
**28 × ep**,不是 22 × ep。多出的 6 是**本来就有的** routed grouped-GEMM 量化
(`expert_slots + fp8*2`,出现在三个稀疏变体里)。也就是说 slot 公式的差值天然分成两半,
**与实测的两个 kernel 模板一一对应**:

| slot 差 / EP rank | 对应实测模板 | 次数/iter |
|---|---|---|
| 22 | `scale_1x128<bf16, fp8_e4m3, float>`(本次新增) | 411 |
| 6 | `scale_1x128<(bool)0, …>`(原本就建模) | 150 |
| **28** | | **561** |

561 是 §4.3 展开主序列数出来的总数。两条独立路径(实测序列展开 / 静态 slot 公式)
得到同一个分解 —— 这不是设计出来的,是核出来的。

### 6.6 端到端验收:timing-predict 实跑通过 ✅

把实验的 `simulation.yaml` 切到 `type: glm52_vllm_dsa_moe` 后重跑 timing-predict
(GPU 2,新增的 `fp8_block_quant` config 现补测,约 18 分钟),282 case 全部产出。
实际 `cost_manifest` 的 slot 普查:

```
slot 实例总数 1250        ← 与 expected_slot_count(8, true, Off) 逐位一致
```

| slot | 实例数 | = 层变体数 × ep |
|---|---|---|
| `fused_qkv_a_proj_input_quant` | 32 | 4 × 8 |
| `q_b_proj_input_quant` | 32 | 4 × 8 |
| `o_proj_input_quant` | 32 | 4 × 8 |
| `gate_up_input_quant` | 32 | 4 × 8 |
| `down_input_quant` | 32 | 4 × 8 |
| `indexer_q_proj_input_quant` | 16 | **2** × 8(只有带 indexer 的两个变体) |
| 新增小计 | **176** | = 22 × 8 ✓ |
| `input_quant`(routed,原有) | 48 | = 6 × 8 ✓ |
| `q_quant`(indexer elementwise,原有) | 16 | = 2 × 8 |

`indexer_q_proj_input_quant` 是 16 而不是 32,正是它只出现在 dense 与 full-index
两个变体上 —— 与实测那 21 次 = 3 dense 层 + 18 full-index 层 同源。

**§4.3 的可证伪验收条件通过。** 这不再是静态公式的自洽,是实跑产物的普查。

分歧清单已写进新 arch 与新 worklet 的头部文档注释,每条指回本报告的实测证据。

### 6.4 测试

`cargo test -p simulator --lib`:**570 passed / 0 failed**。新图带来 13 条测试
(从 native 图复制的图形状契约)+ 1 条新写的 selector 注册测试
`glm52_vllm_selector_is_registered_and_shares_the_glm_parameter_surface`。

---

## 7. Step 5 — 逐 kernel 对应关系(打标依据)

新 arch 的 slot 名单到手后,实测 IndexShare 稀疏层的 42 个 kernel 与 sim 的 38 个 slot
可以逐位对应。下表是打标的依据,按 `operate-run-alignment` 的证据顺序
(相 → 折叠位置 → 有序邻居 → kernel 语义 → slot 契约)得出:

| # | 实测 kernel | sim slot(`body.sparse_cycle_index_share.` 前缀) |
|---|---|---|
| 0 | `triton_red_fused_add_fused_add_rms_norm_mul_1` | `attention.input_add_rms_norm` |
| 1 | `scale_1x128<bf16>` | `attention.fused_qkv_a_proj_input_quant` ← **本次新增** |
| 2 | `deep_gemm N=2624 K=6144` | `attention.fused_qkv_a_proj` |
| 3 | `triton_poi_fused_2` | *(身份未定,见下)* |
| 4 | `triton_red_fused_3` | `attention.q_a_rms_norm` |
| 5 | `scale_1x128<bf16>` | `attention.q_b_proj_input_quant` ← **新增** |
| 6 | `deep_gemm N=16384 K=2048` | `attention.q_b_proj` |
| 7 | `triton_poi_fused_add_copy_index_select_mul_slice_split_stack_sub_unsqueeze` | `attention.main_rope` |
| 8 | `vllm::concat_and_cache_mla_kernel` | `attention.sparse_mla.mla_cache_append` |
| 9 | `nvjet_tst_256x8_64x6_2x1_v_bz_TNT` | `attention.q_absorb` |
| 10 | `vllm::ConcatMLAQKernel` | `attention.sparse_mla.query_concat` |
| 11 | `_convert_req_index_to_global_index_kernel` | `attention.sparse_mla.index_remap` |
| 12 | `sm90::fwd::sparse_attn_fwd_kernel` | `attention.sparse_mla.decode` |
| 13 | `nvjet_tst_128x8_64x12_2x1_v_bz_NNT` | `attention.v_up` |
| 14 | `scale_1x128<bf16>` | `attention.o_proj_input_quant` ← **新增** |
| 15 | `deep_gemm N=6144 K=16384` | `attention.o_proj` |
| 16 | `triton_red_fused_fused_add_rms_norm_moe_forward_shared_0` | `moe.router.post_attn_add_rms_norm` |
| 17 | `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | `moe.router.router_gemm_bf16_proxy` |
| 18 | `cublasLt::splitKreduce_kernel` | 同上(splitK 的归约尾巴,**多对一**) |
| 19 | `scale_1x128<bf16>` | `moe.shared_expert.gate_up_input_quant` ← **新增** |
| 20 | `deep_gemm N=4096 K=6144` | `moe.shared_expert.gate_up_proj` |
| 21 | `triton_poi_fused_mul_silu_slice_0` | `moe.shared_expert.silu_and_mul` |
| 22 | `scale_1x128<bf16>` | `moe.shared_expert.down_input_quant` ← **新增** |
| 23 | `deep_gemm N=6144 K=2048` | `moe.shared_expert.down_proj` |
| 24 | `moe::grouped_topk_fused_small_expert_count_kernel` | `moe.router.router_select` |
| 25–32 | 8 个 dispatch bookkeeping kernel | `moe.dispatch.{dispatch_inter, dispatch_intra}`(**8 对 2**) |
| 33 | `scale_1x128<(bool)0>` | `moe.routed_experts.gate_up.input_quant`(原有) |
| 34 | `deep_gemm N=4096 K=6144` | `moe.routed_experts.gate_up.gemm` |
| 35 | `cutlass_kernels::doActivationKernel` | `moe.routed_experts.activation` |
| 36 | `scale_1x128<(bool)0>` | `moe.routed_experts.down.input_quant`(原有) |
| 37 | `deep_gemm N=6144 K=2048` | `moe.routed_experts.down.gemm` |
| 38 | `cutlass_kernels::finalizeMoeRoutingKernel` | `moe.finalization` |
| 39 | `at::native::vectorized_elementwise_kernel` | *(bookkeeping,`unmapped`)* |
| 40 | `moeAllToAllKernel` | `moe.combine.*`(**2 对 4**) |
| 41 | `at::native::reduce_kernel` | `moe.combine.*` |

**「模拟有、实测无」的 slot(保留为未映射,不硬凑覆盖率)**:

- `moe.router.router_fp32_cast` —— vLLM 把 fp32 cast 融进了 #16 或 router GEMM 路径
- `attention.kv_a_rms_norm` —— 实测 #3/#4 只有一个是 rms_norm;另一个身份未定
- `attention.sparse_mla.prefill` —— 本次 capture 是 decode-only(§8 限制 1)
- `moe.combine` 的 4 个 slot 对实测 2 个 kernel

**「实测有、模拟无」的 kernel**:#3 `triton_poi_fused_2`、#39
`vectorized_elementwise_kernel`。按规则一律 `unmapped`,不挂到邻近 op 上。

### 7.1 打标已产出并通过校验

`kernel_sequences_labeled.json` 已生成(脚本见 `$TMPDIR/label_glm52.py`),
`load_labeled_kernel_sequences` 校验通过。消歧规则是**纯局部、机械**的:
kernel 名 + (对 quant/GEMM 对)后继 GEMM 的 `(N, K)` 模板参数 —— 这个两 kernel 窗口
足以分开五个同名的 quant 位点,以及共享 `(N,K)` 的 shared/routed expert
(靠 quant 模板 `<bf16>` vs `<(bool)0>` 区分)。**局部窗口分不开的一律 `unmapped`**。

```
mapped 5001 / unmapped 5671   逐 kernel 覆盖率 46.9%   operations 31
```

覆盖率偏低是预期的:只标了 forward 相,preprocess/postprocess/sample
以及 dispatch/combine 的 bookkeeping 扇出全部留空。

### 7.2 跑 analyze kernel-align 撞出两个真问题

**问题 A(已修)**:`measured iteration 285 devices {0,1,2,3,4} != labeled inventory
devices {0..7}`。DP rank 可以整步不干活(§2.3 的步数差就是这个现象),所以在并集目录下
实测设备是标注设备的**非空子集**,不必等于。改成 schema 4 走子集断言,
schema 2/3 仍要求每步每卡都在(`is_union_catalog` 开关)。

**问题 B(已修 —— 原因是我打标漏了同步点,不是 analyzer 语义)**:
`recommended_gpu_time_multiplier must be finite and >= 1.0; found 0.9518`。

先排除自己的嫌疑:8 张卡跑的是**同一条** decode 序列(各 277 iteration),
所以 §5 的设备轴归约在这里只有一个组,**逐位退化成改动前的行为**。
0.952 不是本次改动引入的,是**原有语义在 DP 下失效**。

我最初的诊断是「DP 下 rank 不锁步,所以跨 rank 取 max 再相加得到一条没人走过的路径」,
并准备去改 analyzer 的归约语义(代价是会动 llama3/qwen 的既有数值)。

**这个诊断是错的。** EP 下每个稀疏层都有 dispatch/combine 两次 all-to-all,
一次 iteration 有 150 次 —— 各 rank 在每个 combine 处重新对齐,根本没有自由漂移的余地。

回头核自己的打标,原因立刻现形:

```
cross_rank 取值分布:  independent 528,  synchronizing 0
集合通信 kernel 147 个,status 全是 unmapped
```

**这一跑里同步归约一次都没发生。** 每个 occurrence 都走了「跨 rank 取 max」然后相加,
包括那些本该按 `max(end) − max(start)`(末位到达 → 完成)归约的 collective。
Σmax ≥ maxΣ,过计就是这么来的。

修法在打标侧,一行 analyzer 语义都不用动:把 EP 的 all-to-all 标成
`cross_rank: "synchronizing"`。用一个**一 kernel 的后向窗口**区分两者 ——
dispatch 跟在 `memsetExpertIdsDevice` 之后,combine 跟在 routing finalize 的
elementwise 尾巴之后。

```
mapped 5676 / unmapped 4996   覆盖率 46.9% → 53.2%   operations 31 → 33
synchronizing 标记 63 处(折叠后;展开是每稀疏层 2 次)
```

重跑后 **`recommended_gpu_time_multiplier = 1.00603`**,report 正常产出。
既有 golden 一个没动。

---

## 8. 对齐判读(Check 1 逐 kernel + Check 2 duty cycle)

277 个纯 decode iteration(另有 5 个 mixed)。

### 8.1 Check 2 — duty cycle 先说,因为它决定后面怎么读

```
recommended_gpu_time_multiplier = 1.00603
measured_gpu_cycle_ms  p50 = 43.029      measured_ms  p50 = 42.906
```

**倍数贴着 1。** 也就是说这个 workload 下 GPU 几乎没有空转:一次 iteration 的
wall-clock(43.03 ms)与 kernel 关键路径和(42.91 ms)只差 0.3%。
DP8+EP8 的 decode 稳态是**完全 GPU-bound** 的 —— launch gap 和集合通信的到达等待
加起来只占 0.3%。

这跟 §4.4 当时那个「97 ms/iter vs 40.7 ms/iter,倍数会很大」的担心正好相反。
当时那个比值是拿 `kernel_span_ms` 除以 iteration 数得来的,而 `kernel_span` 是
**8 张卡的 kernel 时间总和**;除以 8 就回到 ~12 ms,与真实的 43 ms/iter 不矛盾。
`measured_busy_union_ms` p50 = 126.7 ms(8 卡区间并集)也印证这一点。**§4.4 的疑虑解除。**

倍数贴近 1 意味着:**后面逐 kernel 的偏差不会被 duty-cycle 修正掩盖**,
kernel 成本模型的误差会一比一传到吞吐上。

### 8.2 整体口径

| | 实测 p50 | 模拟 p50 | 偏差 |
|---|---|---|---|
| iteration kernel 关键路径 | 42.906 ms | 42.483 ms | **−0.98%** |
| iteration GPU cycle | 43.029 ms | 42.739 ms | −0.68% |

**总体只差 1%。** 但这个数字会骗人 —— 它是若干处 +10~15% 与若干处 −40~80% 抵消的结果。
逐 operation 才是真话。

覆盖率:实测时长 **70.7%** 已映射,模拟 workload **79.3%** 已映射。
33 个 operation **全部 282/282 配对,两侧零缺失**。

### 8.3 Check 1 — 逐 operation 偏差(按实测 p50 降序)

本表是**修完 §10 第 6 条的接线之后**的值(routed expert 的 grouped GEMM 已按 EP rank
吃各自的 `local_ppm` 分片)。括号里是接线修复前、8 个 rank 共用写死 uniform 分布时的旧值。

| operation | 实测 ms/iter | 模拟 ms/iter | 偏差 |
|---|---|---|---|
| `routed_experts.gate_up` | 12.876 | 13.925 (旧 14.272) | **+8.1%** (旧 +10.8%) |
| `routed_experts.down` | 6.629 | 7.092 (旧 7.379) | **+7.0%** (旧 +11.3%) |
| `attention.o_proj` | 2.361 | 2.733 | +15.7% |
| `moe.combine` | 1.464 | 1.167 | −20.3% |
| `moe.dispatch` | 1.190 | 1.167 | −2.0% |
| `attention.q_b_proj` | 0.939 | 1.011 | +7.7% |
| `shared_expert.gate_up` | 0.845 | 0.975 | +15.3% |
| `moe.finalization` | 0.831 | 0.162 | **−80.5%** |
| `attention.fused_qkv_a_proj` | 0.817 | 0.904 | +10.6% |
| `attention.input_add_rms_norm` | 0.503 | 0.288 | −42.8% |
| `shared_expert.down` | 0.496 | 0.640 | +29.1% |
| `moe.router_select` | 0.323 | 0.177 | −45.3% |
| `routed_experts.activation` | 0.268 | 0.141 | −47.3% |
| `attention.main_rope` | 0.209 | 0.138 | −33.7% |
| `attention.q_a_rms_norm` | 0.128 | 0.198 | +54.5% |
| **合计(已映射)** | **32.523** | **33.694** | **+3.6%** |

倾斜让模拟**变快**,方向符合物理:decode 下每个专家平均只摊到 1.75 个 token,
均匀分布使 32 个本地专家几乎个个非零(一堆极小的 group),而实测倾斜把 token 集中到
少数专家、其余拿 0 —— DeepGEMM 的代价随非零 group 数走,group 少了反而快。
`to_per_expert_counts` 的 `max(1)` 地板只兜底**分片总数**(deficit 只加到残差最大的那
一个专家上),所以单个专家确实可以拿 0,这条路径是通的。

### 8.4 三条可归因的结论

**① dense fp8 GEMM 系统性过预测 +7~16%。** `routed_experts.{gate_up,down}` (+8.1/+7.0%)、
`attention.{fused_qkv_a_proj,q_b_proj,o_proj}` (+10.6/+7.7/+15.7%)、
`shared_expert.gate_up` (+15.3%) —— **方向一致、幅度接近**。
这与 `alignment_gemm_overprediction`(Llama3-8B 上 dense_gemm 过预测 +16%)是同一个现象,
说明它不是模型特定的,是 L1 GEMM 成本模型在 decode 小 M 区间的共性偏差。
按绝对值算,单 `routed_experts.*` 两项就贡献 +1.51 ms/iter,是最大的一块。

**② 本次新增的 6 个 quant 叶子一致过预测 +36~45%。**

| | 偏差 |
|---|---|
| `attention.fused_qkv_a_proj.input_quant` | +36.1% |
| `attention.o_proj.input_quant` | +38.1% |
| `indexer.q_proj.input_quant` | +38.2% |
| `shared_expert.down.input_quant` | +40.4% |
| `shared_expert.gate_up.input_quant` | +40.9% |
| `attention.q_b_proj.input_quant` | +44.7% |

而**原有的** routed 路径 quant 只有 `+4.9%` / `−19.3%`。差别在
`num_problems`:routed 是 32(每卡 32 个 expert),新增的 dense 全是 1。
**L1 `fp8_block_quant` 在 `num_problems=1` 这一端系统性偏高。**
这是一个干净的、指向明确的 L1 修正目标 —— 而且只有把这些叶子切出来才看得见它,
在此之前这部分成本根本不存在。绝对值不大(合计约 +0.3 ms/iter),但它是新增建模的自证。

**③ 小 kernel 一律欠预测 −33~80%。** `moe.finalization` (−80.5%)、
`routed_experts.activation` (−47.3%)、`moe.router_select` (−45.3%)、
`attention.input_add_rms_norm` (−42.8%)、`attention.main_rope` (−33.7%)。
这些实测都在 0.2–0.8 ms/iter 量级,除以 75 层后单次只有几微秒 ——
**已经进入 kernel launch 开销主导区**,而 L1 的成本曲线是纯计算外推的。
合计欠预测约 −1.5 ms/iter,恰好抵消掉 ① 的过预测,这就是「总体只差 1%」的来历。

### 8.5 尚未判读的部分

- **prefill 的样本量**:只有 5 个 iteration(见 §9.5)。而且这 5 步的偏差数字目前
  **不可用** —— prefill kernel 打标不全、collective 的到达等待剥不掉(§10 第 4、5 条)。
  修完那两条之后才能重新判读,要定位到具体 kernel 还需要 capture B 把 prefill 占比提上来。
- **未映射的 29.3% 实测时长**:主要是 dispatch/combine 的 bookkeeping 扇出
  (8 个 kernel 对 2 个 slot)、preprocess/postprocess/sample 三相,以及
  `triton_poi_fused_2` / `vectorized_elementwise_kernel` 两个身份未定的 kernel。
  按规则保持 `unmapped`,没有为覆盖率硬凑。

---

## 9. Check 3 — e2e:吞吐、TTFT、TPOT

sim 已跑通(注入 kernel-align 实测的 `gpu_time_multiplier = 1.00603`),
`analyze alignment analyze` 产出 `alignment_e2e_report.json` 与 6 张对照图。
**256 / 256 请求双向配对,两侧各自零孤儿** —— 配对本身没有可疑之处。

### 9.1 吞吐:+3.4%,很好

| | 实测 | 模拟 | 偏差 |
|---|---|---|---|
| completion tps | 653.15 tok/s | 675.13 tok/s | **+3.4%** |
| 输出 token 总数 | 262,144 | 262,144 | 0 |
| 完成跨度 | 401.4 s | 388.3 s | −3.3% |

考虑到 §8.4 里 dense GEMM 过预测 +10% 与小 kernel 欠预测 −40% 是互相抵消的,
这个 +3.4% 更像是"两个大误差恰好抵消",而不是"成本模型准"。

### 9.2 分布形状对不上 —— 这才是真问题

| 指标 | 实测 p50 | 模拟 p50 | 实测 p99 | 模拟 p99 |
|---|---|---|---|---|
| server TTFT | 7,659 ms | **1,725 ms** | 169,491 ms | **52,892 ms** |
| TPOT | 51.3 ms | **92.7 ms** | 179.1 ms | **93.4 ms** |
| e2e | 63,064 ms | **96,523 ms** | 225,384 ms | 149,890 ms |
| e2e **均值** | 95,963 ms | 96,806 ms | | |

**e2e 均值几乎完全吻合(+0.9%),中位数却差 +53%。** 这是典型的
「总量对、形状不对」:模拟把所有请求磨成了差不多的样子,而实测是
「一半请求跑得很快 + 一条很长的尾巴」。

最刺眼的是 TPOT:

```
实测 TPOT:  p50 51.3   p90 154.3   p99 179.1     ← 3.5 倍跨度
模拟 TPOT:  p50 92.67  p90 92.67   p99 93.41     ← 几乎是个 delta 函数
```

**模拟的 TPOT 分布退化了。** 每个请求看到的每 token 时间几乎一模一样。

### 9.3 归因:sim 的 shard 负载"太均匀了"

TPOT = 该请求所在 shard 的 iteration 时间。模拟里 8 个 shard 走纯轮转
(`placement.rs:11`,`(next+1) % num_partitions`),每个 shard 拿到的请求数
**逐个相等**,于是 batch 大小相等、iteration 时间相等、TPOT 相等 —— 退化成 delta 函数。

实测侧不是这样。§2.3 已经记录:

```
records per dp_rank: {0: 5138, 1: 5265, 2: 5137, 3: 5137,
                      4: 5137, 5: 5136, 6: 5265, 7: 5135}
```

rank 1 和 6 多跑 2.5% 的步数 —— 各 rank 负载**本来就不均**。

**这推翻了计划里 Step 7a 的方向。** 原计划要加 `LoadBalance::LeastLoaded`,理由是
「vLLM 是负载感知的(`score = waiting*4 + running`),sim 是纯轮转,所以 sim 不够均衡」。
数据说的恰恰相反:

- sim 已经**过于**均衡(TPOT 方差近乎为零);
- 加 `LeastLoaded` 只会让它**更**均衡,把差距推向错误的方向。

真正的口径差在**绑定时机**,不在选择策略:

| | 绑定时机 | 依据 |
|---|---|---|
| vLLM | 请求**到达**时就粘到某个 EngineCore | 到达那一刻的 `waiting*4 + running`,**按请求计数,不按 token** |
| sim | **admit** 时才决定 | admit 那一刻的实际负载 |

vLLM 在到达时用一个**更差的信息**(计数而非 token 量)做了一次**不可撤销**的绑定;
8k prompt 下不同请求的实际负担差异巨大,于是长请求会在某个 rank 上堆积,
形成实测里那条长尾。sim 推迟到 admit 才决定,信息更新更全,自然更均匀。

**结论:要复现实测的分布形状,需要的是「到达时粘性放置」,不是 `LeastLoaded`。**
这要动 worker 的 pending membership(计划里已经预判到"改动大得多"),
是一个独立的设计决定,不在本轮范围内。**`LeastLoaded` 这一项应当撤销**,
它建立在一个被数据证伪的前提上。

### 9.4 TTFT 的额外说明

模拟 TTFT 在各分位都更快(p50 快 4.4×,p99 快 3.2×)。
`top-align-with-framework` 允许 sim 的 TTFT 略快(vLLM 异步调度器的固有开销),
**但 4 倍不属于"略快"**。其中一部分同样来自 9.3 的排队形状差异
(模拟没有长请求堆积,所以没有那条尾巴)。剩下的部分需要
`profile_kind: workload_metrics` 的完整调度时间线才能拆开 —— 本实验没跑那一 pass(§10 限制 3)。

**更正**:我一度写成"本次 capture 是 decode-only,TTFT 侧没有 kernel 级证据"。
**这是错的** —— 见 §9.5,prefill 的 kernel 级证据存在,而且暴露出本轮最大的一个偏差。

### 9.5 prefill iteration:−55%,而且原因很具体

282 个 iteration 里有 5 个 `mixed`(iteration 8–12),它们**就是 prefill**。
nsys 侧的序列分配把这件事说得很清楚:

| 序列 | kernel 数 | 卡分布 | iterations |
|---|---|---|---|
| `sequence_4e5023995e6c` | 3586 | 仅 device 0(×4) | 8, 9, 10, 11 |
| `sequence_17c15f8e0fd1` | 3532 | device 0–4(各 ×1) | 8, 12 |
| `sequence_c3757a65a0fd` | 3469 | 8 卡全有(各 ×277) | 13 起 |

**这是 DP 分叉的真实实例**:iteration 8–11 里 device 0 在做 prefill、
device 1–4 做另一种 prefill、device 5–7 纯 decode。§2.4 的并集目录和 §5 的设备轴归约
不是为假想场景写的,就是为这五步写的。

偏差:

| iteration | 实测 | 模拟 | 偏差 |
|---|---|---|---|
| 8 | 1919.53 ms | 863.22 ms | **−55.0%** |
| 9 | 1933.69 ms | 861.72 ms | −55.4% |
| 10 | 1947.61 ms | 861.73 ms | −55.8% |
| 11 | 1942.81 ms | 861.60 ms | −55.7% |
| 12 | 97.94 ms | 49.07 ms | −49.9% |
| 13(首个纯 decode) | 46.75 ms | 42.48 ms | −9.1% |

**先排除输入的嫌疑。** `per_dp_rank` 喂进去的 case 0 是:

```
group 0: prefill 8190 token (2 chunk) + 2 decode      ← 对应 device 0 的 4e50...
group 1-4: prefill 8176 token (1 chunk) + 6 decode    ← 对应 device 1-4 的 17c1...
group 5-7: 0 prefill + 7 decode                        ← 对应 device 5-7 的 decode
```

与实测的序列分配**逐一吻合**。输入是对的,缺口在成本模型。

**缺口的去向(按实测 max 与模拟 max 的差排序)**:

| operation | 实测 max | 模拟 max | 缺口 |
|---|---|---|---|
| `moe.combine` | 339.07 | 5.96 | **+333.11** |
| `routed_experts.gate_up` | 237.72 | 27.84 | **+209.88** |
| `moe.dispatch` | 145.73 | 5.96 | **+139.77** |
| `routed_experts.gate_up.input_quant` | 157.96 | 19.12 | +138.84 |
| `routed_experts.down` | 125.08 | 15.85 | +109.23 |
| `routed_experts.down.input_quant` | 53.08 | 6.52 | +46.56 |
| `attention.main_rope` | 46.07 | 2.44 | +43.63 |
| `moe.finalization` | 54.88 | 17.02 | +37.87 |
| 前 14 项合计 | | | **+1093.4** |

+1093 ms 与 iteration 级 1056 ms 的缺口吻合 —— **整个 −55% 都在 MoE 路径上,
attention 基本没责任**(`main_rope` 的 +43.6 是唯一一个上榜的 attention 项)。

#### 这里曾经写过一条错误结论,已撤回

先前版本据上表判定「MoE 成本模型在 prefill token 量级低估 8 倍」,并把它列为本轮最大缺陷。
**这条不成立**,因为它违反了 `top-align-with-framework` 的 Check 1 前提:
**偏差先假定是打标的错,证据链没走完之前不许归因到成本模型。** 走完之后:

1. **prefill 的 kernel 有一大批根本没打标。** iteration 8 的 measured 分解里,
   未映射的 kernel 合计约 500 ms,单是 `sm90::fwd::KernelTemplate`(FlashAttention 的
   prefill 模板)就 307.3 ms。它们既不在任何 operation 的实测侧,也没有对应的模拟 slot,
   所以上表的「实测 max」包含了模拟侧压根不该负责的量。
   相关的还有 `sm90_fp8_gemm_1d2d_impl`(约 102 ms / 36 ms 两处)、
   `topKPerRowPrefill`、`fp8_mqa_logits`;另外 `sparse_attn_fwd_kernel` 被无条件打成
   `sparse_mla.decode`,prefill 的那部分挂错了 operation。

2. **`moe.combine` 的 339 ms 里绝大部分是等待,不是工作。** `moeAllToAllKernel` 的
   duration 天然包含最后一个 rank 的到达等待;analyzer 的 `synchronizing` 归约
   (`max(end) − max(start)`)本来就是为了剥掉它,但归约是按 `row_id` 做的,而
   `row_id` 里嵌了 `sequence_id`。iteration 8 恰好是 DP 分叉步:device 0 跑
   `4e50…`、device 1–4 跑 `17c1…`、device 5–7 跑 `c375…`,三组的 `row_id` 互不相同,
   **同一次 all-to-all 的 8 张卡永远不会被归到一起归约**,于是剥不掉等待。
   device 5–7 是纯 decode,它们在这一步等的就是 device 0 的 prefill 时间 ——
   这 339 ms 里装的正是那段 prefill 墙钟。

也就是说 −55% 的两个最大分项(`moe.combine` +333、`moe.dispatch` +140)是**度量口径
造成的**,`routed_experts.*` 的那部分则被未打标的 prefill kernel 污染。真实的
prefill 成本模型偏差**目前仍未知**,要等下面两条修完才能重新判读:

- 补齐 prefill kernel 的打标(见 §10 第 4 条);
- 让 `synchronizing` 归约能跨 sequence(见 §10 第 5 条)。

需要说明的取值口径:上表用的是各 operation 在 282 个 case 上的 `max`,
而 5 个 prefill iteration 恰好支配了上尾,所以 max 可以当 prefill 的代理量。
严格的逐 iteration 分解需要 analyzer 输出 per-case 的 operation 明细,现在只有汇总分位。

---

## 10. 已知限制

1. **这次 capture 几乎全是 decode。** 全窗口 282 iteration/卡中 277 个是纯 decode
   (8k prompt / 1k output 下 prefill 只占千分之一)。prefill/TTFT 侧的 kernel 证据基本为空,
   任何关于 prefill kernel 的结论都缺证据支撑,须靠一次 prefill 专项 capture 补。
2. **metrics 是 schema v1。** vLLM fork 在 2026-08-04 才加入 v2 的 `observed_*` EngineCore
   cadence(capture 是 08-03),所以 e2e 的 workload cadence 子项会降级。
3. **没有 `profile_kind: workload_metrics` pass。** e2e workload 子项只能用受 CUPTI
   暂停影响的 NSYS 窗口。
4. **prefill 的 kernel 打标不全。** `sm90::fwd::KernelTemplate`(307.3 ms)、
   `sm90_fp8_gemm_1d2d_impl`、`topKPerRowPrefill`、`fp8_mqa_logits` 都是 unmapped;
   `sparse_attn_fwd_kernel` 被无条件打成 `sparse_mla.decode`,需按相拆成 prefill / decode。
   在补齐之前,§9.5 的 prefill 偏差数字不可用。
5. **`synchronizing` 归约不能跨 sequence。** analyzer 按 `row_id` 归约,而 `row_id`
   里嵌了 `sequence_id`;DP 分叉步上同一次 collective 的 8 张卡分属不同 sequence,
   于是到达等待剥不掉,collective 的实测时间被高估。这是 analyzer 的真实缺口,
   不是打标问题 —— collective 应当按「同一次 all-to-all 的全部参与设备」归约,
   与各自跑的是哪条 sequence 无关。
6. **GLM 的 routed-expert grouped GEMM 曾对路由分布不敏感(已修)。**
   `glm52_dsa_moe` / `glm52_vllm_dsa_moe` 的 `[Max over EP ranks]` 扇出一直存在,
   但 8 个 rank 复用同一个 worklet 对象,`local_ppm` 写死为
   `uniform_local_ppm(256, ep)` = `[3906; 32]`,所以 `expert_popularity_file` 只影响
   dispatch/combine 的通信字节数,grouped GEMM 逐位不变。现已改为
   `MoeExpertComputeLocalWorkletConfig::split_for_ep(template, routing.ppm())`,
   每个 rank 持有自己的分片(与 Qwen 的 vLLM 档一致)。
   **副作用**:`RoutingDistribution::uniform(256)` 把 1e6 的余数 64 摊给前 64 个专家,
   所以即使 `routing: uniform`,rank 0/1 是 `[3907; 32]`、rank 2–7 是 `[3906; 32]` ——
   grouped GEMM 的 cache key 从 1 个变成 2 个,已有的 GLM profile.db 行需要重新 profile。
   `split_for_ep` 切的是**连续**专家段,而 `MoeNetConfig` 声明的是 `Placement::RoundRobin`,
   两者对不上(Qwen 同样如此),暂不改。
   **修复后的效果**:`routed_experts.gate_up` 从 +10.8% 收到 **+8.1%**、
   `down` 从 +11.3% 收到 **+7.0%**,已映射合计从 +5.5% 收到 +3.6%(见 §8.3)。
   **代价**:每个 EP rank 一份分布 → grouped-GEMM 的 cache key 数 ×8,
   本次补测了 ~1040 行 profile.db。
   注意进入 sim 的倾斜幅度是 **0.14× ~ 7.5×**,不是 profile 里 `counts_all_layers`
   看到的 0.79× ~ 1.37×:`canonicalize_layerwise_expert_counts` 是**逐层**把 rank 内专家
   按负载降序、rank 间按总量降序后再累加,保留的是层内倾斜,而 `counts_all_layers`
   跨 75 层平均会把它抹平。判断倾斜量级时不要用后者。
