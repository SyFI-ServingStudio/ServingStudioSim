# Llama3-8B TP1 / TP2 / TP4 alignment 工作日志

## 2026-07-21 当前 checkpoint 摘要（后续判断以本节为准）

### Common-anchor rate32 结果

| TP | 当前配置 | iteration mean APE | signed mean | GPU-span output-rate error | 判断 |
|---:|---|---:|---:|---:|---|
| 1 | `llama3_dense_tp(tp=1)` | 5.31% | -4.55% | +0.66% | clean baseline，达到目标 |
| 2 | pure AllReduce best-of `[nccl,nvshmem]` | 5.85% | +1.08% | -2.64% | aggregate 好，但 fused-boundary 语义仍不正确 |
| 4 | pure AllReduce best-of `[nccl,nvshmem]` | 10.76% | -7.06% | -0.96% | throughput 好，kernel total 略超目标，operation fidelity 不足 |

- TP1 已完成 rate16/24/32/48 unified bridge，四档 simulation 与旧 dense arch 数值
  一致；iteration mean APE 为 4.44%--6.79%，output-rate error 为 +0.18%--+3.74%。
  这是目前可信度最高的 scaling baseline。
- TP2/rate32 从 NCCL-only 的 22.01% mean APE 降至 `[nccl,nvshmem]` 的 5.85%，
  但改善主要来自 pure network curve 偶然接近 vLLM fused
  `AllReduce + residual + RMSNorm` 总耗时。attention/MLP boundary 的
  per-iteration signed mean 仍分别为 +95.19%/+74.72%，存在明显跨 operation
  cancellation，不能把 5.85% 当成 fused modeling 已完成。
- TP4/rate32 profile、simulation、timing-predict、label、analysis 全链路完成。
  vLLM 有 1,800 个 fused iterations 与 240 个超过 2 MiB threshold 的 unfused
  iterations；simulator 尚未表达该 runtime branch。coverage 为 measured 96.026%、
  simulated 99.952%，其中 measured coverage 比 97% 目标低约 0.97 percentage point。
- TP4 retry 与原 capture 高度复现：client throughput 相差 0.13%，multiplier 相差
  0.50%，iteration mean APE 10.55% vs 10.76%。空闲 GPU、稳定 1830 MHz clock、
  无外部 compute process，排除了 capture 损坏、外部 GPU contention 和 thermal
  throttle。高尾误差主要来自多 rank host arrival skew 后早到 rank 在 collective
  kernel 内等待；该等待是真实 deployment runtime 现象，但不应写入 isolated L1
  AllReduce cost curve。

### Iteration 680 当前解释

- 原 replica-wide breakdown 的 `12.439 ms launch sum` 是按 kernel identity union
  后再堆叠的 workload sum，不是 kernel busy union，也不是 NSYS 约 9.4 ms 的
  rank-local forward span，不能直接与 7.393 ms timing-predict 作 total error。
- fastest-rank device0 的 all-phase busy union 为 8.206406 ms，rank-local launch
  sum 为 8.224070 ms，timing-predict 为 7.392504 ms；同口径 workload endpoint
  error 为 -0.831566 ms（-10.1%）。这说明该 iteration 的 isolated kernel cost
  大体接近，而 replica-wide 慢主要包含 rank skew/collective wait。
- diagnostic attribution 已按 folded position 精确拆分：attention network
  AllReduce 1.114974 ms 一对一连接 `S8`，attention residual+RMSNorm 0.183136 ms
  一对一连接 `S9`；embedding AllReduce 显式 unmapped；input norm、MLP AllReduce、
  final norm 也分别连接对应 slots。mixed attention operation 保留 `M10 -> S5+S6`。
- renderer 已修复 one-to-many center overwrite；第三 row 是与 simulated slots
  使用相同宽度、颜色和显式 xlim 的 cumulative-error step。最终 endpoint 必须等于
  iteration workload total error；该图仍是 experiment-local rank diagnostic，不能
  替代 authoritative replica-wide report。

### 尚未完成与下一步

1. 扩展 mapping/analyzer contract，使同一 simulated slots 能按 runtime grammar
   表达 fused one-to-many 与 unfused one-to-one attribution，而不是依赖
   experiment-local diagnostic operation IDs。
2. 在 L3/L4 实现 shape-dependent boundary：TP2 64 MiB、TP4 2 MiB threshold 内
   使用已新增的 `all_reduce_residual_rms_norm` fused leaf，超过 threshold 回退
   pure AllReduce + separate norm；完成 fused profiler GPU smoke 与 cache population。
3. 将 rank-local clean workload、collective wait/rank-arrival skew、replica critical
   path/GPU cycle 分成三个正式统计量，避免一个 aggregate 同时承担三种语义。
4. 修正 vLLM vocab-parallel LM-head（sharded GEMM + all-gather）与 simulator
   replicated full LM-head 的结构差异。
5. 完成 TP2/TP4 pre-knee、knee-onset、knee、post-knee turning sweep；当前只有
   TP1 四档与 TP1/TP2/TP4 rate32 common anchor，尚不足以给出最终 TP scaling 结论。

## 目标

在相同 H200 节点、模型、请求 trace 和 vLLM backend policy 下完成
Llama3-8B unified deployment 的 TP1、TP2、TP4 alignment；按 kernel-only、
GPU cycle/duty ratio、TTFT/TPOT/E2E 的顺序分析，并修复有明确证据支持的异常。

## 不变量与验收口径

- 入口固定为 `uv run python -m launcher alignment {profile,sim,timing-predict,analyze}`。
- 每个 capture 独立计算 `gpu_time_multiplier`，不跨 TP 或 request rate 复用。
- 多 rank kernel 时间不得跨 GPU 求和；simulation 表示一个同步 TP replica，主比较口径必须是 replica critical path，同时保留 per-rank skew。
- 先排查大的 unmapped measured kernel / unmapped simulated slot，再判断 leaf timing。
- 目标 coverage：measured duration >= 97%，simulated workload >= 99%。
- 目标误差：iteration mean APE <= 10%，GPU-cycle/output-rate error <= 5%；饱和区 E2E/TPOT 单独解释。
- 保留所有现有用户改动；本实验不修改 `doc/detailed_design/L7.md`、`profiling/profile.db`、`simulator/src/sim/README.md`、`simulator/src/sim/run.rs` 中已有的非本任务 diff，除非后续证据表明任务必须触及。

## Ground truth 与现有基线

- 现有完成实验：`logs/20260713_5_llama3_8b_poisson_io512_alignment/`。
- 固定 trace：300 requests，input/output mean 512，Poisson arrivals。
- TP1 已有正式 rates：16、24、32、48 req/s。
- 现有 TP1 simulation 使用 `llama3_dense`，不是统一比较所需的
  `llama3_dense_tp {tp_size: 1}`；先做 TP1 bridge，复用已有 measured captures，
  重跑 sim/timing-predict/labels/analyze。
- 当前 alignment v1 文档和代码只允许一个 vLLM model rank；TP2/TP4 前需扩展 multi-rank capture/parser/analyzer contract。

## 2026-07-20 环境审计

- Git HEAD：`5df422f feat(analyzer): add hierarchical request state views`。
- sandbox 内 `nvidia-smi` 无 driver access；经用户明确授权后以 escalated access 查询成功。
- 空闲 H200：GPU 1、4、5、7（约 4 MiB，0% utilization）。
- 忙碌 H200：GPU 0、2、3、6；本实验避开。
- topology：所有八张 H200 互联均为 `NV18`，因此 1/4/5/7 可用于 TP4。

## 执行计划与状态

1. **已完成**：multi-rank profile、normalized trace、GPU ratio、label inventory 与 analyzer critical-path contract；相关 CPU/Rust tests 通过。
2. **已完成阶段验证**：TP2/rate32 已分别完成 NCCL-only 与
   `[nccl,nvshmem]` downstream；fused collective 的 Python/Rust L1 已实现，
   L3/L4 shape-dependent branch 尚未落地。
3. **已完成**：TP1 `llama3_dense_tp(tp=1)` bridge；复用 rate16/24/32/48 captures。
4. **已完成**：TP4/rate32 在 GPU 3,4,5,6 上完成端到端 alignment。
5. **待执行**：分别为 TP2/TP4 simulation turning sweep，选择 pre-knee / knee-onset / knee / post-knee；保留 rate32 common anchor。
6. **进行中**：common-anchor rate32 的 TP1/TP2/TP4 完整链路已齐；turning sweep
   仍待执行。
7. **进行中**：rate32 结果已逐 TP 记录；完整 turning-sweep 后再做最终 TP scaling
   汇总。

## 已确认的代码阻断点

- `alignment/profiler/config.py` 和 `vllm_server.py` 的 contract/docstring 写死 single GPU milestone。
- `alignment/nsys/parse.py::load_metrics` 以 iteration id 为唯一 key，无法表达 rank-local metric duplicates。
- folded kernel inventory 目前没有 rank/device 维度。
- `analyzer/rust/src/alignment_iteration/mod.rs` 明确拒绝一个 iteration 出现多个 device。
- 当前 analyzer 把所有 range kernel interval 做 union；多 GPU 下会把并发 rank 混为一个不明确口径。
- 当前 `gpu_time_multiplier` 虽能按 device 建 cycle，却只输出 pooled GPU-time ratio；需增加 replica critical-path 定义与 per-rank audit。

## 命令与结果日志

### GPU preflight

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi topo -m
```

结果：GPU 1/4/5/7 空闲，互联 `NV18`。下一项是软件 contract 实现与 CPU tests。

### TP2/rate32 profile probe

- Config：`logs/20260720_0_llama3_8b_tp_alignment/tp2/rate32/profile.yaml`
- GPU：4,5；port：18532。
- 目的：先取得真实 multi-rank NSYS artifact，确认当前 external-nsys capture、
  NVTX rank ranges、device identity 和 iteration population，再据此修改 parser/analyzer。
- 状态：config 已创建，待 dry-run/preflight/profile。

首次启动结果：在 GPU 分配前失败。原因是 config 比旧实验多一层 `tp2/`，
`fork_python` 错写成五层 `..`，解析到了 workspace 外；已修正为
`../../../../alignment/profiler/vllm/.venv/bin/python`。这也暴露出当前
profile dry-run 不验证 `fork_python` 存在性，后续纳入明显异常修复候选。

Probe 完成：

- replay 300/300，0 failure；`request_timing_count=300`。
- NSYS validation OK；`forward_ranges=3656=1828×2`。
- normalized iterations=1828，kernel rows=1,440,482。
- 每个 iteration 都同时包含 device 0/1；两设备各 720,241 kernels。
- 两 rank 的 preprocess/forward/postprocess/sample folded sequences 逐项完全一致。
- 明确 anomaly：旧 inventory 按 phase 拼接两个 rank，首个 forward 记录 644
  kernels，而每个 rank 实际各 322。
- 旧 ratio：device0 multiplier 1.122087，device1 1.089238，pooled 1.105418；
  后续改为 replica critical-path population，并保留 per-device audit。

实现后重建结果：

- schema-v3 canonical inventory：首个 forward 322 kernels，不再是错误的 644。
- `device_ids=[0,1]`、`representative_device_id=0`；full ranges 仍在 parsed timeline。
- replica-critical multiplier=1.098374，population=1825 cycles。
- 新增 TP2 simulation/timing-predict configs；simulation 使用
  `llama3_dense_tp(tp_size=2)` 和显式 TP slots/backend policy。

TP2 rate32 downstream：

- simulation complete：300/300，`sim_ms=13580.1`，22.09 req/s，22,621 total tok/s；
  rate32 已处于饱和区。
- timing-predict complete：1828 measured-shape cases。
- TP label grammar 已生成并通过 schema-v3 strict validator：81 forward sequences。
- vLLM 的 row-parallel all-reduce 与下一 RMS norm 是 fused kernel；mapping 将
  attention boundary 对齐到 attn all-reduce + post norm，将全部 MLP boundary
  对齐到 MLP all-reduce + input norms + final norm，避免为 folded 最后一次伪造语义。
- vLLM LM-head 为 sharded GEMM + NCCL all-gather，而 simulator 当前是 replicated
  full LM-head；先作为一个联合 operation 暴露误差，不隐藏该模型差异。

### TP2/rate32 首轮分析（修复前）

- coverage：measured duration 97.05%，simulated workload 99.969%，均达到预设门槛。
- iteration total：mean APE 22.01%，p50 19.84%，p90 44.31%，p99 52.92%；
  signed mean +20.40%，即 simulator 整体偏慢。
- 明显异常集中在两处 TP boundary：
  - attention all-reduce + post norm：measured 0.4447 ms，simulated 1.1816 ms，
    signed error +275.98%。
  - MLP all-reduce + next norm：measured 0.5880 ms，simulated 1.1847 ms，
    signed error +234.79%。
- 其余主要 transformer operation 多数为 simulator 偏快约 8–20%：attention
  -11.83%、qkv -14.52%、o_proj -19.58%、up_gate -7.94%、down -11.01%、
  activation -15.92%、KV append -11.91%。
- E2E（rate32 已饱和）：server TPOT measured 7.371 ms vs simulated 9.356 ms
  （+26.9%）；server TTFT 23.999 ms vs 22.031 ms（-8.2%）；server output
  throughput 12,462 vs 11,311 tok/s/GPU-replica（-9.24%）。
- 根因证据：NSYS 中两处 boundary 均为
  `flashinfer::trtllm_allreduce_fusion::*oneshot_lamport`，且 vLLM 启用了
  `allreduce_rms` fusion；simulator 当前两个 slot 都查询 `all_reduce:nccl`，
  每个约 1.081 ms，再另加约 0.101 ms RMSNorm。不能用 NVSHMEM 冒充；必须按
  本次 vLLM 的 FlashInfer TRT-LLM collective 路径建模后再判断剩余误差。

### Fused collective 修复状态

- 源码追踪确认 capture kernel 为 `Pattern=1` 的
  `kARResidualRMSNorm`，一次 launch 同时完成 all-reduce、residual add 和
  RMSNorm；现有 `all_reduce(num_gpus,message_size_bytes,...)` schema 缺少
  `[num_tokens,hidden_dim]`，且复用会 double-count norm，因此不能作为新 backend
  塞入纯 all-reduce table。
- 已新增独立 Python profiler kind `all_reduce_residual_rms_norm`，backend
  `flashinfer_trtllm`，调用 vLLM 环境中的 FlashInfer public API；registry、lazy
  import、capability 和自动 facade CPU tests 通过（16 tests）。
- 已新增对应 Rust timing kind：静态 config 保留 TP/hidden/dtype/fabric 和
  production flags，runtime axis 为 `num_tokens`；H200 fusion 上限按 vLLM fork
  固定为 TP2=64 MiB、TP4=2 MiB、TP8=0.5 MiB。focused Rust tests 4/4 通过。
- `kernel-query grid` TP4 smoke 通过：`input_fields=[num_tokens]`，H200/TP4
  grid 为 1,2,4,8,16,32,64,128,256，终点严格对应 2 MiB。
- 当前综合 CPU gate：alignment + registry 67 tests、Rust fused kind 4 tests、
  analyzer alignment 4 tests 全部通过，`git diff --check` 通过。
- 尚未写入 shared DB，也尚未声称 GPU smoke 成功。2026-07-20 后续查询时八张
  H200 均被其他任务占满；compute-app audit 显示每卡有外部
  `sglang::scheduler`（约 47--139 GiB）和 `ray::MegatronTrainRayActor`，且当前
  没有遗留 vLLM/torchrun 进程属于本实验。即使瞬时 utilization 降到 0%，显存仍
  被这些进程持有，不能据此抢占。
- 上层正确组合需要调整 L3/L4 语义：TP>1 时 fused leaf 必须替代纯
  `all_reduce + residual + rms_norm`，并处理 TP4 超过 2 MiB 后回退 unfused 的
  runtime 分支。当前 L3/L4 design 只允许 worklet 末尾纯 `tp_allreduce`，没有
  支持跨 boundary fusion/shape-dependent fallback；按 repo 的 before-edit 规则，
  修改该 contract 前需要用户确认设计扩展。

### 当前外部阻塞（连续复查）

- 连续三轮状态复查均显示所有 8 张 H200 被同一批外部
  `sglang::scheduler` + `ray::MegatronTrainRayActor` 持有；GPU 0--5/7 各占
  约 100--141 GiB，GPU6 仍在 100% utilization。不能安全运行 TP2 fused
  profiler smoke 或 TP4 NSYS capture，也不能终止这些不属于本任务的进程。
- canonical `ref/next_gen_design/detailed_design/L3/design.md` 始终带有未知来源
  的未提交 diff；本任务需要新增 cross-boundary fused/unfused runtime branch，
  会与该文件重叠。按仓库规则，在用户明确允许叠加或先处理现有 diff 前不能编辑。
- 已耗尽不依赖上述条件的实质工作：TP1 四档完整 downstream、TP2 capture 与
  diagnosis、multi-rank infrastructure、fused Python/Rust L1、TP4 config dry-run、
  analyzer rank/replica 语义修复及 focused tests 均已完成。
- TP2 per-iteration boundary-substitution oracle（只把两处 simulated
  NCCL+norm 换成同 iteration 的 measured fused-boundary 时间，其他 prediction
  不变）把 mean APE 从 22.01% 降到 7.38%，p50 7.66%、p90 10.75%、p99
  14.90%，signed mean 从 +20.40% 变为 -6.94%。这证明 collective anomaly 是
  当前主误差源；修复后其余 GEMM/attention 偏快会成为下一层主误差，而不是继续
  调 global multiplier 掩盖问题。

### TP1 unified bridge 完成

- 新实验入口：`logs/20260720_0_llama3_8b_tp_alignment/tp1/`；复用
  `20260713_5...` 的四个已验证 TP1 captures，simulation 统一改为
  `llama3_dense_tp(tp_size=1)`，labels 只做旧 dense slot → dense_tp slot 的
  一一重命名。
- rate16/24/32/48 四档 simulation 均 300/300 `DrainComplete`。新旧 arch 的
  `sim_ms`、total throughput、completed req/s 逐项完全相等：
  22220.6 / 17085.9 / 15751.8 / 14019.2 ms。这证明 TP1 degenerate path 无
  数值漂移。
- 四档 measured-shape timing-predict 和 analyzer 均完成；coverage 全部约
  measured 98.0%、simulated 99.97% 以上。
- authoritative current reports：

| rate | iteration mean APE | p90 APE | signed mean | measured GPU-span tok/s | simulated tok/s | output-rate error |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4.44% | 7.06% | -4.07% | 6,899.8 | 6,912.5 | +0.18% |
| 24 | 4.95% | 7.60% | -4.72% | 8,844.9 | 8,989.9 | +1.64% |
| 32 | 5.31% | 8.77% | -4.55% | 9,687.6 | 9,751.3 | +0.66% |
| 48 | 6.79% | 10.30% | -6.17% | 10,561.1 | 10,956.4 | +3.74% |

- 旧 README 中 rate48 的 6.63% / +4.54% 是较早汇总口径；当前旧 analysis
  report 与新 dense_tp report 都是 6.78693% / +3.74%，两者仅有浮点尾差。
  后续跨 TP 表统一以现存 JSON reports 为 authority，避免复用陈旧手写表格。

### TP2 `[nccl, nvshmem]` simulation 重跑

- 用户要求将 TP2/rate32 的两个 `tp_allreduce` backend candidate list 改为
  `[nccl, nvshmem]`。该配置的语义是每个 shape 在两个 cache curve 中取更快值，
  不是分别运行 NCCL-only 与 NVSHMEM-only 两个 deployment。
- 新配置：
  `logs/20260720_0_llama3_8b_tp_alignment/tp2/rate32/simulation_net_nccl_nvshmem.yaml`；
  新 artifact root：同目录下 `simulation_net_nccl_nvshmem/`，不覆盖原始
  `simulation/` NCCL-only 结果。
- dry-run 通过，展开为一个 TP2 run，attention 与 MLP 两个 AllReduce role 均为
  `['nccl', 'nvshmem']`。
- 实际 simulation 在 cache-build 阶段停止，未产生 `.complete`：NCCL curve 的
  19 个点全部命中，NVSHMEM curve 缺少 TP2/H200/NVLink/BF16 的完整 19-point
  grid（4 KiB 到 1 GiB），profiler 报
  `RuntimeError: need 2 idle GPU(s), found 0`。
- 只读 `--cache-report` 按两个 leaf 报 38/1233 misses；两 leaf 共用 cache key，
  唯一缺失 specs 实际为 19。随后以 shared `profiling/profile.db`、显式
  `--gpu-name 'NVIDIA H200'` 运行 `profiling count-missing`，确认
  `missing_count=19/spec_count=19`。spec batch 保存在
  `tp2/rate32/nvshmem_all_reduce_specs.json`。
- 提升权限 GPU audit：8 张 H200 均有外部 `sglang::scheduler` 与 Megatron actor，
  全部为 100% utilization；GPU6 虽仅占约 27.8 GiB，也非 idle，且 TP2 profiling
  需要两张空闲卡。未抢占或终止外部任务，shared DB 未被本次失败运行写入
  NVSHMEM rows。两张 H200 空闲后，重跑同一 alignment sim 命令即可自动 JIT-fill
  19 rows 并继续 simulation。
- 交叉核对已完成的
  `logs/20260720_1_llama3_tp_request_state/preset.json`：该 preset 没有
  `backends:` override，因此使用 arch defaults。其 TP2 cache-build log、运行
  `stdout.log` 和 `raw/kernel_grid_peaks.json` 都明确记录两个 `tp_allreduce` 为
  `backends=['nccl']`，只加载了 NCCL 的 19 samples；该完成 run 没有查询或
  profile NVSHMEM，因而不与当前 19 个 NVSHMEM misses 矛盾。
- 资源恢复后，用户指定 `VIBESIM_PROFILE_GPUS=3,4,5,6`；GPU audit 显示四卡均
  4 MiB / 0% utilization。launcher 将其用于两个双卡 profiling chunks，成功把
  TP2 NVSHMEM 19-point grid 写入 shared `profiling/profile.db`。事后显式
  `count-missing` 为 `0/19`，cache-build log 同时记录 NCCL 与 NVSHMEM 各
  19 samples。
- `[nccl, nvshmem]` simulation 已完成：`.complete` 存在，300/300 requests，
  `DrainComplete`。`sim_ms=12659.3`，23.698 req/s，24,266.7 total tok/s；相对原
  NCCL-only 的 13,580.1 ms、22.091 req/s、22,621.3 tok/s，simulation duration
  缩短 6.78%，吞吐提升约 7.27%。
- 实际 best-of-N sampled selection：两个 AllReduce positions 都是 NVSHMEM
  79/80（98.75%），仅约 9.42 MiB 的 sampled shape 选择 NCCL。profile grid 上
  NVSHMEM 在 8 MiB 为 0.0512 ms vs NCCL 0.0538 ms；到 16 MiB 时 NCCL
  0.0770 ms 已快于 NVSHMEM 0.0968 ms，符合该 crossover。两个 AllReduce 的
  总 kernel-time share 从 NCCL-only 的 28.05% 降到 18.09%。
- 已基于新 simulation root 独立重跑 measured-shape timing-predict 与 alignment
  analyze，未覆盖 NCCL-only baseline。新 roots 为
  `timing_predict_net_nccl_nvshmem/` 和 `analysis_net_nccl_nvshmem/`；复用同一
  TP2 vLLM profile 与 `kernel_sequences_labeled.json`。timing-predict 写出 1,826
  cases，CostTree 同时加载两条 AllReduce curves。
- 新 iteration alignment：mean APE 5.85%、p50 2.32%、p90 15.07%、p99
  28.69%，signed mean +1.08%。相对 NCCL-only 的 22.01% / 19.84% / 44.31% /
  52.92%、signed +20.40%，total-level alignment 大幅改善。coverage 保持
  measured 97.05%，simulated 99.964%。
- E2E：server TPOT error +26.93% -> +5.03%，server GPU-span output-throughput
  error -9.24% -> -2.64%，mean E2E error +21.42% -> +0.57%；server TTFT error
  则从 -8.20% 变为 -13.71%。300 request IDs 全部对应，无 missing。
- 必须保留语义 caveat：NVSHMEM 是 pure AllReduce，仍未表示 vLLM capture 的
  FlashInfer TRT-LLM `AllReduce + residual + RMSNorm` fused kernel，simulator 也
  仍单独累计 norm。operation-level boundary error 虽下降但没有消失：attention
  mean simulated/measured 0.678/0.445 ms，per-iteration signed mean +95.19%；MLP
  0.681/0.588 ms，signed mean +74.72%。因此 5.85% total APE 含跨 operation
  error cancellation，不能替代正确 fused L1/L2/L3/L4 modeling。

### TP4/rate32 完整 alignment（GPU 3,4,5,6）

- 实验 root：`logs/20260720_0_llama3_8b_tp_alignment/tp4/rate32/`；profile、
  gpu-kernel-ratio、simulation、timing-predict、label、analyze 全部完成。
- vLLM+NSYS profile：300/300 replay success，153,600 output tokens，2,040 个
  normalized measured iterations，3,310,232 normalized kernel rows；NSYS validation
  为 8,168 forward ranges、3,313,396 raw kernel rows、11,795.655 ms kernel span，
  `ok=true`。四个 logical device 0--3 对应 physical GPU 3,4,5,6。
- replica critical-path multiplier 为 `1.1721573630980657`；2,039 cycles，GPU
  kernel fraction 85.313%，因此 simulation 使用该 multiplier，而不是跨四卡 pooled
  kernel-time ratio。
- simulation 使用 `llama3_dense_tp(tp_size=4)`，两个 AllReduce role 都配置为
  `[nccl, nvshmem]`。300/300 `DrainComplete`，`sim_ms=11856.7`，25.302 req/s，
  output throughput 12,954.70 tok/s（总 prefill+decode 25,909.40 tok/s；每 GPU
  6,477.35 tok/s）。两个 AllReduce position 的 sampled best-of-N 都是 NVSHMEM
  82/82（100%）。
- 首次 TP4 decode shape cache 需要补齐 162 个 `flashinfer_attn_decode` timing
  specs；显式使用 GPU 3,4,5,6 的四个 profiling worker JIT-fill 后 simulation 完成。
  timing-predict 随后写出 2,040 measured-shape cases。
- TP4 capture 同时存在两种严格 forward grammar：16 个 folded sequence、1,800
  iterations 走 FlashInfer fused AllReduce+residual+RMSNorm；43 个 sequence、240
  iterations 因超过 TP4 2 MiB fusion limit，走独立 multimem AllReduce + Triton
  RMSNorm fallback。实验 label helper 已支持并严格校验两条路径；最终 59 个
  forward unique sequences 全部通过 schema validator。
- coverage：measured duration 96.026%，simulated workload 99.952%。measured coverage
  略低于预设 97% 目标约 0.97 percentage point，主要剩余项是 RoPE/helper、
  preprocess/sample 等没有 simulator leaf 的 kernels。
- iteration total（2,040 paired cases）：mean APE 10.76%，p50 5.43%，p90
  32.82%，p99 50.43%；signed mean -7.06%，说明 simulator 的 kernel-only total
  平均偏快。该结果略高于 10% mean APE 目标。
- 主要 operation signed error（simulated relative to measured）：attention -15.80%，
  qkv -9.97%，o_proj -23.76%，up_gate -12.13%，down -11.94%，activation
  -16.95%，KV append -2.70%。LM-head +79.95%，仍包含 vLLM vocab-parallel
  GEMM/all-gather 与 simulator replicated LM-head 的已知语义差异。
- TP boundary 不能只看 aggregate total：attention boundary measured/simulated mean
  为 0.670/0.749 ms，但 per-iteration signed mean +65.24%、mean APE 76.23%；MLP
  boundary aggregate为 1.111/0.752 ms，signed mean -4.31%、mean APE 33.88%。后者
  的均值接近来自 fused 与 unfused shape 的正负误差抵消，不代表 shape-wise 对齐。
- E2E：server GPU-span measured output throughput 13,079.98 tok/s，simulation
  12,954.70 tok/s（-0.96%）；server TPOT mean 6.122 vs 5.487 ms（-10.38%）；
  server TTFT 22.141 vs 12.891 ms（-41.78%）；mean E2E 3,187.11 vs 2,816.77 ms
  （-11.62%）。300 request IDs 全部对应，无 missing。
- 结论：TP4/rate32 的整体吞吐已经很接近，kernel-only mean APE 为 10.76%；但
  当前 pure `[nccl,nvshmem]` simulator 没有按 2 MiB threshold 表达 vLLM 的
  fused/unfused runtime branch。operation-level TP boundary 和 TTFT 仍是下一步
  修正重点，不能用 total throughput 的 -0.96% 掩盖。

### TP4 profile retry 与 contention 审计

- 因 iteration 680 breakdown 出现 measured `12.439 ms launch sum` vs predict
  `7.393 ms`，没有覆盖旧 capture，而是在
  `logs/20260720_2_llama3_8b_tp4_profile_retry/rate32/` 建立独立 retry。
- retry 起跑前 8 张 H200 均为 4 MiB、0% utilization、SM clock 1830 MHz；运行中
  GPU 3,4,5,6 的 compute process 只有本次 `VLLM::Worker_TP0..3`，没有外部 GPU
  process。四卡采样 clock 均为 1830 MHz，温度约 28--32C；没有 GPU contention、
  thermal throttle 或 clock drop 的证据。
- retry profile 300/300 success，1,992 normalized iterations、3,218,600 normalized
  kernel rows，validation `ok=true`。client output throughput 13,048.90 tok/s，旧
  capture 为 13,065.93 tok/s，仅低 0.13%；retry multiplier 1.177971，旧值
  1.172157，仅高 0.50%。整体 capture 可复现。
- retry measured-shape timing-predict/analyze 写出 1,990 paired cases。iteration
  mean APE 10.55%、p50 4.52%、p90 34.65%、p99 54.70%，signed mean -7.71%；旧
  capture 分别为 10.76%、5.43%、32.82%、50.43%、-7.06%。因此高尾误差不是单次
  capture 损坏。
- analyzer 没有把四卡 duration 相加：total 和 operation measured time 都是跨 rank
  interval union。breakdown 图显示的 `launch sum` 是每个 kernel identity 先 union、
  再把不同 identity 堆叠，因此旧 iter680 图上 12.439 ms 略高于 authoritative
  total union 11.505 ms；但 11.505 vs 7.393（-35.74%）仍是真异常。
- 根因集中在 rank arrival skew 后的 collective wait，而不是所有 kernel 统一变慢。
  旧 iter680（mixed，prefill 518/decode 85）四 rank 的 GEMM、attention、norm 基本
  一致，但 unfused collective sum 为 rank0 2.277 ms、rank1 4.992 ms、rank2
  5.078 ms、rank3 4.643 ms；rank0 forward start 比最早 rank 晚约 1.49 ms。
- retry 更极端地复现：iter443（mixed，prefill 592/decode 79）rank0 collective
  2.387 ms，而 rank1/2/3 为 17.217/14.955/17.319 ms；GEMM 仍都约 4.04--4.10
  ms，attention 都约 1.63 ms。早到 rank 在 unfused collective kernel 内等待晚到
  rank，等待被 NSYS 计入 kernel duration。
- rank skew 是系统性的：旧 capture forward-start skew p50/p90/p99 为
  0.275/1.473/4.071 ms，>1 ms 有 302/2040 iterations；retry 为
  0.254/1.499/4.468 ms，>1 ms 同样 302/1990，retry max 12.600 ms。两次分布高度
  一致，排除偶发外部 GPU contention，但不能排除/忽略 vLLM 多进程 host launch
  skew；它是当前真实 deployment 的运行时现象。
- 后续不能继续重跑 profile 期待消失，也不能把这些等待时间塞进 isolated
  AllReduce L1 curve。alignment 应同时报告：rank-local clean collective cost、
  replica critical-path/union，以及独立 rank-arrival-skew/wait 项；否则 TP4
  operation APE 会把调度/同步等待误判为纯 collective kernel cost error。

### Iteration 680 图表口径更正

- `iter_680_breakdown.png` 顶部的 `12.439 ms launch sum` 不是 NSYS iteration
  span，也不是去除 bubble 后的 authoritative pure-kernel time。renderer 对每个
  folded kernel identity 先跨 rank 做 interval union，再把不同 identity 的结果
  相加并堆叠；当不同 rank 处于不同 operation 时，这个和会重复覆盖同一 wall-clock
  时间。它只能作为 workload-composition 图，不能直接与一个 replica 的 wall span
  或 CostTree critical path 做误差判断。
- iter680 rank0 forward first-kernel→last-kernel span 为 9.488948 ms，正是 NSYS
  看到的约 9.4 ms；rank0 forward kernel busy union 为 7.930022 ms，因此 forward
  bubble 为 1.558926 ms。rank1/2/3 forward span 分别为 10.868075/10.974680/
  10.534718 ms，busy union 为 10.626337/10.728304/10.292144 ms。
- 当前 analyzer 的 authoritative `measured_ms` 也不是图上的 12.439 ms：它对四
  rank、全部 captured phases 的所有 kernel intervals 做一次 global union，iter680
  为 11.504830 ms。forward-only replica union 为 10.898320 ms；all-phase replica
  span 为 13.547504 ms，其中 global-union bubble 为 2.042674 ms。
- 如果该 iteration 的 kernel-only 对照采用 rank0 全 phase busy，则 preprocess
  0.004320 + forward 7.930022 + postprocess 0.191008 + sample 0.081056 =
  8.206406 ms；对 timing-predict 7.392504 ms 的误差约 -9.92%，不是图面
  12.439 vs 7.393 所暗示的 -40.57%。
- 但固定改用 rank0 也不是全局解：旧 capture 2,040 cases 的 device0-busy mean
  APE 为 13.34%、signed mean +9.11%，retry 为 13.93%/+7.71%，反而高于现有
  replica-union mean APE 10.76%/10.55%。需要把 rank-local kernel workload、
  collective wait/rank skew、replica wall critical path 分成三个统计量，而不是选择
  一个聚合量同时承担三种语义。
- 已为 iter680 生成独立 fastest-rank 诊断图：按全 captured phases kernel busy
  union 最小选择 device0，不覆盖原 replica-wide breakdown。device0 busy union
  8.206406 ms、rank-local launch sum 8.224070 ms，对 timing-predict 7.392504 ms；
  图位于
  `tp4/rate32/analysis/plots/iter_520_to_1017/iter_680_fastest_rank_breakdown.png`。
  可复现 helper 为实验目录下 `draw_fastest_rank_breakdown.py`。该图中 GEMM、
  attention 和两个 collective boundary 已在相同量级，支持“此 iteration 的模拟器
  pure-kernel cost 基本合理，而 vLLM replica-wide 结果被 rank skew/collective
  wait 拉坏”的判断；这仍是 iter680 的局部诊断，不替代全 run 的分布统计。
- fastest-rank 图仅在展示层把跨层
  `model.mlp_allreduce_and_norm_boundaries` aggregate 延后到其他 forward groups
  之后、postprocess 之前，使其靠近 simulated AllReduce/final-norm slots，消除横穿
  全图的 mapping arrow；duration、label、phase ownership 均未修改。segment 编号
  按新展示顺序重排，因此该 aggregate 从原图的 M5 显示为新图的 M15。
- iter680 fastest-rank 图的 M15 不是单一 kernel signature，而是 66-launch
  aggregate：33 次
  `vllm::cross_device_reduce_2stage<__nv_bfloat16, 4>(RankData*, RankSignals,
  Signal*, T1*, int, int)`，合计 1.162264 ms；32 次
  `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_2`，合计 0.240640 ms；1 次
  `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1`，0.004064 ms。总计
  1.406968 ms。按 forward 位置判断，33 个 collectives 是 embedding 输出
  AllReduce 1 次 + 32 层 MLP AllReduce；当前 mapping 的 simulated slots 只有
  32 层 MLP AllReduce + 32 个 input norms + final norm，因此 aggregate 隐藏了一个
  measured-only embedding AllReduce，后续应拆开标注。
- 上述 coarse M15 已在 fastest-rank 诊断图中按 folded position 拆开：首个
  embedding AllReduce 为 M5，0.034496 ms，显式 unmapped；S2 input-norm ×32
  对应 M6，由首个 `...rsqrt_1` ×1 加随后 `...rsqrt_2` ×31 组成，合计
  0.237824 ms；S13 MLP AllReduce ×32 对应 M16，合计 1.127768 ms；S14 final
  norm ×1 对应 M18，即最后一个 `...rsqrt_2`，0.006880 ms。这里修正了“全部
  32 个 `rsqrt_2` 都是 input norm”的过粗说法：最后一个按位置属于 final norm。
  该 refinement 仅改变 experiment-local diagnostic arrows，不改 authoritative
  labeled inventory/report；若推广到正式 analyzer，需要先扩展 mixed fused/unfused
  path 共享 simulated slot 的 mapping contract。

### Iteration 680 一对多 mapping arrow 修正

- labeled inventory 的 semantic matching 本身已经把 mixed attention operation
  `layer.attention` 映射到两个 additive simulated slots：prefill attention 与 decode
  attention。原图只显示 `M10 → S6`，不是 `S5` 没有匹配，而是 Python renderer
  用 `operation -> single center` 保存绘图位置；后出现的 `S6` 覆盖了 `S5`。
- renderer 已改为为每个 operation 保留全部 segment centers，并绘制所有合法的
  measured-to-simulated ownership arrows。回归测试覆盖 one-measured-to-two-simulated
  slots。重新生成后，`M10` 同时连接 `S5`、`S6`，`M12` 同时连接 `S8`、`S9`。
- fastest-rank device0 的 `M10` 共 96 launches、1.490302 ms：32 次 FlashInfer
  persistent prefill kernel（0.477760 ms）、32 次 FlashInfer paged decode kernel
  （0.888062 ms）、32 次 `PersistentVariableLengthMergeStatesKernel`（0.124480
  ms）。它与 `S5` 0.372057 ms 加 `S6` 0.914565 ms 作 operation-workload 对比。
- `M12` 共 64 launches、1.298110 ms：32 次
  `vllm::cross_device_reduce_2stage`（1.114974 ms）加 32 次
  `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0`（0.183136 ms）。语义上分别
  对应 `S8` attention AllReduce 与 `S9` post-norm。fastest-rank diagnostic 已按
  32 组严格交替的 `collective -> norm` folded positions 拆成两个 measured segments，
  分别作一对一连接；不再用一个 coarse segment 同时连接两个 slots。
- 当前 matching 是按 phase、folded position、ordered neighbors 和 kernel semantics
  人工赋予稳定 operation，再由 operation 连接 CostTree slots；它不是按 kernel
  signature 自动模糊匹配。此前图中一对多箭头缺失会让 matching quality 看起来比
  label 实际质量更低，但 coarse boundary label 仍需像 M5/M6/M16/M18 那样按位置
  审计，不能只凭同色 operation bucket 判断正确。
- iteration breakdown 另增加紧凑的 cumulative error step row。它与 simulated stack
  共享毫秒口径：每个 interval 宽度严格等于对应 `S` slot 的 `folded_ms`，背景色复用
  该 slot 的颜色，而不是等宽 categorical bins；终点标注 signed `Δ ms` 与 percentage。
  一对多 operation 的 measured workload 按其 simulated slot 宽度成比例分摊，仅作为
  可视化 convention，保证 operation total 与 iteration endpoint 不变；所有
  measured-only workload 进入 step 的初始负 baseline。该 row 不使用 replica
  kernel-busy union 或 GPU-cycle wall time，避免把 workload composition error 与
  rank-skew/idle 混合。
- cumulative step row 的 vertical allocation 已增至原先约两倍，并同步增加整张
  breakdown canvas 高度，保持上方 measured/simulated stacks 的可读空间。
- 修正 top stack 与 cumulative row 的 x-position drift：此前 cumulative axes 使用
  显式 `0..1.03*max(total)`，top axes 使用 Matplotlib auto limits，造成相同 simulated
  slot boundary 有轻微横向偏移；现在两个 axes 由同一个显式 workload xlim 驱动，
  第三 row 的 variable-width 色块与上方 `S` segments 垂直对齐。
- authoritative inventory 暂时仍保留
  `layer.attention_allreduce_post_norm -> [tp_allreduce, post_norm]`，因为同一 run 的
  fused path 用一个 CUDA kernel 同时拥有两个 slots，而 unfused path 用两个 kernels。
  现有 contract 要求一个 slot 在全 run 只属于一个 operation，无法同时表达
  fused one-to-many 与 unfused one-to-one operation IDs；本次精确拆分因此限定在已确认
  为 unfused grammar 的 iter680 diagnostic。
