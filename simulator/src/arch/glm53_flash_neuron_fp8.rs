//! GLM-5.3-Flash (FP8 checkpoint) on AWS Trainium2 logical cores (LNC2),
//! served by the glm5-trn2 seed engine (`glmflash/neuron/DESIGN.md`,
//! `engine/engine.py`): one process per logical core, one compiled region per
//! layer, collectives traced into the regions. Every compute leaf is an L1 row
//! of that engine's own op modules (`doc/pr/trn2-glm53-flash/L1_CONTRACT.md`).
//!
//! ```text
//! replica   world ranks (32 = 8 chips); attention DP = world / attn_tp groups of
//!           attn_tp contiguous ranks, one KV partition each; FFN EP over the world
//! prologue  embed (vocab-parallel over the TP group) + reduce_scatter(tp)
//! layer     attention: mhc_pre(own rows) -> all_gather(tp) -> KDA | DSA (heads
//!           over TP; indexer and latent cache replicated) -> reduce_scatter(tp)
//!           -> mhc_post
//!           FFN: mhc_pre -> all_gather(world) -> local experts + shared-expert
//!           shard (MoE) | dense 1/world shard (layers 0-2) -> reduce_scatter(world)
//!           -> mhc_post
//! epilogue  head region: all_gather(tp) of the final hidden rows, lm_head over the
//!           sampled rows + local argmax; combine region: (value, id) gathers
//! mtp       (speculative selector, placeholder: the engine has no MTP yet) one
//!           DSA + MoE layer with embed and head per pass
//! launch    one leaf per forward: host dispatch and launch beyond its leaves
//! ```
//!
//! "Own rows": the residual streams are sequence-parallel inside the TP group,
//! so a rank holds `R / attn_tp` of its group's `R` rows; the attention mixer
//! runs on all `R` rows (gathered) with `64 / attn_tp` heads; the FFN runs on
//! the world-gathered `N = sum over groups of R` rows.
//!
//! Step (the engine's `serve.py`): a decode forward over every group's running
//! requests, then a prefill forward with at most one chunk per DP group
//! (worker `max_prefill_chunks_per_partition: 1`). Shapes are bucketed and
//! SPMD (`spmd_padding`, default on): every group runs the decode forward at the
//! bucket of the largest group batch and the context bucket of the longest KV,
//! and the prefill forward at the bucket of the longest chunk and the context
//! bucket of the furthest `start + T`, idle groups included. A per-group leaf
//! (attention, mHC, head) costs the slowest group's shape; with SPMD every group
//! has the same shape.
//!
//! Composition (`launch_model` pipelined, the trn2-flashnext fit): every
//! compute leaf is its L1 row times `region_device_factor` (f_g). A forward
//! costs `max(sum over regions of max(device_r + region_gap_ms,
//! region_host_ms), regions x region_host_ms + forward_overhead_ms)`; the step
//! adds `step_overhead_ms`. Each forward's launch leaf holds everything beyond
//! the sum of its leaves. Regions per forward: embed + one per layer (or one per
//! `block_layers` layers under `region_granularity: block`, a what-if) + head +
//! combine.
//!
//! Not modeled: host scheduling beyond the fitted constants, first-use stalls
//! of a new shape, sampling (greedy only), the MTP `eh_proj` fusion GEMM.

use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use serde::Deserialize;

use crate::arch::contract::{
    IterwiseUnifiedModel, SpeculativeArchInput, SpeculativeUnifiedModel, UnifiedArchInput,
};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::Glm53FlashModelCfg;
use crate::arch::kv_layout::{EngineOverhead, KvCacheLayout, KvCachePart};
use crate::common::Fabric;
use crate::timing::bridge::intern_backend;
use crate::timing::bridge::DType;
use crate::timing::kernels::engine::{Kernel, KernelSpec};
use crate::timing::kernels::*;
use crate::timing::slot_input::NeuronLaunchLog;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, FlatCostNode, LeafMetrics,
    PerfApiBridge, Probe, SlotInput,
};

const BF16: u64 = 2;
const FP32: u64 = 4;
const FP8: u64 = 1;
/// Logical cores per Trainium2 chip at LNC2.
pub const CORES_PER_CHIP: u32 = 4;
/// mHC Sinkhorn iterations (`hc_sinkhorn_iters`).
const SINKHORN_ITERS: u32 = 20;
/// The MTP placeholder's acceptance curve source: InferenceX `glm-5.3-flash`
/// golden acceptance length (accepted tokens per verify step, the target's own
/// included) at k = 1..5 drafts.
pub const GOLDEN_ACCEPTANCE_LENGTH: [f64; 5] = [1.86, 2.52, 3.00, 3.35, 3.60];

// ── knobs ────────────────────────────────────────────────────────────────────

/// How the engine cuts a forward into compiled regions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionGranularity {
    /// One region per layer (the seed engine).
    #[default]
    Layer,
    /// One region per `block_layers` layers (what-if).
    Block,
}

impl RegionGranularity {
    pub const CHOICES: [&'static str; 2] = ["layer", "block"];

    /// Host dispatch per region (ms): the trn2-flashnext engine fit with the
    /// NRT execution barrier off (layer 1.3 ms, 4-layer block 3.7 ms).
    pub fn region_host_ms(self) -> f32 {
        match self {
            Self::Layer => 1.3,
            Self::Block => 3.7,
        }
    }
}

/// One what-if time scale: compute leaves of L1 `kind` at calls of at least
/// `min_rows` rows run `factor` times their measured device time.
#[derive(Clone, Debug, PartialEq)]
pub struct KindTimeScale {
    pub kind: String,
    pub min_rows: u32,
    pub factor: f32,
}

impl KindTimeScale {
    /// Parse `kind=factor` or `kind@min_rows=factor`.
    pub fn parse(entry: &str) -> Result<Self> {
        let Some((lhs, factor)) = entry.split_once('=') else {
            bail!("kind_time_scale entry {entry:?} is not `kind=factor` or `kind@min_rows=factor`");
        };
        let (kind, min_rows) = match lhs.split_once('@') {
            Some((kind, rows)) => (
                kind,
                rows.trim().parse::<u32>().map_err(|_| {
                    anyhow::anyhow!("kind_time_scale entry {entry:?}: bad min_rows {rows:?}")
                })?,
            ),
            None => (lhs, 0),
        };
        let factor: f32 = factor
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("kind_time_scale entry {entry:?}: bad factor"))?;
        ensure!(
            !kind.trim().is_empty(),
            "kind_time_scale entry {entry:?} names no kind"
        );
        ensure!(
            factor.is_finite() && factor > 0.0,
            "kind_time_scale entry {entry:?}: factor must be positive"
        );
        Ok(Self {
            kind: kind.trim().to_string(),
            min_rows,
            factor,
        })
    }
}

/// Per-op implementation keys (`neuron_<impl>` backends). Several keys = the
/// best measured row per shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm53NeuronImpls {
    pub kda_prefill: Vec<String>,
    pub kda_decode: Vec<String>,
    pub dsa: Vec<String>,
    pub moe: Vec<String>,
    pub dense_ffn: Vec<String>,
    pub mhc_pre: Vec<String>,
    pub mhc_post: Vec<String>,
    pub lm_head: Vec<String>,
    pub embed: Vec<String>,
}

impl Default for Glm53NeuronImpls {
    fn default() -> Self {
        let v = |x: &str| vec![x.to_string()];
        Self {
            kda_prefill: v("torch"),
            kda_decode: v("torch"),
            dsa: v("radix"),
            moe: vec!["dense_local".to_string(), "cte_hybrid".to_string()],
            dense_ffn: v("dense"),
            mhc_pre: v("torch_bf16"),
            mhc_post: v("torch"),
            lm_head: v("torch"),
            embed: v("torch"),
        }
    }
}

/// The layout and engine knobs of one replica.
#[derive(Clone, Debug)]
pub struct Glm53NeuronParallel {
    pub gpu_name: String,
    pub max_model_len: u32,
    /// Ranks (LNC2 logical cores) of one replica.
    pub world: u32,
    /// Attention TP group size; `world / attn_tp` attention-DP groups.
    pub attn_tp: u32,
    /// FFN expert parallelism (the engine: every rank of the replica).
    pub moe_ep: u32,
    /// MTP drafts per step (0: no MTP).
    pub draft_tokens: u32,
    pub region_granularity: RegionGranularity,
    /// Layers per region under `block`.
    pub block_layers: u32,
    pub spmd_padding: bool,
    pub prefill_buckets: Vec<u32>,
    pub decode_buckets: Vec<u32>,
    pub context_buckets: Vec<u32>,
    pub region_device_factor: f32,
    pub region_host_ms: f32,
    pub region_gap_ms: f32,
    pub forward_overhead_ms: f32,
    pub mtp_forward_overhead_ms: f32,
    pub step_overhead_ms: f32,
    /// HBM of one LNC2 core (GB) and the runtime reserve kept out of the pools.
    pub hbm_per_core_gb: f64,
    pub hbm_reserve_gb: f64,
    /// Engine request slots per DP group (slot 0 is scratch) and the per-slot
    /// DSA cache length: the pools the engine preallocates.
    pub slots: u32,
    pub cmax: u32,
    pub impls: Glm53NeuronImpls,
    pub kind_time_scale: Vec<KindTimeScale>,
}

impl Glm53NeuronParallel {
    /// The seed engine's layout (`EngineConfig` / `serve.py` defaults).
    pub fn seed(gpu_name: &str) -> Self {
        Self {
            gpu_name: gpu_name.to_string(),
            max_model_len: 9216,
            world: 32,
            attn_tp: 8,
            moe_ep: 32,
            draft_tokens: 0,
            region_granularity: RegionGranularity::Layer,
            block_layers: 4,
            spmd_padding: true,
            prefill_buckets: vec![2048],
            decode_buckets: vec![8, 16],
            context_buckets: vec![2048, 4096, 8192, 9216],
            region_device_factor: 1.0,
            region_host_ms: RegionGranularity::Layer.region_host_ms(),
            region_gap_ms: 0.17,
            forward_overhead_ms: 19.0,
            mtp_forward_overhead_ms: 2.0,
            step_overhead_ms: 2.5,
            hbm_per_core_gb: 25.769_803_776,
            hbm_reserve_gb: 2.0,
            slots: 17,
            cmax: 9216,
            impls: Glm53NeuronImpls::default(),
            kind_time_scale: Vec::new(),
        }
    }

    pub fn dp(&self) -> u32 {
        self.world / self.attn_tp
    }

    pub fn validate(&self, m: &Glm53FlashModelCfg) -> Result<()> {
        ensure!(
            [1, 2, 4, 8, 16, 32].contains(&self.attn_tp),
            "attn_tp must be 1, 2, 4, 8, 16 or 32, got {}",
            self.attn_tp
        );
        ensure!(
            self.world % self.attn_tp == 0,
            "world {} is not a multiple of attn_tp {}",
            self.world,
            self.attn_tp
        );
        ensure!(
            m.kda_num_heads % self.attn_tp == 0 && m.num_attention_heads % self.attn_tp == 0,
            "attn_tp {} does not split the 64 KDA / MLA heads",
            self.attn_tp
        );
        ensure!(
            self.moe_ep == self.world,
            "the engine's FFN spans the world: moe_ep must equal world ({}), got {}",
            self.world,
            self.moe_ep
        );
        ensure!(
            m.n_routed_experts % self.moe_ep == 0,
            "moe_ep {} does not split {} experts",
            self.moe_ep,
            m.n_routed_experts
        );
        for (name, b) in [
            ("prefill_buckets", &self.prefill_buckets),
            ("decode_buckets", &self.decode_buckets),
            ("context_buckets", &self.context_buckets),
        ] {
            ensure!(!b.is_empty(), "{name} is empty");
            ensure!(
                b.windows(2).all(|w| w[0] < w[1]),
                "{name} must be ascending"
            );
        }
        // Rows of a forward split over the TP group (sequence-parallel streams).
        for &b in self.prefill_buckets.iter().chain(&self.decode_buckets) {
            ensure!(
                b % self.attn_tp == 0,
                "bucket {b} is not a multiple of attn_tp {} (the engine splits every \
                 forward's rows over its TP group)",
                self.attn_tp
            );
        }
        ensure!(self.block_layers >= 1, "block_layers must be positive");
        ensure!(
            self.slots >= 2,
            "slots must leave one request slot after the scratch slot"
        );
        Ok(())
    }

    fn cores_fabric(n: u32) -> Fabric {
        // Contiguous ranks: n < 4 shares one chip with n ranks; n >= 4 fills chips.
        match n.min(CORES_PER_CHIP) {
            0..=1 => Fabric::NeuronC1,
            2 | 3 => Fabric::NeuronC2,
            _ => Fabric::NeuronC4,
        }
    }
}

fn bucket(n: u32, buckets: &[u32]) -> u32 {
    buckets.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

/// `n` as calls of its bucket: several largest-bucket calls above it.
fn bucket_calls(n: u32, buckets: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    let max = *buckets.last().expect("validated nonempty");
    let mut rest = n;
    while rest > max {
        out.push(max);
        rest -= max;
    }
    out.push(bucket(rest, buckets));
    out
}

// ── probes ───────────────────────────────────────────────────────────────────

trait ShapeProbe<X>: Send + Sync {
    fn kind(&self) -> &'static str;
    fn config(&self) -> serde_json::Value;
    fn eval_at(&self, x: X) -> LeafMetrics;
    fn input_at(&self, x: X) -> SlotInput;
}

struct Shaped<S: KernelSpec, X> {
    kernel: Kernel<S>,
    input: fn(X) -> S::Input,
}

impl<S, X> ShapeProbe<X> for Shaped<S, X>
where
    S: KernelSpec,
    S::Input: Clone + Into<SlotInput>,
    Kernel<S>: Probe<Input = S::Input> + Send + Sync,
    X: Copy + Send + Sync,
{
    fn kind(&self) -> &'static str {
        Probe::kind(&self.kernel)
    }
    fn config(&self) -> serde_json::Value {
        Probe::describe_config(&self.kernel)
    }
    fn eval_at(&self, x: X) -> LeafMetrics {
        Probe::eval(&self.kernel, &(self.input)(x))
    }
    fn input_at(&self, x: X) -> SlotInput {
        (self.input)(x).into()
    }
}

type Rows = Arc<dyn ShapeProbe<u32>>;
type Pair = Arc<dyn ShapeProbe<(u32, u32)>>;
type Comm = Arc<dyn ShapeProbe<u64>>;

fn shaped<S, X>(
    name: &str,
    config: S::Config,
    input: fn(X) -> S::Input,
    bridge: &PerfApiBridge,
) -> Result<Arc<dyn ShapeProbe<X>>, BuildError>
where
    S: KernelSpec,
    S::Input: Clone + Into<SlotInput>,
    Kernel<S>: Probe<Input = S::Input> + Send + Sync,
    X: Copy + Send + Sync + 'static,
{
    Ok(Arc::new(Shaped::<S, X> {
        kernel: Kernel::<S>::build(name.to_string(), config, bridge)?,
        input,
    }))
}

fn backends(impls: &[String]) -> Vec<&'static str> {
    impls
        .iter()
        .map(|i| intern_backend(&format!("neuron_{i}")))
        .collect()
}

/// Every kernel the step uses.
struct Kernels {
    embed: Rows,
    mhc_pre: Rows,
    mhc_post: Rows,
    kda_prefill: Rows,
    kda_decode: Rows,
    dsa_prefill: Pair,
    dsa_decode: Pair,
    moe: Rows,
    dense: Rows,
    lm_head: Rows,
    ag_tp: Option<Comm>,
    rs_tp: Option<Comm>,
    ag_world: Comm,
    rs_world: Comm,
}

impl Kernels {
    fn build(
        name: &str,
        m: &Glm53FlashModelCfg,
        p: &Glm53NeuronParallel,
        bridge: &PerfApiBridge,
    ) -> Result<Self> {
        let gpu = p.gpu_name.clone();
        let n = |leaf: &str| format!("{name}.{leaf}");
        let tp = p.attn_tp;
        let bf16 = DType::Bf16;
        let wbf16 = DType::Bf16;
        let wfp8 = DType::Fp8E4m3;
        let vocab_local = m.vocab_size.div_ceil(tp);
        let kda = |b: &[String]| KdaPrefillKernelConfig {
            backends: backends(b),
            gpu_name: gpu.clone(),
            hidden_size: m.hidden,
            num_heads: m.kda_num_heads / tp,
            head_dim: m.kda_head_dim,
            conv_kernel_size: m.short_conv_kernel_size,
            dtype: bf16,
            weight_dtype: wbf16,
        };
        let kda_prefill = shaped::<KdaPrefillSpec, u32>(
            &n("kda_prefill"),
            kda(&p.impls.kda_prefill),
            |num_tokens| KdaPrefillKernelInput { num_tokens },
            bridge,
        )?;
        let c = kda(&p.impls.kda_decode);
        let kda_decode = shaped::<KdaDecodeSpec, u32>(
            &n("kda_decode"),
            KdaDecodeKernelConfig {
                backends: c.backends,
                gpu_name: c.gpu_name,
                hidden_size: c.hidden_size,
                num_heads: c.num_heads,
                head_dim: c.head_dim,
                conv_kernel_size: c.conv_kernel_size,
                dtype: c.dtype,
                weight_dtype: c.weight_dtype,
            },
            |batch_size| KdaDecodeKernelInput { batch_size },
            bridge,
        )?;
        let dsa_cfg = DsaPrefillKernelConfig {
            backends: backends(&p.impls.dsa),
            gpu_name: gpu.clone(),
            hidden_size: m.hidden,
            num_heads: m.num_attention_heads / tp,
            kv_lora_rank: m.kv_lora_rank,
            head_dim: m.qk_nope_head_dim,
            index_heads: m.index_n_heads,
            index_head_dim: m.index_head_dim,
            index_topk: m.index_topk,
            dtype: bf16,
            weight_dtype: wbf16,
        };
        let dsa_decode = shaped::<DsaDecodeSpec, (u32, u32)>(
            &n("dsa_decode"),
            DsaDecodeKernelConfig {
                backends: dsa_cfg.backends.clone(),
                gpu_name: dsa_cfg.gpu_name.clone(),
                hidden_size: dsa_cfg.hidden_size,
                num_heads: dsa_cfg.num_heads,
                kv_lora_rank: dsa_cfg.kv_lora_rank,
                head_dim: dsa_cfg.head_dim,
                index_heads: dsa_cfg.index_heads,
                index_head_dim: dsa_cfg.index_head_dim,
                index_topk: dsa_cfg.index_topk,
                dtype: bf16,
                weight_dtype: wbf16,
            },
            |(batch_size, context_len)| DsaDecodeKernelInput {
                batch_size,
                context_len,
            },
            bridge,
        )?;
        let dsa_prefill = shaped::<DsaPrefillSpec, (u32, u32)>(
            &n("dsa_prefill"),
            dsa_cfg,
            |(num_tokens, context_len)| DsaPrefillKernelInput {
                num_tokens,
                context_len,
            },
            bridge,
        )?;
        let ep = p.moe_ep;
        let moe = shaped::<MoePartialSpec, u32>(
            &n("moe_partial"),
            MoePartialKernelConfig {
                backends: backends(&p.impls.moe),
                gpu_name: gpu.clone(),
                hidden_size: m.hidden,
                num_experts: m.n_routed_experts,
                top_k: m.num_experts_per_tok,
                local_experts: m.n_routed_experts / ep,
                intermediate_size: m.moe_intermediate_size,
                shared_intermediate_size: (m.moe_intermediate_size * m.n_shared_experts)
                    .div_ceil(ep),
                dtype: bf16,
                weight_dtype: wfp8,
            },
            |num_tokens| MoePartialKernelInput { num_tokens },
            bridge,
        )?;
        let dense = shaped::<DenseFfnPartialSpec, u32>(
            &n("dense_ffn_partial"),
            DenseFfnPartialKernelConfig {
                backends: backends(&p.impls.dense_ffn),
                gpu_name: gpu.clone(),
                hidden_size: m.hidden,
                intermediate_size: m.intermediate_size.div_ceil(ep),
                dtype: bf16,
                weight_dtype: wfp8,
            },
            |num_tokens| DenseFfnPartialKernelInput { num_tokens },
            bridge,
        )?;
        let mhc_pre = shaped::<MhcPreSpec, u32>(
            &n("mhc_pre"),
            MhcPreKernelConfig {
                backends: backends(&p.impls.mhc_pre),
                gpu_name: gpu.clone(),
                hidden_size: m.hidden,
                hc_mult: m.hc_mult,
                sinkhorn_iters: SINKHORN_ITERS,
                dtype: bf16,
            },
            |num_tokens| MhcPreKernelInput { num_tokens },
            bridge,
        )?;
        let mhc_post = shaped::<MhcPostSpec, u32>(
            &n("mhc_post"),
            MhcPostKernelConfig {
                backends: backends(&p.impls.mhc_post),
                gpu_name: gpu.clone(),
                hidden_size: m.hidden,
                hc_mult: m.hc_mult,
                dtype: bf16,
            },
            |num_tokens| MhcPostKernelInput { num_tokens },
            bridge,
        )?;
        let lm_head = shaped::<LmHeadSpec, u32>(
            &n("lm_head"),
            LmHeadKernelConfig {
                backends: backends(&p.impls.lm_head),
                gpu_name: gpu.clone(),
                hidden_size: m.hidden,
                vocab_size: vocab_local,
                hc_mult: m.hc_mult,
                dtype: bf16,
            },
            |num_tokens| LmHeadKernelInput { num_tokens },
            bridge,
        )?;
        let embed = shaped::<EmbedSpec, u32>(
            &n("embed"),
            EmbedKernelConfig {
                backends: backends(&p.impls.embed),
                gpu_name: gpu.clone(),
                hidden_size: m.hidden,
                vocab_size: vocab_local,
                hc_mult: m.hc_mult,
                dtype: bf16,
            },
            |num_tokens| EmbedKernelInput { num_tokens },
            bridge,
        )?;
        let cc = vec![intern_backend("neuron_cc")];
        let ag = |label: &str, ranks: u32| -> Result<Comm> {
            Ok(shaped::<AllGatherSpec, u64>(
                &n(label),
                AllGatherKernelConfig {
                    backends: cc.clone(),
                    gpu_name: gpu.clone(),
                    num_gpus: ranks,
                    fabric: Glm53NeuronParallel::cores_fabric(ranks),
                },
                |message_size_bytes| AllGatherKernelInput { message_size_bytes },
                bridge,
            )?)
        };
        let rs = |label: &str, ranks: u32| -> Result<Comm> {
            Ok(shaped::<ReduceScatterSpec, u64>(
                &n(label),
                ReduceScatterKernelConfig {
                    backends: cc.clone(),
                    gpu_name: gpu.clone(),
                    num_gpus: ranks,
                    fabric: Glm53NeuronParallel::cores_fabric(ranks),
                },
                |message_size_bytes| ReduceScatterKernelInput { message_size_bytes },
                bridge,
            )?)
        };
        let (ag_tp, rs_tp) = if tp > 1 {
            (
                Some(ag("all_gather_tp", tp)?),
                Some(rs("reduce_scatter_tp", tp)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            embed,
            mhc_pre,
            mhc_post,
            kda_prefill,
            kda_decode,
            dsa_prefill,
            dsa_decode,
            moe,
            dense,
            lm_head,
            ag_tp,
            rs_tp,
            ag_world: ag("all_gather_world", p.world)?,
            rs_world: rs("reduce_scatter_world", p.world)?,
        })
    }
}

// ── the step's shapes ────────────────────────────────────────────────────────

/// One DP group's padded share of one forward.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GroupShape {
    /// Decode requests (padded) and rows each presents (1, or k + 1 verifying).
    decode_reqs: u32,
    query_len: u32,
    /// Prefill chunk sizes (padded).
    chunks: Vec<u32>,
    /// Context bucket the attention regions read.
    context: u32,
}

impl GroupShape {
    fn rows(&self) -> u32 {
        self.decode_reqs * self.query_len + self.chunks.iter().sum::<u32>()
    }
    /// lm_head rows: every decode row, one per prefill chunk.
    fn sample_rows(&self) -> u32 {
        self.decode_reqs * self.query_len + self.chunks.len() as u32
    }
}

/// The group shapes of one forward; empty when the forward does not run.
#[derive(Clone, Debug, Default)]
struct ForwardShape {
    groups: Vec<GroupShape>,
}

impl ForwardShape {
    fn runs(&self) -> bool {
        self.groups.iter().any(|g| g.rows() > 0)
    }
    /// Rows the world all-gather delivers to every rank.
    fn gathered(&self) -> u32 {
        self.groups.iter().map(GroupShape::rows).sum()
    }
}

/// One group's unpadded decode requests and prefill chunks.
#[derive(Clone, Debug, Default)]
struct GroupStep {
    decode_reqs: u32,
    query_len: u32,
    decode_max_kv: u32,
    /// `(prefix, append)` per chunk.
    chunks: Vec<(u32, u32)>,
}

/// The passes of one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Decode,
    MtpDecode,
    MtpDecodeDraft,
    Prefill,
    MtpPrefill,
    MtpPrefillDraft,
}

// ── leaves ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mixer {
    Kda,
    Dsa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ffn {
    Dense,
    Moe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Embed,
    EmbedRs,
    MhcPre,
    AgTp,
    Mixer(Mixer),
    RsTp,
    MhcPost,
    AgWorld,
    Ffn(Ffn),
    RsWorld,
    HeadAg,
    LmHead,
    Combine,
    Launch,
    Step,
}

/// Which region of its forward a leaf runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    Embed,
    /// A layer of layer group `g` (folded `n` times).
    Layer(usize),
    Head,
    Combine,
    None,
}

#[derive(Clone, Debug)]
struct LeafPlan {
    op: Op,
    forward: usize,
    region: Region,
    /// What-if `(min_rows, factor)` scales of this leaf's kind.
    scales: Vec<(u32, f32)>,
}

#[derive(Clone, Debug)]
struct ForwardPlan {
    view: View,
    mtp: bool,
    /// Instances of this forward per step (MTP draft passes).
    repeat: u32,
    /// Layer groups: (mixer, ffn, layers).
    layers: Vec<(Mixer, Ffn, u32)>,
}

impl ForwardPlan {
    fn layer_count(&self) -> u32 {
        self.layers.iter().map(|l| l.2).sum()
    }
}

/// The checkpoint's layers as (mixer, ffn, count) groups.
fn layer_groups(m: &Glm53FlashModelCfg) -> Vec<(Mixer, Ffn, u32)> {
    let mut groups: Vec<(Mixer, Ffn, u32)> = Vec::new();
    for l in 0..m.num_layers {
        let mixer = if m.kda_layers.contains(&l) {
            Mixer::Kda
        } else {
            Mixer::Dsa
        };
        let ffn = if l < m.first_k_dense_replace {
            Ffn::Dense
        } else {
            Ffn::Moe
        };
        match groups.iter_mut().find(|g| (g.0, g.1) == (mixer, ffn)) {
            Some(g) => g.2 += 1,
            None => groups.push((mixer, ffn, 1)),
        }
    }
    groups
}

fn op_label(op: Op) -> &'static str {
    match op {
        Op::Embed => "embed",
        Op::EmbedRs => "embed_reduce_scatter_tp",
        Op::MhcPre => "mhc_pre",
        Op::AgTp => "all_gather_tp",
        Op::Mixer(Mixer::Kda) => "kda",
        Op::Mixer(Mixer::Dsa) => "dsa",
        Op::RsTp => "reduce_scatter_tp",
        Op::MhcPost => "mhc_post",
        Op::AgWorld => "all_gather_world",
        Op::Ffn(Ffn::Moe) => "moe_partial",
        Op::Ffn(Ffn::Dense) => "dense_ffn_partial",
        Op::RsWorld => "reduce_scatter_world",
        Op::HeadAg => "head_all_gather_tp",
        Op::LmHead => "lm_head",
        Op::Combine => "combine_all_gather",
        Op::Launch => "launch",
        Op::Step => "step",
    }
}

fn mixer_name(m: Mixer) -> &'static str {
    match m {
        Mixer::Kda => "kda",
        Mixer::Dsa => "dsa",
    }
}

fn ffn_name(f: Ffn) -> &'static str {
    match f {
        Ffn::Dense => "dense",
        Ffn::Moe => "moe",
    }
}

struct PlanBuilder<'a> {
    name: String,
    b: CostTreeBuilder,
    leaves: Vec<LeafPlan>,
    forwards: Vec<ForwardPlan>,
    k: &'a Kernels,
    p: &'a Glm53NeuronParallel,
}

impl PlanBuilder<'_> {
    fn kind_config(&self, op: Op) -> (&'static str, serde_json::Value) {
        let k = self.k;
        let rows = |r: &Rows| (r.kind(), r.config());
        let comm = |c: &Option<Comm>| match c {
            Some(c) => (c.kind(), c.config()),
            None => ("all_gather", serde_json::json!({"num_gpus": 1})),
        };
        match op {
            Op::Embed => rows(&k.embed),
            Op::EmbedRs | Op::RsTp => comm(&k.rs_tp),
            Op::MhcPre => rows(&k.mhc_pre),
            Op::MhcPost => rows(&k.mhc_post),
            Op::AgTp | Op::HeadAg => comm(&k.ag_tp),
            Op::Mixer(Mixer::Kda) => rows(&k.kda_prefill),
            Op::Mixer(Mixer::Dsa) => (k.dsa_prefill.kind(), k.dsa_prefill.config()),
            Op::AgWorld | Op::Combine => (k.ag_world.kind(), k.ag_world.config()),
            Op::Ffn(Ffn::Moe) => rows(&k.moe),
            Op::Ffn(Ffn::Dense) => rows(&k.dense),
            Op::RsWorld => (k.rs_world.kind(), k.rs_world.config()),
            Op::LmHead => rows(&k.lm_head),
            Op::Launch | Op::Step => (
                "neuron_launch",
                serde_json::json!({
                    "region_host_ms": self.p.region_host_ms,
                    "region_gap_ms": self.p.region_gap_ms,
                    "forward_overhead_ms": self.p.forward_overhead_ms,
                    "step_overhead_ms": self.p.step_overhead_ms,
                }),
            ),
        }
    }

    fn leaf(&mut self, scope: &str, op: Op, forward: usize, region: Region) -> CostNode {
        let (kind, config) = self.kind_config(op);
        let kind_name = match op {
            Op::Mixer(Mixer::Kda) => "kda",
            Op::Mixer(Mixer::Dsa) => "dsa",
            _ => kind,
        };
        let scales = self
            .p
            .kind_time_scale
            .iter()
            .filter(|s| s.kind == kind || s.kind == kind_name)
            .map(|s| (s.min_rows, s.factor))
            .collect();
        self.leaves.push(LeafPlan {
            op,
            forward,
            region,
            scales,
        });
        let label = if scope.is_empty() {
            format!("{}.{}", self.name, op_label(op))
        } else {
            format!("{}.{scope}.{}", self.name, op_label(op))
        };
        self.b.leaf(label, kind, config)
    }

    fn forward(&mut self, label: &str, scope: &str, plan: ForwardPlan) -> CostNode {
        let f = self.forwards.len();
        let repeat = plan.repeat;
        let layers = plan.layers.clone();
        self.forwards.push(plan);
        let mut parts = Vec::new();
        let pro = vec![
            self.leaf(scope, Op::Embed, f, Region::Embed),
            self.leaf(scope, Op::EmbedRs, f, Region::Embed),
        ];
        parts.push(CostNode::Labeled {
            label: "prologue".into(),
            child: Box::new(CostNode::Sum(pro)),
        });
        for (gi, &(mixer, ffn, n)) in layers.iter().enumerate() {
            let s = format!(
                "{}{}{}_{}",
                scope,
                if scope.is_empty() { "" } else { "." },
                mixer_name(mixer),
                ffn_name(ffn)
            );
            let r = Region::Layer(gi);
            let ops = [
                Op::MhcPre,
                Op::AgTp,
                Op::Mixer(mixer),
                Op::RsTp,
                Op::MhcPost,
                Op::MhcPre,
                Op::AgWorld,
                Op::Ffn(ffn),
                Op::RsWorld,
                Op::MhcPost,
            ];
            let mut layer = Vec::new();
            for (i, op) in ops.into_iter().enumerate() {
                let sub = if i < 5 { "attn" } else { "ffn" };
                layer.push(self.leaf(&format!("{s}.{sub}"), op, f, r));
            }
            parts.push(CostNode::Labeled {
                label: format!("{n} x {} + {}", mixer_name(mixer), ffn_name(ffn)),
                child: Box::new(CostNode::Scale {
                    n,
                    child: Box::new(CostNode::Sum(layer)),
                }),
            });
        }
        let epi = vec![
            self.leaf(scope, Op::HeadAg, f, Region::Head),
            self.leaf(scope, Op::LmHead, f, Region::Head),
            self.leaf(scope, Op::Combine, f, Region::Combine),
        ];
        parts.push(CostNode::Labeled {
            label: "epilogue".into(),
            child: Box::new(CostNode::Sum(epi)),
        });
        parts.push(self.leaf(scope, Op::Launch, f, Region::None));
        let node = CostNode::Sum(parts);
        let node = if repeat > 1 {
            CostNode::Scale {
                n: repeat,
                child: Box::new(node),
            }
        } else {
            node
        };
        CostNode::Labeled {
            label: label.into(),
            child: Box::new(node),
        }
    }
}

fn build_plan(
    name: &str,
    m: &Glm53FlashModelCfg,
    p: &Glm53NeuronParallel,
    k: &Kernels,
) -> (CostTree, Vec<LeafPlan>, Vec<ForwardPlan>) {
    let mut pb = PlanBuilder {
        name: name.to_string(),
        b: CostTreeBuilder::new(),
        leaves: Vec::new(),
        forwards: Vec::new(),
        k,
        p,
    };
    let layers = layer_groups(m);
    let mtp_layers = vec![(Mixer::Dsa, Ffn::Moe, 1)];
    let kd = p.draft_tokens;
    let fw = |view, mtp, repeat, layers: &Vec<(Mixer, Ffn, u32)>| ForwardPlan {
        view,
        mtp,
        repeat,
        layers: layers.clone(),
    };
    let mut root = Vec::new();
    root.push(pb.forward(
        "decode forward",
        "decode",
        fw(View::Decode, false, 1, &layers),
    ));
    if kd > 0 {
        root.push(pb.forward(
            "decode mtp",
            "decode.mtp",
            fw(View::MtpDecode, true, 1, &mtp_layers),
        ));
        if kd > 1 {
            root.push(pb.forward(
                "decode mtp drafts",
                "decode.mtp_draft",
                fw(View::MtpDecodeDraft, true, kd - 1, &mtp_layers),
            ));
        }
    }
    root.push(pb.forward(
        "prefill forward",
        "prefill",
        fw(View::Prefill, false, 1, &layers),
    ));
    if kd > 0 {
        root.push(pb.forward(
            "prefill mtp",
            "prefill.mtp",
            fw(View::MtpPrefill, true, 1, &mtp_layers),
        ));
        if kd > 1 {
            root.push(pb.forward(
                "prefill mtp drafts",
                "prefill.mtp_draft",
                fw(View::MtpPrefillDraft, true, kd - 1, &mtp_layers),
            ));
        }
    }
    root.push(pb.leaf("", Op::Step, usize::MAX, Region::None));
    let tree = pb.b.finish(CostNode::Labeled {
        label: format!(
            "{name} [Trainium2 LNC2, world {} = dp{} x attn tp{}, ffn ep{}, {:?}{}]",
            p.world,
            p.dp(),
            p.attn_tp,
            p.moe_ep,
            p.region_granularity,
            if kd > 0 {
                format!(", mtp k={kd}")
            } else {
                String::new()
            }
        ),
        child: Box::new(CostNode::Sum(root)),
    });
    (tree, pb.leaves, pb.forwards)
}

// ── memory ───────────────────────────────────────────────────────────────────

/// Per-rank HBM: weights by part and the per-request / per-token pools.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemoryReport {
    pub kda_weights: u64,
    pub dsa_weights: u64,
    pub mhc_weights: u64,
    pub ffn_weights: u64,
    pub head_weights: u64,
    pub mtp_weights: u64,
    /// KDA conv + SSM state of one request on one rank (all KDA layers).
    pub kda_state_per_request: u64,
    /// MLA latent + indexer key + gate of one token on one rank (all DSA layers;
    /// replicated over the TP group).
    pub dsa_cache_per_token: u64,
    /// The engine's preallocated pools on one rank: `slots` KDA states and
    /// `slots x cmax` DSA cache rows.
    pub pools: u64,
}

impl MemoryReport {
    pub fn weights(&self) -> u64 {
        self.kda_weights
            + self.dsa_weights
            + self.mhc_weights
            + self.ffn_weights
            + self.head_weights
            + self.mtp_weights
    }
}

/// Weight bytes one rank holds: attention, mHC, embeddings and lm_head BF16;
/// dense FFN, shared and routed experts FP8 (1 byte, block scales counted).
pub fn memory_report(m: &Glm53FlashModelCfg, p: &Glm53NeuronParallel) -> MemoryReport {
    let h = u64::from(m.hidden);
    let tp = u64::from(p.attn_tp);
    let ep = u64::from(p.moe_ep);
    let kda_layers = m.kda_layers.len() as u64;
    let layers = u64::from(m.num_layers);
    let dsa_layers = layers - kda_layers;
    let dense_layers = u64::from(m.first_k_dense_replace);
    let moe_layers = layers - dense_layers;
    // KDA: q, k, v, o and the per-head f_b, g_b, b, conv sharded; f_a, g_a replicated.
    let kh = u64::from(m.kda_num_heads) * u64::from(m.kda_head_dim);
    let kda_sharded = 4 * h * kh
        + 2 * u64::from(m.kda_head_dim) * kh
        + h * u64::from(m.kda_num_heads)
        + 3 * kh * u64::from(m.short_conv_kernel_size);
    let kda_repl = 2 * h * u64::from(m.kda_head_dim);
    let kda = kda_layers * (kda_sharded / tp + kda_repl) * BF16;
    // DSA: q_b, kv_b, o sharded; q_a, kv_a, the indexer replicated.
    let heads = u64::from(m.num_attention_heads);
    let (qd, vd) = (u64::from(m.qk_nope_head_dim), u64::from(m.v_head_dim));
    let (ql, kl) = (u64::from(m.q_lora_rank), u64::from(m.kv_lora_rank));
    let dsa_sharded = ql * heads * qd + kl * heads * (qd + vd) + heads * vd * h;
    let ih = u64::from(m.index_n_heads) * u64::from(m.index_head_dim);
    let dsa_repl = h * ql
        + h * kl
        + ql * ih
        + h * u64::from(m.index_head_dim)
        + h * u64::from(m.index_n_heads);
    let dsa = dsa_layers * (dsa_sharded / tp + dsa_repl) * BF16;
    // mHC: two modules per layer, fn [hc*(2 + hc), hc*h].
    let hc = u64::from(m.hc_mult);
    let mhc = layers * 2 * (hc * (2 + hc)) * hc * h * BF16;
    // FFN over the world: 1/ep of the dense intermediate, 1/ep of the experts
    // and the shared expert; the router replicated. FP8 + 128x128 block scales.
    let fp8 = |elems: u64| elems * FP8 + elems.div_ceil(128 * 128) * FP32;
    let dense = dense_layers * fp8(3 * h * u64::from(m.intermediate_size) / ep);
    let moe_i = u64::from(m.moe_intermediate_size);
    let experts = u64::from(m.n_routed_experts) * 3 * h * moe_i / ep;
    let shared = 3 * h * moe_i * u64::from(m.n_shared_experts) / ep;
    let router = u64::from(m.n_routed_experts) * h * FP32;
    let ffn = dense + moe_layers * (fp8(experts + shared) + router);
    let head = 2 * u64::from(m.vocab_size).div_ceil(tp) * h * BF16 + h * BF16;
    let mtp = if p.draft_tokens > 0 {
        // One DSA + MoE layer, its eh_proj and norms.
        (dsa_sharded / tp + dsa_repl) * BF16
            + fp8(experts + shared)
            + router
            + 2 * h * h * BF16
            + 2 * (hc * (2 + hc)) * hc * h * BF16
    } else {
        0
    };
    let kd = u64::from(m.kda_head_dim);
    let hl = u64::from(m.kda_num_heads) / tp;
    let kda_state = kda_layers
        * (hl * kd * kd * FP32 + u64::from(m.short_conv_kernel_size - 1) * 3 * hl * kd * BF16);
    let mtp_dsa = u64::from(p.draft_tokens > 0);
    let dsa_tok = (dsa_layers + mtp_dsa) * (kl + 2 * u64::from(m.index_head_dim)) * BF16;
    let pools = u64::from(p.slots) * (kda_state + u64::from(p.cmax) * dsa_tok);
    MemoryReport {
        kda_weights: kda,
        dsa_weights: dsa,
        mhc_weights: mhc,
        ffn_weights: ffn,
        head_weights: head,
        mtp_weights: mtp,
        kda_state_per_request: kda_state,
        dsa_cache_per_token: dsa_tok,
        pools,
    }
}

/// The KV layout of one DP partition (its `attn_tp` ranks): the DSA cache
/// replicated on every rank, the KDA state sharded by head.
fn kv_layout(mem: &MemoryReport, p: &Glm53NeuronParallel, fixed_per_rank: u64) -> KvCacheLayout {
    let tp = p.attn_tp;
    KvCacheLayout::new()
        .with_part(
            KvCachePart::per_token("dsa_latent_indexer", mem.dsa_cache_per_token).replicated(tp),
        )
        .with_part(
            KvCachePart::per_request("kda_state", mem.kda_state_per_request * u64::from(tp))
                .sharded(tp)
                .not_prefix_reusable(),
        )
        .with_engine(EngineOverhead {
            block_tokens: None,
            fixed_device_bytes: fixed_per_rank * u64::from(tp),
        })
}

// ── the model ────────────────────────────────────────────────────────────────

/// The compiled step shared by the ordinary and the speculative contract.
pub struct Glm53NeuronCore {
    pub name: String,
    pub parallel: Glm53NeuronParallel,
    pub memory: MemoryReport,
    kernels: Kernels,
    leaves: Vec<LeafPlan>,
    forwards: Vec<ForwardPlan>,
    cost_flat: Vec<FlatCostNode>,
    manifest: CostManifest,
    n_slots: usize,
    kv: KvCacheLayout,
    hidden: u32,
}

impl Glm53NeuronCore {
    pub fn build(
        name: String,
        model: &Glm53FlashModelCfg,
        parallel: Glm53NeuronParallel,
        bridge: &PerfApiBridge,
    ) -> Result<Self> {
        parallel.validate(model)?;
        let memory = memory_report(model, &parallel);
        let reserve = (parallel.hbm_reserve_gb * 1e9) as u64;
        let hbm = (parallel.hbm_per_core_gb * 1e9) as u64;
        let need = memory.weights() + memory.pools + reserve;
        if need > hbm {
            bail!(
                "glm53_flash_neuron_fp8: {:.2} GB weights + {:.2} GB pools ({} slots x {} \
                 tokens) + {:.2} GB reserve per rank exceed the {:.2} GB of one LNC2 core",
                memory.weights() as f64 / 1e9,
                memory.pools as f64 / 1e9,
                parallel.slots,
                parallel.cmax,
                parallel.hbm_reserve_gb,
                parallel.hbm_per_core_gb
            );
        }
        let kernels = Kernels::build(&name, model, &parallel, bridge)?;
        let (tree, leaves, forwards) = build_plan(&name, model, &parallel, &kernels);
        let kv = kv_layout(&memory, &parallel, memory.weights() + reserve);
        tracing::info!(
            "[build] {name}: {} leaf slots; per rank {:.2} GB weights (kda {:.2}, dsa {:.2}, \
             ffn {:.2}, head {:.2}), {:.2} GB pools, KDA state {:.1} MB/request, DSA cache \
             {:.1} KB/token\n{}",
            tree.n_slots(),
            memory.weights() as f64 / 1e9,
            memory.kda_weights as f64 / 1e9,
            memory.dsa_weights as f64 / 1e9,
            memory.ffn_weights as f64 / 1e9,
            memory.head_weights as f64 / 1e9,
            memory.pools as f64 / 1e9,
            memory.kda_state_per_request as f64 / 1e6,
            memory.dsa_cache_per_token as f64 / 1e3,
            tree.describe()
        );
        Ok(Self {
            name,
            memory,
            kernels,
            leaves,
            forwards,
            cost_flat: tree.flatten(),
            manifest: tree.manifest(),
            n_slots: tree.n_slots(),
            kv,
            hidden: model.hidden,
            parallel,
        })
    }

    fn check_groups(&self, groups: usize) -> Result<(), String> {
        if groups != self.parallel.dp() as usize && groups != 1 {
            return Err(format!(
                "{}: the batch has {groups} attention-DP groups, the arch {}",
                self.name,
                self.parallel.dp()
            ));
        }
        Ok(())
    }

    fn ctx_bucket(&self, n: u32) -> u32 {
        // Past the largest compiled bucket the engine has no shape: price the
        // context itself, rounded up to 128.
        let b = bucket(n.max(1), &self.parallel.context_buckets);
        if b < n {
            n.div_ceil(128) * 128
        } else {
            b
        }
    }

    /// Pad each forward's group shapes the way the engine compiles them.
    fn shapes(&self, mut target: Vec<GroupStep>) -> Vec<ForwardShape> {
        let p = &self.parallel;
        let dp = p.dp() as usize;
        let mut spmd = p.spmd_padding;
        if target.len() == 1 && dp > 1 {
            target = spread_over_groups(&target[0], dp);
            spmd = false;
        }
        let pad_b = |b: u32| -> u32 { bucket_calls(b, &p.decode_buckets).iter().sum() };
        // Decode forward.
        let decode: Vec<GroupShape> = if spmd {
            let b = target.iter().map(|g| g.decode_reqs).max().unwrap_or(0);
            let q = target.iter().map(|g| g.query_len).max().unwrap_or(1).max(1);
            let kv = target.iter().map(|g| g.decode_max_kv).max().unwrap_or(0);
            let shape = GroupShape {
                decode_reqs: pad_b(b),
                query_len: q,
                chunks: Vec::new(),
                context: self.ctx_bucket(kv),
            };
            vec![shape; dp]
        } else {
            target
                .iter()
                .map(|g| GroupShape {
                    decode_reqs: pad_b(g.decode_reqs),
                    query_len: g.query_len.max(1),
                    chunks: Vec::new(),
                    context: self.ctx_bucket(g.decode_max_kv),
                })
                .collect()
        };
        // Prefill forward: chunk buckets, context of the furthest start + T.
        let pad_chunks = |chunks: &[(u32, u32)]| -> (Vec<u32>, u32) {
            let mut out = Vec::new();
            let mut end = 0;
            for &(prefix, append) in chunks {
                let calls = bucket_calls(append, &p.prefill_buckets);
                end = end.max(prefix + calls.iter().sum::<u32>());
                out.extend(calls);
            }
            (out, end)
        };
        let prefill: Vec<GroupShape> = if spmd {
            let n = target.iter().map(|g| g.chunks.len()).max().unwrap_or(0);
            let mut chunks = Vec::new();
            let mut end = 0;
            for i in 0..n {
                let (t, prefix) = target
                    .iter()
                    .filter_map(|g| {
                        let mut c = g.chunks.clone();
                        c.sort_by(|a, b| b.1.cmp(&a.1));
                        c.get(i).copied()
                    })
                    .fold((0, 0), |(t, pr), c| (t.max(c.1), pr.max(c.0)));
                let (padded, e) = pad_chunks(&[(prefix, t)]);
                chunks.extend(padded);
                end = end.max(e);
            }
            // Every group's own ends set the context bucket (the furthest).
            for g in &target {
                end = end.max(pad_chunks(&g.chunks).1);
            }
            let shape = GroupShape {
                decode_reqs: 0,
                query_len: 1,
                chunks,
                context: self.ctx_bucket(end),
            };
            vec![shape; dp]
        } else {
            target
                .iter()
                .map(|g| {
                    let (chunks, end) = pad_chunks(&g.chunks);
                    GroupShape {
                        decode_reqs: 0,
                        query_len: 1,
                        context: self.ctx_bucket(end),
                        chunks,
                    }
                })
                .collect()
        };
        let as_rows = |src: &[GroupShape], q1: bool| -> Vec<GroupShape> {
            src.iter()
                .map(|g| GroupShape {
                    decode_reqs: g.decode_reqs,
                    query_len: if q1 { 1 } else { g.query_len },
                    chunks: Vec::new(),
                    context: g.context,
                })
                .collect()
        };
        let any_prefill = prefill.iter().any(|g| !g.chunks.is_empty());
        let prefill_draft: Vec<GroupShape> = prefill
            .iter()
            .map(|g| GroupShape {
                decode_reqs: if !g.chunks.is_empty() || (spmd && any_prefill) {
                    p.attn_tp
                } else {
                    0
                },
                query_len: 1,
                chunks: Vec::new(),
                context: g.context,
            })
            .collect();
        // View order: Decode, MtpDecode, MtpDecodeDraft, Prefill, MtpPrefill, MtpPrefillDraft.
        vec![
            ForwardShape {
                groups: decode.clone(),
            },
            ForwardShape {
                groups: decode.clone(),
            },
            ForwardShape {
                groups: as_rows(&decode, true),
            },
            ForwardShape {
                groups: prefill.clone(),
            },
            ForwardShape { groups: prefill },
            ForwardShape {
                groups: prefill_draft,
            },
        ]
    }

    fn view_index(v: View) -> usize {
        match v {
            View::Decode => 0,
            View::MtpDecode => 1,
            View::MtpDecodeDraft => 2,
            View::Prefill => 3,
            View::MtpPrefill => 4,
            View::MtpPrefillDraft => 5,
        }
    }

    /// One leaf at one forward shape: the slowest group's per-group leaves,
    /// the common world leaves; time already scaled.
    fn eval_op(
        &self,
        plan: &LeafPlan,
        shape: &ForwardShape,
        record: bool,
    ) -> (LeafMetrics, Option<SlotInput>) {
        let k = &self.kernels;
        let tp = self.parallel.attn_tp;
        let row_bytes = u64::from(self.hidden) * BF16;
        let factor = self.parallel.region_device_factor;
        let scale = |rows: u32| -> f32 {
            plan.scales
                .iter()
                .filter(|(min_rows, _)| rows >= *min_rows)
                .map(|(_, f)| f)
                .product::<f32>()
                * factor
        };
        let mut input = None;
        let mut note = |i: SlotInput| {
            if record && input.is_none() {
                input = Some(i);
            }
        };
        // Max over groups of a per-group evaluation.
        let mut worst = LeafMetrics::ZERO;
        let mut take = |m: LeafMetrics| {
            if m.m.time_ms > worst.m.time_ms || worst.m.time_ms == 0.0 && m.m.time_ms > 0.0 {
                worst = m;
            }
        };
        let rows_eval = |r: &Rows, n: u32, note: &mut dyn FnMut(SlotInput)| -> LeafMetrics {
            note(r.input_at(n));
            if n == 0 {
                return LeafMetrics::ZERO;
            }
            let mut m = r.eval_at(n);
            m.m.time_ms *= scale(n);
            m
        };
        let comm_eval = |c: &Option<Comm>, bytes: u64, note: &mut dyn FnMut(SlotInput)| match c {
            Some(c) => {
                note(c.input_at(bytes));
                if bytes == 0 {
                    LeafMetrics::ZERO
                } else {
                    c.eval_at(bytes)
                }
            }
            None => LeafMetrics::ZERO,
        };
        let gathered = shape.gathered();
        match plan.op {
            Op::Embed => {
                for g in &shape.groups {
                    take(rows_eval(&k.embed, g.rows(), &mut note));
                }
            }
            Op::EmbedRs | Op::RsTp => {
                for g in &shape.groups {
                    take(comm_eval(
                        &k.rs_tp,
                        u64::from(g.rows()) * row_bytes,
                        &mut note,
                    ));
                }
            }
            Op::MhcPre | Op::MhcPost => {
                let r = if plan.op == Op::MhcPre {
                    &k.mhc_pre
                } else {
                    &k.mhc_post
                };
                for g in &shape.groups {
                    take(rows_eval(r, g.rows().div_ceil(tp), &mut note));
                }
            }
            Op::AgTp | Op::HeadAg => {
                for g in &shape.groups {
                    let own = u64::from(g.rows().div_ceil(tp));
                    take(comm_eval(&k.ag_tp, own * row_bytes, &mut note));
                }
            }
            Op::Mixer(mixer) => {
                for g in &shape.groups {
                    let mut m = LeafMetrics::ZERO;
                    let decode_rows = g.decode_reqs * g.query_len;
                    if decode_rows > 0 {
                        let x = match mixer {
                            Mixer::Kda => rows_eval(&k.kda_decode, decode_rows, &mut note),
                            Mixer::Dsa => {
                                let at = (decode_rows, g.context);
                                note(k.dsa_decode.input_at(at));
                                let mut x = k.dsa_decode.eval_at(at);
                                x.m.time_ms *= scale(decode_rows);
                                x
                            }
                        };
                        m.add(x);
                    }
                    for &t in &g.chunks {
                        let x = match mixer {
                            Mixer::Kda => rows_eval(&k.kda_prefill, t, &mut note),
                            Mixer::Dsa => {
                                let at = (t, g.context.max(t));
                                note(k.dsa_prefill.input_at(at));
                                let mut x = k.dsa_prefill.eval_at(at);
                                x.m.time_ms *= scale(t);
                                x
                            }
                        };
                        m.add(x);
                    }
                    take(m);
                }
            }
            Op::AgWorld => {
                let own = shape
                    .groups
                    .iter()
                    .map(|g| g.rows().div_ceil(tp))
                    .max()
                    .unwrap_or(0);
                take(comm_eval(
                    &Some(k.ag_world.clone()),
                    u64::from(own) * row_bytes,
                    &mut note,
                ));
            }
            Op::Ffn(ffn) => {
                let r = match ffn {
                    Ffn::Moe => &k.moe,
                    Ffn::Dense => &k.dense,
                };
                take(rows_eval(r, gathered, &mut note));
            }
            Op::RsWorld => {
                take(comm_eval(
                    &Some(k.rs_world.clone()),
                    u64::from(gathered) * row_bytes,
                    &mut note,
                ));
            }
            Op::LmHead => {
                for g in &shape.groups {
                    take(rows_eval(&k.lm_head, g.sample_rows(), &mut note));
                }
            }
            Op::Combine => {
                // (max value, id) pairs gathered over the TP group, then the
                // group's ids over the world.
                for g in &shape.groups {
                    let n = u64::from(g.sample_rows());
                    let mut m = comm_eval(&k.ag_tp, n * 8, &mut note);
                    m.add(comm_eval(&Some(k.ag_world.clone()), n * 4, &mut note));
                    take(m);
                }
            }
            Op::Launch | Op::Step => unreachable!("priced per forward"),
        }
        (worst, input)
    }

    /// The launch leaf of forward `f` per instance: host and launch time
    /// beyond the sum of its leaves (`regions`: device ms per region kind and
    /// the instances of each).
    fn forward_launch(&self, f: usize, regions: &[(f32, u32)]) -> (f32, u32) {
        let p = &self.parallel;
        let fw = &self.forwards[f];
        let h = p.region_host_ms;
        let gap = p.region_gap_ms;
        let layer_regions = match p.region_granularity {
            RegionGranularity::Layer => fw.layer_count(),
            RegionGranularity::Block => fw.layer_count().div_ceil(p.block_layers),
        };
        let n_regions = layer_regions + 3; // embed, head, combine
        let device: f32 = regions.iter().map(|&(d, n)| d * n as f32).sum();
        let floored = match p.region_granularity {
            RegionGranularity::Layer => regions
                .iter()
                .map(|&(d, n)| n as f32 * (d + gap).max(h))
                .sum::<f32>(),
            RegionGranularity::Block => (device + n_regions as f32 * gap).max(n_regions as f32 * h),
        };
        let overhead = if fw.mtp {
            p.mtp_forward_overhead_ms
        } else {
            p.forward_overhead_ms
        };
        let host = n_regions as f32 * h + overhead;
        ((floored - device) + (host - floored).max(0.0), n_regions)
    }

    fn eval(
        &self,
        target: Vec<GroupStep>,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        mut inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let record = inputs.is_some();
        if let Some(v) = inputs.as_deref_mut() {
            v.clear();
        }
        let shapes = self.shapes(target);
        let empty = ForwardShape::default();
        // Per forward: device per instance of each region (embed, layer groups, head, combine).
        let mut region_dev: Vec<Vec<f32>> = self
            .forwards
            .iter()
            .map(|f| vec![0.0; f.layers.len() + 3])
            .collect();
        let mut any = false;
        for (slot, plan) in self.leaves.iter().enumerate() {
            match plan.op {
                Op::Launch => {
                    let f = plan.forward;
                    let fw = &self.forwards[f];
                    let runs = shapes[Self::view_index(fw.view)].runs();
                    let mut regions: Vec<(f32, u32)> = Vec::new();
                    let rd = &region_dev[f];
                    regions.push((rd[0], 1));
                    for (gi, l) in fw.layers.iter().enumerate() {
                        regions.push((rd[1 + gi], l.2));
                    }
                    regions.push((rd[fw.layers.len() + 1], 1));
                    regions.push((rd[fw.layers.len() + 2], 1));
                    let (t, n) = if runs {
                        any = true;
                        self.forward_launch(f, &regions)
                    } else {
                        (0.0, 0)
                    };
                    slots[slot].m.time_ms = t;
                    if let Some(v) = inputs.as_deref_mut() {
                        v.push(SlotInput::from(NeuronLaunchLog {
                            regions: n,
                            forward: f as u32,
                        }));
                    }
                }
                Op::Step => {
                    if let Some(v) = inputs.as_deref_mut() {
                        v.push(SlotInput::from(NeuronLaunchLog {
                            regions: 0,
                            forward: u32::MAX,
                        }));
                    }
                }
                _ => {
                    let fw = &self.forwards[plan.forward];
                    let shape = &shapes[Self::view_index(fw.view)];
                    let shape = if shape.runs() { shape } else { &empty };
                    let (m, input) = self.eval_op(plan, shape, record);
                    let ri = match plan.region {
                        Region::Embed => 0,
                        Region::Layer(g) => 1 + g,
                        Region::Head => fw.layers.len() + 1,
                        Region::Combine => fw.layers.len() + 2,
                        Region::None => unreachable!("compute leaves have a region"),
                    };
                    region_dev[plan.forward][ri] += m.m.time_ms;
                    slots[slot] = m;
                    if let Some(v) = inputs.as_deref_mut() {
                        v.push(input.unwrap_or_else(|| {
                            SlotInput::from(NeuronLaunchLog {
                                regions: 0,
                                forward: plan.forward as u32,
                            })
                        }));
                    }
                }
            }
        }
        if let Some(slot) = self.leaves.iter().position(|l| l.op == Op::Step) {
            slots[slot].m.time_ms = if any {
                self.parallel.step_overhead_ms
            } else {
                0.0
            };
        }
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

/// A worker without DP partitions hands one group: deal its requests over the
/// DP groups, prefill chunks largest first.
fn spread_over_groups(one: &GroupStep, dp: usize) -> Vec<GroupStep> {
    let mut groups = vec![
        GroupStep {
            query_len: one.query_len,
            decode_max_kv: one.decode_max_kv,
            ..Default::default()
        };
        dp
    ];
    for i in 0..one.decode_reqs as usize {
        groups[i % dp].decode_reqs += 1;
    }
    let mut chunks = one.chunks.clone();
    chunks.sort_by(|a, b| b.1.cmp(&a.1));
    for (i, c) in chunks.into_iter().enumerate() {
        groups[i % dp].chunks.push(c);
    }
    for g in &mut groups {
        if g.decode_reqs == 0 {
            g.decode_max_kv = 0;
        }
    }
    groups
}

fn unified_targets(batch: &UnifiedArchInput) -> Vec<GroupStep> {
    batch
        .groups
        .iter()
        .map(|g| GroupStep {
            decode_reqs: g.decode_kv_lens.len() as u32,
            query_len: 1,
            decode_max_kv: g.decode_kv_lens.iter().copied().max().unwrap_or(0),
            chunks: g.prefill_chunk_pairs.clone(),
        })
        .collect()
}

fn speculative_targets(batch: &SpeculativeArchInput) -> Vec<GroupStep> {
    batch
        .groups
        .iter()
        .map(|g| GroupStep {
            decode_reqs: g.decode_requests.len() as u32,
            query_len: g
                .decode_requests
                .iter()
                .map(|r| r.query_len)
                .max()
                .unwrap_or(1)
                .max(1),
            decode_max_kv: g
                .decode_requests
                .iter()
                .map(|r| r.kv_len)
                .max()
                .unwrap_or(0),
            chunks: g.prefill_chunk_pairs.clone(),
        })
        .collect()
}

/// GLM-5.3-Flash FP8 on Trainium2 without MTP.
pub struct Glm53FlashNeuronFp8Model {
    pub core: Glm53NeuronCore,
}

impl IterwiseUnifiedModel for Glm53FlashNeuronFp8Model {
    fn logs_decode_kv_lens(&self) -> bool {
        true
    }

    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        self.core.check_groups(batch.groups.len())
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.core.eval(unified_targets(batch), slots, scratch, None)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.core
            .eval(unified_targets(batch), slots, scratch, Some(inputs))
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.core.manifest.clone()
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.core.kv.clone()
    }

    fn gpus_per_replica(&self) -> u16 {
        self.core.parallel.world as u16
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.core.parallel.dp() as u16
    }
}

/// GLM-5.3-Flash FP8 on Trainium2 with an MTP drafter (placeholder: the seed
/// engine has none yet; acceptance comes from the trace's `accept_rate`).
pub struct Glm53FlashNeuronFp8SpeculativeModel {
    pub core: Glm53NeuronCore,
}

impl SpeculativeUnifiedModel for Glm53FlashNeuronFp8SpeculativeModel {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        if batch.draft_tokens != self.core.parallel.draft_tokens {
            return Err(format!(
                "{}: built for {} draft tokens, the batch drafts {}",
                self.core.name, self.core.parallel.draft_tokens, batch.draft_tokens
            ));
        }
        self.core.check_groups(batch.groups.len())
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.core
            .eval(speculative_targets(batch), slots, scratch, None)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.core
            .eval(speculative_targets(batch), slots, scratch, Some(inputs))
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.core.manifest.clone()
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.core.kv.clone()
    }

    fn max_model_len(&self) -> u32 {
        self.core.parallel.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        self.core.parallel.world as u16
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.core.parallel.dp() as u16
    }
}

/// Per-draft acceptance `p` whose chain acceptance length `1 + p + ... + p^k`
/// equals the golden `GOLDEN_ACCEPTANCE_LENGTH[k - 1]` (bisection).
pub fn golden_accept_rate(k: u32) -> f64 {
    assert!((1..=5).contains(&k), "golden acceptance covers k = 1..5");
    let target = GOLDEN_ACCEPTANCE_LENGTH[k as usize - 1];
    let al = |p: f64| (0..=k).map(|i| p.powi(i as i32)).sum::<f64>();
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if al(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::{
        ArchGroupInput, SpeculativeArchGroupInput, SpeculativeDecodeInput,
    };
    use std::path::Path;

    fn model() -> Glm53FlashModelCfg {
        Glm53FlashModelCfg::from_json(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/glm53_flash.json"),
        )
        .unwrap()
    }

    /// A probe whose time encodes its shape: rows (ms) for row kinds,
    /// rows + context / 1000 for DSA, bytes / 1e6 for collectives.
    struct FakeRows;
    impl ShapeProbe<u32> for FakeRows {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn config(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        fn eval_at(&self, n: u32) -> LeafMetrics {
            let mut m = LeafMetrics::ZERO;
            m.m.time_ms = n as f32;
            m
        }
        fn input_at(&self, num_tokens: u32) -> SlotInput {
            SlotInput::from(MhcPreKernelInput { num_tokens })
        }
    }
    struct FakePair;
    impl ShapeProbe<(u32, u32)> for FakePair {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn config(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        fn eval_at(&self, (n, c): (u32, u32)) -> LeafMetrics {
            let mut m = LeafMetrics::ZERO;
            m.m.time_ms = n as f32 + c as f32 / 1000.0;
            m
        }
        fn input_at(&self, (batch_size, context_len): (u32, u32)) -> SlotInput {
            SlotInput::from(DsaDecodeKernelInput {
                batch_size,
                context_len,
            })
        }
    }
    struct FakeComm;
    impl ShapeProbe<u64> for FakeComm {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn config(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        fn eval_at(&self, b: u64) -> LeafMetrics {
            let mut m = LeafMetrics::ZERO;
            m.m.time_ms = b as f32 / 1e6;
            m
        }
        fn input_at(&self, message_size_bytes: u64) -> SlotInput {
            SlotInput::from(AllGatherKernelInput { message_size_bytes })
        }
    }

    fn fakes(tp: u32) -> Kernels {
        let r = || -> Rows { Arc::new(FakeRows) };
        let p = || -> Pair { Arc::new(FakePair) };
        let c = || -> Comm { Arc::new(FakeComm) };
        Kernels {
            embed: r(),
            mhc_pre: r(),
            mhc_post: r(),
            kda_prefill: r(),
            kda_decode: r(),
            dsa_prefill: p(),
            dsa_decode: p(),
            moe: r(),
            dense: r(),
            lm_head: r(),
            ag_tp: (tp > 1).then(c),
            rs_tp: (tp > 1).then(c),
            ag_world: c(),
            rs_world: c(),
        }
    }

    fn parallel(tp: u32) -> Glm53NeuronParallel {
        let mut p = Glm53NeuronParallel::seed("AWS Trainium2 LNC2");
        p.attn_tp = tp;
        p.decode_buckets = vec![8, 16, 32];
        p.prefill_buckets = vec![512, 2048];
        // Shape-only tests: the host leaves are zero unless a test sets them.
        p.region_host_ms = 0.0;
        p.region_gap_ms = 0.0;
        p.forward_overhead_ms = 0.0;
        p.step_overhead_ms = 0.0;
        p
    }

    fn built(p: Glm53NeuronParallel) -> Glm53NeuronCore {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let tp = p.attn_tp;
        let mut core = Glm53NeuronCore::build("unified".into(), &model(), p, &bridge).unwrap();
        core.kernels = fakes(tp);
        core
    }

    fn decode(b: u32, kv: u32) -> ArchGroupInput {
        ArchGroupInput {
            batch_tokens: b,
            decode_tokens: b,
            decode_kv_lens: vec![kv; b as usize],
            total_kv_len: kv * b,
            ..Default::default()
        }
    }

    fn prefill(prefix: u32, append: u32) -> ArchGroupInput {
        ArchGroupInput {
            batch_tokens: append,
            prefill_tokens: append,
            prefill_chunk_pairs: vec![(prefix, append)],
            ..Default::default()
        }
    }

    /// (slot name, ms) of one evaluation, and the step total.
    fn eval(core: &Glm53NeuronCore, groups: Vec<ArchGroupInput>) -> (Vec<(String, f32)>, f32) {
        let batch = UnifiedArchInput {
            groups,
            tokens_per_source_rank: Vec::new(),
        };
        let (mut slots, mut scratch, mut inputs) = (Vec::new(), Vec::new(), Vec::new());
        let total = core.eval(
            unified_targets(&batch),
            &mut slots,
            &mut scratch,
            Some(&mut inputs),
        );
        assert_eq!(inputs.len(), core.n_slots, "one recorded input per slot");
        let named = core
            .manifest
            .slots
            .iter()
            .zip(&slots)
            .map(|(s, m)| (s.name.clone(), m.m.time_ms))
            .collect();
        (named, total.m.time_ms)
    }

    fn ms(named: &[(String, f32)], name: &str) -> f32 {
        named
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no slot {name}"))
            .1
    }

    /// Catches a layer schedule that drifts from the checkpoint: layers 0-2
    /// KDA + dense FFN, then 31 KDA and 11 DSA layers with MoE.
    #[test]
    fn layer_groups_follow_the_checkpoint() {
        assert_eq!(
            layer_groups(&model()),
            [
                (Mixer::Kda, Ffn::Dense, 3),
                (Mixer::Dsa, Ffn::Moe, 11),
                (Mixer::Kda, Ffn::Moe, 31)
            ]
        );
    }

    /// Catches SPMD padding and the sequence-parallel row split going wrong
    /// in a decode step: groups of 2 / 0 / 0 / 5 requests all run the bucket
    /// of the largest (8); the mixer sees the group's 8 rows, mHC its rank's
    /// 8 / tp, the FFN the world's 4 x 8 rows; DSA reads the context bucket
    /// of the longest KV (5000 -> 8192).
    #[test]
    fn decode_pads_every_group_to_the_largest_batch() {
        let core = built(parallel(8));
        let (n, _) = eval(
            &core,
            vec![decode(2, 3000), decode(0, 0), decode(0, 0), decode(5, 5000)],
        );
        assert_eq!(ms(&n, "unified.decode.kda_moe.attn.kda"), 8.0);
        assert_eq!(ms(&n, "unified.decode.kda_moe.attn.mhc_pre"), 1.0);
        assert_eq!(ms(&n, "unified.decode.dsa_moe.attn.dsa"), 8.0 + 8.192);
        assert_eq!(ms(&n, "unified.decode.kda_moe.ffn.moe_partial"), 32.0);
        assert_eq!(
            ms(&n, "unified.decode.kda_dense.ffn.dense_ffn_partial"),
            32.0
        );
        let row = 4096.0 * 2.0 / 1e6;
        assert_eq!(
            ms(&n, "unified.decode.kda_moe.ffn.reduce_scatter_world"),
            32.0 * row
        );
        assert_eq!(
            ms(&n, "unified.decode.kda_moe.ffn.all_gather_world"),
            1.0 * row
        );
        assert_eq!(
            ms(&n, "unified.decode.kda_moe.attn.reduce_scatter_tp"),
            8.0 * row
        );
        assert_eq!(ms(&n, "unified.decode.lm_head"), 8.0);
        // No prefill chunk: the prefill forward does not run.
        assert_eq!(ms(&n, "unified.prefill.kda_moe.ffn.moe_partial"), 0.0);
    }

    /// Catches the prefill forward losing the engine's shape: one chunk of
    /// 1500 tokens at prefix 2048 in one group runs every group at the 2048
    /// bucket, the context bucket of 2048 + 2048 (4096), the FFN over 4 x 2048
    /// rows, and the head over one row per group.
    #[test]
    fn prefill_runs_the_longest_chunk_bucket_in_every_group() {
        let core = built(parallel(8));
        let (n, _) = eval(
            &core,
            vec![
                prefill(2048, 1500),
                decode(0, 0),
                decode(0, 0),
                decode(0, 0),
            ],
        );
        assert_eq!(ms(&n, "unified.prefill.kda_moe.attn.kda"), 2048.0);
        assert_eq!(ms(&n, "unified.prefill.dsa_moe.attn.dsa"), 2048.0 + 4.096);
        assert_eq!(ms(&n, "unified.prefill.kda_moe.attn.mhc_post"), 256.0);
        assert_eq!(ms(&n, "unified.prefill.kda_moe.ffn.moe_partial"), 8192.0);
        assert_eq!(ms(&n, "unified.prefill.lm_head"), 1.0);
        assert_eq!(ms(&n, "unified.decode.kda_moe.attn.kda"), 0.0);
    }

    /// Catches the what-if `spmd_padding: false` still padding idle groups:
    /// each group pays its own bucket and the FFN sees the sum of them.
    #[test]
    fn without_spmd_each_group_pays_its_own_shape() {
        let mut p = parallel(8);
        p.spmd_padding = false;
        let core = built(p);
        let (n, _) = eval(
            &core,
            vec![decode(2, 100), decode(0, 0), decode(0, 0), decode(20, 100)],
        );
        // The slowest group (20 -> 32) sets the per-group leaves.
        assert_eq!(ms(&n, "unified.decode.kda_moe.attn.kda"), 32.0);
        assert_eq!(ms(&n, "unified.decode.kda_moe.ffn.moe_partial"), 40.0);
    }

    /// Catches the pipelined launch model drifting from the trn2-flashnext
    /// fit: a forward costs max(sum_r max(device_r + gap, host), regions x
    /// host + overhead); its launch leaf is the excess over the leaves.
    #[test]
    fn launch_leaf_is_the_host_excess_over_device() {
        let mut p = parallel(8);
        p.region_host_ms = 100.0;
        p.forward_overhead_ms = 10.0;
        p.step_overhead_ms = 2.5;
        let core = built(p);
        let (n, total) = eval(&core, vec![decode(8, 100); 4]);
        let device: f32 = n
            .iter()
            .filter(|(name, _)| name.starts_with("unified.decode.") && !name.ends_with("launch"))
            .map(|(name, t)| {
                let fold = if name.contains("kda_dense") {
                    3.0
                } else if name.contains("kda_moe") {
                    31.0
                } else if name.contains("dsa_moe") {
                    11.0
                } else {
                    1.0
                };
                t * fold
            })
            .sum();
        // 45 layer regions + embed + head + combine, each under the 100 ms
        // dispatch: the host sets the forward.
        let host = 48.0 * 100.0 + 10.0;
        assert!((ms(&n, "unified.decode.launch") - (host - device)).abs() < 1e-2);
        assert!((total - (host + 2.5)).abs() < 1e-2, "{total}");
        // Block regions of 15 layers: 3 + 3 regions, each 100 ms of dispatch;
        // the same device work now exceeds the host's 600 ms.
        let mut p = parallel(8);
        p.region_granularity = RegionGranularity::Block;
        p.block_layers = 15;
        p.region_host_ms = 100.0;
        let core = built(p);
        let (_, total) = eval(&core, vec![decode(8, 100); 4]);
        assert!(device > 600.0);
        assert!((total - device).abs() < 1e-2, "{total} vs {device}");
    }

    /// Catches an idle step being charged host time.
    #[test]
    fn an_empty_step_costs_nothing() {
        let mut p = parallel(8);
        p.step_overhead_ms = 2.5;
        p.forward_overhead_ms = 19.0;
        let core = built(p);
        let (_, total) = eval(&core, vec![decode(0, 0); 4]);
        assert_eq!(total, 0.0);
    }

    /// Catches the HBM check and the pool sizes drifting: TP8 holds ~11.7 GB
    /// of FP8/BF16 weights per rank; a DSA token is 11 x (512 + 128 + 128)
    /// bf16 values on every rank of the group; a request's KDA state is 34
    /// layers of 8 heads x 128 x 128 fp32 (+ conv) at TP8. TP1 with 64
    /// requests per group does not fit a core.
    #[test]
    fn memory_report_and_hbm_check() {
        let m = model();
        let r = memory_report(&m, &parallel(8));
        assert_eq!(r.dsa_cache_per_token, 11 * 768 * 2);
        assert_eq!(
            r.kda_state_per_request,
            34 * (8 * 128 * 128 * 4 + 3 * 3 * 8 * 128 * 2)
        );
        let gb = r.weights() as f64 / 1e9;
        assert!((10.5..13.0).contains(&gb), "TP8 weights {gb:.2} GB/rank");
        let mut p = parallel(1);
        p.slots = 65;
        p.decode_buckets = vec![64];
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let err = Glm53NeuronCore::build("u".into(), &m, p, &bridge)
            .err()
            .expect("TP1 x 64 slots must not fit");
        assert!(err.to_string().contains("exceed"), "{err}");
    }

    /// Catches layouts the engine cannot run.
    #[test]
    fn layouts_are_validated() {
        let m = model();
        let mut p = parallel(16);
        assert!(p.validate(&m).is_err(), "decode bucket 8 < tp 16");
        p.decode_buckets = vec![16, 32];
        p.prefill_buckets = vec![2048];
        p.validate(&m).unwrap();
        let mut p = parallel(8);
        p.attn_tp = 3;
        assert!(p.validate(&m).is_err());
        let mut p = parallel(8);
        p.moe_ep = 16;
        assert!(p.validate(&m).is_err(), "the FFN spans the world");
    }

    /// Catches the MTP placeholder's passes: verify rows B x (k + 1) in the
    /// decode forward, the MTP pass over them, k - 1 draft passes of B rows.
    #[test]
    fn speculative_adds_mtp_passes() {
        let mut p = parallel(8);
        p.draft_tokens = 3;
        let core = built(p);
        let names: Vec<_> = core.manifest.slots.iter().map(|s| s.name.clone()).collect();
        assert!(names
            .iter()
            .any(|n| n == "unified.decode.mtp.dsa_moe.attn.dsa"));
        assert!(names
            .iter()
            .any(|n| n == "unified.decode.mtp_draft.lm_head"));
        let batch = SpeculativeArchInput {
            draft_tokens: 3,
            groups: vec![
                SpeculativeArchGroupInput {
                    batch_tokens: 32,
                    decode_tokens: 32,
                    decode_requests: vec![
                        SpeculativeDecodeInput {
                            kv_len: 1000,
                            query_len: 4,
                        };
                        8
                    ],
                    ..Default::default()
                };
                4
            ],
            tokens_per_source_rank: Vec::new(),
        };
        let (mut slots, mut scratch) = (Vec::new(), Vec::new());
        core.eval(speculative_targets(&batch), &mut slots, &mut scratch, None);
        let at = |name: &str| {
            slots[names.iter().position(|n| n == name).unwrap()]
                .m
                .time_ms
        };
        assert_eq!(at("unified.decode.kda_moe.attn.kda"), 32.0);
        assert_eq!(at("unified.decode.mtp.dsa_moe.attn.mhc_pre"), 4.0);
        assert_eq!(at("unified.decode.mtp_draft.dsa_moe.attn.mhc_pre"), 1.0);
    }

    /// Catches the golden acceptance inversion: k = 1 accepts 0.86 per draft.
    #[test]
    fn golden_accept_rate_matches_the_acceptance_length() {
        assert!((golden_accept_rate(1) - 0.86).abs() < 1e-9);
        for k in 1..=5 {
            let p = golden_accept_rate(k);
            let al: f64 = (0..=k).map(|i| p.powi(i as i32)).sum();
            assert!((al - GOLDEN_ACCEPTANCE_LENGTH[k as usize - 1]).abs() < 1e-9);
        }
    }

    #[test]
    fn kind_time_scale_parses() {
        let s = KindTimeScale::parse("moe_partial@256=0.5").unwrap();
        assert_eq!(
            (s.kind.as_str(), s.min_rows, s.factor),
            ("moe_partial", 256, 0.5)
        );
        assert!(KindTimeScale::parse("moe_partial").is_err());
        assert!(KindTimeScale::parse("x=0").is_err());
    }
}
