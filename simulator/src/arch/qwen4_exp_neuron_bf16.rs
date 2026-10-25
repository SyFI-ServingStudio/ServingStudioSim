//! Qwen4Exp (Qwen3.8-Flash-Next, BF16) on AWS Trainium2 logical cores (LNC2),
//! served by the flashnext-neuron engine: static-shape compiled regions per
//! bucket, one process per logical core, collectives traced into the regions.
//! Every leaf is a measured L1 row of that engine's own op modules
//! (`doc/pr/trn2-flashnext/L1_CONTRACT.md`).
//!
//! ```text
//! replica      attn_dp x G ranks, G = max(attn_tp, gdn_tp); one KV partition per
//!              attention-DP group. QSA runs G/attn_tp sub-replicas of attn_tp
//!              ranks, GDN G/gdn_tp sub-replicas of gdn_tp ranks, each serving
//!              a share of the group's requests. MoE spans every rank
//!              (moe_ep x moe_tp = attn_dp x G).
//! prologue     embed (vocab-parallel over G) + all-reduce
//! per layer    attn HC module, GDN or QSA mixer (+ TP collective), mlp HC
//!              module, MoE block over every DP group's tokens (+ dispatch
//!              collectives); PLE block before layer 1's attention module
//! epilogue     final HC combine + mixer, lm_head + greedy argmax (vocab-parallel)
//! mtp          (speculative) the MTP layer, k passes: the first over every
//!              target row, the rest over one row per decode request
//! launch       one leaf: the host launch and inter-region gap of every region
//!              the step launches (`region_launch_overhead`)
//! ```
//!
//! Shapes are bucketed: a prefill chunk runs at the next `prefill_buckets`
//! size, a decode batch at the next `decode_buckets` size, a token-wise
//! region (HC, MoE, PLE, embed) at the next size of their union, and QSA
//! context at the next `context_buckets` size; padding costs its kernel time.
//! A mixer runs once per prefill chunk (one request chunk per region) and once
//! for the decode batch.
//!
//! Region granularity (`region_granularity`):
//! - `per_op`: every op is its own region (op-level L1 leaves).
//! - `mixer`: the engine's fused regions (whole GDN mixer, the two-region QSA
//!   mixer, the fused MoE block, the fused HC combine+mix and PLE post).
//! - `layer`: the mixer-level leaves inside one region per layer (one per
//!   mixer call).
//!
//! Composition (the engine-slice calibration, flashnext-neuron
//! `engine/README.md` "Composition rule for the simulator"): every compute
//! leaf is scaled by the granularity's device factor f_g (op 1.10, mixer 0.94,
//! layer 0.88: fused regions are cheaper than their isolated ops). With
//! `launch_model = calibrated` (default) each region costs
//! `max(f_g * device, L_r)`, `L_r` = `launch_coll_ms` for a region holding a
//! collective and `launch_plain_ms` otherwise, and the step adds
//! `step_overhead_ms` (input H2D, next-id readback, collective skew). With
//! `l1_additive` the measured `region_launch_overhead` row prices the step's
//! region count instead. The wall constants were fitted on a loaded host and
//! are parameters.
//!
//! Not modeled: the host scheduler and sampling bookkeeping, KV block-table
//! upload, the vision tower.

use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use serde::Deserialize;

use crate::arch::contract::{
    IterwiseUnifiedModel, SpeculativeArchInput, SpeculativeUnifiedModel, UnifiedArchInput,
};
use crate::arch::kv_layout::{EngineOverhead, KvCacheLayout, KvCachePart};
use crate::arch::qwen4_exp_model_cfg::{Qwen4ExpMixer, Qwen4ExpModelCfg};
use crate::common::Fabric;
use crate::timing::bridge::payload::intern_backend;
use crate::timing::bridge::DType;
use crate::timing::kernels::engine::{Kernel, KernelSpec};
use crate::timing::kernels::*;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, FlatCostNode, LeafMetrics,
    PerfApiBridge, Probe, SlotInput,
};

const BF16_BYTES: u64 = 2;
const FP32_BYTES: u64 = 4;
/// One greedy-argmax result per row and rank: an fp32 value and an int32 index.
const ARGMAX_PAIR_BYTES: u64 = 8;

// ── knobs ────────────────────────────────────────────────────────────────────

/// How finely the engine cuts a step into compiled regions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionGranularity {
    PerOp,
    #[default]
    Mixer,
    Layer,
}

impl RegionGranularity {
    pub const CHOICES: [&'static str; 3] = ["per_op", "mixer", "layer"];

    /// The engine-slice device factor f_g: step device time over the sum of
    /// the isolated L1 rows of its ops (`logs/engine/calib_summary.md`).
    pub fn calibrated_device_factor(self) -> f32 {
        match self {
            Self::PerOp => 1.10,
            Self::Mixer => 0.94,
            Self::Layer => 0.88,
        }
    }
}

/// The collective after a tensor-parallel mixer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TpCollective {
    /// All-reduce the row-parallel output; every rank keeps every token.
    #[default]
    AllReduce,
    /// Reduce-scatter along tokens; the token-wise regions (HC, MoE dispatch)
    /// run sequence-parallel on 1/G of the tokens, and an all-gather brings
    /// the tokens back before the next mixer.
    ReduceScatterAllGather,
}

impl TpCollective {
    pub const CHOICES: [&'static str; 2] = ["all_reduce", "reduce_scatter_all_gather"];
}

/// How tokens reach the experts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeDispatch {
    /// Every rank sees every token of every DP group (all-gather before, the
    /// fused MoE block over the local experts, reduce-scatter after).
    #[default]
    AllGather,
    /// Each rank sends its tokens' routed rows to the experts' ranks and back
    /// (all-to-all); the shared expert runs where the token lives.
    AllToAll,
}

impl MoeDispatch {
    pub const CHOICES: [&'static str; 2] = ["all_gather", "all_to_all"];
}

/// How host launch cost enters the step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaunchModel {
    /// Per region `max(f_g * device, L_r)` plus a per-step constant (the
    /// engine-slice fit).
    #[default]
    Calibrated,
    /// The measured `region_launch_overhead` row at the step's region count,
    /// added to the device time.
    L1Additive,
}

impl LaunchModel {
    pub const CHOICES: [&'static str; 2] = ["calibrated", "l1_additive"];
}

/// Where the PLE n-gram table lives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PleTableShards {
    /// Row-sharded over every rank of the replica.
    #[default]
    Replica,
    /// Row-sharded over the ranks of one attention-DP group (one copy per group).
    DpGroup,
    /// Host memory. Unsupported: a compiled region cannot read a host tensor
    /// (flashnext-neuron `ops/ple.md`).
    Host,
}

impl PleTableShards {
    pub const CHOICES: [&'static str; 3] = ["replica", "dp_group", "host"];
}

/// Per-op implementation choice: the op module `IMPLS` keys each role may
/// use. More than one key lets the kernel pick the fastest measured row per
/// shape (best-of-N), as an engine that picks its implementation per bucket
/// would.
#[derive(Clone, Debug, PartialEq)]
pub struct NeuronFlashNextImpls {
    pub gdn_prefill: Vec<String>,
    pub gdn_decode: Vec<String>,
    pub qsa_prefill: Vec<String>,
    pub qsa_decode: Vec<String>,
    pub qsa_select: Vec<String>,
    pub qsa_scores: Vec<String>,
    pub moe_routed: Vec<String>,
    pub moe_router: Vec<String>,
    pub moe_shared: Vec<String>,
    pub hc_mix: Vec<String>,
    pub head: Vec<String>,
}

impl Default for NeuronFlashNextImpls {
    fn default() -> Self {
        let v = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        Self {
            gdn_prefill: v(&["nki"]),
            gdn_decode: v(&["torch"]),
            qsa_prefill: v(&["torch_sparse"]),
            qsa_decode: v(&["torch_sparse"]),
            qsa_select: v(&["nki"]),
            qsa_scores: v(&["torch"]),
            moe_routed: v(&["auto"]),
            moe_router: v(&["torch_topk_first"]),
            moe_shared: v(&["torch"]),
            hc_mix: v(&["torch"]),
            head: v(&["torch"]),
        }
    }
}

/// The deployment layout and engine knobs of one replica.
#[derive(Clone, Debug)]
pub struct NeuronFlashNextParallel {
    pub gpu_name: String,
    pub max_model_len: u32,
    /// Logical NeuronCore config: 2 = two physical cores per rank (4 ranks per
    /// Trainium2 chip). Every profiled row is LNC2.
    pub lnc: u32,
    pub attn_dp: u32,
    pub attn_tp: u32,
    pub gdn_tp: u32,
    pub moe_ep: u32,
    pub moe_tp: u32,
    /// MTP drafts per step; 0 builds no MTP layer.
    pub draft_tokens: u32,
    pub tp_collective: TpCollective,
    pub moe_dispatch: MoeDispatch,
    pub region_granularity: RegionGranularity,
    /// Device factor applied to every compute leaf (f_g).
    pub region_device_factor: f32,
    pub launch_model: LaunchModel,
    pub launch_coll_ms: f32,
    pub launch_plain_ms: f32,
    pub step_overhead_ms: f32,
    pub prefill_buckets: Vec<u32>,
    pub decode_buckets: Vec<u32>,
    pub context_buckets: Vec<u32>,
    pub ple_table_shards: PleTableShards,
    /// Per-core HBM the build checks the weights against (GB).
    pub hbm_per_core_gb: f64,
    /// Per-core runtime and activation scratch kept out of the KV budget (GB).
    pub hbm_reserve_gb: f64,
    pub impls: NeuronFlashNextImpls,
}

impl NeuronFlashNextParallel {
    /// Ranks in one attention-DP group.
    pub fn group_size(&self) -> u32 {
        self.attn_tp.max(self.gdn_tp)
    }

    pub fn ranks(&self) -> u32 {
        self.attn_dp * self.group_size()
    }

    /// Logical cores per chip at this LNC (Trainium2: 8 physical cores).
    pub fn cores_per_chip(&self) -> u32 {
        8 / self.lnc
    }

    fn sequence_parallel(&self) -> bool {
        self.tp_collective == TpCollective::ReduceScatterAllGather
    }

    pub fn validate(&self, model: &Qwen4ExpModelCfg) -> Result<()> {
        ensure!(
            self.lnc == 2,
            "qwen4_exp_neuron_bf16: only LNC2 rows are profiled (lnc = {})",
            self.lnc
        );
        let pow2 = |x: u32| x > 0 && x.is_power_of_two();
        ensure!(
            pow2(self.attn_tp) && pow2(self.gdn_tp) && pow2(self.attn_dp),
            "attn_tp, gdn_tp and attn_dp must be powers of two"
        );
        ensure!(
            model.qsa_heads % self.attn_tp == 0,
            "attn_tp {} must divide the {} QSA heads (tp16 would leave idle padded heads)",
            self.attn_tp,
            model.qsa_heads
        );
        ensure!(
            model.gdn_key_heads % self.gdn_tp == 0 && model.gdn_value_heads % self.gdn_tp == 0,
            "gdn_tp {} must divide the GDN heads",
            self.gdn_tp
        );
        ensure!(
            self.moe_ep * self.moe_tp == self.ranks(),
            "moe_ep x moe_tp ({} x {}) must equal the replica's {} ranks (attn_dp x max(attn_tp, gdn_tp))",
            self.moe_ep,
            self.moe_tp,
            self.ranks()
        );
        ensure!(
            model.num_experts % self.moe_ep == 0 && model.moe_intermediate % self.moe_tp == 0,
            "moe_ep must divide the experts and moe_tp the expert intermediate size"
        );
        ensure!(
            self.moe_dispatch == MoeDispatch::AllGather || self.moe_tp == 1,
            "all-to-all dispatch is modeled for moe_tp = 1"
        );
        ensure!(
            self.ranks() <= 64,
            "a trn2.48xlarge has 64 LNC2 cores; the replica needs {}",
            self.ranks()
        );
        ensure!(
            self.ple_table_shards != PleTableShards::Host,
            "ple_table_shards = host is unsupported on Neuron: a compiled region cannot read a \
             host-resident table (flashnext-neuron ops/ple.md); shard it over replica or dp_group"
        );
        for (name, buckets) in [
            ("prefill_buckets", &self.prefill_buckets),
            ("decode_buckets", &self.decode_buckets),
            ("context_buckets", &self.context_buckets),
        ] {
            ensure!(
                !buckets.is_empty() && buckets.windows(2).all(|w| w[0] < w[1]) && buckets[0] > 0,
                "{name} must be a nonempty strictly increasing list of positive sizes"
            );
        }
        ensure!(
            self.region_device_factor > 0.0
                && self.launch_coll_ms >= 0.0
                && self.launch_plain_ms >= 0.0
                && self.step_overhead_ms >= 0.0,
            "region_device_factor must be positive and the launch constants nonnegative"
        );
        Ok(())
    }
}

// ── bucket padding ───────────────────────────────────────────────────────────

/// Pad `n` to the next bucket; above the largest bucket, the work runs as
/// several largest-bucket calls plus one padded remainder. Pushes each call's
/// padded size into `out`.
fn bucket_calls(n: u32, buckets: &[u32], out: &mut Vec<u32>) {
    if n == 0 {
        return;
    }
    let max = *buckets.last().expect("validated nonempty");
    let mut rest = n;
    while rest > max {
        out.push(max);
        rest -= max;
    }
    out.push(*buckets.iter().find(|&&b| b >= rest).unwrap_or(&max));
}

fn bucket(n: u32, buckets: &[u32]) -> u32 {
    buckets.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

// ── probes: one evaluation shape per leaf family ─────────────────────────────

/// A built kernel evaluated at one argument shape `X`.
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

// ── the step, seen per DP group ──────────────────────────────────────────────

/// One attention-DP group's share of a step, in one view (target pass, MTP
/// first pass, or an MTP recurrent pass).
#[derive(Clone, Debug, Default)]
struct GroupStep {
    /// Decode requests and the rows each presents (1, or k + 1 when verifying).
    decode_reqs: u32,
    query_len: u32,
    /// Largest KV length among the decode requests.
    decode_max_kv: u32,
    /// `(prefix, append)` per prefill chunk.
    chunks: Vec<(u32, u32)>,
    /// Requests in the group (decode + prefill).
    requests: u32,
}

impl GroupStep {
    fn decode_rows(&self) -> u32 {
        self.decode_reqs * self.query_len
    }
    fn tokens(&self) -> u32 {
        self.decode_rows() + self.chunks.iter().map(|c| c.1).sum::<u32>()
    }
    /// lm_head rows: every verify row, one row per prefill chunk.
    fn sample_rows(&self) -> u32 {
        self.decode_rows() + self.chunks.len() as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Target,
    MtpFirst,
    MtpRecurrent,
}

struct StepViews {
    target: Vec<GroupStep>,
    recurrent: Vec<GroupStep>,
}

impl StepViews {
    fn get(&self, view: View) -> &[GroupStep] {
        match view {
            View::Target | View::MtpFirst => &self.target,
            View::MtpRecurrent => &self.recurrent,
        }
    }
}

// ── leaf plans ───────────────────────────────────────────────────────────────

/// Which rows a token-wise leaf or a collective sees.
#[derive(Clone, Copy, Debug)]
enum RowSrc {
    /// The group's tokens, padded to the token buckets.
    Group(usize),
    /// Every group's padded tokens.
    All,
    /// The largest group's padded tokens.
    MaxGroup,
    /// The group's lm_head rows, padded to the token buckets.
    Sample(usize),
    /// Every group's padded tokens over `moe_ep`: the routed rows one expert
    /// rank receives under all-to-all dispatch, as an equivalent token count
    /// (each with `top_k` assignments).
    MoeEquivalent,
}

#[derive(Clone)]
enum LeafEval {
    /// A token-wise op at `rows / div` rows.
    Tokens { k: Rows, src: RowSrc, div: u32 },
    /// The decode batch at `ceil(B / sub)` requests (`rows`: times query_len).
    Decode {
        k: Rows,
        d: usize,
        sub: u32,
        rows: bool,
    },
    /// One call per prefill chunk, chunks spread over `sub` sub-replicas.
    Chunks { k: Rows, d: usize, sub: u32 },
    /// QSA decode at (rows, context bucket).
    QsaDecode { k: Pair, d: usize, sub: u32 },
    /// QSA prefill, one call per chunk at (chunk bucket, context bucket).
    QsaChunks { k: Pair, d: usize, sub: u32 },
    /// PLE n-gram ids: (tokens, requests).
    Ngram { k: Pair, d: usize },
    /// PLE short conv over the decode batch: (requests, query_len).
    ConvDecode { k: Pair, d: usize },
    /// PLE short conv, one request chunk per call: (1, chunk bucket).
    ConvChunks { k: Pair, d: usize },
    /// A collective of `rows x bytes_per_row / div` bytes per rank.
    Comm {
        k: Comm,
        src: RowSrc,
        bytes_per_row: u64,
        div: u32,
    },
    /// The step's region launches.
    Launch { k: Rows },
}

#[derive(Clone)]
struct LeafPlan {
    eval: LeafEval,
    view: View,
    /// Compiled regions per call this leaf stands for.
    regions_per_call: u32,
    /// Product of the enclosing `Scale` folds (for the region count).
    fold: u32,
    /// A compute leaf: scaled by f_g and, per region, floored by its launch.
    compute: bool,
    /// The leaf's region traces a collective (launch `launch_coll_ms`).
    coll: bool,
}

impl LeafEval {
    fn group(&self) -> Option<usize> {
        match self {
            LeafEval::Tokens { src, .. } | LeafEval::Comm { src, .. } => match src {
                RowSrc::Group(d) | RowSrc::Sample(d) => Some(*d),
                _ => None,
            },
            LeafEval::Decode { d, .. }
            | LeafEval::Chunks { d, .. }
            | LeafEval::QsaDecode { d, .. }
            | LeafEval::QsaChunks { d, .. }
            | LeafEval::Ngram { d, .. }
            | LeafEval::ConvDecode { d, .. }
            | LeafEval::ConvChunks { d, .. } => Some(*d),
            LeafEval::Launch { .. } => None,
        }
    }
}

// ── kernels ──────────────────────────────────────────────────────────────────

/// `neuron_<impl>` backend names, interned for the kernel configs.
fn backends(impls: &[String]) -> Vec<&'static str> {
    impls
        .iter()
        .map(|i| intern_backend(&format!("neuron_{i}")))
        .collect()
}

fn torch() -> Vec<&'static str> {
    vec!["neuron_torch"]
}

/// Whether a collective of `n` ranks spaced `stride` apart crosses chips, and
/// how many of its ranks share each chip (`Fabric::NeuronC{1,2,4}`).
fn neuron_fabric(n: u32, stride: u32, cores_per_chip: u32) -> Fabric {
    let per_chip = if stride >= cores_per_chip {
        1
    } else {
        n.min(cores_per_chip / stride.max(1))
    };
    match per_chip {
        0 | 1 => Fabric::NeuronC1,
        2 | 3 => Fabric::NeuronC2,
        _ => Fabric::NeuronC4,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CommOp {
    AllReduce,
    AllGather,
    ReduceScatter,
    AllToAll,
}

impl CommOp {
    fn label(self) -> &'static str {
        match self {
            CommOp::AllReduce => "all_reduce",
            CommOp::AllGather => "all_gather",
            CommOp::ReduceScatter => "reduce_scatter",
            CommOp::AllToAll => "all_to_all",
        }
    }
}

/// Every kernel the step uses, built once and shared by the per-group leaves.
struct Kernels {
    // Token-wise.
    embed: Rows,
    hc_mix: Rows,
    hc_combine: Rows,
    hc_combine_mix: Option<Rows>,
    hc_mix_final: Rows,
    // GDN, fused.
    gdn_prefill_mixer: Option<Rows>,
    gdn_decode_mixer: Option<Rows>,
    // GDN, per op.
    gdn_ops: Option<GdnOps>,
    // QSA.
    qsa_mixer2_prefill: Option<Pair>,
    qsa_mixer2_decode: Option<Pair>,
    qsa_ops_prefill: Option<QsaOps>,
    qsa_ops_decode: Option<QsaOps>,
    // MoE.
    moe_block: Option<Rows>,
    moe_router: Option<Rows>,
    moe_routed: Option<Rows>,
    moe_shared: Option<Rows>,
    // PLE.
    ple_ngram: Option<Pair>,
    ple_gather: Option<Rows>,
    ple_post: Option<Rows>,
    ple_project: Option<Rows>,
    ple_gate: Option<Rows>,
    ple_residual: Option<Rows>,
    ple_conv_decode: Option<Pair>,
    ple_conv_prefill: Option<Pair>,
    // Head.
    lm_head_argmax: Option<Rows>,
    lm_head_gemm: Option<Rows>,
    vocab_argmax: Option<Rows>,
    // MTP.
    mtp_fc: Option<Rows>,
    launch: Rows,
    comm: std::collections::HashMap<(CommOp, u32, Fabric), Comm>,
}

struct GdnOps {
    in_proj: Rows,
    conv_prefill: Rows,
    split_gating: Rows,
    prefill_core: Rows,
    output: Rows,
    conv_decode: Rows,
    decode_core: Rows,
    state_io: Rows,
}

#[derive(Clone)]
struct QsaOps {
    project: Rows,
    indexer_project: Rows,
    cache_write: Rows,
    indexer_compress: Rows,
    indexer_scores: Pair,
    indexer_select: Pair,
    attention: Pair,
    output: Rows,
}

/// Per-rank geometry of every op, derived from the checkpoint and the layout.
#[derive(Clone, Debug)]
struct Geometry {
    hidden: u32,
    gdn_k_heads: u32,
    gdn_v_heads: u32,
    gdn_head_dim: u32,
    gdn_conv: u32,
    qsa_q_heads: u32,
    qsa_kv_heads: u32,
    qsa_head_dim: u32,
    index_heads: u32,
    index_head_dim: u32,
    index_topk_tokens: u32,
    compress_ratio: u32,
    num_experts: u32,
    top_k: u32,
    local_experts: u32,
    moe_intermediate: u32,
    shared_intermediate_split: u32,
    shared_intermediate_full: u32,
    hc_mult: u32,
    hc_lowrank: u32,
    ple_heads: u32,
    ple_head_dim: u32,
    ple_table_rows_local: u64,
    ple_conv: u32,
    ple_dilation: u32,
    vocab_local: u32,
}

impl Geometry {
    fn new(m: &Qwen4ExpModelCfg, p: &NeuronFlashNextParallel) -> Self {
        let g = p.group_size();
        let ple_heads = (m.ngram_size - 1) * m.heads_per_ngram;
        let ple_shards = match p.ple_table_shards {
            PleTableShards::Replica => p.ranks(),
            PleTableShards::DpGroup | PleTableShards::Host => g,
        };
        Self {
            hidden: m.hidden,
            gdn_k_heads: m.gdn_key_heads / p.gdn_tp,
            gdn_v_heads: m.gdn_value_heads / p.gdn_tp,
            gdn_head_dim: m.gdn_key_head_dim,
            gdn_conv: m.gdn_conv_kernel,
            qsa_q_heads: m.qsa_heads / p.attn_tp,
            // Each rank holds the KV head(s) its Q heads read (`qsa_rank_layout`).
            qsa_kv_heads: (m.qsa_kv_heads / p.attn_tp).max(1),
            qsa_head_dim: m.qsa_head_dim,
            index_heads: m.indexer_heads,
            index_head_dim: m.indexer_head_dim,
            index_topk_tokens: m.indexer_token_topk,
            compress_ratio: m.indexer_compress_ratio,
            num_experts: m.num_experts,
            top_k: m.experts_per_token,
            local_experts: m.num_experts / p.moe_ep,
            moe_intermediate: m.moe_intermediate / p.moe_tp,
            shared_intermediate_split: (m.shared_expert_intermediate / p.ranks()).max(1),
            shared_intermediate_full: m.shared_expert_intermediate,
            hc_mult: m.hc_count,
            hc_lowrank: m.hc_lowrank,
            ple_heads,
            ple_head_dim: m.ple_embed_dim / ple_heads,
            ple_table_rows_local: m.ple_table_rows.div_ceil(u64::from(ple_shards)),
            ple_conv: m.ple_conv_kernel,
            ple_dilation: m.ple_conv_dilation(),
            vocab_local: m.vocab_size / g,
        }
    }
}

fn bf16() -> DType {
    DType::Bf16
}

impl Kernels {
    #[allow(clippy::too_many_lines)]
    fn build(
        name: &str,
        model: &Qwen4ExpModelCfg,
        p: &NeuronFlashNextParallel,
        bridge: &PerfApiBridge,
    ) -> Result<Self> {
        let gpu = p.gpu_name.clone();
        let g = Geometry::new(model, p);
        let per_op = p.region_granularity == RegionGranularity::PerOp;
        let spec = p.draft_tokens > 0;
        let query_len = p.draft_tokens + 1;
        let has_ple = !model.ple_layers.is_empty();
        let n = |leaf: &str| format!("{name}.{leaf}");
        let gdn_cfg = |backends: Vec<&'static str>| GdnCommon {
            backends,
            gpu_name: gpu.clone(),
            hidden_size: g.hidden,
            num_key_heads: g.gdn_k_heads,
            num_value_heads: g.gdn_v_heads,
            head_dim: g.gdn_head_dim,
            conv_kernel_size: g.gdn_conv,
            dtype: bf16(),
        };
        let qsa_cfg = |mode: &str, backends: Vec<&'static str>| QsaCommon {
            backends,
            gpu_name: gpu.clone(),
            mode: mode.to_string(),
            hidden_size: g.hidden,
            num_q_heads: g.qsa_q_heads,
            num_kv_heads: g.qsa_kv_heads,
            head_dim: g.qsa_head_dim,
            index_heads: g.index_heads,
            index_head_dim: g.index_head_dim,
            index_topk_tokens: g.index_topk_tokens,
            compress_ratio: g.compress_ratio,
            dtype: bf16(),
        };
        let hc = |backends: Vec<&'static str>| {
            (backends, gpu.clone(), g.hidden, g.hc_mult, g.hc_lowrank)
        };

        let embed = shaped::<EmbeddingRowGatherSpec, u32>(
            &n("embed"),
            EmbeddingRowGatherKernelConfig {
                backends: torch(),
                gpu_name: gpu.clone(),
                num_heads: 1,
                head_dim: g.hidden,
                table_rows: u64::from(g.vocab_local),
                residency: "device".to_string(),
                weight_dtype: bf16(),
            },
            |num_tokens| EmbeddingRowGatherKernelInput { num_tokens },
            bridge,
        )?;
        let (hb, hg, hh, hm, hr) = hc(backends(&p.impls.hc_mix));
        let hc_mix = shaped::<HcMixSpec, u32>(
            &n("hc_mix"),
            HcMixKernelConfig {
                backends: hb.clone(),
                gpu_name: hg.clone(),
                hidden_size: hh,
                hc_mult: hm,
                lowrank: hr,
                dtype: bf16(),
            },
            |num_tokens| HcMixKernelInput { num_tokens },
            bridge,
        )?;
        let hc_combine = shaped::<HcCombineSpec, u32>(
            &n("hc_combine"),
            MhcRmsNormKernelConfig {
                backends: torch(),
                gpu_name: gpu.clone(),
                hidden_size: Dim::lit(g.hidden),
                hc_mult: g.hc_mult,
                hidden_dtype: bf16(),
            },
            |num_tokens| MhcRmsNormKernelInput { num_tokens },
            bridge,
        )?;
        let hc_combine_mix = if per_op {
            None
        } else {
            Some(shaped::<HcCombineMixSpec, u32>(
                &n("hc_combine_mix"),
                HcCombineMixKernelConfig {
                    backends: hb.clone(),
                    gpu_name: hg.clone(),
                    hidden_size: hh,
                    hc_mult: hm,
                    lowrank: hr,
                    dtype: bf16(),
                },
                |num_tokens| HcCombineMixKernelInput { num_tokens },
                bridge,
            )?)
        };
        let hc_mix_final = shaped::<HcMixFinalSpec, u32>(
            &n("hc_mix_final"),
            HcMixFinalKernelConfig {
                backends: torch(),
                gpu_name: hg.clone(),
                hidden_size: hh,
                hc_mult: hm,
                lowrank: hr,
                dtype: bf16(),
            },
            |num_tokens| HcMixFinalKernelInput { num_tokens },
            bridge,
        )?;

        // GDN.
        let (gdn_prefill_mixer, gdn_decode_mixer, gdn_ops) = if per_op {
            let c = |b| gdn_cfg(b);
            let ops = GdnOps {
                in_proj: shaped::<GdnInProjSpec, u32>(
                    &n("gdn.in_proj"),
                    c(torch()).into_in_proj(),
                    |num_tokens| GdnInProjKernelInput { num_tokens },
                    bridge,
                )?,
                conv_prefill: shaped::<GdnConv1dPrefillSpec, u32>(
                    &n("gdn.conv1d_prefill"),
                    c(backends(&p.impls.gdn_prefill)).into_conv1d_prefill(),
                    |num_tokens| GdnConv1dPrefillKernelInput { num_tokens },
                    bridge,
                )?,
                split_gating: shaped::<GdnSplitGatingSpec, u32>(
                    &n("gdn.split_gating"),
                    c(torch()).into_split_gating(),
                    |num_tokens| GdnSplitGatingKernelInput { num_tokens },
                    bridge,
                )?,
                prefill_core: shaped::<GdnPrefillCoreSpec, u32>(
                    &n("gdn.prefill_core"),
                    c(backends(&p.impls.gdn_prefill)).into_prefill_core(),
                    |num_tokens| GdnPrefillCoreKernelInput { num_tokens },
                    bridge,
                )?,
                output: shaped::<GdnOutputSpec, u32>(
                    &n("gdn.output"),
                    c(torch()).into_output(),
                    |num_tokens| GdnOutputKernelInput { num_tokens },
                    bridge,
                )?,
                conv_decode: shaped::<GdnConv1dDecodeSpec, u32>(
                    &n("gdn.conv1d_decode"),
                    c(backends(&p.impls.gdn_decode))
                        .into_conv1d_decode(query_len_or_one(spec, query_len)),
                    |batch_size| GdnConv1dDecodeKernelInput { batch_size },
                    bridge,
                )?,
                decode_core: shaped::<GdnDecodeCoreSpec, u32>(
                    &n("gdn.decode_core"),
                    c(backends(&p.impls.gdn_decode))
                        .into_decode_core(query_len_or_one(spec, query_len)),
                    |batch_size| GdnDecodeCoreKernelInput { batch_size },
                    bridge,
                )?,
                state_io: shaped::<GdnStateIoSpec, u32>(
                    &n("gdn.state_io"),
                    c(torch()).into_state_io(),
                    |batch_size| GdnStateIoKernelInput { batch_size },
                    bridge,
                )?,
            };
            (None, None, Some(ops))
        } else {
            (
                Some(shaped::<GdnPrefillMixerSpec, u32>(
                    &n("gdn.prefill_mixer"),
                    gdn_cfg(backends(&p.impls.gdn_prefill)).into_prefill_mixer(),
                    |num_tokens| GdnPrefillMixerKernelInput { num_tokens },
                    bridge,
                )?),
                Some(shaped::<GdnDecodeMixerSpec, u32>(
                    &n("gdn.decode_mixer"),
                    gdn_cfg(backends(&p.impls.gdn_decode))
                        .into_decode_mixer(query_len_or_one(spec, query_len)),
                    |batch_size| GdnDecodeMixerKernelInput { batch_size },
                    bridge,
                )?),
                None,
            )
        };

        // QSA.
        let qsa_ops = |mode: &str, attn: &[String]| -> Result<QsaOps> {
            let tag = |op: &str| n(&format!("qsa.{mode}.{op}"));
            Ok(QsaOps {
                project: shaped::<QsaProjectSpec, u32>(
                    &tag("project"),
                    qsa_cfg(mode, torch()).into_project(),
                    |num_rows| QsaProjectKernelInput { num_rows },
                    bridge,
                )?,
                indexer_project: shaped::<QsaIndexerProjectSpec, u32>(
                    &tag("indexer_project"),
                    qsa_cfg(mode, torch()).into_indexer_project(),
                    |num_rows| QsaIndexerProjectKernelInput { num_rows },
                    bridge,
                )?,
                cache_write: shaped::<QsaCacheWriteSpec, u32>(
                    &tag("cache_write"),
                    qsa_cfg(mode, torch()).into_cache_write(),
                    |num_rows| QsaCacheWriteKernelInput { num_rows },
                    bridge,
                )?,
                indexer_compress: shaped::<QsaIndexerCompressSpec, u32>(
                    &tag("indexer_compress"),
                    qsa_cfg(mode, torch()).into_indexer_compress(),
                    |num_rows| QsaIndexerCompressKernelInput { num_rows },
                    bridge,
                )?,
                indexer_scores: shaped::<QsaIndexerScoresSpec, (u32, u32)>(
                    &tag("indexer_scores"),
                    qsa_cfg(mode, backends(&p.impls.qsa_scores)).into_indexer_scores(),
                    |(num_rows, context_len)| QsaIndexerScoresKernelInput {
                        num_rows,
                        context_len,
                    },
                    bridge,
                )?,
                indexer_select: shaped::<QsaIndexerSelectSpec, (u32, u32)>(
                    &tag("indexer_select"),
                    qsa_cfg(mode, backends(&p.impls.qsa_select)).into_indexer_select(),
                    |(num_rows, context_len)| QsaIndexerSelectKernelInput {
                        num_rows,
                        context_len,
                    },
                    bridge,
                )?,
                attention: shaped::<QsaAttentionSpec, (u32, u32)>(
                    &tag("attention"),
                    qsa_cfg(mode, backends(attn)).into_attention(),
                    |(num_rows, context_len)| QsaAttentionKernelInput {
                        num_rows,
                        context_len,
                    },
                    bridge,
                )?,
                output: shaped::<QsaOutputSpec, u32>(
                    &tag("output"),
                    qsa_cfg(mode, torch()).into_output(),
                    |num_rows| QsaOutputKernelInput { num_rows },
                    bridge,
                )?,
            })
        };
        let mixer2 = |mode: &str, impls: &[String]| {
            shaped::<QsaMixer2Spec, (u32, u32)>(
                &n(&format!("qsa.{mode}.mixer2")),
                qsa_cfg(mode, backends(impls)).into_mixer2(),
                |(num_rows, context_len)| QsaMixer2KernelInput {
                    num_rows,
                    context_len,
                },
                bridge,
            )
        };
        let (qsa_mixer2_prefill, qsa_mixer2_decode, qsa_ops_prefill, qsa_ops_decode) = if per_op {
            (
                None,
                None,
                Some(qsa_ops("prefill", &p.impls.qsa_prefill)?),
                Some(qsa_ops("decode", &p.impls.qsa_decode)?),
            )
        } else {
            (
                Some(mixer2("prefill", &p.impls.qsa_prefill)?),
                Some(mixer2("decode", &p.impls.qsa_decode)?),
                None,
                None,
            )
        };

        // MoE.
        let routed_cfg = |name_: &str| -> Result<Rows> {
            Ok(shaped::<MoeRoutedExpertsSpec, u32>(
                &n(name_),
                MoeRoutedExpertsKernelConfig {
                    backends: backends(&p.impls.moe_routed),
                    gpu_name: gpu.clone(),
                    hidden_size: g.hidden,
                    num_experts: g.num_experts,
                    top_k: g.top_k,
                    local_experts: g.local_experts,
                    intermediate_size: g.moe_intermediate,
                    dtype: bf16(),
                },
                |num_tokens| MoeRoutedExpertsKernelInput { num_tokens },
                bridge,
            )?)
        };
        let router = || -> Result<Rows> {
            Ok(shaped::<MoeRouterSpec, u32>(
                &n("moe.router"),
                MoeRouterKernelConfig {
                    backends: backends(&p.impls.moe_router),
                    gpu_name: gpu.clone(),
                    hidden_size: g.hidden,
                    num_experts: g.num_experts,
                    top_k: g.top_k,
                    dtype: bf16(),
                },
                |num_tokens| MoeRouterKernelInput { num_tokens },
                bridge,
            )?)
        };
        let shared = |inter: u32| -> Result<Rows> {
            Ok(shaped::<MoeSharedExpertSpec, u32>(
                &n("moe.shared_expert"),
                MoeSharedExpertKernelConfig {
                    backends: backends(&p.impls.moe_shared),
                    gpu_name: gpu.clone(),
                    hidden_size: g.hidden,
                    intermediate_size: inter,
                    dtype: bf16(),
                },
                |num_tokens| MoeSharedExpertKernelInput { num_tokens },
                bridge,
            )?)
        };
        let (moe_block, moe_router, moe_routed, moe_shared) = match (p.moe_dispatch, per_op) {
            (MoeDispatch::AllGather, false) => (
                Some(shaped::<MoeBlockSpec, u32>(
                    &n("moe.block"),
                    MoeBlockKernelConfig {
                        backends: backends(&p.impls.moe_routed),
                        gpu_name: gpu.clone(),
                        hidden_size: g.hidden,
                        num_experts: g.num_experts,
                        top_k: g.top_k,
                        local_experts: g.local_experts,
                        intermediate_size: g.moe_intermediate,
                        shared_intermediate_size: g.shared_intermediate_split,
                        dtype: bf16(),
                    },
                    |num_tokens| MoeBlockKernelInput { num_tokens },
                    bridge,
                )?),
                None,
                None,
                None,
            ),
            (MoeDispatch::AllGather, true) => (
                None,
                Some(router()?),
                Some(routed_cfg("moe.routed_experts")?),
                Some(shared(g.shared_intermediate_split)?),
            ),
            (MoeDispatch::AllToAll, _) => (
                None,
                Some(router()?),
                Some(routed_cfg("moe.routed_experts")?),
                Some(shared(g.shared_intermediate_full)?),
            ),
        };

        // PLE.
        let ple_cfg_hidden = g.hidden;
        let (
            ple_ngram,
            ple_gather,
            ple_post,
            ple_project,
            ple_gate,
            ple_residual,
            ple_conv_decode,
            ple_conv_prefill,
        ) = if has_ple {
            let ngram = shaped::<PleNgramIdsSpec, (u32, u32)>(
                &n("ple.ngram_ids"),
                PleNgramIdsKernelConfig {
                    backends: torch(),
                    gpu_name: gpu.clone(),
                    ngram_context_len: model.ngram_size - 1,
                    heads_per_ngram: model.heads_per_ngram,
                },
                |(num_tokens, num_reqs)| PleNgramIdsKernelInput {
                    num_tokens,
                    num_reqs,
                },
                bridge,
            )?;
            let gather = shaped::<EmbeddingRowGatherSpec, u32>(
                &n("ple.gather"),
                EmbeddingRowGatherKernelConfig {
                    backends: torch(),
                    gpu_name: gpu.clone(),
                    num_heads: g.ple_heads,
                    head_dim: g.ple_head_dim,
                    table_rows: g.ple_table_rows_local,
                    residency: "device".to_string(),
                    weight_dtype: model.ple_table_dtype,
                },
                |num_tokens| EmbeddingRowGatherKernelInput { num_tokens },
                bridge,
            )?;
            if per_op {
                let conv = |mode: &str, label: &str| {
                    shaped::<PleShortConvSpec, (u32, u32)>(
                        &n(label),
                        PleShortConvKernelConfig {
                            backends: torch(),
                            gpu_name: gpu.clone(),
                            channels: ple_cfg_hidden * g.hc_mult,
                            kernel_size: g.ple_conv,
                            dilation: g.ple_dilation,
                            mode: mode.to_string(),
                            dtype: bf16(),
                        },
                        |(batch_size, query_len)| PleShortConvKernelInput {
                            batch_size,
                            query_len,
                        },
                        bridge,
                    )
                };
                (
                    Some(ngram),
                    Some(gather),
                    None,
                    Some(shaped::<PleProjectSpec, u32>(
                        &n("ple.project"),
                        PleProjectKernelConfig {
                            backends: torch(),
                            gpu_name: gpu.clone(),
                            embed_dim: g.ple_heads * g.ple_head_dim,
                            hidden_size: g.hidden,
                            hc_mult: g.hc_mult,
                            dtype: bf16(),
                        },
                        |num_tokens| PleProjectKernelInput { num_tokens },
                        bridge,
                    )?),
                    Some(shaped::<PleGateSpec, u32>(
                        &n("ple.gate"),
                        PleGateKernelConfig {
                            backends: torch(),
                            gpu_name: gpu.clone(),
                            hidden_size: g.hidden,
                            hc_mult: g.hc_mult,
                            dtype: bf16(),
                        },
                        |num_tokens| PleGateKernelInput { num_tokens },
                        bridge,
                    )?),
                    Some(shaped::<PleResidualSpec, u32>(
                        &n("ple.residual"),
                        PleResidualKernelConfig {
                            backends: torch(),
                            gpu_name: gpu.clone(),
                            hidden_size: g.hidden,
                            hc_mult: g.hc_mult,
                            dtype: bf16(),
                        },
                        |num_tokens| PleResidualKernelInput { num_tokens },
                        bridge,
                    )?),
                    Some(conv(
                        if spec { "spec" } else { "decode" },
                        "ple.short_conv_decode",
                    )?),
                    Some(conv("prefill", "ple.short_conv_prefill")?),
                )
            } else {
                (
                    Some(ngram),
                    Some(gather),
                    Some(shaped::<PlePostSpec, u32>(
                        &n("ple.post"),
                        PlePostKernelConfig {
                            backends: torch(),
                            gpu_name: gpu.clone(),
                            embed_dim: g.ple_heads * g.ple_head_dim,
                            hidden_size: g.hidden,
                            hc_mult: g.hc_mult,
                            conv_kernel_size: g.ple_conv,
                            dilation: g.ple_dilation,
                            dtype: bf16(),
                        },
                        |num_tokens| PlePostKernelInput { num_tokens },
                        bridge,
                    )?),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            }
        } else {
            (None, None, None, None, None, None, None, None)
        };

        // Head.
        let (lm_head_argmax, lm_head_gemm, vocab_argmax) = if per_op {
            (
                None,
                Some(shaped::<GemmFp32OutputSpec, u32>(
                    &n("head.lm_head"),
                    GemmFp32OutputKernelConfig {
                        backends: torch(),
                        gpu_name: gpu.clone(),
                        n: Dim::lit(g.vocab_local),
                        k: Dim::lit(g.hidden),
                        input_dtype: bf16(),
                    },
                    |m| GemmFp32OutputKernelInput { m },
                    bridge,
                )?),
                Some(shaped::<VocabArgmaxSpec, u32>(
                    &n("head.argmax"),
                    VocabArgmaxKernelConfig {
                        backends: backends(&p.impls.head),
                        gpu_name: gpu.clone(),
                        vocab_size: g.vocab_local,
                        dtype: bf16(),
                    },
                    |num_tokens| VocabArgmaxKernelInput { num_tokens },
                    bridge,
                )?),
            )
        } else {
            (
                Some(shaped::<LmHeadArgmaxSpec, u32>(
                    &n("head.lm_head_argmax"),
                    LmHeadArgmaxKernelConfig {
                        backends: backends(&p.impls.head),
                        gpu_name: gpu.clone(),
                        hidden_size: g.hidden,
                        vocab_size: g.vocab_local,
                        dtype: bf16(),
                    },
                    |num_tokens| LmHeadArgmaxKernelInput { num_tokens },
                    bridge,
                )?),
                None,
                None,
            )
        };
        let mtp_fc = if spec {
            Some(shaped::<SingleGemmSpec, u32>(
                &n("mtp.fc"),
                SingleGemmKernelConfig {
                    backends: torch(),
                    gpu_name: gpu.clone(),
                    n: Dim::lit(g.hidden),
                    k: Dim::lit(2 * g.hidden),
                    dtype: bf16(),
                },
                |m| SingleGemmKernelInput { m },
                bridge,
            )?)
        } else {
            None
        };
        let launch = shaped::<RegionLaunchOverheadSpec, u32>(
            &n("launch"),
            RegionLaunchOverheadKernelConfig {
                backends: torch(),
                gpu_name: gpu.clone(),
                dtype: bf16(),
            },
            |num_regions| RegionLaunchOverheadKernelInput { num_regions },
            bridge,
        )?;
        Ok(Self {
            embed,
            hc_mix,
            hc_combine,
            hc_combine_mix,
            hc_mix_final,
            gdn_prefill_mixer,
            gdn_decode_mixer,
            gdn_ops,
            qsa_mixer2_prefill,
            qsa_mixer2_decode,
            qsa_ops_prefill,
            qsa_ops_decode,
            moe_block,
            moe_router,
            moe_routed,
            moe_shared,
            ple_ngram,
            ple_gather,
            ple_post,
            ple_project,
            ple_gate,
            ple_residual,
            ple_conv_decode,
            ple_conv_prefill,
            lm_head_argmax,
            lm_head_gemm,
            vocab_argmax,
            mtp_fc,
            launch,
            comm: std::collections::HashMap::new(),
        })
    }

    fn comm(
        &mut self,
        op: CommOp,
        ranks: u32,
        fabric: Fabric,
        name: &str,
        gpu: &str,
        bridge: &PerfApiBridge,
    ) -> Result<Comm> {
        if let Some(k) = self.comm.get(&(op, ranks, fabric)) {
            return Ok(k.clone());
        }
        let label = format!("{name}.{}.n{ranks}.{}", op.label(), fabric.as_str());
        let backends = vec!["neuron_cc"];
        let k: Comm = match op {
            CommOp::AllReduce => shaped::<AllReduceSpec, u64>(
                &label,
                AllReduceKernelConfig {
                    backends,
                    gpu_name: gpu.to_string(),
                    num_gpus: ranks,
                    fabric,
                },
                |message_size_bytes| AllReduceKernelInput { message_size_bytes },
                bridge,
            )?,
            CommOp::AllGather => shaped::<AllGatherSpec, u64>(
                &label,
                AllGatherKernelConfig {
                    backends,
                    gpu_name: gpu.to_string(),
                    num_gpus: ranks,
                    fabric,
                },
                |message_size_bytes| AllGatherKernelInput { message_size_bytes },
                bridge,
            )?,
            CommOp::ReduceScatter => shaped::<ReduceScatterSpec, u64>(
                &label,
                ReduceScatterKernelConfig {
                    backends,
                    gpu_name: gpu.to_string(),
                    num_gpus: ranks,
                    fabric,
                },
                |message_size_bytes| ReduceScatterKernelInput { message_size_bytes },
                bridge,
            )?,
            CommOp::AllToAll => shaped::<AllToAllSpec, u64>(
                &label,
                AllToAllKernelConfig {
                    backends,
                    gpu_name: gpu.to_string(),
                    num_gpus: ranks,
                    fabric,
                },
                |message_size_bytes| AllToAllKernelInput { message_size_bytes },
                bridge,
            )?,
        };
        self.comm.insert((op, ranks, fabric), k.clone());
        Ok(k)
    }
}

fn query_len_or_one(spec: bool, query_len: u32) -> u32 {
    if spec {
        query_len
    } else {
        1
    }
}

/// The GDN args every GDN kind shares; `into_*` adds the kind's own fields.
struct GdnCommon {
    backends: Vec<&'static str>,
    gpu_name: String,
    hidden_size: u32,
    num_key_heads: u32,
    num_value_heads: u32,
    head_dim: u32,
    conv_kernel_size: u32,
    dtype: DType,
}

macro_rules! gdn_into {
    ($fn:ident, $cfg:ident) => {
        fn $fn(self) -> $cfg {
            $cfg {
                backends: self.backends,
                gpu_name: self.gpu_name,
                hidden_size: self.hidden_size,
                num_key_heads: self.num_key_heads,
                num_value_heads: self.num_value_heads,
                head_dim: self.head_dim,
                conv_kernel_size: self.conv_kernel_size,
                dtype: self.dtype,
            }
        }
    };
    ($fn:ident, $cfg:ident, query_len) => {
        fn $fn(self, query_len: u32) -> $cfg {
            $cfg {
                backends: self.backends,
                gpu_name: self.gpu_name,
                query_len,
                hidden_size: self.hidden_size,
                num_key_heads: self.num_key_heads,
                num_value_heads: self.num_value_heads,
                head_dim: self.head_dim,
                conv_kernel_size: self.conv_kernel_size,
                dtype: self.dtype,
            }
        }
    };
}

impl GdnCommon {
    gdn_into!(into_prefill_mixer, GdnPrefillMixerKernelConfig);
    gdn_into!(into_decode_mixer, GdnDecodeMixerKernelConfig, query_len);
    gdn_into!(into_in_proj, GdnInProjKernelConfig);
    gdn_into!(into_conv1d_prefill, GdnConv1dPrefillKernelConfig);
    gdn_into!(into_conv1d_decode, GdnConv1dDecodeKernelConfig, query_len);
    gdn_into!(into_split_gating, GdnSplitGatingKernelConfig);
    gdn_into!(into_prefill_core, GdnPrefillCoreKernelConfig);
    gdn_into!(into_decode_core, GdnDecodeCoreKernelConfig, query_len);
    gdn_into!(into_output, GdnOutputKernelConfig);
    gdn_into!(into_state_io, GdnStateIoKernelConfig);
}

/// The QSA args every QSA kind shares.
struct QsaCommon {
    backends: Vec<&'static str>,
    gpu_name: String,
    mode: String,
    hidden_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    index_heads: u32,
    index_head_dim: u32,
    index_topk_tokens: u32,
    compress_ratio: u32,
    dtype: DType,
}

macro_rules! qsa_into {
    ($fn:ident, $cfg:ident) => {
        fn $fn(self) -> $cfg {
            $cfg {
                backends: self.backends,
                gpu_name: self.gpu_name,
                mode: self.mode,
                hidden_size: self.hidden_size,
                num_q_heads: self.num_q_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                index_heads: self.index_heads,
                index_head_dim: self.index_head_dim,
                index_topk_tokens: self.index_topk_tokens,
                compress_ratio: self.compress_ratio,
                dtype: self.dtype,
            }
        }
    };
}

impl QsaCommon {
    qsa_into!(into_mixer2, QsaMixer2KernelConfig);
    qsa_into!(into_project, QsaProjectKernelConfig);
    qsa_into!(into_indexer_project, QsaIndexerProjectKernelConfig);
    qsa_into!(into_cache_write, QsaCacheWriteKernelConfig);
    qsa_into!(into_indexer_compress, QsaIndexerCompressKernelConfig);
    qsa_into!(into_indexer_scores, QsaIndexerScoresKernelConfig);
    qsa_into!(into_indexer_select, QsaIndexerSelectKernelConfig);
    qsa_into!(into_attention, QsaAttentionKernelConfig);
    qsa_into!(into_output, QsaOutputKernelConfig);
}

// ── tree construction ────────────────────────────────────────────────────────

/// What a leaf counts toward the step's region launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegionRole {
    /// A mixer call: `regions` per call at every granularity but `layer`,
    /// where the layer's single region is charged here, once per call.
    Mixer(u32),
    /// A token-wise op inside a layer: one region per call, folded into the
    /// layer's region (no launch) at `layer` granularity.
    InLayer(u32),
    /// Outside the layers (embed, head): its own region at every granularity.
    Standalone(u32),
    /// Fused into a neighbouring region, or a collective traced into one.
    None,
}

struct PlanBuilder<'a> {
    name: String,
    b: CostTreeBuilder,
    leaves: Vec<LeafPlan>,
    k: &'a mut Kernels,
    p: &'a NeuronFlashNextParallel,
    model: &'a Qwen4ExpModelCfg,
    bridge: &'a PerfApiBridge,
    /// Product of the enclosing Scale folds.
    fold: u32,
    view: View,
    /// The location prefix of the leaves being minted (`gdn_first`, `qsa`,
    /// `mtp.final_mixer`, ...); empty at the root.
    scope: String,
}

fn join(base: &str, leaf: &str) -> String {
    if base.is_empty() {
        leaf.to_string()
    } else {
        format!("{base}.{leaf}")
    }
}

impl PlanBuilder<'_> {
    /// Run `f` with the scope extended by `sub`.
    fn scoped<T>(&mut self, sub: &str, f: impl FnOnce(&mut Self) -> T) -> T {
        let saved = std::mem::take(&mut self.scope);
        self.scope = join(&saved, sub);
        let out = f(self);
        self.scope = saved;
        out
    }

    fn leaf(&mut self, label: &str, eval: LeafEval, role: RegionRole) -> CostNode {
        let (kind, config) = match &eval {
            LeafEval::Tokens { k, .. }
            | LeafEval::Decode { k, .. }
            | LeafEval::Chunks { k, .. }
            | LeafEval::Launch { k } => (k.kind(), k.config()),
            LeafEval::QsaDecode { k, .. }
            | LeafEval::QsaChunks { k, .. }
            | LeafEval::Ngram { k, .. }
            | LeafEval::ConvDecode { k, .. }
            | LeafEval::ConvChunks { k, .. } => (k.kind(), k.config()),
            LeafEval::Comm { k, .. } => (k.kind(), k.config()),
        };
        let layer = self.p.region_granularity == RegionGranularity::Layer;
        let regions_per_call = match role {
            RegionRole::Mixer(n) => {
                if layer {
                    1
                } else {
                    n
                }
            }
            RegionRole::InLayer(n) => {
                if layer {
                    0
                } else {
                    n
                }
            }
            RegionRole::Standalone(n) => n,
            RegionRole::None => 0,
        };
        let compute = !matches!(eval, LeafEval::Comm { .. } | LeafEval::Launch { .. });
        self.leaves.push(LeafPlan {
            eval,
            view: self.view,
            regions_per_call,
            fold: self.fold,
            compute,
            coll: false,
        });
        self.b.leaf(
            format!("{}.{}", self.name, join(&self.scope, label)),
            kind,
            config,
        )
    }

    fn comm_leaf(
        &mut self,
        label: &str,
        op: CommOp,
        ranks: u32,
        stride: u32,
        src: RowSrc,
        bytes_per_row: u64,
        div: u32,
    ) -> Result<Option<CostNode>> {
        if ranks <= 1 {
            return Ok(None);
        }
        let fabric = neuron_fabric(ranks, stride, self.p.cores_per_chip());
        let k = self
            .k
            .comm(op, ranks, fabric, &self.name, &self.p.gpu_name, self.bridge)?;
        Ok(Some(self.leaf(
            label,
            LeafEval::Comm {
                k,
                src,
                bytes_per_row,
                div,
            },
            RegionRole::None,
        )))
    }

    /// Mark the compute regions minted since `start` as tracing a collective.
    fn mark_coll(&mut self, start: usize, coll: bool) {
        for l in &mut self.leaves[start..] {
            if l.compute && l.regions_per_call > 0 {
                l.coll |= coll;
            }
        }
    }

    fn g(&self) -> u32 {
        self.p.group_size()
    }

    fn hc_div(&self) -> u32 {
        if self.p.sequence_parallel() {
            self.g()
        } else {
            1
        }
    }

    fn h2(&self) -> u64 {
        u64::from(self.model.hidden) * BF16_BYTES
    }

    /// The combine of the previous block fused with this block's mix (`mixer`
    /// granularity), or both as separate regions (`per_op`).
    fn combine_mix(&mut self, d: usize, label: &str, out: &mut Vec<CostNode>) {
        let div = self.hc_div();
        let src = RowSrc::Group(d);
        if let Some(k) = self.k.hc_combine_mix.clone() {
            out.push(self.leaf(
                label,
                LeafEval::Tokens { k, src, div },
                RegionRole::InLayer(1),
            ));
        } else {
            let k = self.k.hc_combine.clone();
            out.push(self.leaf(
                &format!("{label}.combine"),
                LeafEval::Tokens { k, src, div },
                RegionRole::InLayer(1),
            ));
            let k = self.k.hc_mix.clone();
            out.push(self.leaf(
                &format!("{label}.mix"),
                LeafEval::Tokens { k, src, div },
                RegionRole::InLayer(1),
            ));
        }
    }

    fn tokens(
        &mut self,
        label: &str,
        k: Rows,
        src: RowSrc,
        div: u32,
        role: RegionRole,
    ) -> CostNode {
        self.leaf(label, LeafEval::Tokens { k, src, div }, role)
    }

    /// A mixer's compute leaves for group `d`, its TP collectives around it.
    fn mixer(&mut self, d: usize, mixer: Qwen4ExpMixer, out: &mut Vec<CostNode>) -> Result<()> {
        let start = self.leaves.len();
        self.mixer_leaves(d, mixer, out)?;
        let coll = self.g() > 1;
        self.mark_coll(start, coll);
        Ok(())
    }

    fn mixer_leaves(
        &mut self,
        d: usize,
        mixer: Qwen4ExpMixer,
        out: &mut Vec<CostNode>,
    ) -> Result<()> {
        let g = self.g();
        let m = match mixer {
            Qwen4ExpMixer::Gdn => self.p.gdn_tp,
            Qwen4ExpMixer::Qsa => self.p.attn_tp,
        };
        let sub = g / m;
        let h2 = self.h2();
        let sp = self.p.sequence_parallel();
        let tag = match mixer {
            Qwen4ExpMixer::Gdn => "gdn",
            Qwen4ExpMixer::Qsa => "qsa",
        };
        let src = RowSrc::Group(d);
        if sp {
            if let Some(n) = self.comm_leaf(
                &format!("{tag}.all_gather"),
                CommOp::AllGather,
                m,
                1,
                src,
                h2,
                g,
            )? {
                out.push(n);
            }
        }
        match mixer {
            Qwen4ExpMixer::Gdn => {
                if let (Some(dec), Some(pre)) = (
                    self.k.gdn_decode_mixer.clone(),
                    self.k.gdn_prefill_mixer.clone(),
                ) {
                    out.push(self.leaf(
                        "gdn.decode_mixer",
                        LeafEval::Decode {
                            k: dec,
                            d,
                            sub,
                            rows: false,
                        },
                        RegionRole::Mixer(1),
                    ));
                    if self.view != View::MtpRecurrent {
                        out.push(self.leaf(
                            "gdn.prefill_mixer",
                            LeafEval::Chunks { k: pre, d, sub },
                            RegionRole::Mixer(1),
                        ));
                    }
                } else {
                    let ops = self.k.gdn_ops.as_ref().expect("per-op GDN kernels built");
                    let (in_proj, conv_d, core_d, output, state_io) = (
                        ops.in_proj.clone(),
                        ops.conv_decode.clone(),
                        ops.decode_core.clone(),
                        ops.output.clone(),
                        ops.state_io.clone(),
                    );
                    let (conv_p, split, core_p) = (
                        ops.conv_prefill.clone(),
                        ops.split_gating.clone(),
                        ops.prefill_core.clone(),
                    );
                    let dec = |k: Rows, rows: bool| LeafEval::Decode { k, d, sub, rows };
                    out.push(self.leaf(
                        "gdn.decode.in_proj",
                        dec(in_proj.clone(), true),
                        RegionRole::Mixer(1),
                    ));
                    out.push(self.leaf(
                        "gdn.decode.conv1d",
                        dec(conv_d, false),
                        RegionRole::InLayer(1),
                    ));
                    out.push(self.leaf(
                        "gdn.decode.core",
                        dec(core_d, false),
                        RegionRole::InLayer(1),
                    ));
                    out.push(self.leaf(
                        "gdn.decode.output",
                        dec(output.clone(), true),
                        RegionRole::InLayer(1),
                    ));
                    out.push(self.leaf(
                        "gdn.decode.state_io",
                        dec(state_io, false),
                        RegionRole::InLayer(1),
                    ));
                    let ch = |k: Rows| LeafEval::Chunks { k, d, sub };
                    out.push(self.leaf("gdn.prefill.in_proj", ch(in_proj), RegionRole::Mixer(1)));
                    out.push(self.leaf("gdn.prefill.conv1d", ch(conv_p), RegionRole::InLayer(1)));
                    out.push(self.leaf(
                        "gdn.prefill.split_gating",
                        ch(split),
                        RegionRole::InLayer(1),
                    ));
                    out.push(self.leaf("gdn.prefill.core", ch(core_p), RegionRole::InLayer(1)));
                    out.push(self.leaf("gdn.prefill.output", ch(output), RegionRole::InLayer(1)));
                }
            }
            Qwen4ExpMixer::Qsa => {
                if let (Some(dec), Some(pre)) = (
                    self.k.qsa_mixer2_decode.clone(),
                    self.k.qsa_mixer2_prefill.clone(),
                ) {
                    out.push(self.leaf(
                        "qsa.decode_mixer2",
                        LeafEval::QsaDecode { k: dec, d, sub },
                        RegionRole::Mixer(2),
                    ));
                    if self.view != View::MtpRecurrent {
                        out.push(self.leaf(
                            "qsa.prefill_mixer2",
                            LeafEval::QsaChunks { k: pre, d, sub },
                            RegionRole::Mixer(2),
                        ));
                    }
                } else {
                    let modes = [
                        ("decode", self.k.qsa_ops_decode.clone()),
                        ("prefill", self.k.qsa_ops_prefill.clone()),
                    ];
                    for (mode, ops) in modes {
                        if mode == "prefill" && self.view == View::MtpRecurrent {
                            continue;
                        }
                        let ops = ops.expect("per-op QSA kernels built");
                        let rows: [(&str, Rows); 5] = [
                            ("project", ops.project.clone()),
                            ("indexer_project", ops.indexer_project.clone()),
                            ("cache_write", ops.cache_write.clone()),
                            ("indexer_compress", ops.indexer_compress.clone()),
                            ("output", ops.output.clone()),
                        ];
                        let ctx: [(&str, Pair); 3] = [
                            ("indexer_scores", ops.indexer_scores.clone()),
                            ("indexer_select", ops.indexer_select.clone()),
                            ("attention", ops.attention.clone()),
                        ];
                        let mut first = true;
                        for (op, k) in rows.into_iter().take(4) {
                            let eval = if mode == "decode" {
                                LeafEval::Decode {
                                    k,
                                    d,
                                    sub,
                                    rows: true,
                                }
                            } else {
                                LeafEval::Chunks { k, d, sub }
                            };
                            let role = if first {
                                RegionRole::Mixer(1)
                            } else {
                                RegionRole::InLayer(1)
                            };
                            first = false;
                            out.push(self.leaf(&format!("qsa.{mode}.{op}"), eval, role));
                        }
                        for (op, k) in ctx {
                            let eval = if mode == "decode" {
                                LeafEval::QsaDecode { k, d, sub }
                            } else {
                                LeafEval::QsaChunks { k, d, sub }
                            };
                            out.push(self.leaf(
                                &format!("qsa.{mode}.{op}"),
                                eval,
                                RegionRole::InLayer(1),
                            ));
                        }
                        let k = ops.output.clone();
                        let eval = if mode == "decode" {
                            LeafEval::Decode {
                                k,
                                d,
                                sub,
                                rows: true,
                            }
                        } else {
                            LeafEval::Chunks { k, d, sub }
                        };
                        out.push(self.leaf(
                            &format!("qsa.{mode}.output"),
                            eval,
                            RegionRole::InLayer(1),
                        ));
                    }
                }
            }
        }
        // The row-parallel output: per sub-replica partial sums.
        if sp {
            if let Some(n) = self.comm_leaf(
                &format!("{tag}.reduce_scatter"),
                CommOp::ReduceScatter,
                m,
                1,
                src,
                h2,
                sub,
            )? {
                out.push(n);
            }
        } else {
            if let Some(n) = self.comm_leaf(
                &format!("{tag}.all_reduce"),
                CommOp::AllReduce,
                m,
                1,
                src,
                h2,
                sub,
            )? {
                out.push(n);
            }
            // Sub-replicas exchange their tokens so every rank of the group
            // holds the group's residual stream.
            if let Some(n) = self.comm_leaf(
                &format!("{tag}.sub_all_gather"),
                CommOp::AllGather,
                sub,
                m,
                src,
                h2,
                sub,
            )? {
                out.push(n);
            }
        }
        Ok(())
    }

    /// The MoE block over every group's tokens, with its dispatch collectives.
    fn moe(&mut self, out: &mut Vec<CostNode>) -> Result<()> {
        let start = self.leaves.len();
        self.moe_leaves(out)?;
        let coll = self.p.ranks() > 1;
        self.mark_coll(start, coll);
        Ok(())
    }

    fn moe_leaves(&mut self, out: &mut Vec<CostNode>) -> Result<()> {
        let r = self.p.ranks();
        let g = self.g();
        let h2 = self.h2();
        let sp = self.p.sequence_parallel();
        let dp = self.p.attn_dp;
        match self.p.moe_dispatch {
            MoeDispatch::AllGather => {
                let replicated = dp == 1 && !sp;
                if !replicated {
                    if let Some(n) = self.comm_leaf(
                        "moe.all_gather",
                        CommOp::AllGather,
                        r,
                        1,
                        RowSrc::All,
                        h2,
                        r,
                    )? {
                        out.push(n);
                    }
                }
                if let Some(k) = self.k.moe_block.clone() {
                    out.push(self.tokens("moe.block", k, RowSrc::All, 1, RegionRole::InLayer(1)));
                } else {
                    let router = self.k.moe_router.clone().expect("router");
                    let routed = self.k.moe_routed.clone().expect("routed");
                    let shared = self.k.moe_shared.clone().expect("shared");
                    out.push(self.tokens(
                        "moe.router",
                        router,
                        RowSrc::All,
                        1,
                        RegionRole::InLayer(1),
                    ));
                    out.push(self.tokens(
                        "moe.routed_experts",
                        routed,
                        RowSrc::All,
                        1,
                        RegionRole::InLayer(1),
                    ));
                    out.push(self.tokens(
                        "moe.shared_expert",
                        shared,
                        RowSrc::All,
                        1,
                        RegionRole::InLayer(1),
                    ));
                }
                if replicated {
                    if let Some(n) = self.comm_leaf(
                        "moe.all_reduce",
                        CommOp::AllReduce,
                        r,
                        1,
                        RowSrc::All,
                        h2,
                        1,
                    )? {
                        out.push(n);
                    }
                } else {
                    if let Some(n) = self.comm_leaf(
                        "moe.reduce_scatter",
                        CommOp::ReduceScatter,
                        r,
                        1,
                        RowSrc::All,
                        h2,
                        1,
                    )? {
                        out.push(n);
                    }
                    if !sp {
                        if let Some(n) = self.comm_leaf(
                            "moe.group_all_gather",
                            CommOp::AllGather,
                            g,
                            1,
                            RowSrc::MaxGroup,
                            h2,
                            g,
                        )? {
                            out.push(n);
                        }
                    }
                }
            }
            MoeDispatch::AllToAll => {
                let k = u64::from(self.model.experts_per_token);
                let router = self.k.moe_router.clone().expect("router");
                let routed = self.k.moe_routed.clone().expect("routed");
                let shared = self.k.moe_shared.clone().expect("shared");
                out.push(self.tokens(
                    "moe.router",
                    router,
                    RowSrc::MaxGroup,
                    g,
                    RegionRole::InLayer(1),
                ));
                if let Some(n) = self.comm_leaf(
                    "moe.dispatch",
                    CommOp::AllToAll,
                    r,
                    1,
                    RowSrc::MaxGroup,
                    k * h2,
                    g,
                )? {
                    out.push(n);
                }
                out.push(self.tokens(
                    "moe.routed_experts",
                    routed,
                    RowSrc::MoeEquivalent,
                    1,
                    RegionRole::InLayer(1),
                ));
                if let Some(n) = self.comm_leaf(
                    "moe.combine",
                    CommOp::AllToAll,
                    r,
                    1,
                    RowSrc::MaxGroup,
                    k * h2,
                    g,
                )? {
                    out.push(n);
                }
                out.push(self.tokens(
                    "moe.shared_expert",
                    shared,
                    RowSrc::MaxGroup,
                    g,
                    RegionRole::None,
                ));
                if !sp {
                    if let Some(n) = self.comm_leaf(
                        "moe.group_all_gather",
                        CommOp::AllGather,
                        g,
                        1,
                        RowSrc::MaxGroup,
                        h2,
                        g,
                    )? {
                        out.push(n);
                    }
                }
            }
        }
        Ok(())
    }

    /// The PLE block of group `d` (before its layer's attention module).
    fn ple(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        let start = self.leaves.len();
        self.ple_leaves(d, out)?;
        let coll = self.p.ranks() > 1;
        self.mark_coll(start, coll);
        Ok(())
    }

    fn ple_leaves(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        let r = self.p.ranks();
        let g = self.g();
        let sp = self.p.sequence_parallel();
        let div = self.hc_div();
        let ple_bytes = u64::from(self.model.ple_embed_dim) * BF16_BYTES;
        let ids_bytes =
            u64::from((self.model.ngram_size - 1) * self.model.heads_per_ngram) * FP32_BYTES;
        let ngram = self.k.ple_ngram.clone().expect("ple kernels");
        let gather = self.k.ple_gather.clone().expect("ple kernels");
        out.push(self.leaf(
            "ple.ngram_ids",
            LeafEval::Ngram { k: ngram, d },
            RegionRole::InLayer(1),
        ));
        match self.p.ple_table_shards {
            PleTableShards::Replica => {
                if let Some(n) = self.comm_leaf(
                    "ple.ids_all_gather",
                    CommOp::AllGather,
                    r,
                    1,
                    RowSrc::All,
                    ids_bytes,
                    r,
                )? {
                    out.push(n);
                }
                out.push(self.tokens("ple.gather", gather, RowSrc::All, 1, RegionRole::None));
                if let Some(n) = self.comm_leaf(
                    "ple.reduce_scatter",
                    CommOp::ReduceScatter,
                    r,
                    1,
                    RowSrc::All,
                    ple_bytes,
                    1,
                )? {
                    out.push(n);
                }
                if !sp {
                    if let Some(n) = self.comm_leaf(
                        "ple.group_all_gather",
                        CommOp::AllGather,
                        g,
                        1,
                        RowSrc::Group(d),
                        ple_bytes,
                        g,
                    )? {
                        out.push(n);
                    }
                }
            }
            PleTableShards::DpGroup | PleTableShards::Host => {
                out.push(self.tokens("ple.gather", gather, RowSrc::Group(d), 1, RegionRole::None));
                let op = if sp {
                    CommOp::ReduceScatter
                } else {
                    CommOp::AllReduce
                };
                if let Some(n) =
                    self.comm_leaf("ple.group_reduce", op, g, 1, RowSrc::Group(d), ple_bytes, 1)?
                {
                    out.push(n);
                }
            }
        }
        if let Some(k) = self.k.ple_post.clone() {
            out.push(self.tokens("ple.post", k, RowSrc::Group(d), div, RegionRole::InLayer(1)));
        } else {
            let project = self.k.ple_project.clone().expect("ple per-op");
            let gate = self.k.ple_gate.clone().expect("ple per-op");
            let residual = self.k.ple_residual.clone().expect("ple per-op");
            let conv_d = self.k.ple_conv_decode.clone().expect("ple per-op");
            let conv_p = self.k.ple_conv_prefill.clone().expect("ple per-op");
            out.push(self.tokens(
                "ple.project",
                project,
                RowSrc::Group(d),
                div,
                RegionRole::InLayer(1),
            ));
            out.push(self.tokens(
                "ple.gate",
                gate,
                RowSrc::Group(d),
                div,
                RegionRole::InLayer(1),
            ));
            out.push(self.leaf(
                "ple.short_conv_decode",
                LeafEval::ConvDecode { k: conv_d, d },
                RegionRole::InLayer(1),
            ));
            out.push(self.leaf(
                "ple.short_conv_prefill",
                LeafEval::ConvChunks { k: conv_p, d },
                RegionRole::InLayer(1),
            ));
            out.push(self.tokens(
                "ple.residual",
                residual,
                RowSrc::Group(d),
                div,
                RegionRole::InLayer(1),
            ));
        }
        Ok(())
    }

    /// Embedding for group `d`: vocab-parallel gather + the group reduction.
    fn embed(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        let start = self.leaves.len();
        self.embed_leaves(d, out)?;
        let coll = self.g() > 1;
        self.mark_coll(start, coll);
        Ok(())
    }

    fn embed_leaves(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        let k = self.k.embed.clone();
        out.push(self.tokens(
            "embedding",
            k,
            RowSrc::Group(d),
            1,
            RegionRole::Standalone(1),
        ));
        let op = if self.p.sequence_parallel() {
            CommOp::ReduceScatter
        } else {
            CommOp::AllReduce
        };
        let (g, h2) = (self.g(), self.h2());
        if let Some(n) = self.comm_leaf("embedding.reduce", op, g, 1, RowSrc::Group(d), h2, 1)? {
            out.push(n);
        }
        Ok(())
    }

    /// Final HC combine and mixer, lm_head and greedy argmax for group `d`.
    fn head(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        self.scoped("final_mixer", |s| {
            let div = s.hc_div();
            let k = s.k.hc_combine.clone();
            out.push(s.tokens(
                "combine",
                k,
                RowSrc::Group(d),
                div,
                RegionRole::Standalone(1),
            ));
            let k = s.k.hc_mix_final.clone();
            out.push(s.tokens("mix", k, RowSrc::Group(d), div, RegionRole::Standalone(1)));
        });
        self.scoped("head", |s| s.head_rows(d, out))
    }

    fn head_rows(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        let start = self.leaves.len();
        self.head_rows_leaves(d, out)?;
        let coll = self.g() > 1;
        self.mark_coll(start, coll);
        Ok(())
    }

    fn head_rows_leaves(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        let (g, h2) = (self.g(), self.h2());
        if self.p.sequence_parallel() {
            if let Some(n) = self.comm_leaf(
                "rows_all_gather",
                CommOp::AllGather,
                g,
                1,
                RowSrc::Sample(d),
                h2,
                g,
            )? {
                out.push(n);
            }
        }
        if let Some(k) = self.k.lm_head_argmax.clone() {
            out.push(self.tokens(
                "lm_head_argmax",
                k,
                RowSrc::Sample(d),
                1,
                RegionRole::Standalone(1),
            ));
        } else {
            let gemm = self.k.lm_head_gemm.clone().expect("per-op head");
            let argmax = self.k.vocab_argmax.clone().expect("per-op head");
            out.push(self.tokens(
                "lm_head",
                gemm,
                RowSrc::Sample(d),
                1,
                RegionRole::Standalone(1),
            ));
            out.push(self.tokens(
                "argmax",
                argmax,
                RowSrc::Sample(d),
                1,
                RegionRole::Standalone(1),
            ));
        }
        if let Some(n) = self.comm_leaf(
            "argmax_all_gather",
            CommOp::AllGather,
            g,
            1,
            RowSrc::Sample(d),
            ARGMAX_PAIR_BYTES,
            1,
        )? {
            out.push(n);
        }
        Ok(())
    }

    fn per_group<F>(&mut self, mut f: F) -> Result<CostNode>
    where
        F: FnMut(&mut Self, usize, &mut Vec<CostNode>) -> Result<()>,
    {
        let mut groups = Vec::new();
        for d in 0..self.p.attn_dp as usize {
            let mut out = Vec::new();
            f(self, d, &mut out)?;
            groups.push(CostNode::Sum(out));
        }
        Ok(CostNode::Max {
            overlap: 1.0,
            children: groups,
        })
    }

    /// One decoder layer of `mixer`: attention module, mixer, mlp module (per
    /// group), then the MoE block over every group.
    fn layer(&mut self, mixer: Qwen4ExpMixer, first: bool, ple: bool) -> Result<CostNode> {
        let attn = self.per_group(|s, d, out| {
            let div = s.hc_div();
            if ple {
                if !first {
                    let k = s.k.hc_combine.clone();
                    out.push(s.tokens(
                        "attn_hc.pending_combine",
                        k,
                        RowSrc::Group(d),
                        div,
                        RegionRole::InLayer(1),
                    ));
                }
                s.ple(d, out)?;
                let k = s.k.hc_mix.clone();
                out.push(s.tokens(
                    "attn_hc.mix",
                    k,
                    RowSrc::Group(d),
                    div,
                    RegionRole::InLayer(1),
                ));
            } else if first {
                let k = s.k.hc_mix.clone();
                out.push(s.tokens(
                    "attn_hc.mix",
                    k,
                    RowSrc::Group(d),
                    div,
                    RegionRole::InLayer(1),
                ));
            } else {
                s.combine_mix(d, "attn_hc", out);
            }
            s.mixer(d, mixer, out)?;
            s.combine_mix(d, "mlp_hc", out);
            Ok(())
        })?;
        let mut moe = Vec::new();
        self.moe(&mut moe)?;
        Ok(CostNode::Sum(vec![attn, CostNode::Sum(moe)]))
    }

    /// One MTP pass in `view`.
    fn mtp_pass(&mut self) -> Result<CostNode> {
        let pre = self.per_group(|s, d, out| {
            s.embed(d, out)?;
            let fc = s.k.mtp_fc.clone().expect("mtp fc");
            out.push(s.tokens("fc", fc, RowSrc::Group(d), 1, RegionRole::InLayer(1)));
            let k = s.k.hc_mix.clone();
            let div = s.hc_div();
            out.push(s.tokens(
                "attn_hc.mix",
                k,
                RowSrc::Group(d),
                div,
                RegionRole::InLayer(1),
            ));
            s.mixer(d, Qwen4ExpMixer::Qsa, out)?;
            s.combine_mix(d, "mlp_hc", out);
            Ok(())
        })?;
        let mut moe = Vec::new();
        self.moe(&mut moe)?;
        let post = self.per_group(|s, d, out| s.head(d, out))?;
        Ok(CostNode::Sum(vec![pre, CostNode::Sum(moe), post]))
    }
}

/// The checkpoint's layers as (mixer, first layer, carries PLE) runs with
/// their counts, in first-appearance order.
fn layer_groups(model: &Qwen4ExpModelCfg) -> Vec<(Qwen4ExpMixer, bool, bool, u32)> {
    let mut groups: Vec<(Qwen4ExpMixer, bool, bool, u32)> = Vec::new();
    for (l, &mixer) in model.mixers.iter().enumerate() {
        let key = (mixer, l == 0, model.ple_layers.contains(&(l as u32)));
        match groups.iter_mut().find(|(m, f, p, _)| (*m, *f, *p) == key) {
            Some(group) => group.3 += 1,
            None => groups.push((key.0, key.1, key.2, 1)),
        }
    }
    groups
}

fn mixer_name(m: Qwen4ExpMixer) -> &'static str {
    match m {
        Qwen4ExpMixer::Gdn => "gdn",
        Qwen4ExpMixer::Qsa => "qsa",
    }
}

fn build_plan(
    name: &str,
    model: &Qwen4ExpModelCfg,
    p: &NeuronFlashNextParallel,
    kernels: &mut Kernels,
    bridge: &PerfApiBridge,
) -> Result<(CostTree, Vec<LeafPlan>)> {
    let mut pb = PlanBuilder {
        name: name.to_string(),
        b: CostTreeBuilder::new(),
        leaves: Vec::new(),
        k: kernels,
        p,
        model,
        bridge,
        fold: 1,
        view: View::Target,
        scope: String::new(),
    };
    let mut root = Vec::new();
    root.push(CostNode::Labeled {
        label: "prologue".to_string(),
        child: Box::new(pb.per_group(|s, d, out| s.embed(d, out))?),
    });
    for (mixer, first, ple, n) in layer_groups(model) {
        pb.fold = n;
        let scope = match (mixer, first, ple) {
            (Qwen4ExpMixer::Gdn, _, true) => "gdn_ple",
            (Qwen4ExpMixer::Gdn, true, false) => "gdn_first",
            (Qwen4ExpMixer::Gdn, false, false) => "gdn",
            (Qwen4ExpMixer::Qsa, _, true) => "qsa_ple",
            (Qwen4ExpMixer::Qsa, true, false) => "qsa_first",
            (Qwen4ExpMixer::Qsa, false, false) => "qsa",
        };
        let layer = pb.scoped(scope, |pb| pb.layer(mixer, first, ple))?;
        let label = format!(
            "{n} x {}{}{}",
            mixer_name(mixer),
            if first { " (first)" } else { "" },
            if ple { " + ple" } else { "" }
        );
        root.push(CostNode::Labeled {
            label,
            child: Box::new(CostNode::Scale {
                n,
                child: Box::new(layer),
            }),
        });
    }
    pb.fold = 1;
    root.push(CostNode::Labeled {
        label: "epilogue".to_string(),
        child: Box::new(pb.per_group(|s, d, out| s.head(d, out))?),
    });
    if p.draft_tokens > 0 {
        ensure!(
            model.mtp_layers == 1,
            "the MTP pass models one QSA MTP layer; the checkpoint has {}",
            model.mtp_layers
        );
        pb.view = View::MtpFirst;
        let first = pb.scoped("mtp", |pb| pb.mtp_pass())?;
        let mut mtp = vec![CostNode::Labeled {
            label: "mtp pass 0".to_string(),
            child: Box::new(first),
        }];
        if p.draft_tokens > 1 {
            pb.view = View::MtpRecurrent;
            pb.fold = p.draft_tokens - 1;
            let rec = pb.scoped("mtp_recurrent", |pb| pb.mtp_pass())?;
            mtp.push(CostNode::Labeled {
                label: "mtp recurrent pass".to_string(),
                child: Box::new(CostNode::Scale {
                    n: p.draft_tokens - 1,
                    child: Box::new(rec),
                }),
            });
            pb.fold = 1;
        }
        pb.view = View::Target;
        root.push(CostNode::Labeled {
            label: "mtp".to_string(),
            child: Box::new(CostNode::Sum(mtp)),
        });
    }
    let launch = pb.k.launch.clone();
    root.push(pb.leaf("launch", LeafEval::Launch { k: launch }, RegionRole::None));
    let tree = pb.b.finish(CostNode::Labeled {
        label: format!(
            "{name} [Trainium2 LNC{}, dp{} x (attn tp{}, gdn tp{}), moe ep{} x tp{}, {:?}]",
            p.lnc, p.attn_dp, p.attn_tp, p.gdn_tp, p.moe_ep, p.moe_tp, p.region_granularity
        ),
        child: Box::new(CostNode::Sum(root)),
    });
    Ok((tree, pb.leaves))
}

// ── evaluation ───────────────────────────────────────────────────────────────

/// Bucket inventories, resolved once.
struct Buckets {
    prefill: Vec<u32>,
    decode: Vec<u32>,
    context: Vec<u32>,
    /// Token-wise regions: the union of the prefill and decode buckets.
    tokens: Vec<u32>,
}

impl Buckets {
    fn new(p: &NeuronFlashNextParallel) -> Self {
        let mut tokens: Vec<u32> = p
            .prefill_buckets
            .iter()
            .chain(&p.decode_buckets)
            .copied()
            .collect();
        tokens.sort_unstable();
        tokens.dedup();
        Self {
            prefill: p.prefill_buckets.clone(),
            decode: p.decode_buckets.clone(),
            context: p.context_buckets.clone(),
            tokens,
        }
    }

    fn tok(&self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            bucket(n, &self.tokens)
        }
    }
}

/// One leaf's evaluated work: metrics and the calls (regions) it launched.
struct LeafOut {
    m: LeafMetrics,
    calls: u32,
    input: Option<SlotInput>,
}

fn rows_of(src: RowSrc, groups: &[GroupStep], bk: &Buckets, ep: u32) -> u32 {
    match src {
        RowSrc::Group(d) => bk.tok(groups[d].tokens()),
        RowSrc::All => groups.iter().map(|g| bk.tok(g.tokens())).sum(),
        RowSrc::MaxGroup => groups.iter().map(|g| bk.tok(g.tokens())).max().unwrap_or(0),
        RowSrc::Sample(d) => bk.tok(groups[d].sample_rows()),
        RowSrc::MoeEquivalent => groups
            .iter()
            .map(|g| bk.tok(g.tokens()))
            .sum::<u32>()
            .div_ceil(ep.max(1)),
    }
}

/// Spread per-call metrics over `sub` sub-replicas, largest first: time is the
/// busiest sub-replica's, work sums over all of them.
fn spread(mut calls: Vec<LeafMetrics>, sub: u32) -> LeafMetrics {
    let mut total = LeafMetrics::ZERO;
    if calls.is_empty() {
        return total;
    }
    calls.sort_by(|a, b| b.m.time_ms.total_cmp(&a.m.time_ms));
    let mut bins = vec![0f32; sub.max(1) as usize];
    for c in &calls {
        let (i, _) = bins
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("at least one bin");
        bins[i] += c.m.time_ms;
        total.add(*c);
    }
    total.m.time_ms = bins.iter().copied().fold(0.0, f32::max);
    total
}

fn eval_leaf(
    plan: &LeafPlan,
    views: &StepViews,
    bk: &Buckets,
    ep: u32,
    regions: u32,
    record: bool,
    adj: &dyn Fn(LeafMetrics) -> LeafMetrics,
) -> LeafOut {
    let groups = views.get(plan.view);
    let mut m = LeafMetrics::ZERO;
    let mut calls = 0u32;
    let mut input = None;
    let mut sizes = Vec::new();
    match &plan.eval {
        LeafEval::Tokens { k, src, div } => {
            let rows = rows_of(*src, groups, bk, ep).div_ceil((*div).max(1));
            if rows > 0 {
                let max = *bk.tokens.last().expect("buckets");
                let mut rest = rows;
                while rest > 0 {
                    let n = rest.min(max);
                    m.add(adj(k.eval_at(n)));
                    if record && input.is_none() {
                        input = Some(k.input_at(n));
                    }
                    calls += 1;
                    rest -= n;
                }
            }
            if record && input.is_none() {
                input = Some(k.input_at(0));
            }
        }
        LeafEval::Decode { k, d, sub, rows } => {
            let g = &groups[*d];
            if g.decode_reqs > 0 {
                bucket_calls(g.decode_reqs.div_ceil(*sub), &bk.decode, &mut sizes);
                for &b in &sizes {
                    let n = if *rows { b * g.query_len } else { b };
                    m.add(adj(k.eval_at(n)));
                    calls += 1;
                    if record && input.is_none() {
                        input = Some(k.input_at(n));
                    }
                }
            }
            if record && input.is_none() {
                input = Some(k.input_at(0));
            }
        }
        LeafEval::Chunks { k, d, sub } => {
            let mut per = Vec::new();
            for &(_, append) in &groups[*d].chunks {
                sizes.clear();
                bucket_calls(append, &bk.prefill, &mut sizes);
                for &c in &sizes {
                    per.push(adj(k.eval_at(c)));
                    calls += 1;
                    if record && input.is_none() {
                        input = Some(k.input_at(c));
                    }
                }
            }
            m = spread(per, *sub);
            if record && input.is_none() {
                input = Some(k.input_at(0));
            }
        }
        LeafEval::QsaDecode { k, d, sub } => {
            let g = &groups[*d];
            if g.decode_reqs > 0 {
                let ctx = bucket(g.decode_max_kv.max(1), &bk.context);
                bucket_calls(g.decode_reqs.div_ceil(*sub), &bk.decode, &mut sizes);
                for &b in &sizes {
                    let x = (b * g.query_len, ctx);
                    m.add(adj(k.eval_at(x)));
                    calls += 1;
                    if record && input.is_none() {
                        input = Some(k.input_at(x));
                    }
                }
            }
            if record && input.is_none() {
                input = Some(k.input_at((0, 0)));
            }
        }
        LeafEval::QsaChunks { k, d, sub } => {
            let mut per = Vec::new();
            for &(prefix, append) in &groups[*d].chunks {
                sizes.clear();
                bucket_calls(append, &bk.prefill, &mut sizes);
                let mut done = prefix;
                for &c in &sizes {
                    let ctx = bucket(done + c, &bk.context).max(c);
                    per.push(adj(k.eval_at((c, ctx))));
                    calls += 1;
                    if record && input.is_none() {
                        input = Some(k.input_at((c, ctx)));
                    }
                    done += c;
                }
            }
            m = spread(per, *sub);
            if record && input.is_none() {
                input = Some(k.input_at((0, 0)));
            }
        }
        LeafEval::Ngram { k, d } => {
            let g = &groups[*d];
            let tokens = bk.tok(g.tokens());
            let x = (tokens, g.requests.clamp(1, tokens.max(1)));
            if tokens > 0 {
                m = adj(k.eval_at(x));
                calls = 1;
            }
            if record {
                input = Some(k.input_at(x));
            }
        }
        LeafEval::ConvDecode { k, d } => {
            let g = &groups[*d];
            if g.decode_reqs > 0 {
                bucket_calls(g.decode_reqs, &bk.decode, &mut sizes);
                for &b in &sizes {
                    m.add(adj(k.eval_at((b, g.query_len))));
                    calls += 1;
                }
            }
            if record {
                input = Some(k.input_at((bucket(g.decode_reqs.max(1), &bk.decode), g.query_len)));
            }
        }
        LeafEval::ConvChunks { k, d } => {
            for &(_, append) in &groups[*d].chunks {
                sizes.clear();
                bucket_calls(append, &bk.prefill, &mut sizes);
                for &c in &sizes {
                    m.add(adj(k.eval_at((1, c))));
                    calls += 1;
                }
            }
            if record {
                input = Some(k.input_at((1, 0)));
            }
        }
        LeafEval::Comm {
            k,
            src,
            bytes_per_row,
            div,
        } => {
            let rows = u64::from(rows_of(*src, groups, bk, ep));
            let bytes = (rows * bytes_per_row).div_ceil(u64::from((*div).max(1)));
            if bytes > 0 {
                m = k.eval_at(bytes);
            }
            if record {
                input = Some(k.input_at(bytes));
            }
        }
        LeafEval::Launch { k } => {
            m = k.eval_at(regions);
            if record {
                input = Some(k.input_at(regions));
            }
        }
    }
    LeafOut { m, calls, input }
}

// ── weights and KV ───────────────────────────────────────────────────────────

/// BF16 weight bytes one rank of the replica holds.
fn weight_bytes_per_rank(m: &Qwen4ExpModelCfg, p: &NeuronFlashNextParallel) -> u64 {
    let h = u64::from(m.hidden);
    let (g, r) = (u64::from(p.group_size()), u64::from(p.ranks()));
    let hc = u64::from(m.hc_count);
    let d = hc * h;
    // One HC module: norm, down, up, inject.
    let hc_module = d + 2 * u64::from(m.hc_lowrank) * d + hc * d;
    let gdn = {
        let (kh, vh) = (u64::from(m.gdn_key_heads), u64::from(m.gdn_value_heads));
        let (dk, dv) = (
            u64::from(m.gdn_key_head_dim),
            u64::from(m.gdn_value_head_dim),
        );
        let in_proj = h * (2 * kh * dk + vh * dv + vh * dv + 2 * vh);
        let conv = u64::from(m.gdn_conv_channels()) * u64::from(m.gdn_conv_kernel);
        (in_proj + conv + vh * dv * h) / u64::from(p.gdn_tp)
    };
    let qsa = {
        let (qh, d_) = (u64::from(m.qsa_heads), u64::from(m.qsa_head_dim));
        let kv_local = (u64::from(m.qsa_kv_heads) / u64::from(p.attn_tp)).max(1);
        let q_o = (h * qh * d_ * 2 + qh * d_ * h) / u64::from(p.attn_tp);
        let kv = h * kv_local * d_ * 2;
        let indexer = h * u64::from(m.indexer_heads * m.indexer_head_dim)
            + h * u64::from(m.indexer_kv_heads * m.indexer_head_dim);
        q_o + kv + indexer
    };
    let moe = {
        let e = u64::from(m.num_experts);
        let i = u64::from(m.moe_intermediate);
        let experts = e * 3 * i * h / (u64::from(p.moe_ep) * u64::from(p.moe_tp));
        let shared = 3 * u64::from(m.shared_expert_intermediate) * h;
        let shared = match p.moe_dispatch {
            MoeDispatch::AllGather => shared / r,
            MoeDispatch::AllToAll => shared,
        };
        experts + e * h + shared + h
    };
    let layers = m.mixers.iter().fold(0u64, |acc, mixer| {
        acc + 2 * hc_module
            + moe
            + match mixer {
                Qwen4ExpMixer::Gdn => gdn,
                Qwen4ExpMixer::Qsa => qsa,
            }
    });
    let vocab = 2 * u64::from(m.vocab_size) * h / g;
    let ple = if m.ple_layers.is_empty() {
        0
    } else {
        let shards = match p.ple_table_shards {
            PleTableShards::Replica => r,
            _ => g,
        };
        let table = m.ple_table_rows
            * u64::from(m.ple_embed_dim / ((m.ngram_size - 1) * m.heads_per_ngram))
            / shards;
        // key/value projections, norms and the short conv: replicated.
        table + u64::from(m.ple_embed_dim) * (d + h) + d * u64::from(m.ple_conv_kernel)
    };
    let mtp = if p.draft_tokens > 0 {
        u64::from(m.mtp_layers) * (2 * h * h + qsa + moe + 2 * hc_module)
    } else {
        0
    };
    let final_mixer = hc_module;
    (layers + vocab + ple + mtp + final_mixer) * BF16_BYTES
}

fn kv_layout(
    m: &Qwen4ExpModelCfg,
    p: &NeuronFlashNextParallel,
    fixed_device_bytes_per_rank: u64,
) -> KvCacheLayout {
    let tp = u64::from(p.attn_tp);
    let kv_local = (u64::from(m.qsa_kv_heads) / tp).max(1);
    let qsa_layers = u64::from(m.layer_count(Qwen4ExpMixer::Qsa))
        + if p.draft_tokens > 0 {
            u64::from(m.mtp_layers)
        } else {
            0
        };
    let d = u64::from(m.qsa_head_dim);
    let index = u64::from(m.indexer_head_dim);
    // K and V of the rank's KV head(s), on each of the sub-replica's ranks.
    let kv = qsa_layers * tp * kv_local * d * 2 * BF16_BYTES;
    // The indexer's raw key cache is replicated on every rank of the sub-replica.
    let raw = qsa_layers * tp * index * BF16_BYTES;
    let comp = qsa_layers * tp * index * BF16_BYTES;
    let gdn_layers = u64::from(m.layer_count(Qwen4ExpMixer::Gdn));
    let state_bytes = u64::from(m.gdn_state_dtype.size_bytes());
    let ssm = u64::from(m.gdn_value_heads)
        * u64::from(m.gdn_key_head_dim)
        * u64::from(m.gdn_value_head_dim)
        * state_bytes;
    let conv = u64::from(m.gdn_conv_channels()) * u64::from(m.gdn_conv_kernel - 1) * BF16_BYTES;
    let mut layout = KvCacheLayout::new()
        .with_part(KvCachePart::per_token("qsa_kv", kv))
        .with_part(KvCachePart::per_token("qsa_indexer_raw", raw))
        .with_part(KvCachePart::compressed(
            "qsa_indexer_comp",
            m.indexer_compress_ratio,
            comp,
        ))
        .with_part(
            KvCachePart::per_request("gdn_state", gdn_layers * (ssm + conv)).not_prefix_reusable(),
        );
    if !m.ple_layers.is_empty() {
        let ple_state = u64::from(m.ple_conv_state_len())
            * u64::from(m.hc_count * m.hidden)
            * BF16_BYTES
            * u64::from(p.group_size());
        layout = layout
            .with_part(KvCachePart::per_request("ple_conv_state", ple_state).not_prefix_reusable());
    }
    layout.with_engine(EngineOverhead {
        block_tokens: None,
        fixed_device_bytes: fixed_device_bytes_per_rank * u64::from(p.group_size()),
    })
}

// ── the model ────────────────────────────────────────────────────────────────

/// The compiled step shared by the ordinary and the speculative contract.
pub struct NeuronFlashNextCore {
    pub name: String,
    pub parallel: NeuronFlashNextParallel,
    /// BF16 weights plus the runtime reserve, per rank.
    pub fixed_bytes_per_rank: u64,
    pub weight_bytes_per_rank: u64,
    leaves: Vec<LeafPlan>,
    buckets: Buckets,
    cost_flat: Vec<FlatCostNode>,
    manifest: CostManifest,
    n_slots: usize,
    kv: KvCacheLayout,
}

impl NeuronFlashNextCore {
    pub fn build(
        name: String,
        model: &Qwen4ExpModelCfg,
        parallel: NeuronFlashNextParallel,
        bridge: &PerfApiBridge,
    ) -> Result<Self> {
        parallel.validate(model)?;
        ensure!(
            model.ple_table_dtype == DType::Bf16,
            "qwen4_exp_neuron_bf16 serves the BF16 checkpoint (read it with from_bf16_json)"
        );
        let weight_bytes_per_rank = weight_bytes_per_rank(model, &parallel);
        let reserve = (parallel.hbm_reserve_gb * 1e9) as u64;
        let fixed_bytes_per_rank = weight_bytes_per_rank + reserve;
        let hbm = (parallel.hbm_per_core_gb * 1e9) as u64;
        if fixed_bytes_per_rank >= hbm {
            bail!(
                "qwen4_exp_neuron_bf16: {:.2} GB of BF16 weights + {:.2} GB reserve per rank exceed \
                 the {:.2} GB HBM of one LNC{} core; spread the experts (moe_ep) or the PLE table \
                 over more ranks",
                weight_bytes_per_rank as f64 / 1e9,
                parallel.hbm_reserve_gb,
                parallel.hbm_per_core_gb,
                parallel.lnc
            );
        }
        let mut kernels = Kernels::build(&name, model, &parallel, bridge)?;
        let (tree, leaves) = build_plan(&name, model, &parallel, &mut kernels, bridge)?;
        let kv = kv_layout(model, &parallel, fixed_bytes_per_rank);
        tracing::info!(
            "[build] {name}: {} leaf slots, {:.2} GB weights/rank\n{}",
            tree.n_slots(),
            weight_bytes_per_rank as f64 / 1e9,
            tree.describe()
        );
        Ok(Self {
            buckets: Buckets::new(&parallel),
            name,
            parallel,
            fixed_bytes_per_rank,
            weight_bytes_per_rank,
            leaves,
            cost_flat: tree.flatten(),
            manifest: tree.manifest(),
            n_slots: tree.n_slots(),
            kv,
        })
    }

    fn check_groups(&self, groups: usize) -> Result<(), String> {
        if groups != self.parallel.attn_dp as usize {
            return Err(format!(
                "{}: the batch has {groups} attention-DP groups, the arch {}",
                self.name, self.parallel.attn_dp
            ));
        }
        Ok(())
    }

    fn eval(
        &self,
        views: &StepViews,
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
        let ep = self.parallel.moe_ep;
        let par = &self.parallel;
        let factor = par.region_device_factor;
        let calibrated_launch = par.launch_model == LaunchModel::Calibrated;
        let mut group_regions = vec![0u64; self.parallel.attn_dp as usize];
        let mut shared_regions = 0u64;
        for (slot, plan) in self.leaves.iter().enumerate() {
            let regions = if matches!(plan.eval, LeafEval::Launch { .. }) {
                let total = group_regions.iter().copied().max().unwrap_or(0) + shared_regions;
                u32::try_from(total).unwrap_or(u32::MAX)
            } else {
                0
            };
            // Per call: f_g x device, floored by the region's launch.
            let floor = if calibrated_launch && plan.compute {
                plan.regions_per_call as f32
                    * if plan.coll {
                        par.launch_coll_ms
                    } else {
                        par.launch_plain_ms
                    }
            } else {
                0.0
            };
            let compute = plan.compute;
            let adj = move |mut x: LeafMetrics| {
                if compute {
                    x.m.time_ms = (x.m.time_ms * factor).max(floor);
                }
                x
            };
            let mut out = eval_leaf(plan, views, &self.buckets, ep, regions, record, &adj);
            if calibrated_launch && matches!(plan.eval, LeafEval::Launch { .. }) {
                // The per-step constant; the regions already carry their launch.
                out.m = LeafMetrics::ZERO;
                out.m.m.time_ms = par.step_overhead_ms;
            }
            let launched =
                u64::from(out.calls) * u64::from(plan.regions_per_call) * u64::from(plan.fold);
            match plan.eval.group() {
                Some(d) => group_regions[d] += launched,
                None => shared_regions += launched,
            }
            slots[slot] = out.m;
            if let (Some(v), Some(input)) = (inputs.as_deref_mut(), out.input) {
                v.push(input);
            }
        }
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

fn iter_views(batch: &UnifiedArchInput, attn_dp: usize) -> StepViews {
    let mut target: Vec<GroupStep> = batch
        .groups
        .iter()
        .map(|g| GroupStep {
            decode_reqs: g.decode_kv_lens.len() as u32,
            query_len: 1,
            decode_max_kv: g.decode_kv_lens.iter().copied().max().unwrap_or(0),
            chunks: g.prefill_chunk_pairs.clone(),
            requests: g.request_count(),
        })
        .collect();
    if target.len() == 1 && attn_dp > 1 {
        target = spread_over_groups(&target[0], attn_dp);
    }
    StepViews {
        recurrent: Vec::new(),
        target,
    }
}

/// A worker without attention-DP partitions (the PD prefill worker) hands one
/// group; its requests are dealt round-robin over the DP groups, prefill
/// chunks largest first, as the engine's DP placement would.
fn spread_over_groups(one: &GroupStep, attn_dp: usize) -> Vec<GroupStep> {
    let mut groups = vec![
        GroupStep {
            query_len: one.query_len,
            decode_max_kv: one.decode_max_kv,
            ..Default::default()
        };
        attn_dp
    ];
    for i in 0..one.decode_reqs as usize {
        groups[i % attn_dp].decode_reqs += 1;
    }
    let mut chunks = one.chunks.clone();
    chunks.sort_by(|a, b| b.1.cmp(&a.1));
    for (i, chunk) in chunks.into_iter().enumerate() {
        groups[i % attn_dp].chunks.push(chunk);
    }
    for g in &mut groups {
        g.requests = g.decode_reqs + g.chunks.len() as u32;
        if g.decode_reqs == 0 {
            g.decode_max_kv = 0;
        }
    }
    groups
}

fn speculative_views(batch: &SpeculativeArchInput) -> StepViews {
    let target: Vec<GroupStep> = batch
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
            requests: g.request_count(),
        })
        .collect();
    let recurrent = target
        .iter()
        .map(|g| GroupStep {
            decode_reqs: g.decode_reqs,
            query_len: 1,
            decode_max_kv: g.decode_max_kv + 1,
            chunks: Vec::new(),
            requests: g.decode_reqs,
        })
        .collect();
    StepViews { target, recurrent }
}

/// Qwen4Exp BF16 on Trainium2 without MTP drafting.
pub struct Qwen4ExpNeuronBf16Model {
    pub core: NeuronFlashNextCore,
}

impl IterwiseUnifiedModel for Qwen4ExpNeuronBf16Model {
    /// QSA's per-row top-k budget and the PLE short conv's history make the
    /// necessary work per-request in context, which a decode KV total cannot
    /// recover.
    fn logs_decode_kv_lens(&self) -> bool {
        true
    }

    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        if batch.groups.len() == 1 {
            return Ok(()); // spread over the DP groups (spread_over_groups)
        }
        self.core.check_groups(batch.groups.len())
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.core.eval(
            &iter_views(batch, self.core.parallel.attn_dp as usize),
            slots,
            scratch,
            None,
        )
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.core.eval(
            &iter_views(batch, self.core.parallel.attn_dp as usize),
            slots,
            scratch,
            Some(inputs),
        )
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.core.manifest.clone()
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.core.kv.clone()
    }

    fn gpus_per_replica(&self) -> u16 {
        self.core.parallel.ranks() as u16
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.core.parallel.attn_dp as u16
    }
}

/// Qwen4Exp BF16 on Trainium2 with the checkpoint's MTP layer drafting.
pub struct Qwen4ExpNeuronBf16SpeculativeModel {
    pub core: NeuronFlashNextCore,
}

impl SpeculativeUnifiedModel for Qwen4ExpNeuronBf16SpeculativeModel {
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
            .eval(&speculative_views(batch), slots, scratch, None)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.core
            .eval(&speculative_views(batch), slots, scratch, Some(inputs))
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
        self.core.parallel.ranks() as u16
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.core.parallel.attn_dp as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::{
        ArchGroupInput, SpeculativeArchGroupInput, SpeculativeDecodeInput,
    };
    use std::path::Path;

    fn model() -> Qwen4ExpModelCfg {
        Qwen4ExpModelCfg::from_bf16_json(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("model/config/qwen3_8_flash_next_bf16.json"),
        )
        .unwrap()
    }

    fn parallel(attn_dp: u32, attn_tp: u32, moe_ep: u32) -> NeuronFlashNextParallel {
        NeuronFlashNextParallel {
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            max_model_len: 131072,
            lnc: 2,
            attn_dp,
            attn_tp,
            gdn_tp: attn_tp,
            moe_ep,
            moe_tp: 1,
            draft_tokens: 0,
            tp_collective: TpCollective::AllReduce,
            moe_dispatch: MoeDispatch::AllGather,
            region_granularity: RegionGranularity::Mixer,
            region_device_factor: 1.0,
            launch_model: LaunchModel::L1Additive,
            launch_coll_ms: 1.25,
            launch_plain_ms: 0.075,
            step_overhead_ms: 4.0,
            prefill_buckets: vec![128, 256, 512, 1024, 2048, 4096, 8192],
            decode_buckets: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
            context_buckets: vec![4096, 8192, 16384, 32768, 65536, 131072],
            ple_table_shards: PleTableShards::Replica,
            hbm_per_core_gb: 25.769_803_776,
            hbm_reserve_gb: 2.0,
            impls: NeuronFlashNextImpls::default(),
        }
    }

    /// Structure-only builds: one DP group of 8 ranks does not fit the
    /// weights, so these lift the HBM check.
    fn built(mut p: NeuronFlashNextParallel) -> NeuronFlashNextCore {
        p.hbm_per_core_gb = 1e6;
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        NeuronFlashNextCore::build("unified".to_string(), &model(), p, &bridge).unwrap()
    }

    fn decode_group(b: usize, kv: u32) -> ArchGroupInput {
        ArchGroupInput {
            batch_tokens: b as u32,
            decode_tokens: b as u32,
            decode_kv_lens: vec![kv; b],
            total_kv_len: kv * b as u32,
            ..Default::default()
        }
    }

    /// The region count the launch leaf is evaluated at (its slot input).
    fn launched_regions(
        core: &NeuronFlashNextCore,
        groups: Vec<ArchGroupInput>,
    ) -> serde_json::Value {
        let model = Qwen4ExpNeuronBf16Model {
            core: built_like(core),
        };
        let (mut slots, mut scratch, mut inputs) = (Vec::new(), Vec::new(), Vec::new());
        model.eval_iter_with_inputs(
            &UnifiedArchInput {
                groups,
                tokens_per_source_rank: Vec::new(),
            },
            &mut slots,
            &mut scratch,
            &mut inputs,
        );
        assert_eq!(inputs.len(), model.core.n_slots, "one input per slot");
        serde_json::to_value(inputs.last().unwrap()).unwrap()
    }

    /// A probe that costs 1 ms per call and records its shape.
    struct Fake;
    impl<X: Copy + Send + Sync + serde::Serialize> ShapeProbe<X> for Fake {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn config(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        fn eval_at(&self, _: X) -> LeafMetrics {
            let mut m = LeafMetrics::ZERO;
            m.m.time_ms = 1.0;
            m
        }
        fn input_at(&self, _: X) -> SlotInput {
            SlotInput::from(crate::timing::kernels::RegionLaunchOverheadKernelInput {
                num_regions: 0,
            })
        }
    }

    /// The same compiled plan with every kernel replaced by [`Fake`], except
    /// the launch leaf, whose recorded input carries the region count.
    fn built_like(core: &NeuronFlashNextCore) -> NeuronFlashNextCore {
        struct Regions;
        impl ShapeProbe<u32> for Regions {
            fn kind(&self) -> &'static str {
                "region_launch_overhead"
            }
            fn config(&self) -> serde_json::Value {
                serde_json::Value::Null
            }
            fn eval_at(&self, _: u32) -> LeafMetrics {
                LeafMetrics::ZERO
            }
            fn input_at(&self, num_regions: u32) -> SlotInput {
                SlotInput::from(crate::timing::kernels::RegionLaunchOverheadKernelInput {
                    num_regions,
                })
            }
        }
        let mut c = built(core.parallel.clone());
        let rows: Rows = Arc::new(Fake);
        let pair: Pair = Arc::new(Fake);
        let comm: Comm = Arc::new(Fake);
        for leaf in &mut c.leaves {
            match &mut leaf.eval {
                LeafEval::Tokens { k, .. }
                | LeafEval::Decode { k, .. }
                | LeafEval::Chunks { k, .. } => *k = rows.clone(),
                LeafEval::QsaDecode { k, .. }
                | LeafEval::QsaChunks { k, .. }
                | LeafEval::Ngram { k, .. }
                | LeafEval::ConvDecode { k, .. }
                | LeafEval::ConvChunks { k, .. } => *k = pair.clone(),
                LeafEval::Comm { k, .. } => *k = comm.clone(),
                LeafEval::Launch { k } => *k = Arc::new(Regions),
            }
        }
        c
    }

    /// Catches a region miscount: a decode step at mixer granularity launches
    /// embed, 48 layers of (attn HC, mixer regions, mlp HC, MoE block), PLE's
    /// ids + post + its pending combine, and the three epilogue regions.
    #[test]
    fn decode_step_launches_the_engines_regions() {
        let core = built(parallel(1, 8, 8));
        // 1 embed + layer 0 (4) + layer 1 with PLE (7) + 34 GDN (4 each)
        // + 12 QSA (5 each: mixer2 is two regions) + 3 epilogue.
        let expected = 1 + 4 + 7 + 34 * 4 + 12 * 5 + 3;
        assert_eq!(
            launched_regions(&core, vec![decode_group(4, 4096)]),
            serde_json::json!({ "num_regions": expected })
        );
        let mut p = parallel(1, 8, 8);
        p.region_granularity = RegionGranularity::Layer;
        // One region per mixer call per layer, plus embed and three epilogue.
        assert_eq!(
            launched_regions(&built(p), vec![decode_group(4, 4096)]),
            serde_json::json!({ "num_regions": 1 + 48 + 3 })
        );
    }

    /// Catches a granularity that keeps fused leaves (or the reverse).
    #[test]
    fn per_op_granularity_uses_op_level_kinds_only() {
        let mut p = parallel(4, 8, 32);
        p.region_granularity = RegionGranularity::PerOp;
        let kinds: Vec<String> = built(p)
            .manifest
            .slots
            .iter()
            .map(|s| s.kind.clone())
            .collect();
        for fused in [
            "gdn_decode_mixer",
            "qsa_mixer2",
            "moe_block",
            "hc_combine_mix",
            "ple_post",
            "lm_head_argmax",
        ] {
            assert!(!kinds.iter().any(|k| k == fused), "per_op kept {fused}");
        }
        for op in [
            "gdn_decode_core",
            "qsa_attention",
            "moe_routed_experts",
            "hc_combine",
            "ple_short_conv",
            "vocab_argmax",
        ] {
            assert!(kinds.iter().any(|k| k == op), "per_op lacks {op}");
        }
        let kinds: Vec<String> = built(parallel(4, 8, 32))
            .manifest
            .slots
            .iter()
            .map(|s| s.kind.clone())
            .collect();
        assert!(!kinds
            .iter()
            .any(|k| k == "qsa_attention" || k == "moe_routed_experts"));
    }

    /// Catches the knob validation letting an unbuildable layout through.
    #[test]
    fn layouts_are_validated() {
        let m = model();
        let mut p = parallel(1, 16, 16);
        assert!(p
            .validate(&m)
            .unwrap_err()
            .to_string()
            .contains("QSA heads"));
        p = parallel(2, 8, 8);
        assert!(p.validate(&m).unwrap_err().to_string().contains("moe_ep"));
        p = parallel(4, 8, 32);
        p.ple_table_shards = PleTableShards::Host;
        assert!(p
            .validate(&m)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
        p = parallel(4, 8, 32);
        p.lnc = 1;
        assert!(p.validate(&m).is_err());
        p = parallel(4, 8, 32);
        p.moe_dispatch = MoeDispatch::AllToAll;
        p.moe_ep = 16;
        p.moe_tp = 2;
        assert!(p.validate(&m).is_err());
    }

    /// Catches a weight count that would let an overfull rank build: 16 ranks
    /// hold 1/16 of 241 GB of experts and of the 102 GB PLE table.
    #[test]
    fn hbm_check_rejects_layouts_whose_weights_do_not_fit() {
        let m = model();
        let gb = |p: &NeuronFlashNextParallel| weight_bytes_per_rank(&m, p) as f64 / 1e9;
        let sixteen = gb(&parallel(2, 8, 16));
        let thirty_two = gb(&parallel(4, 8, 32));
        assert!((23.0..25.0).contains(&sixteen), "{sixteen}");
        assert!((11.0..14.0).contains(&thirty_two), "{thirty_two}");
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let err = NeuronFlashNextCore::build("unified".into(), &m, parallel(2, 8, 16), &bridge)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("exceed"), "{err}");
    }

    /// Pins the per-request and per-token state of one DP partition.
    #[test]
    fn kv_layout_counts_qsa_kv_indexer_and_recurrent_state() {
        let m = model();
        let p = parallel(4, 8, 32);
        let kv = kv_layout(&m, &p, 1);
        let part = |name: &str| {
            kv.parts
                .iter()
                .find(|x| x.name == name)
                .unwrap()
                .bytes_per_entry
        };
        // 12 QSA layers x 8 ranks x 1 KV head x 256 x (K, V) x 2 B.
        assert_eq!(part("qsa_kv"), 12 * 8 * 256 * 2 * 2);
        // 36 GDN layers x (48 x 128 x 128 fp32 + 10240 x 3 bf16).
        assert_eq!(part("gdn_state"), 36 * (48 * 128 * 128 * 4 + 10240 * 3 * 2));
        assert_eq!(kv.engine.fixed_device_bytes, 8);
    }

    #[test]
    fn bucket_calls_pad_and_split() {
        let mut out = Vec::new();
        bucket_calls(300, &[128, 512], &mut out);
        assert_eq!(out, vec![512]);
        out.clear();
        bucket_calls(1100, &[128, 512], &mut out);
        assert_eq!(out, vec![512, 512, 128]);
    }

    #[test]
    fn collectives_follow_the_strategy() {
        let kinds = |p: NeuronFlashNextParallel| -> Vec<String> {
            built(p)
                .manifest
                .slots
                .iter()
                .map(|s| s.name.clone())
                .collect()
        };
        let mut p = parallel(4, 8, 32);
        p.tp_collective = TpCollective::ReduceScatterAllGather;
        let names = kinds(p);
        assert!(names.iter().any(|n| n.ends_with("gdn.reduce_scatter")));
        assert!(!names.iter().any(|n| n.ends_with("gdn.all_reduce")));
        let mut p = parallel(4, 8, 32);
        p.moe_dispatch = MoeDispatch::AllToAll;
        let names = kinds(p);
        assert!(names.iter().any(|n| n.ends_with("moe.dispatch")));
        assert!(!names.iter().any(|n| n.ends_with("moe.block")));
        assert_eq!(neuron_fabric(8, 1, 4), Fabric::NeuronC4);
        assert_eq!(neuron_fabric(2, 1, 4), Fabric::NeuronC2);
        assert_eq!(neuron_fabric(4, 8, 4), Fabric::NeuronC1);
    }

    /// Catches the MTP passes not reading the verify rows.
    #[test]
    fn speculative_model_adds_mtp_passes() {
        let mut p = parallel(1, 8, 8);
        p.draft_tokens = 3;
        let core = built(p);
        assert!(core.manifest.slots.iter().any(|s| s.kind == "single_gemm"));
        let model = Qwen4ExpNeuronBf16SpeculativeModel {
            core: built_like(&core),
        };
        let batch = SpeculativeArchInput {
            draft_tokens: 3,
            groups: vec![SpeculativeArchGroupInput {
                batch_tokens: 16,
                decode_tokens: 16,
                decode_requests: vec![
                    SpeculativeDecodeInput {
                        kv_len: 4096,
                        query_len: 4
                    };
                    4
                ],
                ..Default::default()
            }],
            tokens_per_source_rank: Vec::new(),
        };
        assert!(model.check_input(&batch).is_ok());
        let (mut slots, mut scratch) = (Vec::new(), Vec::new());
        model.eval_speculative_iter(&batch, &mut slots, &mut scratch);
        assert_eq!(slots.len(), model.core.n_slots);
    }

    /// Catches a location rename or a new leaf that the necessary-work map
    /// (`model/work/location_maps/qwen4_exp_neuron_bf16_unified.json`) misses:
    /// the analyzer attributes R6 per location only on an exact cover.
    #[test]
    fn necessary_work_map_covers_the_default_layout() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/work/location_maps/qwen4_exp_neuron_bf16_unified.json");
        let map: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(
            map["arch_types"],
            serde_json::json!(["qwen4_exp_neuron_bf16"])
        );
        let mapped: std::collections::BTreeSet<String> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap().to_string())
            .collect();
        let comm = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"];
        for attn_dp in [1, 4] {
            let locations: std::collections::BTreeSet<String> =
                built(parallel(attn_dp, 8, 8 * attn_dp))
                    .manifest
                    .slots
                    .into_iter()
                    .filter(|slot| !comm.contains(&slot.kind.as_str()))
                    .map(|slot| slot.name)
                    .collect();
            assert_eq!(mapped, locations, "attn_dp {attn_dp}");
        }
    }

    /// Catches a one-group batch (the PD prefill worker) costing every request
    /// on DP group 0 while the other groups idle.
    #[test]
    fn a_one_group_batch_is_spread_over_the_dp_groups() {
        let one = GroupStep {
            decode_reqs: 5,
            query_len: 1,
            decode_max_kv: 900,
            chunks: vec![(0, 100), (0, 2048), (0, 512)],
            requests: 8,
        };
        let groups = spread_over_groups(&one, 4);
        assert_eq!(
            groups.iter().map(|g| g.decode_reqs).collect::<Vec<_>>(),
            [2, 1, 1, 1]
        );
        assert_eq!(groups[0].chunks, [(0, 2048)]);
        assert_eq!(groups[3].chunks, []);
        assert_eq!(groups.iter().map(|g| g.requests).sum::<u32>(), 8);
    }

    /// Catches the calibrated launch model not flooring a region by its launch
    /// or not adding the per-step constant: with every fake region at 1 ms
    /// device and f_g = 0.5, a region costs max(0.5, L_r).
    #[test]
    fn calibrated_launch_floors_each_region_and_adds_the_step_constant() {
        let mut p = parallel(1, 8, 8);
        p.launch_model = LaunchModel::Calibrated;
        p.region_device_factor = 0.5;
        p.launch_coll_ms = 2.0;
        p.launch_plain_ms = 0.1;
        p.step_overhead_ms = 4.0;
        let core = built_like(&built(p));
        let views = iter_views(
            &UnifiedArchInput {
                groups: vec![decode_group(4, 4096)],
                tokens_per_source_rank: Vec::new(),
            },
            1,
        );
        let (mut slots, mut scratch) = (Vec::new(), Vec::new());
        core.eval(&views, &mut slots, &mut scratch, None);
        for (plan, slot) in core.leaves.iter().zip(&slots) {
            if matches!(plan.eval, LeafEval::Launch { .. }) {
                assert_eq!(slot.m.time_ms, 4.0);
            } else if plan.compute && plan.regions_per_call > 0 && slot.m.time_ms > 0.0 {
                let floor = plan.regions_per_call as f32 * if plan.coll { 2.0 } else { 0.1 };
                assert!(
                    slot.m.time_ms >= floor.max(0.5) - 1e-6,
                    "{}",
                    slot.m.time_ms
                );
            }
        }
        // The decode mixer and the MoE block trace collectives, the HC modules do not.
        let names = &core.manifest.slots;
        let coll = |suffix: &str| {
            core.leaves
                .iter()
                .zip(names)
                .find(|(_, s)| s.name.ends_with(suffix))
                .map(|(l, _)| l.coll)
                .unwrap()
        };
        assert!(coll("gdn.gdn.decode_mixer") && coll("gdn.moe.block"));
        assert!(!coll("gdn.attn_hc"));
    }
}
