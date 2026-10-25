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
//! epilogue     final HC combine + mixer, lm_head + greedy argmax (vocab-parallel),
//!              the (value, index) gather in its own region
//! mtp          (speculative) the MTP layer, k passes per side: the first over
//!              the target rows, the rest over one row per request
//! launch       one leaf per forward: its host and launch time beyond its leaves
//! ```
//!
//! Step structure (`step_forwards`, default `separate`, the engine's): a step
//! is a decode forward (+ the MTP catch-up over the verify rows and `k - 1`
//! draft passes) then a prefill forward (+ the MTP over the chunk and `k - 1`
//! first-draft passes), separate compiled programs; either may be absent.
//! `fused` runs one forward over every row instead (not in the engine).
//!
//! Shapes are bucketed and SPMD (`spmd_padding`, default on): every DP group
//! runs the forward's padded shape, idle groups included. The decode forward
//! runs every group at the bucket of the largest group batch x the verify
//! width; the prefill forward at the bucket of the longest chunk, from the
//! furthest chunk start (its context bucket is that of the longest start +
//! T). Token-wise regions run at exactly those rows. The MoE block and PLE
//! all-gather every group's rows over the replica and reduce-scatter them
//! back; a group batch that does not split over its TP ranks is gathered from
//! each of them ("dup", 8x the rows). Under `reduce_scatter_all_gather` a
//! forward runs sequence-parallel only when its rows split over the TP group,
//! else the all-reduce form. All-to-all dispatch sends each destination S
//! rows (S = a rank's token slice; exact capacity, so the experts see the
//! all-gather's rows) or `min(S, ceil(cf S top_k / ranks))` with a capacity
//! factor, and pays the engine's packing per received row.
//!
//! Region granularity (`region_granularity`), cut as `program._keys`:
//! - `per_op`: every op is its own region (op-level L1 leaves).
//! - `mixer`: [embed], [PLE], per block [HC mix + mixer + collective +
//!   combine], [HC mix + MoE + collectives + combine], QSA front | back.
//! - `layer`: one region per layer, QSA layers split front | back: 63 per
//!   target forward with the head and its gather.
//! - `block`: cut only at QSA front | back and the head gather: 14.
//!
//! Composition (`launch_model = pipelined`, default; flashnext-neuron
//! `engine/README.md` with the NRT execution barrier off): every compute leaf
//! is its L1 row times f_g (1.0). A forward costs `max(sum over regions of
//! max(device_r + region_gap_ms, region_host_ms), regions x region_host_ms +
//! forward_overhead_ms)`: the host dispatches each region in
//! `region_host_ms` and does its per-forward work (`forward_overhead_ms`;
//! `mtp_forward_overhead_ms` for an MTP pass) while the device runs; each
//! launch costs the device `region_gap_ms` outside the region, and a region
//! shorter than its dispatch waits for the next launch. The step adds
//! `step_overhead_ms`. Each forward's launch leaf holds everything beyond the
//! sum of its leaves. `calibrated` (per region `max(device, L_r)`) and
//! `l1_additive` (the `region_launch_overhead` row per forward) remain.
//!
//! What-if scaling (`kind_time_scale`): entries `kind=factor` or
//! `kind@min_rows=factor` multiply the device time of every compute leaf of
//! that L1 kind (at calls of at least `min_rows` rows: tokens, chunk tokens or
//! decode rows) before f_g and the launch floor. Default none. It is for
//! sensitivity studies ("MoE prefill 3x faster" = `moe_block@256=0.333`), not
//! calibration; collectives and the launch leaf are never scaled.
//!
//! Not modeled: the host scheduler and sampling bookkeeping, KV block-table
//! upload, the engine's first execution of a shape after warm-up (1-3 s per
//! run), its 16-request cap per DP group, the vision tower.

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
    /// Cut only at the QSA front | back boundary: [embed .. L3 front],
    /// [L3 back .. L7 front], ..., [L47 back .. head] and the head gather.
    Block,
}

impl RegionGranularity {
    pub const CHOICES: [&'static str; 4] = ["per_op", "mixer", "layer", "block"];

    /// f_g: step device time over the sum of the isolated L1 rows of its
    /// ops. On the full 48-layer engine the prefill forward at layer and block
    /// granularity is the sum of its rows within -5% (T = 512) .. +3%
    /// (T = 2048), so 1.0 (`logs/engine/full_prefill_bench.jsonl`, serving
    /// prefill forwards); decode is set by the region floors instead.
    pub fn calibrated_device_factor(self) -> f32 {
        1.0
    }

    /// Host time to dispatch one region (input assembly + NRT launch) on the
    /// 32-rank engine with the NRT execution barrier off: op granularity is
    /// host-bound at 0.55 ms per region (`logs/engine/full_decode_bench.jsonl`);
    /// larger regions take more inputs. The device of a region holding a
    /// replica-wide collective is floored at it, because that collective waits
    /// for the slowest rank's launch.
    pub fn region_host_ms(self) -> f32 {
        match self {
            // 0.55 ms per engine op region (457 per decode step); the Sim's
            // per_op leaves split the engine's ops further (~820 regions).
            Self::PerOp => 0.3,
            Self::Mixer => 1.0,
            Self::Layer => 1.3,
            Self::Block => 3.7,
        }
    }
}

/// How one engine step runs its decode rows and prefill chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepForwards {
    /// The flashnext engine: a decode forward (+ its MTP passes), then a
    /// prefill forward (+ its MTP passes), as separate compiled programs.
    #[default]
    Separate,
    /// One fused forward over the decode rows and prefill chunks (not in the
    /// engine; a what-if that shares the weight streaming).
    Fused,
}

impl StepForwards {
    pub const CHOICES: [&'static str; 2] = ["separate", "fused"];
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
    /// The engine with the NRT execution barrier off: a forward costs
    /// `max(sum of region device, regions x region_host_ms +
    /// forward_overhead_ms)` (the host's dispatch and per-forward work overlap
    /// the device), where a region holding a collective across DP groups has
    /// device time of at least `region_host_ms`; the step adds
    /// `step_overhead_ms`.
    #[default]
    Pipelined,
    /// Per region `max(f_g * device, L_r)` plus a per-step constant (the
    /// barrier-on engine-slice fit).
    Calibrated,
    /// The measured `region_launch_overhead` row at each forward's region
    /// count, added to the device time.
    L1Additive,
}

impl LaunchModel {
    pub const CHOICES: [&'static str; 3] = ["pipelined", "calibrated", "l1_additive"];
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
                    anyhow::anyhow!(
                        "kind_time_scale entry {entry:?}: min_rows {rows:?} is not an integer"
                    )
                })?,
            ),
            None => (lhs, 0),
        };
        let factor: f32 = factor.trim().parse().map_err(|_| {
            anyhow::anyhow!("kind_time_scale entry {entry:?}: factor {factor:?} is not a number")
        })?;
        let kind = kind.trim();
        ensure!(
            !kind.is_empty(),
            "kind_time_scale entry {entry:?} names no kind"
        );
        ensure!(
            factor.is_finite() && factor > 0.0,
            "kind_time_scale entry {entry:?}: factor must be positive"
        );
        Ok(Self {
            kind: kind.to_string(),
            min_rows,
            factor,
        })
    }
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
    /// Separate decode and prefill forwards (the engine) or one fused forward.
    pub step_forwards: StepForwards,
    /// Every DP group runs the forward's padded shape (SPMD), idle or not.
    pub spmd_padding: bool,
    /// Device factor applied to every compute leaf (f_g).
    pub region_device_factor: f32,
    pub launch_model: LaunchModel,
    pub launch_coll_ms: f32,
    pub launch_plain_ms: f32,
    /// Pipelined launch: host dispatch per region and the device floor of a
    /// region holding a replica-wide collective (ms).
    pub region_host_ms: f32,
    /// Pipelined launch: device time per region launch outside the regions'
    /// own executions (NEFF switch, launch skew, helper copies; ms).
    pub region_gap_ms: f32,
    /// Pipelined launch: host work per target forward (input assembly, H2D,
    /// id readback, helper executions), overlapping the device, and the same
    /// per MTP forward (ms).
    pub forward_overhead_ms: f32,
    pub mtp_forward_overhead_ms: f32,
    pub step_overhead_ms: f32,
    /// All-to-all dispatch: rows each rank sends each destination are
    /// `min(S, ceil(cf * S * top_k / ranks))`; 0 = exact (S, no drops).
    pub moe_a2a_capacity_factor: f32,
    /// All-to-all: the engine's slot packing (dispatch) and scatter-add
    /// (combine) per received row, on each of the two all-to-all leaves (µs).
    /// Fitted, not an L1 row.
    pub moe_a2a_pack_us_per_row: f32,
    pub prefill_buckets: Vec<u32>,
    pub decode_buckets: Vec<u32>,
    pub context_buckets: Vec<u32>,
    pub ple_table_shards: PleTableShards,
    /// Per-core HBM the build checks the weights against (GB).
    pub hbm_per_core_gb: f64,
    /// Per-core runtime and activation scratch kept out of the KV budget (GB).
    pub hbm_reserve_gb: f64,
    pub impls: NeuronFlashNextImpls,
    /// What-if device-time scales per L1 kind (empty = as measured).
    pub kind_time_scale: Vec<KindTimeScale>,
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
                && self.region_host_ms >= 0.0
                && self.region_gap_ms >= 0.0
                && self.forward_overhead_ms >= 0.0
                && self.mtp_forward_overhead_ms >= 0.0
                && self.step_overhead_ms >= 0.0,
            "region_device_factor must be positive and the launch constants nonnegative"
        );
        ensure!(
            self.moe_a2a_capacity_factor >= 0.0 && self.moe_a2a_capacity_factor.is_finite(),
            "moe_a2a_capacity_factor must be >= 0 (0 = exact capacity)"
        );
        ensure!(
            self.moe_a2a_pack_us_per_row >= 0.0,
            "moe_a2a_pack_us_per_row must be nonnegative"
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

/// One attention-DP group's share of one pass of a forward.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    fn is_empty(&self) -> bool {
        self.decode_reqs == 0 && self.chunks.is_empty()
    }
}

/// The pass a leaf belongs to. `Fused*` are the one-forward step; the rest
/// are the engine's decode forward and prefill forward and their MTP passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    /// The fused forward's target pass: decode rows and prefill chunks.
    Fused,
    /// MTP pass 0 over every target row of the fused forward.
    MtpFused,
    /// The decode forward's target pass (the verify rows under MTP).
    Decode,
    /// MTP catch-up over the decode forward's verify rows.
    MtpDecode,
    /// An MTP draft pass: one row per decode request.
    MtpRecurrent,
    /// The prefill forward's target pass.
    Prefill,
    /// MTP over the prefill chunks.
    MtpPrefill,
    /// A first-draft pass after a prompt's last chunk: one row per group.
    MtpPrefillDraft,
}

const N_VIEWS: usize = 8;

impl View {
    fn index(self) -> usize {
        self as usize
    }
}

/// The group shapes of every pass of one step.
struct StepViews {
    groups: [Vec<GroupStep>; N_VIEWS],
    /// Shapes are already padded to their buckets (the separate forwards):
    /// token-wise regions run at exactly the group's rows. Otherwise (fused)
    /// token-wise regions pad to the union of the token buckets.
    exact: bool,
}

impl StepViews {
    fn get(&self, view: View) -> &[GroupStep] {
        &self.groups[view.index()]
    }
}

// ── leaf plans ───────────────────────────────────────────────────────────────

/// Which rows a token-wise leaf or a collective sees.
#[derive(Clone, Copy, Debug)]
enum RowSrc {
    /// The group's tokens.
    Group(usize),
    /// Every group's tokens (a replicated, dp = 1 MoE block sees its group's).
    All,
    /// The rows a replica-wide all-gather delivers to each rank: every group's
    /// tokens, each group's whole batch per rank when its rows do not split
    /// over its TP ranks (flashnext `world_gather_rows`, "dup").
    WorldGather,
    /// The largest group's tokens.
    MaxGroup,
    /// The group's lm_head rows.
    Sample(usize),
    /// All-to-all: the token slice a rank routes (S), or under the
    /// all-gather fallback every rank's rows.
    A2aRouter,
    /// All-to-all: the rows a rank's experts receive (ranks x capacity).
    A2aRecv,
    /// All-to-all: the rows the local shared expert runs (S, or the group's
    /// rows under the fallback).
    A2aShared,
}

/// When a leaf runs under `reduce_scatter_all_gather`: the engine applies
/// sequence parallelism only to a forward whose rows split over the TP group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpCond {
    Always,
    /// Only with sequence parallelism active.
    Sp,
    /// Only without it (the all-reduce form).
    NoSp,
}

#[derive(Clone)]
enum LeafEval {
    /// A token-wise op at `rows / div` rows; `sp` divides by the group size
    /// when sequence parallelism is active.
    Tokens {
        k: Rows,
        src: RowSrc,
        div: u32,
        sp: bool,
    },
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
    /// The launch and host cost of forward `forward`, or (None) the
    /// per-step host constant.
    Launch { k: Rows, forward: Option<usize> },
}

#[derive(Clone)]
struct LeafPlan {
    eval: LeafEval,
    view: View,
    /// The forward this leaf runs in.
    forward: usize,
    /// The compiled region (one per forward-local cut) this leaf runs in.
    region: usize,
    /// The DP group whose per-group part minted this leaf (None: the MoE
    /// block and the launch leaves, outside the per-group Max).
    group: Option<usize>,
    /// Product of the enclosing `Scale` folds (region instances).
    fold: u32,
    /// A compute leaf: scaled by f_g.
    compute: bool,
    /// Runs only with / without sequence parallelism.
    sp_cond: SpCond,
    /// What-if `(min_rows, factor)` scales of this leaf's kind.
    scales: Vec<(u32, f32)>,
}

impl LeafPlan {
    /// The what-if scale of one call of `rows` rows.
    fn scale_at(&self, rows: u32) -> f32 {
        self.scales
            .iter()
            .filter(|(min_rows, _)| rows >= *min_rows)
            .map(|(_, f)| f)
            .product()
    }
}

/// One forward of the step: its passes' views and its regions.
#[derive(Clone, Debug)]
struct ForwardPlan {
    /// The pass whose shapes decide whether the forward runs.
    view: View,
    mtp: bool,
    /// Region launches of one instance of the forward.
    regions: u32,
    /// Instances of the forward per step (its Scale fold).
    fold: u32,
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

/// A point where the engine's granularity may start a new compiled region
/// (flashnext `program._keys`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cut {
    /// The embedding.
    Embed,
    /// A decoder layer (its first leaf: the previous mlp module's combine).
    Layer,
    /// The PLE block before layer 1's attention module.
    Ple,
    /// An attention module: HC mix + mixer (+ TP collective + combine).
    Attn,
    /// A QSA layer's back region: attention + o_proj onward.
    QsaBack,
    /// An mlp module: HC mix + MoE (+ collectives + combine).
    Mlp,
    /// The final HC mixer and lm_head.
    Head,
    /// The head's gather + argmax combine (its own region in the engine).
    HeadGather,
}

/// Which pass a view's leaves run: its target phase rows.
fn has_decode(view: View) -> bool {
    !matches!(view, View::Prefill | View::MtpPrefill)
}

fn has_prefill(view: View) -> bool {
    matches!(
        view,
        View::Fused | View::MtpFused | View::Prefill | View::MtpPrefill
    )
}

struct PlanBuilder<'a> {
    name: String,
    b: CostTreeBuilder,
    leaves: Vec<LeafPlan>,
    forwards: Vec<ForwardPlan>,
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
    /// The forward being built, its own fold and whether it is an MTP pass.
    forward: usize,
    forward_fold: u32,
    in_mtp: bool,
    /// The region the next leaf joins, and whether a cut is pending.
    region: usize,
    pending_cut: bool,
    /// Stamped on minted leaves.
    sp_cond: SpCond,
    /// The DP group being minted inside `per_group`.
    group: Option<usize>,
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

    /// Run `f` with leaves stamped `cond`.
    fn when<T>(&mut self, cond: SpCond, f: impl FnOnce(&mut Self) -> T) -> T {
        let saved = self.sp_cond;
        self.sp_cond = cond;
        let out = f(self);
        self.sp_cond = saved;
        out
    }

    /// Request a region boundary if the granularity cuts at `cut`.
    fn cut(&mut self, cut: Cut) {
        let cuts = match self.p.region_granularity {
            RegionGranularity::PerOp => false, // every leaf is a region
            RegionGranularity::Mixer => cut != Cut::Layer,
            RegionGranularity::Layer => matches!(
                cut,
                Cut::Embed | Cut::Layer | Cut::QsaBack | Cut::Head | Cut::HeadGather
            ),
            RegionGranularity::Block => {
                cut == Cut::QsaBack || (cut == Cut::HeadGather && !self.in_mtp)
            }
        };
        self.pending_cut |= cuts;
    }

    fn new_region(&mut self) {
        self.region += 1;
        self.pending_cut = false;
        let instances = self.fold / self.forward_fold.max(1);
        self.forwards[self.forward].regions += instances;
    }

    fn leaf(&mut self, label: &str, eval: LeafEval) -> CostNode {
        let (kind, config) = match &eval {
            LeafEval::Tokens { k, .. }
            | LeafEval::Decode { k, .. }
            | LeafEval::Chunks { k, .. }
            | LeafEval::Launch { k, .. } => (k.kind(), k.config()),
            LeafEval::QsaDecode { k, .. }
            | LeafEval::QsaChunks { k, .. }
            | LeafEval::Ngram { k, .. }
            | LeafEval::ConvDecode { k, .. }
            | LeafEval::ConvChunks { k, .. } => (k.kind(), k.config()),
            LeafEval::Comm { k, .. } => (k.kind(), k.config()),
        };
        let launch = matches!(eval, LeafEval::Launch { .. });
        if !launch && (self.pending_cut || self.p.region_granularity == RegionGranularity::PerOp) {
            self.new_region();
        }
        let compute = !matches!(eval, LeafEval::Comm { .. } | LeafEval::Launch { .. });
        let scales = if compute {
            self.p
                .kind_time_scale
                .iter()
                .filter(|s| s.kind == kind)
                .map(|s| (s.min_rows, s.factor))
                .collect()
        } else {
            Vec::new()
        };
        self.leaves.push(LeafPlan {
            eval,
            view: self.view,
            forward: self.forward,
            region: self.region,
            group: self.group,
            fold: self.fold,
            compute,
            sp_cond: self.sp_cond,
            scales,
        });
        self.b.leaf(
            format!("{}.{}", self.name, join(&self.scope, label)),
            kind,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
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
        )))
    }

    fn g(&self) -> u32 {
        self.p.group_size()
    }

    fn h2(&self) -> u64 {
        u64::from(self.model.hidden) * BF16_BYTES
    }

    fn tokens(&mut self, label: &str, k: Rows, src: RowSrc, div: u32, sp: bool) -> CostNode {
        self.leaf(label, LeafEval::Tokens { k, src, div, sp })
    }

    /// The previous module's HC combine, a cut, then this module's HC mix
    /// (the engine's stage list: `*_combine`, then `*_mix`).
    fn combine_mix(&mut self, d: usize, label: &str, cut: Cut, out: &mut Vec<CostNode>) {
        let src = RowSrc::Group(d);
        let k = self.k.hc_combine.clone();
        out.push(self.tokens(&format!("{label}.combine"), k, src, 1, true));
        self.cut(cut);
        let k = self.k.hc_mix.clone();
        out.push(self.tokens(&format!("{label}.mix"), k, src, 1, true));
    }

    /// A mixer's compute leaves for group `d`, its TP collectives around it.
    #[allow(clippy::too_many_lines)]
    fn mixer(&mut self, d: usize, mixer: Qwen4ExpMixer, out: &mut Vec<CostNode>) -> Result<()> {
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
        let (dec_v, pre_v) = (has_decode(self.view), has_prefill(self.view));
        if sp {
            self.when(SpCond::Sp, |s| -> Result<()> {
                if let Some(n) = s.comm_leaf(
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
                Ok(())
            })?;
        }
        match mixer {
            Qwen4ExpMixer::Gdn => {
                if let (Some(dec), Some(pre)) = (
                    self.k.gdn_decode_mixer.clone(),
                    self.k.gdn_prefill_mixer.clone(),
                ) {
                    if dec_v {
                        out.push(self.leaf(
                            "gdn.decode_mixer",
                            LeafEval::Decode {
                                k: dec,
                                d,
                                sub,
                                rows: false,
                            },
                        ));
                    }
                    if pre_v {
                        out.push(
                            self.leaf("gdn.prefill_mixer", LeafEval::Chunks { k: pre, d, sub }),
                        );
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
                    if dec_v {
                        let dec = |k: Rows, rows: bool| LeafEval::Decode { k, d, sub, rows };
                        for (op, k, rows) in [
                            ("in_proj", in_proj.clone(), true),
                            ("conv1d", conv_d, false),
                            ("core", core_d, false),
                            ("output", output.clone(), true),
                            ("state_io", state_io, false),
                        ] {
                            out.push(self.leaf(&format!("gdn.decode.{op}"), dec(k, rows)));
                        }
                    }
                    if pre_v {
                        let ch = |k: Rows| LeafEval::Chunks { k, d, sub };
                        for (op, k) in [
                            ("in_proj", in_proj),
                            ("conv1d", conv_p),
                            ("split_gating", split),
                            ("core", core_p),
                            ("output", output),
                        ] {
                            out.push(self.leaf(&format!("gdn.prefill.{op}"), ch(k)));
                        }
                    }
                }
            }
            Qwen4ExpMixer::Qsa => {
                if let (Some(dec), Some(pre)) = (
                    self.k.qsa_mixer2_decode.clone(),
                    self.k.qsa_mixer2_prefill.clone(),
                ) {
                    // The two-region mixer: its leaf stands for front + back,
                    // charged to the front region.
                    if dec_v {
                        out.push(
                            self.leaf("qsa.decode_mixer2", LeafEval::QsaDecode { k: dec, d, sub }),
                        );
                    }
                    if pre_v {
                        out.push(
                            self.leaf("qsa.prefill_mixer2", LeafEval::QsaChunks { k: pre, d, sub }),
                        );
                    }
                    self.cut(Cut::QsaBack);
                } else {
                    let modes = [
                        ("decode", self.k.qsa_ops_decode.clone(), dec_v),
                        ("prefill", self.k.qsa_ops_prefill.clone(), pre_v),
                    ];
                    for (mode, ops, run) in modes {
                        if !run {
                            continue;
                        }
                        let ops = ops.expect("per-op QSA kernels built");
                        let rows: [(&str, Rows); 4] = [
                            ("project", ops.project.clone()),
                            ("indexer_project", ops.indexer_project.clone()),
                            ("cache_write", ops.cache_write.clone()),
                            ("indexer_compress", ops.indexer_compress.clone()),
                        ];
                        let ctx: [(&str, Pair); 3] = [
                            ("indexer_scores", ops.indexer_scores.clone()),
                            ("indexer_select", ops.indexer_select.clone()),
                            ("attention", ops.attention.clone()),
                        ];
                        let row_eval = |k: Rows| {
                            if mode == "decode" {
                                LeafEval::Decode {
                                    k,
                                    d,
                                    sub,
                                    rows: true,
                                }
                            } else {
                                LeafEval::Chunks { k, d, sub }
                            }
                        };
                        for (op, k) in rows {
                            out.push(self.leaf(&format!("qsa.{mode}.{op}"), row_eval(k)));
                        }
                        for (op, k) in ctx {
                            if op == "attention" {
                                self.cut(Cut::QsaBack);
                            }
                            let eval = if mode == "decode" {
                                LeafEval::QsaDecode { k, d, sub }
                            } else {
                                LeafEval::QsaChunks { k, d, sub }
                            };
                            out.push(self.leaf(&format!("qsa.{mode}.{op}"), eval));
                        }
                        let k = ops.output.clone();
                        out.push(self.leaf(&format!("qsa.{mode}.output"), row_eval(k)));
                    }
                }
            }
        }
        // The row-parallel output: per sub-replica partial sums.
        if sp {
            self.when(SpCond::Sp, |s| -> Result<()> {
                if let Some(n) = s.comm_leaf(
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
                Ok(())
            })?;
        }
        self.when(
            if sp { SpCond::NoSp } else { SpCond::Always },
            |s| -> Result<()> {
                if let Some(n) = s.comm_leaf(
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
                if let Some(n) = s.comm_leaf(
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
                Ok(())
            },
        )
    }

    /// The MoE block over every group's tokens, with its dispatch collectives.
    #[allow(clippy::too_many_lines)]
    fn moe(&mut self, out: &mut Vec<CostNode>) -> Result<()> {
        let r = self.p.ranks();
        let g = self.g();
        let h2 = self.h2();
        let sp = self.p.sequence_parallel();
        let dp = self.p.attn_dp;
        let no_sp = if sp { SpCond::NoSp } else { SpCond::Always };
        match self.p.moe_dispatch {
            MoeDispatch::AllGather => {
                let rows = if dp == 1 {
                    RowSrc::All
                } else {
                    RowSrc::WorldGather
                };
                // dp = 1 without sequence parallelism: replicated tokens, the
                // partials all-reduce. Otherwise all-gather the (sliced) rows.
                if dp > 1 || sp {
                    let cond = if dp > 1 { SpCond::Always } else { SpCond::Sp };
                    self.when(cond, |s| -> Result<()> {
                        if let Some(n) =
                            s.comm_leaf("moe.all_gather", CommOp::AllGather, r, 1, rows, h2, r)?
                        {
                            out.push(n);
                        }
                        Ok(())
                    })?;
                }
                if let Some(k) = self.k.moe_block.clone() {
                    out.push(self.tokens("moe.block", k, rows, 1, false));
                } else {
                    let router = self.k.moe_router.clone().expect("router");
                    let routed = self.k.moe_routed.clone().expect("routed");
                    let shared = self.k.moe_shared.clone().expect("shared");
                    out.push(self.tokens("moe.router", router, rows, 1, false));
                    out.push(self.tokens("moe.routed_experts", routed, rows, 1, false));
                    out.push(self.tokens("moe.shared_expert", shared, rows, 1, false));
                }
                if dp == 1 {
                    self.when(no_sp, |s| -> Result<()> {
                        if let Some(n) =
                            s.comm_leaf("moe.all_reduce", CommOp::AllReduce, r, 1, rows, h2, 1)?
                        {
                            out.push(n);
                        }
                        Ok(())
                    })?;
                }
                if dp > 1 || sp {
                    let cond = if dp > 1 { SpCond::Always } else { SpCond::Sp };
                    self.when(cond, |s| -> Result<()> {
                        if let Some(n) = s.comm_leaf(
                            "moe.reduce_scatter",
                            CommOp::ReduceScatter,
                            r,
                            1,
                            rows,
                            h2,
                            1,
                        )? {
                            out.push(n);
                        }
                        Ok(())
                    })?;
                }
                if dp > 1 {
                    self.when(no_sp, |s| -> Result<()> {
                        if let Some(n) = s.comm_leaf(
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
                        Ok(())
                    })?;
                }
            }
            MoeDispatch::AllToAll => {
                // flashnext `moe_a2a`: route the own token slice, send each
                // destination its rows (capacity per destination), experts on
                // every received row, send back, add the full shared expert.
                let k = u64::from(self.model.experts_per_token);
                let router = self.k.moe_router.clone().expect("router");
                let routed = self.k.moe_routed.clone().expect("routed");
                let shared = self.k.moe_shared.clone().expect("shared");
                out.push(self.tokens("moe.router", router, RowSrc::A2aRouter, 1, false));
                // Each received row carries its hidden state and its routing
                // (top-k weights and ids as fp32).
                if let Some(n) = self.comm_leaf(
                    "moe.dispatch",
                    CommOp::AllToAll,
                    r,
                    1,
                    RowSrc::A2aRecv,
                    h2 + 2 * k * FP32_BYTES,
                    1,
                )? {
                    out.push(n);
                }
                out.push(self.tokens("moe.routed_experts", routed, RowSrc::A2aRecv, 1, false));
                if let Some(n) = self.comm_leaf(
                    "moe.combine",
                    CommOp::AllToAll,
                    r,
                    1,
                    RowSrc::A2aRecv,
                    h2,
                    1,
                )? {
                    out.push(n);
                }
                out.push(self.tokens("moe.shared_expert", shared, RowSrc::A2aShared, 1, false));
                self.when(no_sp, |s| -> Result<()> {
                    if let Some(n) = s.comm_leaf(
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
                    Ok(())
                })?;
            }
        }
        Ok(())
    }

    /// The PLE block of group `d` (before its layer's attention module).
    /// Under sequence parallelism it still runs on the group's full rows.
    #[allow(clippy::too_many_lines)]
    fn ple(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        let r = self.p.ranks();
        let g = self.g();
        let sp = self.p.sequence_parallel();
        let no_sp = if sp { SpCond::NoSp } else { SpCond::Always };
        let ple_bytes = u64::from(self.model.ple_embed_dim) * BF16_BYTES;
        let ids_bytes =
            u64::from((self.model.ngram_size - 1) * self.model.heads_per_ngram) * FP32_BYTES;
        let ngram = self.k.ple_ngram.clone().expect("ple kernels");
        let gather = self.k.ple_gather.clone().expect("ple kernels");
        out.push(self.leaf("ple.ngram_ids", LeafEval::Ngram { k: ngram, d }));
        match self.p.ple_table_shards {
            PleTableShards::Replica => {
                let rows = if self.p.attn_dp == 1 {
                    RowSrc::All
                } else {
                    RowSrc::WorldGather
                };
                if let Some(n) = self.comm_leaf(
                    "ple.ids_all_gather",
                    CommOp::AllGather,
                    r,
                    1,
                    rows,
                    ids_bytes,
                    r,
                )? {
                    out.push(n);
                }
                out.push(self.tokens("ple.gather", gather, rows, 1, false));
                if let Some(n) = self.comm_leaf(
                    "ple.reduce_scatter",
                    CommOp::ReduceScatter,
                    r,
                    1,
                    rows,
                    ple_bytes,
                    1,
                )? {
                    out.push(n);
                }
                self.when(no_sp, |s| -> Result<()> {
                    if let Some(n) = s.comm_leaf(
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
                    Ok(())
                })?;
            }
            PleTableShards::DpGroup | PleTableShards::Host => {
                out.push(self.tokens("ple.gather", gather, RowSrc::Group(d), 1, false));
                if sp {
                    self.when(SpCond::Sp, |s| -> Result<()> {
                        if let Some(n) = s.comm_leaf(
                            "ple.group_reduce_scatter",
                            CommOp::ReduceScatter,
                            g,
                            1,
                            RowSrc::Group(d),
                            ple_bytes,
                            1,
                        )? {
                            out.push(n);
                        }
                        Ok(())
                    })?;
                }
                self.when(no_sp, |s| -> Result<()> {
                    if let Some(n) = s.comm_leaf(
                        "ple.group_reduce",
                        CommOp::AllReduce,
                        g,
                        1,
                        RowSrc::Group(d),
                        ple_bytes,
                        1,
                    )? {
                        out.push(n);
                    }
                    Ok(())
                })?;
            }
        }
        let src = RowSrc::Group(d);
        if let Some(k) = self.k.ple_post.clone() {
            out.push(self.tokens("ple.post", k, src, 1, false));
        } else {
            let project = self.k.ple_project.clone().expect("ple per-op");
            let gate = self.k.ple_gate.clone().expect("ple per-op");
            let residual = self.k.ple_residual.clone().expect("ple per-op");
            let conv_d = self.k.ple_conv_decode.clone().expect("ple per-op");
            let conv_p = self.k.ple_conv_prefill.clone().expect("ple per-op");
            out.push(self.tokens("ple.project", project, src, 1, false));
            out.push(self.tokens("ple.gate", gate, src, 1, false));
            if has_decode(self.view) {
                out.push(self.leaf(
                    "ple.short_conv_decode",
                    LeafEval::ConvDecode { k: conv_d, d },
                ));
            }
            if has_prefill(self.view) {
                out.push(self.leaf(
                    "ple.short_conv_prefill",
                    LeafEval::ConvChunks { k: conv_p, d },
                ));
            }
            out.push(self.tokens("ple.residual", residual, src, 1, false));
        }
        Ok(())
    }

    /// Embedding for group `d`: vocab-parallel gather + the group reduction.
    fn embed(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        self.cut(Cut::Embed);
        let k = self.k.embed.clone();
        out.push(self.tokens("embedding", k, RowSrc::Group(d), 1, false));
        let (g, h2) = (self.g(), self.h2());
        if self.p.sequence_parallel() {
            self.when(SpCond::Sp, |s| -> Result<()> {
                if let Some(n) = s.comm_leaf(
                    "embedding.reduce_scatter",
                    CommOp::ReduceScatter,
                    g,
                    1,
                    RowSrc::Group(d),
                    h2,
                    1,
                )? {
                    out.push(n);
                }
                Ok(())
            })?;
        }
        let cond = if self.p.sequence_parallel() {
            SpCond::NoSp
        } else {
            SpCond::Always
        };
        self.when(cond, |s| -> Result<()> {
            if let Some(n) = s.comm_leaf(
                "embedding.reduce",
                CommOp::AllReduce,
                g,
                1,
                RowSrc::Group(d),
                h2,
                1,
            )? {
                out.push(n);
            }
            Ok(())
        })
    }

    /// Final HC combine and mixer, lm_head and greedy argmax for group `d`.
    fn head(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        self.scoped("final_mixer", |s| {
            let k = s.k.hc_combine.clone();
            out.push(s.tokens("combine", k, RowSrc::Group(d), 1, true));
            s.cut(Cut::Head);
            let k = s.k.hc_mix_final.clone();
            out.push(s.tokens("mix", k, RowSrc::Group(d), 1, true));
        });
        self.scoped("head", |s| s.head_rows(d, out))
    }

    fn head_rows(&mut self, d: usize, out: &mut Vec<CostNode>) -> Result<()> {
        let (g, h2) = (self.g(), self.h2());
        if self.p.sequence_parallel() {
            self.when(SpCond::Sp, |s| -> Result<()> {
                if let Some(n) = s.comm_leaf(
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
                Ok(())
            })?;
        }
        if let Some(k) = self.k.lm_head_argmax.clone() {
            out.push(self.tokens("lm_head_argmax", k, RowSrc::Sample(d), 1, false));
        } else {
            let gemm = self.k.lm_head_gemm.clone().expect("per-op head");
            let argmax = self.k.vocab_argmax.clone().expect("per-op head");
            out.push(self.tokens("lm_head", gemm, RowSrc::Sample(d), 1, false));
            out.push(self.tokens("argmax", argmax, RowSrc::Sample(d), 1, false));
        }
        self.cut(Cut::HeadGather);
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

    /// `f` once per DP group (SPMD: each group's leaves join the same
    /// regions), composed as the slowest group.
    fn per_group<F>(&mut self, mut f: F) -> Result<CostNode>
    where
        F: FnMut(&mut Self, usize, &mut Vec<CostNode>) -> Result<()>,
    {
        let saved = (
            self.region,
            self.pending_cut,
            self.forwards[self.forward].regions,
        );
        let mut end = saved;
        let mut groups = Vec::new();
        for d in 0..self.p.attn_dp as usize {
            (self.region, self.pending_cut) = (saved.0, saved.1);
            self.forwards[self.forward].regions = saved.2;
            let mut out = Vec::new();
            self.group = Some(d);
            f(self, d, &mut out)?;
            self.group = None;
            groups.push(CostNode::Sum(out));
            end = (
                self.region,
                self.pending_cut,
                self.forwards[self.forward].regions,
            );
        }
        (self.region, self.pending_cut) = (end.0, end.1);
        self.forwards[self.forward].regions = end.2;
        Ok(CostNode::Max {
            overlap: 1.0,
            children: groups,
        })
    }

    /// One decoder layer of `mixer`: attention module, mixer, mlp module (per
    /// group), then the MoE block over every group.
    fn layer(&mut self, mixer: Qwen4ExpMixer, first: bool, ple: bool) -> Result<CostNode> {
        let attn = self.per_group(|s, d, out| {
            s.cut(Cut::Layer);
            if ple {
                if !first {
                    let k = s.k.hc_combine.clone();
                    out.push(s.tokens("attn_hc.pending_combine", k, RowSrc::Group(d), 1, true));
                }
                s.cut(Cut::Ple);
                s.ple(d, out)?;
                s.cut(Cut::Attn);
                let k = s.k.hc_mix.clone();
                out.push(s.tokens("attn_hc.mix", k, RowSrc::Group(d), 1, true));
            } else if first {
                s.cut(Cut::Attn);
                let k = s.k.hc_mix.clone();
                out.push(s.tokens("attn_hc.mix", k, RowSrc::Group(d), 1, true));
            } else {
                s.combine_mix(d, "attn_hc", Cut::Attn, out);
            }
            s.mixer(d, mixer, out)?;
            s.combine_mix(d, "mlp_hc", Cut::Mlp, out);
            Ok(())
        })?;
        let mut moe = Vec::new();
        self.moe(&mut moe)?;
        Ok(CostNode::Sum(vec![attn, CostNode::Sum(moe)]))
    }

    /// One MTP pass in the current view.
    fn mtp_pass(&mut self) -> Result<CostNode> {
        let pre = self.per_group(|s, d, out| {
            s.embed(d, out)?;
            let fc = s.k.mtp_fc.clone().expect("mtp fc");
            out.push(s.tokens("fc", fc, RowSrc::Group(d), 1, false));
            s.cut(Cut::Attn);
            let k = s.k.hc_mix.clone();
            out.push(s.tokens("attn_hc.mix", k, RowSrc::Group(d), 1, true));
            s.mixer(d, Qwen4ExpMixer::Qsa, out)?;
            s.combine_mix(d, "mlp_hc", Cut::Mlp, out);
            Ok(())
        })?;
        let mut moe = Vec::new();
        self.moe(&mut moe)?;
        let post = self.per_group(|s, d, out| s.head(d, out))?;
        Ok(CostNode::Sum(vec![pre, CostNode::Sum(moe), post]))
    }

    /// The target pass of the current view: prologue, every layer, epilogue.
    fn target_pass(&mut self) -> Result<Vec<CostNode>> {
        let mut root = Vec::new();
        root.push(CostNode::Labeled {
            label: "prologue".to_string(),
            child: Box::new(self.per_group(|s, d, out| s.embed(d, out))?),
        });
        let outer = self.fold;
        for (mixer, first, ple, n) in layer_groups(self.model) {
            self.fold = outer * n;
            let scope = match (mixer, first, ple) {
                (Qwen4ExpMixer::Gdn, _, true) => "gdn_ple",
                (Qwen4ExpMixer::Gdn, true, false) => "gdn_first",
                (Qwen4ExpMixer::Gdn, false, false) => "gdn",
                (Qwen4ExpMixer::Qsa, _, true) => "qsa_ple",
                (Qwen4ExpMixer::Qsa, true, false) => "qsa_first",
                (Qwen4ExpMixer::Qsa, false, false) => "qsa",
            };
            let layer = self.scoped(scope, |pb| pb.layer(mixer, first, ple))?;
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
        self.fold = outer;
        root.push(CostNode::Labeled {
            label: "epilogue".to_string(),
            child: Box::new(self.per_group(|s, d, out| s.head(d, out))?),
        });
        Ok(root)
    }

    /// One forward: `body` in `view`, closed by its launch leaf, `n` times.
    fn forward<F>(
        &mut self,
        label: &str,
        scope: &str,
        view: View,
        mtp: bool,
        n: u32,
        body: F,
    ) -> Result<CostNode>
    where
        F: FnOnce(&mut Self) -> Result<Vec<CostNode>>,
    {
        self.forwards.push(ForwardPlan {
            view,
            mtp,
            regions: 0,
            fold: self.fold * n,
        });
        self.forward = self.forwards.len() - 1;
        let outer = self.fold;
        self.fold = outer * n;
        self.forward_fold = self.fold;
        self.view = view;
        self.in_mtp = mtp;
        self.pending_cut = true;
        let node = self.scoped(scope, |s| -> Result<CostNode> {
            let mut parts = body(s)?;
            let launch = s.k.launch.clone();
            let forward = Some(s.forward);
            parts.push(s.leaf("launch", LeafEval::Launch { k: launch, forward }));
            Ok(CostNode::Sum(parts))
        })?;
        self.fold = outer;
        let node = if n > 1 {
            CostNode::Scale {
                n,
                child: Box::new(node),
            }
        } else {
            node
        };
        Ok(CostNode::Labeled {
            label: label.to_string(),
            child: Box::new(node),
        })
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
) -> Result<(CostTree, Vec<LeafPlan>, Vec<ForwardPlan>)> {
    let mut pb = PlanBuilder {
        name: name.to_string(),
        b: CostTreeBuilder::new(),
        leaves: Vec::new(),
        forwards: Vec::new(),
        k: kernels,
        p,
        model,
        bridge,
        fold: 1,
        view: View::Fused,
        scope: String::new(),
        forward: 0,
        forward_fold: 1,
        in_mtp: false,
        region: 0,
        pending_cut: true,
        sp_cond: SpCond::Always,
        group: None,
    };
    let k = p.draft_tokens;
    if k > 0 {
        ensure!(
            model.mtp_layers == 1,
            "the MTP pass models one QSA MTP layer; the checkpoint has {}",
            model.mtp_layers
        );
    }
    let mut root = Vec::new();
    match p.step_forwards {
        StepForwards::Fused => {
            // One forward: the target pass over every row, then the MTP
            // passes (pass 0 over every target row, k - 1 over one row per
            // decode request), one launch leaf with the step constant.
            let node = pb.forward("fused forward", "", View::Fused, false, 1, |s| {
                let mut parts = s.target_pass()?;
                if k > 0 {
                    s.view = View::MtpFused;
                    s.in_mtp = true;
                    let first = s.scoped("mtp", |s| s.mtp_pass())?;
                    parts.push(CostNode::Labeled {
                        label: "mtp pass 0".to_string(),
                        child: Box::new(first),
                    });
                    if k > 1 {
                        s.view = View::MtpRecurrent;
                        let outer = s.fold;
                        s.fold = outer * (k - 1);
                        let rec = s.scoped("mtp_recurrent", |s| s.mtp_pass())?;
                        s.fold = outer;
                        parts.push(CostNode::Labeled {
                            label: "mtp recurrent pass".to_string(),
                            child: Box::new(CostNode::Scale {
                                n: k - 1,
                                child: Box::new(rec),
                            }),
                        });
                    }
                    s.view = View::Fused;
                    s.in_mtp = false;
                }
                Ok(parts)
            })?;
            root.push(node);
        }
        StepForwards::Separate => {
            root.push(
                pb.forward("decode forward", "decode", View::Decode, false, 1, |s| {
                    s.target_pass()
                })?,
            );
            if k > 0 {
                root.push(pb.forward(
                    "decode mtp",
                    "decode.mtp",
                    View::MtpDecode,
                    true,
                    1,
                    |s| Ok(vec![s.mtp_pass()?]),
                )?);
                if k > 1 {
                    root.push(pb.forward(
                        "decode mtp drafts",
                        "decode.mtp_recurrent",
                        View::MtpRecurrent,
                        true,
                        k - 1,
                        |s| Ok(vec![s.mtp_pass()?]),
                    )?);
                }
            }
            root.push(
                pb.forward("prefill forward", "prefill", View::Prefill, false, 1, |s| {
                    s.target_pass()
                })?,
            );
            if k > 0 {
                root.push(pb.forward(
                    "prefill mtp",
                    "prefill.mtp",
                    View::MtpPrefill,
                    true,
                    1,
                    |s| Ok(vec![s.mtp_pass()?]),
                )?);
                if k > 1 {
                    root.push(pb.forward(
                        "prefill mtp drafts",
                        "prefill.mtp_draft",
                        View::MtpPrefillDraft,
                        true,
                        k - 1,
                        |s| Ok(vec![s.mtp_pass()?]),
                    )?);
                }
            }
            let launch = pb.k.launch.clone();
            root.push(pb.leaf(
                "step",
                LeafEval::Launch {
                    k: launch,
                    forward: None,
                },
            ));
        }
    }
    let tree = pb.b.finish(CostNode::Labeled {
        label: format!(
            "{name} [Trainium2 LNC{}, dp{} x (attn tp{}, gdn tp{}), moe ep{} x tp{}, {:?}, {:?}]",
            p.lnc,
            p.attn_dp,
            p.attn_tp,
            p.gdn_tp,
            p.moe_ep,
            p.moe_tp,
            p.region_granularity,
            p.step_forwards
        ),
        child: Box::new(CostNode::Sum(root)),
    });
    Ok((tree, pb.leaves, pb.forwards))
}

// ── evaluation ───────────────────────────────────────────────────────────────

/// Bucket inventories, resolved once.
struct Buckets {
    prefill: Vec<u32>,
    decode: Vec<u32>,
    context: Vec<u32>,
    /// Token-wise regions of the fused forward: the union of the prefill and
    /// decode buckets.
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

/// `n` padded to its bucket calls (several largest-bucket calls above it).
fn padded(n: u32, buckets: &[u32]) -> u32 {
    let mut out = Vec::new();
    bucket_calls(n, buckets, &mut out);
    out.iter().sum()
}

/// One leaf's evaluated work and its recorded input.
struct LeafOut {
    m: LeafMetrics,
    input: Option<SlotInput>,
}

/// What row counts depend on besides the group shapes.
#[derive(Clone, Copy)]
struct RowCtx {
    /// Shapes are padded already: token-wise leaves take them as they are.
    exact: bool,
    /// Sequence parallelism is active in this pass.
    sp: bool,
    /// Ranks per DP group and in the replica.
    g: u32,
    ranks: u32,
    top_k: u32,
    a2a_cf: f32,
    /// ms per received row added to each all-to-all leaf (packing).
    a2a_pack_ms: f32,
}

impl RowCtx {
    /// All-to-all: the slice a rank routes, or None when the engine falls
    /// back to the all-gather form (fewer than 2 rows per rank, or rows that
    /// do not split over the TP group).
    fn a2a_slice(&self, n: u32) -> Option<u32> {
        let s = if self.sp || n % self.g == 0 {
            n / self.g
        } else {
            0
        };
        (s >= 2).then_some(s)
    }

    /// Rows each rank sends each destination (flashnext `a2a_capacity`).
    fn a2a_capacity(&self, s: u32) -> u32 {
        if self.a2a_cf <= 0.0 {
            return s;
        }
        let want = (self.a2a_cf * s as f32 * self.top_k as f32 / self.ranks as f32).ceil() as u32;
        want.clamp(1, s)
    }

    /// Rows one rank holds before the fallback all-gather.
    fn local_rows(&self, n: u32) -> u32 {
        if self.sp {
            n / self.g
        } else {
            n
        }
    }
}

fn rows_of(src: RowSrc, groups: &[GroupStep], bk: &Buckets, c: RowCtx) -> u32 {
    let rows = |g: &GroupStep| {
        if c.exact {
            g.tokens()
        } else {
            bk.tok(g.tokens())
        }
    };
    let max = || groups.iter().map(rows).max().unwrap_or(0);
    match src {
        RowSrc::Group(d) => rows(&groups[d]),
        RowSrc::All => groups.iter().map(rows).sum(),
        RowSrc::WorldGather => groups
            .iter()
            .map(|g| {
                let n = rows(g);
                // "dup": a group batch that does not split over its TP ranks
                // is gathered from every one of them.
                if c.exact && !c.sp && n % c.g != 0 {
                    n * c.g
                } else {
                    n
                }
            })
            .sum(),
        RowSrc::MaxGroup => max(),
        RowSrc::Sample(d) => {
            let n = groups[d].sample_rows();
            if c.exact {
                n
            } else {
                bk.tok(n)
            }
        }
        RowSrc::A2aRouter => {
            let n = max();
            match c.a2a_slice(n) {
                Some(s) => s,
                None => c.ranks * c.local_rows(n),
            }
        }
        RowSrc::A2aRecv => {
            let n = max();
            match c.a2a_slice(n) {
                Some(s) => c.ranks * c.a2a_capacity(s),
                None => c.ranks * c.local_rows(n),
            }
        }
        RowSrc::A2aShared => {
            let n = max();
            match c.a2a_slice(n) {
                Some(s) => s,
                None => c.local_rows(n),
            }
        }
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

#[allow(clippy::too_many_lines)]
fn eval_leaf(
    plan: &LeafPlan,
    groups: &[GroupStep],
    bk: &Buckets,
    rc: RowCtx,
    record: bool,
    adj: &dyn Fn(LeafMetrics, u32) -> LeafMetrics,
) -> LeafOut {
    let mut m = LeafMetrics::ZERO;
    let mut input = None;
    let mut sizes = Vec::new();
    match &plan.eval {
        LeafEval::Tokens { k, src, div, sp } => {
            let div = (*div).max(1) * if *sp && rc.sp { rc.g } else { 1 };
            let rows = rows_of(*src, groups, bk, rc).div_ceil(div);
            if rows > 0 {
                let max = if rc.exact {
                    u32::MAX
                } else {
                    *bk.tokens.last().expect("buckets")
                };
                let mut rest = rows;
                while rest > 0 {
                    let n = rest.min(max);
                    m.add(adj(k.eval_at(n), n));
                    if record && input.is_none() {
                        input = Some(k.input_at(n));
                    }
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
                    m.add(adj(k.eval_at(n), n));
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
                    per.push(adj(k.eval_at(c), c));
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
                    m.add(adj(k.eval_at(x), x.0));
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
                    per.push(adj(k.eval_at((c, ctx)), c));
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
            let tokens = if rc.exact {
                g.tokens()
            } else {
                bk.tok(g.tokens())
            };
            let x = (tokens, g.requests.clamp(1, tokens.max(1)));
            if tokens > 0 {
                m = adj(k.eval_at(x), x.0);
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
                    m.add(adj(k.eval_at((b, g.query_len)), b * g.query_len));
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
                    m.add(adj(k.eval_at((1, c)), c));
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
            let rows = u64::from(rows_of(*src, groups, bk, rc));
            let bytes = (rows * bytes_per_row).div_ceil(u64::from((*div).max(1)));
            if bytes > 0 {
                m = k.eval_at(bytes);
                if matches!(src, RowSrc::A2aRecv) {
                    // flashnext `a2a_dispatch` / `a2a_combine`: slot packing
                    // and the scatter-add back, traced into the same stage.
                    m.m.time_ms += rows as f32 * rc.a2a_pack_ms;
                }
            }
            if record {
                input = Some(k.input_at(bytes));
            }
        }
        LeafEval::Launch { .. } => unreachable!("launch leaves are priced per forward"),
    }
    LeafOut { m, input }
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
    forwards: Vec<ForwardPlan>,
    buckets: Buckets,
    cost_flat: Vec<FlatCostNode>,
    manifest: CostManifest,
    n_slots: usize,
    kv: KvCacheLayout,
    top_k: u32,
}

/// One forward's region bookkeeping during an evaluation.
#[derive(Clone, Default)]
struct RegionSum {
    /// Device per step (all instances) per DP group, and outside the
    /// per-group part (MoE, shared).
    per_group: Vec<f32>,
    shared: f32,
    instances: u32,
    coll: bool,
}

impl RegionSum {
    /// Device time of one instance of the region.
    fn device(&self) -> f32 {
        let total = self.per_group.iter().copied().fold(0.0, f32::max) + self.shared;
        total / self.instances.max(1) as f32
    }
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
        let (tree, leaves, forwards) = build_plan(&name, model, &parallel, &mut kernels, bridge)?;
        let kv = kv_layout(model, &parallel, fixed_bytes_per_rank);
        tracing::info!(
            "[build] {name}: {} leaf slots, {:.2} GB weights/rank, regions per forward {:?}\n{}",
            tree.n_slots(),
            weight_bytes_per_rank as f64 / 1e9,
            forwards.iter().map(|f| f.regions).collect::<Vec<_>>(),
            tree.describe()
        );
        Ok(Self {
            buckets: Buckets::new(&parallel),
            name,
            parallel,
            fixed_bytes_per_rank,
            weight_bytes_per_rank,
            leaves,
            forwards,
            cost_flat: tree.flatten(),
            manifest: tree.manifest(),
            n_slots: tree.n_slots(),
            kv,
            top_k: model.experts_per_token,
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

    /// The step's passes from each DP group's own decode rows and chunks.
    fn views(&self, mut target: Vec<GroupStep>) -> StepViews {
        let p = &self.parallel;
        let bk = &self.buckets;
        let dp = p.attn_dp as usize;
        // A worker without DP partitions (the PD prefill worker) hands one
        // group, one prompt per iteration: padding the idle groups would charge
        // each prompt four groups' work, where the engine runs one prompt per
        // group per step. Its spread groups pay their own shapes instead.
        let mut spmd = p.spmd_padding;
        if target.len() == 1 && dp > 1 {
            target = spread_over_groups(&target[0], dp);
            spmd = false;
        }
        let mut groups: [Vec<GroupStep>; N_VIEWS] = Default::default();
        let recurrent = |pass: &[GroupStep]| -> Vec<GroupStep> {
            pass.iter()
                .map(|g| GroupStep {
                    decode_reqs: g.decode_reqs,
                    query_len: 1,
                    decode_max_kv: g.decode_max_kv + 1,
                    chunks: Vec::new(),
                    requests: g.decode_reqs,
                })
                .collect()
        };
        match p.step_forwards {
            StepForwards::Fused => {
                if spmd {
                    pad_decode(&mut target, true, bk);
                    pad_prefill(&mut target, true, bk);
                }
                groups[View::MtpRecurrent.index()] = recurrent(&target);
                groups[View::MtpFused.index()] = target.clone();
                groups[View::Fused.index()] = target;
                StepViews {
                    groups,
                    exact: false,
                }
            }
            StepForwards::Separate => {
                let mut dec: Vec<GroupStep> = target
                    .iter()
                    .map(|g| GroupStep {
                        decode_reqs: g.decode_reqs,
                        query_len: g.query_len,
                        decode_max_kv: g.decode_max_kv,
                        chunks: Vec::new(),
                        requests: g.decode_reqs,
                    })
                    .collect();
                let mut pre: Vec<GroupStep> = target
                    .iter()
                    .map(|g| GroupStep {
                        query_len: 1,
                        chunks: g.chunks.clone(),
                        requests: g.chunks.len() as u32,
                        ..Default::default()
                    })
                    .collect();
                pad_decode(&mut dec, spmd, bk);
                pad_prefill(&mut pre, spmd, bk);
                // The first drafts after a prompt's last chunk: one row per
                // group (every group under SPMD).
                let any_pre = pre.iter().any(|g| !g.chunks.is_empty());
                let draft: Vec<GroupStep> = pre
                    .iter()
                    .map(|g| {
                        let run = if spmd { any_pre } else { !g.chunks.is_empty() };
                        let end = g.chunks.iter().map(|c| c.0 + c.1).max().unwrap_or(0);
                        if run {
                            GroupStep {
                                decode_reqs: 1,
                                query_len: 1,
                                decode_max_kv: end + 1,
                                chunks: Vec::new(),
                                requests: 1,
                            }
                        } else {
                            GroupStep::default()
                        }
                    })
                    .collect();
                groups[View::MtpRecurrent.index()] = recurrent(&dec);
                groups[View::MtpDecode.index()] = dec.clone();
                groups[View::Decode.index()] = dec;
                groups[View::MtpPrefill.index()] = pre.clone();
                groups[View::Prefill.index()] = pre;
                groups[View::MtpPrefillDraft.index()] = draft;
                StepViews {
                    groups,
                    exact: true,
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
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
        let par = &self.parallel;
        let factor = par.region_device_factor;
        let dp = par.attn_dp as usize;
        let g = par.group_size();
        // Per view: does sequence parallelism apply (rows split over the TP
        // group, flashnext `Layout.sliced`)?
        let mut sp = [false; N_VIEWS];
        for (i, groups) in views.groups.iter().enumerate() {
            sp[i] = par.sequence_parallel()
                && (!views.exact || groups.iter().all(|x| x.tokens() > 0 && x.tokens() % g == 0));
        }
        let runs: Vec<bool> = self
            .forwards
            .iter()
            .map(|f| views.get(f.view).iter().any(|x| !x.is_empty()))
            .collect();
        let mut regions: Vec<std::collections::BTreeMap<usize, RegionSum>> =
            vec![Default::default(); self.forwards.len()];
        let mut step_slot = None;
        let idle = vec![GroupStep::default(); dp];
        for (slot, plan) in self.leaves.iter().enumerate() {
            if let LeafEval::Launch { k, forward } = &plan.eval {
                let value = match forward {
                    Some(f) => self.forward_launch(*f, runs[*f], &regions[*f], k),
                    None => {
                        step_slot = Some(slot);
                        0.0
                    }
                };
                slots[slot] = LeafMetrics::ZERO;
                slots[slot].m.time_ms = value;
                if let Some(v) = inputs.as_deref_mut() {
                    let n = forward.map_or(0, |f| self.forwards[f].regions);
                    v.push(k.input_at(n));
                }
                continue;
            }
            let vi = plan.view.index();
            let active = runs[plan.forward]
                && match plan.sp_cond {
                    SpCond::Always => true,
                    SpCond::Sp => sp[vi],
                    SpCond::NoSp => !sp[vi],
                };
            let rc = RowCtx {
                exact: views.exact,
                sp: sp[vi],
                g,
                ranks: par.ranks(),
                top_k: self.top_k(),
                a2a_cf: par.moe_a2a_capacity_factor,
                a2a_pack_ms: par.moe_a2a_pack_us_per_row * 1e-3,
            };
            let compute = plan.compute;
            let adj = move |mut x: LeafMetrics, rows: u32| {
                if compute {
                    x.m.time_ms *= plan.scale_at(rows) * factor;
                }
                x
            };
            // An inactive leaf sees empty groups: no work, only its input.
            let groups = if active {
                views.get(plan.view)
            } else {
                &idle[..]
            };
            let out = eval_leaf(plan, groups, &self.buckets, rc, record, &adj);
            let r = regions[plan.forward]
                .entry(plan.region)
                .or_insert_with(|| RegionSum {
                    per_group: vec![0.0; dp],
                    ..Default::default()
                });
            // A region may take a leaf from outside its layers' Scale (the
            // next layer group's leading combine): sums are per step, and the
            // instances are those of the leaf that opened the region.
            let instances = plan.fold / self.forwards[plan.forward].fold.max(1);
            if r.instances == 0 {
                r.instances = instances;
            }
            r.coll |= !compute && active;
            let total = instances as f32 * out.m.m.time_ms;
            match plan.group {
                Some(d) => r.per_group[d] += total,
                None => r.shared += total,
            }
            slots[slot] = out.m;
            if let (Some(v), Some(input)) = (inputs.as_deref_mut(), out.input) {
                v.push(input);
            }
        }
        if let Some(slot) = step_slot {
            let any = runs.iter().any(|&r| r);
            slots[slot].m.time_ms = match par.launch_model {
                LaunchModel::Pipelined | LaunchModel::Calibrated if any => par.step_overhead_ms,
                _ => 0.0,
            };
        }
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn top_k(&self) -> u32 {
        self.top_k
    }

    #[cfg(test)]
    fn k_launch(&self) -> Rows {
        self.leaves
            .iter()
            .find_map(|l| match &l.eval {
                LeafEval::Launch { k, .. } => Some(k.clone()),
                _ => None,
            })
            .expect("a launch leaf")
    }

    /// The launch leaf of forward `f`: host and launch time beyond the sum of
    /// its leaves, per instance of the forward.
    fn forward_launch(
        &self,
        f: usize,
        runs: bool,
        regions: &std::collections::BTreeMap<usize, RegionSum>,
        k: &Rows,
    ) -> f32 {
        if !runs {
            return 0.0;
        }
        let par = &self.parallel;
        let fw = &self.forwards[f];
        let fused_step = if par.step_forwards == StepForwards::Fused {
            par.step_overhead_ms
        } else {
            0.0
        };
        match par.launch_model {
            LaunchModel::Pipelined => {
                let h = par.region_host_ms;
                let (mut device, mut floored) = (0f32, 0f32);
                let gap = par.region_gap_ms;
                for r in regions.values() {
                    let n = r.instances as f32;
                    let dev = r.device();
                    device += n * dev;
                    // Each launch costs the device `gap` outside the region;
                    // a region shorter than its dispatch leaves the device
                    // waiting for the next launch (idle, or inside a
                    // cross-group collective for the slowest rank's launch).
                    floored += n * (dev + gap).max(h);
                }
                if par.region_granularity == RegionGranularity::Block {
                    // Block regions span layers of differently folded groups:
                    // floor the forward's regions as equal shares of it.
                    let n = fw.regions as f32;
                    floored = (device + n * gap).max(n * h);
                }
                // The forward's host work (dispatch + its per-forward input
                // assembly, H2D and readback) overlaps the device; whichever
                // is longer sets the forward.
                let overhead = if fw.mtp {
                    par.mtp_forward_overhead_ms
                } else {
                    par.forward_overhead_ms
                };
                let host = fw.regions as f32 * h + overhead;
                fused_step + (floored - device) + (host - floored).max(0.0)
            }
            LaunchModel::Calibrated => {
                let slack: f32 = regions
                    .values()
                    .map(|r| {
                        let floor = if r.coll {
                            par.launch_coll_ms
                        } else {
                            par.launch_plain_ms
                        };
                        r.instances as f32 * (floor - r.device()).max(0.0)
                    })
                    .sum();
                slack + fused_step
            }
            LaunchModel::L1Additive => k.eval_at(fw.regions).m.time_ms,
        }
    }
}

/// Pad every group's decode batch to its bucket; under SPMD every group runs
/// the largest group's padded batch, the longest query and the longest KV.
fn pad_decode(groups: &mut [GroupStep], spmd: bool, bk: &Buckets) {
    if spmd {
        let b = groups.iter().map(|g| g.decode_reqs).max().unwrap_or(0);
        if b == 0 {
            return;
        }
        let b = padded(b, &bk.decode);
        let q = groups.iter().map(|g| g.query_len).max().unwrap_or(1).max(1);
        let kv = groups.iter().map(|g| g.decode_max_kv).max().unwrap_or(0);
        for g in groups {
            g.decode_reqs = b;
            g.query_len = q;
            g.decode_max_kv = kv;
            g.requests = g.requests.max(b);
        }
    } else {
        for g in groups.iter_mut().filter(|g| g.decode_reqs > 0) {
            g.decode_reqs = padded(g.decode_reqs, &bk.decode);
        }
    }
}

/// Pad every prefill chunk to its bucket; under SPMD every group runs the
/// longest chunk's bucket from the furthest prefix (the engine's context
/// bucket is that of the longest `start + T`), idle groups included.
fn pad_prefill(groups: &mut [GroupStep], spmd: bool, bk: &Buckets) {
    if spmd {
        let n = groups.iter().map(|g| g.chunks.len()).max().unwrap_or(0);
        if n == 0 {
            return;
        }
        let all = groups.iter().flat_map(|g| g.chunks.iter());
        let t = all.clone().map(|c| c.1).max().unwrap_or(0);
        let prefix = all.map(|c| c.0).max().unwrap_or(0);
        let chunk = (prefix, padded(t, &bk.prefill));
        for g in groups {
            g.chunks = vec![chunk; n];
            g.requests = g.requests.max(n as u32);
        }
    } else {
        for g in groups.iter_mut() {
            for c in &mut g.chunks {
                c.1 = padded(c.1, &bk.prefill);
            }
        }
    }
}

/// Each DP group's own decode rows and chunks of a unified batch.
fn unified_targets(batch: &UnifiedArchInput) -> Vec<GroupStep> {
    batch
        .groups
        .iter()
        .map(|g| GroupStep {
            decode_reqs: g.decode_kv_lens.len() as u32,
            query_len: 1,
            decode_max_kv: g.decode_kv_lens.iter().copied().max().unwrap_or(0),
            chunks: g.prefill_chunk_pairs.clone(),
            requests: g.request_count(),
        })
        .collect()
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

/// Each DP group's verify rows and chunks of a speculative batch.
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
            requests: g.request_count(),
        })
        .collect()
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
            &self.core.views(unified_targets(batch)),
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
            &self.core.views(unified_targets(batch)),
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
        self.core.eval(
            &self.core.views(speculative_targets(batch)),
            slots,
            scratch,
            None,
        )
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.core.eval(
            &self.core.views(speculative_targets(batch)),
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
            step_forwards: StepForwards::Separate,
            spmd_padding: true,
            region_device_factor: 1.0,
            launch_model: LaunchModel::Pipelined,
            launch_coll_ms: 1.25,
            launch_plain_ms: 0.075,
            region_host_ms: 1.0,
            region_gap_ms: 0.0,
            forward_overhead_ms: 12.0,
            mtp_forward_overhead_ms: 3.0,
            step_overhead_ms: 1.0,
            moe_a2a_capacity_factor: 0.0,
            moe_a2a_pack_us_per_row: 0.0,
            prefill_buckets: vec![128, 256, 512, 1024, 2048, 4096, 8192],
            decode_buckets: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
            context_buckets: vec![4096, 8192, 16384, 32768, 65536, 131072],
            ple_table_shards: PleTableShards::Replica,
            hbm_per_core_gb: 25.769_803_776,
            hbm_reserve_gb: 2.0,
            impls: NeuronFlashNextImpls::default(),
            kind_time_scale: Vec::new(),
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

    fn unified(groups: Vec<ArchGroupInput>) -> UnifiedArchInput {
        UnifiedArchInput {
            groups,
            tokens_per_source_rank: Vec::new(),
        }
    }

    /// Evaluate `batch` and return (slot names, slot times, slot inputs, total).
    fn eval_unified(
        core: &NeuronFlashNextCore,
        batch: &UnifiedArchInput,
    ) -> (Vec<String>, Vec<f32>, Vec<serde_json::Value>, f32) {
        let (mut slots, mut scratch, mut inputs) = (Vec::new(), Vec::new(), Vec::new());
        let total = core.eval(
            &core.views(unified_targets(batch)),
            &mut slots,
            &mut scratch,
            Some(&mut inputs),
        );
        assert_eq!(inputs.len(), core.n_slots, "one input per slot");
        (
            core.manifest.slots.iter().map(|s| s.name.clone()).collect(),
            slots.iter().map(|m| m.m.time_ms).collect(),
            inputs
                .iter()
                .map(|i| serde_json::to_value(i).unwrap())
                .collect(),
            total.m.time_ms,
        )
    }

    fn slot<'a, T>(names: &[String], values: &'a [T], suffix: &str) -> &'a T {
        let i = names
            .iter()
            .position(|n| n.ends_with(suffix))
            .unwrap_or_else(|| panic!("no slot ending in {suffix}"));
        &values[i]
    }

    /// A row probe that costs 1 ms per call and records its row count.
    struct FakeRows;
    impl ShapeProbe<u32> for FakeRows {
        fn kind(&self) -> &'static str {
            "fake"
        }
        fn config(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        fn eval_at(&self, _: u32) -> LeafMetrics {
            let mut m = LeafMetrics::ZERO;
            m.m.time_ms = 1.0;
            m
        }
        fn input_at(&self, num_tokens: u32) -> SlotInput {
            SlotInput::from(crate::timing::kernels::HcMixKernelInput { num_tokens })
        }
    }

    /// A probe that costs 1 ms per call.
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
        let rows: Rows = Arc::new(FakeRows);
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
                LeafEval::Launch { k, .. } => *k = Arc::new(Regions),
            }
        }
        c
    }

    /// Catches a region miscount against the engine (`engine/README.md`
    /// "Region granularity", plus the head gather region): layer 63, block 14,
    /// mixer 112 per target forward; the MTP pass is 4 regions at layer and 2
    /// at block.
    #[test]
    fn forwards_launch_the_engines_regions() {
        let regions = |gran: RegionGranularity, k: u32| -> Vec<u32> {
            let mut p = parallel(4, 8, 32);
            p.region_granularity = gran;
            p.draft_tokens = k;
            built(p).forwards.iter().map(|f| f.regions).collect()
        };
        // [decode, decode mtp, prefill, prefill mtp]
        assert_eq!(regions(RegionGranularity::Layer, 1), [63, 4, 63, 4]);
        assert_eq!(regions(RegionGranularity::Block, 1), [14, 2, 14, 2]);
        assert_eq!(regions(RegionGranularity::Mixer, 0), [112, 112]);
        // k = 2 adds one draft pass per side, each its own forward.
        assert_eq!(regions(RegionGranularity::Layer, 2), [63, 4, 4, 63, 4, 4]);
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
        // Both forms: sequence parallelism applies only to a forward whose
        // rows split over the TP group (`sequence_parallelism_needs_rows_that_split`).
        let names = kinds(p);
        assert!(names.iter().any(|n| n.ends_with("gdn.reduce_scatter")));
        assert!(names.iter().any(|n| n.ends_with("gdn.all_reduce")));
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
    fn necessary_work_map_covers_the_fused_mixer_layout() {
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
        // The map covers the one-forward step (`step_forwards: fused`); the
        // engine's separate forwards split each semantic row's tokens over two
        // locations, which a static map cannot attribute.
        for attn_dp in [1, 4] {
            let mut p = parallel(attn_dp, 8, 8 * attn_dp);
            p.step_forwards = StepForwards::Fused;
            let locations: std::collections::BTreeSet<String> = built(p)
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

    fn prefill_group(chunks: Vec<(u32, u32)>) -> ArchGroupInput {
        let tokens = chunks.iter().map(|c| c.1).sum();
        ArchGroupInput {
            batch_tokens: tokens,
            prefill_tokens: tokens,
            prefill_chunk_pairs: chunks,
            ..Default::default()
        }
    }

    /// Catches the pipelined launch model mispricing a forward: with every
    /// fake leaf at 1 ms, a host cost per region far above the device makes
    /// the decode forward host-bound (regions x region_host_ms + the forward
    /// constant), and one far below leaves the leaves' sum + the constant.
    #[test]
    fn pipelined_launch_prices_a_forward_as_max_of_device_and_host() {
        let mut p = parallel(4, 8, 32);
        p.region_granularity = RegionGranularity::Layer;
        let batch = unified(vec![decode_group(4, 4096); 4]);
        p.region_host_ms = 100.0;
        let core = built_like(&built(p.clone()));
        let (_, _, _, total) = eval_unified(&core, &batch);
        let host = 63.0 * 100.0 + 12.0 + 1.0;
        assert!((total - host).abs() < 1e-2, "{total} vs {host}");
        // At mixer granularity the floored regions must count each region
        // once: a region that takes the next layer group's leading combine
        // keeps its own instances (112 regions, not 112 + 33).
        let mut m = p.clone();
        m.region_granularity = RegionGranularity::Mixer;
        let core = built_like(&built(m));
        let (_, _, _, total) = eval_unified(&core, &batch);
        let host = 112.0 * 100.0 + 12.0 + 1.0;
        assert!((total - host).abs() < 1e-1, "{total} vs {host}");
        // The host's 63 x 0.001 + 12 ms hides under the device.
        p.region_host_ms = 1e-3;
        let core = built_like(&built(p));
        let (names, times, _, total) = eval_unified(&core, &batch);
        assert_eq!(*slot(&names, &times, "decode.launch"), 0.0);
        assert_eq!(
            *slot(&names, &times, "prefill.launch"),
            0.0,
            "no prefill forward"
        );
        assert_eq!(*slot(&names, &times, ".step"), 1.0);
        assert!(total > 1.0 + 48.0, "{total}");
    }

    /// Catches a forward's launch leaf mispricing the regions shorter than
    /// their host dispatch (floored), or the host work exceeding the device.
    #[test]
    fn regions_are_floored_at_their_dispatch() {
        let mut p = parallel(4, 8, 32);
        p.region_granularity = RegionGranularity::Layer;
        p.region_host_ms = 0.1;
        let core = built_like(&built(p));
        // 10 instances at 0.05 ms floor to 0.1 ms each; a 20 ms region does
        // not; the host (63 x 0.1 + 12 ms) stays under the device.
        let region = |device: f32, instances: u32| RegionSum {
            per_group: vec![device * instances as f32, 0.0, 0.0, 0.0],
            shared: 0.0,
            instances,
            coll: false,
        };
        let regions = [(0, region(0.05, 10)), (1, region(20.0, 1))]
            .into_iter()
            .collect();
        let k = core.k_launch();
        let launch = core.forward_launch(0, true, &regions, &k);
        assert!((launch - 0.5).abs() < 1e-4, "{launch}");
        assert_eq!(core.forward_launch(0, false, &regions, &k), 0.0);
        // A 1 ms region: the host's 63 x 0.1 + 12 = 18.3 ms sets the forward.
        let regions = [(0, region(1.0, 1))].into_iter().collect();
        let launch = core.forward_launch(0, true, &regions, &k);
        assert!((launch - 17.3).abs() < 1e-4, "{launch}");
    }

    /// Catches an idle DP group running free under SPMD, or an engine-padded
    /// shape leaking into `spmd_padding: false`.
    #[test]
    fn idle_groups_pay_the_padded_shape_under_spmd() {
        let p = parallel(4, 8, 32);
        let batch = vec![
            prefill_group(vec![(0, 1500)]),
            prefill_group(vec![(2000, 48)]),
            ArchGroupInput::default(),
            ArchGroupInput::default(),
        ];
        let core = built(p.clone());
        let views = core.views(unified_targets(&unified(batch.clone())));
        for g in views.get(View::Prefill) {
            // The longest chunk's bucket from the furthest prefix: context
            // bucket of 2000 + 2048.
            assert_eq!(g.chunks, [(2000, 2048)]);
        }
        assert!(views.get(View::Decode).iter().all(GroupStep::is_empty));
        let mut q = p;
        q.spmd_padding = false;
        let core = built(q);
        let views = core.views(unified_targets(&unified(batch)));
        let pre = views.get(View::Prefill);
        assert_eq!(pre[0].chunks, [(0, 2048)]);
        assert_eq!(pre[1].chunks, [(2000, 128)]);
        assert!(pre[2].is_empty() && pre[3].is_empty());
    }

    /// Catches decode and prefill sharing one forward under `separate`: a
    /// mixed step costs its decode-only and prefill-only steps minus one step
    /// constant, and the fused forward costs less than that.
    #[test]
    fn separate_forwards_add_and_fused_shares_one_forward() {
        let mut p = parallel(4, 8, 32);
        p.region_granularity = RegionGranularity::Layer;
        let core = built_like(&built(p.clone()));
        let dec = vec![decode_group(4, 4096); 4];
        let pre = vec![prefill_group(vec![(0, 512)]); 4];
        let mixed: Vec<ArchGroupInput> = (0..4)
            .map(|_| ArchGroupInput {
                batch_tokens: 516,
                decode_tokens: 4,
                prefill_tokens: 512,
                decode_kv_lens: vec![4096; 4],
                total_kv_len: 4 * 4096,
                prefill_chunk_pairs: vec![(0, 512)],
                ..Default::default()
            })
            .collect();
        let t = |b: Vec<ArchGroupInput>| eval_unified(&core, &unified(b)).3;
        let (d, f, m) = (t(dec), t(pre), t(mixed.clone()));
        assert!((m - (d + f - 1.0)).abs() < 1e-2, "{m} vs {d} + {f} - 1");
        p.step_forwards = StepForwards::Fused;
        let fused = built_like(&built(p));
        let one = eval_unified(&fused, &unified(mixed)).3;
        assert!(one < m, "fused {one} vs separate {m}");
    }

    /// Catches the world all-gather row count: a decode batch that does not
    /// split over the 8 TP ranks is gathered from every rank (flashnext
    /// `world_gather_rows` "dup"), one that does is sliced.
    #[test]
    fn moe_gathers_whole_group_batches_that_do_not_split() {
        let core = built_like(&built(parallel(4, 8, 32)));
        let rows = |b: usize| {
            let (names, _, inputs, _) =
                eval_unified(&core, &unified(vec![decode_group(b, 4096); 4]));
            slot(&names, &inputs, "decode.gdn.moe.block")["num_tokens"].clone()
        };
        assert_eq!(rows(1), 32, "1 row x 32 ranks");
        assert_eq!(rows(8), 32, "8 rows x 4 groups");
        assert_eq!(rows(16), 64);
    }

    /// Catches sequence parallelism applied to a forward whose rows do not
    /// split over the TP group (the engine falls back to all-reduce there).
    #[test]
    fn sequence_parallelism_needs_rows_that_split() {
        let mut p = parallel(4, 8, 32);
        p.tp_collective = TpCollective::ReduceScatterAllGather;
        let core = built_like(&built(p));
        for (b, sp) in [(4usize, false), (16, true)] {
            let (names, times, _, _) =
                eval_unified(&core, &unified(vec![decode_group(b, 4096); 4]));
            let rs = *slot(&names, &times, "decode.gdn.gdn.reduce_scatter");
            let ar = *slot(&names, &times, "decode.gdn.gdn.all_reduce");
            assert_eq!((rs > 0.0, ar > 0.0), (sp, !sp), "B={b}");
        }
    }

    /// Catches the all-to-all experts seeing fewer rows than the engine's
    /// exact capacity (every rank receives S rows from each of 32 ranks, the
    /// all-gather's 4N), or the capacity factor not cutting them.
    #[test]
    fn all_to_all_exact_capacity_matches_all_gather_rows() {
        let mut p = parallel(4, 8, 32);
        p.moe_dispatch = MoeDispatch::AllToAll;
        let core = built_like(&built(p.clone()));
        let rows = |core: &NeuronFlashNextCore| {
            let (names, _, inputs, _) =
                eval_unified(core, &unified(vec![prefill_group(vec![(0, 512)]); 4]));
            (
                slot(&names, &inputs, "prefill.gdn.moe.routed_experts")["num_tokens"].clone(),
                slot(&names, &inputs, "prefill.gdn.moe.router")["num_tokens"].clone(),
            )
        };
        assert_eq!(
            rows(&core),
            (serde_json::json!(4 * 512), serde_json::json!(64))
        );
        p.moe_a2a_capacity_factor = 1.0;
        let core = built_like(&built(p));
        // cap = ceil(1.0 x 64 x 10 / 32) = 20 rows from each of 32 ranks.
        assert_eq!(rows(&core).0, 32 * 20);
    }

    #[test]
    fn kind_time_scale_parses_and_rejects_bad_entries() {
        assert_eq!(
            KindTimeScale::parse("moe_block@256=0.5").unwrap(),
            KindTimeScale {
                kind: "moe_block".into(),
                min_rows: 256,
                factor: 0.5
            }
        );
        assert_eq!(KindTimeScale::parse("hc_mix=2").unwrap().min_rows, 0);
        for bad in [
            "moe_block",
            "moe_block=0",
            "moe_block=-1",
            "=0.5",
            "moe_block@x=0.5",
        ] {
            assert!(KindTimeScale::parse(bad).is_err(), "{bad}");
        }
    }

    /// Catches the what-if scale leaking onto other kinds or onto calls below
    /// its row threshold: `moe_block@256=0.25` quarters the MoE block of a
    /// 2048-token prefill chunk and leaves a 4-request decode batch alone.
    #[test]
    fn kind_time_scale_scales_only_its_kind_above_the_threshold() {
        let base = parallel(1, 8, 8);
        let mut scaled = base.clone();
        scaled.kind_time_scale = vec![KindTimeScale::parse("moe_block@256=0.25").unwrap()];
        let eval = |p: &NeuronFlashNextParallel, group: ArchGroupInput| {
            let core = built_like(&built(p.clone()));
            let views = core.views(unified_targets(&unified(vec![group])));
            let (mut slots, mut scratch) = (Vec::new(), Vec::new());
            core.eval(&views, &mut slots, &mut scratch, None);
            let names: Vec<String> = core.manifest.slots.iter().map(|s| s.name.clone()).collect();
            (names, slots.iter().map(|m| m.m.time_ms).collect::<Vec<_>>())
        };
        let prefill = ArchGroupInput {
            batch_tokens: 2048,
            prefill_tokens: 2048,
            prefill_chunk_pairs: vec![(0, 2048)],
            ..Default::default()
        };
        for (group, moe_ratio) in [(prefill, 0.25f32), (decode_group(4, 4096), 1.0)] {
            let (names, a) = eval(&base, group.clone());
            let (_, b) = eval(&scaled, group);
            let mut moe_seen = false;
            for ((name, x), y) in names.iter().zip(&a).zip(&b) {
                let want = if name.ends_with("moe.block") {
                    moe_seen |= *x > 0.0;
                    x * moe_ratio
                } else {
                    *x
                };
                assert!((y - want).abs() < 1e-5, "{name}: {y} vs {want}");
            }
            assert!(moe_seen);
        }
    }
}
