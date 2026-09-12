//! Shared GLM-5.2 checkpoint identity and MTP configuration.
//!
//! These definitions are independent of a serving framework's kernel graph.
//! Every GLM-5.2 architecture parses the same checkpoint contract here.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::timing::bridge::DType;
use crate::timing::Dim;

const NUM_LAYERS: u32 = 78;
const NUM_DENSE_LAYERS: u32 = 3;
const HIDDEN_DIM: u32 = 6_144;
const DENSE_INTERMEDIATE_DIM: u32 = 12_288;
const NUM_ATTN_HEADS: u32 = 64;
const RAW_NUM_KV_HEADS: u32 = 64;
const Q_LORA_RANK: u32 = 2_048;
const KV_LORA_RANK: u32 = 512;
const QK_NOPE_HEAD_DIM: u32 = 192;
const ROPE_DIM: u32 = 64;
const V_HEAD_DIM: u32 = 256;
const MODEL_INDEX_HEADS: u32 = 32;
const INDEX_HEAD_DIM: u32 = 128;
const INDEX_TOP_K: u32 = 2_048;
const CHECKPOINT_MAX_CONTEXT: u32 = 1_048_576;
const NUM_EXPERTS: u32 = 256;
const ROUTER_TOP_K: u32 = 8;
const MOE_INTERMEDIATE_DIM: u32 = 2_048;
const NUM_SHARED_EXPERTS: u32 = 1;
const VOCAB_SIZE: u32 = 154_880;
const NUM_MTP_LAYERS: u32 = 1;

const FULL_INDEX_LAYERS: [u32; 21] = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
];

/// Parsed and validated GLM-5.2 checkpoint identity.
#[derive(Clone, Debug)]
pub struct Glm52ModelCfg {
    pub hidden_dim: Dim,
    pub dense_intermediate_dim: Dim,
    pub num_attention_heads: Dim,
    pub raw_num_kv_heads: Dim,
    pub q_lora_rank: Dim,
    pub kv_lora_rank: Dim,
    pub qk_nope_head_dim: Dim,
    pub rope_dim: Dim,
    pub v_head_dim: Dim,
    pub model_num_index_heads: Dim,
    pub index_head_dim: Dim,
    pub index_top_k: u32,
    pub max_context: Dim,
    pub num_experts: Dim,
    pub router_top_k: u32,
    pub moe_intermediate_dim: Dim,
    pub num_shared_experts: u32,
    pub vocab_size: Dim,
    pub num_layers: u32,
    pub num_mtp_layers: u32,
    pub dtype: DType,
    pub router_semantic_dtype: DType,
    pub full_index_layers: Vec<u32>,
    pub indexer_types: Vec<String>,
    pub mlp_layer_types: Vec<String>,
    pub index_share_for_mtp_iteration: bool,
}

impl Glm52ModelCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading GLM-5.2 config {}", path.display()))?;
        parse_model_json(&text)
            .with_context(|| format!("validating GLM-5.2 config {}", path.display()))
    }
}

#[derive(Deserialize)]
struct JsonGlm52Config {
    architectures: Vec<String>,
    model_type: String,
    dtype: String,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    first_k_dense_replace: u32,
    mlp_layer_types: Vec<String>,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    qk_head_dim: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    index_skip_topk_offset: u32,
    index_topk_freq: u32,
    index_topk_pattern: Option<String>,
    indexer_types: Vec<String>,
    index_share_for_mtp_iteration: bool,
    indexer_rope_interleave: bool,
    rope_interleave: bool,
    max_position_embeddings: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    n_shared_experts: u32,
    moe_layer_freq: u32,
    moe_router_dtype: String,
    scoring_func: String,
    topk_method: String,
    n_group: u32,
    topk_group: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    vocab_size: u32,
    num_nextn_predict_layers: u32,
}

/// Parse the checkpoint identity shared by all GLM-5.2 execution graphs.
pub(super) fn parse_model_json(text: &str) -> Result<Glm52ModelCfg> {
    let raw: JsonGlm52Config = serde_json::from_str(text).context("parsing JSON")?;
    ensure!(
        raw.architectures == ["GlmMoeDsaForCausalLM"],
        "architectures must be [GlmMoeDsaForCausalLM]"
    );
    ensure!(
        raw.model_type == "glm_moe_dsa",
        "model_type must be glm_moe_dsa"
    );
    ensure!(raw.dtype == "bfloat16", "dtype must be bfloat16");
    for (name, actual, required) in [
        ("hidden_size", raw.hidden_size, HIDDEN_DIM),
        (
            "intermediate_size",
            raw.intermediate_size,
            DENSE_INTERMEDIATE_DIM,
        ),
        ("num_hidden_layers", raw.num_hidden_layers, NUM_LAYERS),
        (
            "first_k_dense_replace",
            raw.first_k_dense_replace,
            NUM_DENSE_LAYERS,
        ),
        (
            "num_attention_heads",
            raw.num_attention_heads,
            NUM_ATTN_HEADS,
        ),
        (
            "num_key_value_heads",
            raw.num_key_value_heads,
            RAW_NUM_KV_HEADS,
        ),
        ("head_dim", raw.head_dim, QK_NOPE_HEAD_DIM),
        ("q_lora_rank", raw.q_lora_rank, Q_LORA_RANK),
        ("kv_lora_rank", raw.kv_lora_rank, KV_LORA_RANK),
        ("qk_head_dim", raw.qk_head_dim, QK_NOPE_HEAD_DIM + ROPE_DIM),
        ("qk_nope_head_dim", raw.qk_nope_head_dim, QK_NOPE_HEAD_DIM),
        ("qk_rope_head_dim", raw.qk_rope_head_dim, ROPE_DIM),
        ("v_head_dim", raw.v_head_dim, V_HEAD_DIM),
        ("index_n_heads", raw.index_n_heads, MODEL_INDEX_HEADS),
        ("index_head_dim", raw.index_head_dim, INDEX_HEAD_DIM),
        ("index_topk", raw.index_topk, INDEX_TOP_K),
        ("index_skip_topk_offset", raw.index_skip_topk_offset, 3),
        ("index_topk_freq", raw.index_topk_freq, 4),
        (
            "max_position_embeddings",
            raw.max_position_embeddings,
            CHECKPOINT_MAX_CONTEXT,
        ),
        ("n_routed_experts", raw.n_routed_experts, NUM_EXPERTS),
        ("num_experts_per_tok", raw.num_experts_per_tok, ROUTER_TOP_K),
        (
            "moe_intermediate_size",
            raw.moe_intermediate_size,
            MOE_INTERMEDIATE_DIM,
        ),
        ("n_shared_experts", raw.n_shared_experts, NUM_SHARED_EXPERTS),
        ("moe_layer_freq", raw.moe_layer_freq, 1),
        ("n_group", raw.n_group, 1),
        ("topk_group", raw.topk_group, 1),
        ("vocab_size", raw.vocab_size, VOCAB_SIZE),
        (
            "num_nextn_predict_layers",
            raw.num_nextn_predict_layers,
            NUM_MTP_LAYERS,
        ),
    ] {
        ensure!(
            actual == required,
            "{name} must be {required}, got {actual}"
        );
    }
    ensure!(
        raw.index_topk_pattern.is_none(),
        "index_topk_pattern must be null"
    );
    ensure!(
        raw.index_share_for_mtp_iteration,
        "index_share_for_mtp_iteration must be enabled"
    );
    ensure!(
        raw.indexer_rope_interleave,
        "indexer_rope_interleave must be enabled"
    );
    ensure!(raw.rope_interleave, "rope_interleave must be enabled");
    ensure!(
        raw.moe_router_dtype == "float32",
        "moe_router_dtype must be float32"
    );
    ensure!(
        raw.scoring_func == "sigmoid",
        "scoring_func must be sigmoid"
    );
    ensure!(
        raw.topk_method == "noaux_tc",
        "topk_method must be noaux_tc"
    );
    ensure!(raw.norm_topk_prob, "norm_topk_prob must be enabled");
    ensure!(
        raw.routed_scaling_factor == 2.5,
        "routed_scaling_factor must be 2.5"
    );

    let expected_mlp: Vec<String> = (0..NUM_LAYERS)
        .map(|layer| {
            if layer < NUM_DENSE_LAYERS {
                "dense"
            } else {
                "sparse"
            }
            .to_string()
        })
        .collect();
    ensure!(
        raw.mlp_layer_types == expected_mlp,
        "mlp_layer_types must be dense for layers 0..2 and sparse for 3..77"
    );
    let expected_indexers: Vec<String> = (0..NUM_LAYERS)
        .map(|layer| {
            if FULL_INDEX_LAYERS.contains(&layer) {
                "full"
            } else {
                "shared"
            }
            .to_string()
        })
        .collect();
    ensure!(
        raw.indexer_types == expected_indexers,
        "indexer_types do not match the GLM-5.2 21-full/57-shared schedule"
    );

    Ok(Glm52ModelCfg {
        hidden_dim: Dim::param("hidden_size", raw.hidden_size),
        dense_intermediate_dim: Dim::param("intermediate_size", raw.intermediate_size),
        num_attention_heads: Dim::param("num_attention_heads", raw.num_attention_heads),
        raw_num_kv_heads: Dim::param("num_key_value_heads", raw.num_key_value_heads),
        q_lora_rank: Dim::param("q_lora_rank", raw.q_lora_rank),
        kv_lora_rank: Dim::param("kv_lora_rank", raw.kv_lora_rank),
        qk_nope_head_dim: Dim::param("qk_nope_head_dim", raw.qk_nope_head_dim),
        rope_dim: Dim::param("qk_rope_head_dim", raw.qk_rope_head_dim),
        v_head_dim: Dim::param("v_head_dim", raw.v_head_dim),
        model_num_index_heads: Dim::param("index_n_heads", raw.index_n_heads),
        index_head_dim: Dim::param("index_head_dim", raw.index_head_dim),
        index_top_k: raw.index_topk,
        max_context: Dim::param("max_position_embeddings", raw.max_position_embeddings),
        num_experts: Dim::param("n_routed_experts", raw.n_routed_experts),
        router_top_k: raw.num_experts_per_tok,
        moe_intermediate_dim: Dim::param("moe_intermediate_size", raw.moe_intermediate_size),
        num_shared_experts: raw.n_shared_experts,
        vocab_size: Dim::param("vocab_size", raw.vocab_size),
        num_layers: raw.num_hidden_layers,
        num_mtp_layers: raw.num_nextn_predict_layers,
        dtype: DType::Bf16,
        router_semantic_dtype: DType::Fp32,
        full_index_layers: FULL_INDEX_LAYERS.to_vec(),
        indexer_types: raw.indexer_types,
        mlp_layer_types: raw.mlp_layer_types,
        index_share_for_mtp_iteration: raw.index_share_for_mtp_iteration,
    })
}

/// MTP execution identity shared by regular and speculative GLM-5.2 graphs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Glm52MtpMode {
    #[default]
    Off,
    FullIndex,
    IndexShare,
}
