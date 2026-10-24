//! The DFlash2 drafter checkpoint that TensorFold pairs with Qwen3.8-27B
//! (`DFlash2DraftModel`, model_type `qwen3`), as its config.json describes it.
//! The config is kept verbatim (`model/config/qwen3_8_27b_dflash2.json` is
//! `Qwen3.8-27B-DFlash2/config.json`).
//!
//! It is read the way TensorFold's `DFlash2.__init__` reads it
//! (`tensorfold/families/qwen3_5/cuda/dflash2.py`):
//!
//! - `num_hidden_layers` sliding-attention layers, each wrapped in two
//!   two-tap dynamic causal convolutions (`attention_conv`, `mlp_conv`). A
//!   conv's `kernel_projection` maps a row onto `2 x 2 x hidden /
//!   conv_group_size` taps.
//! - GQA attention: `num_attention_heads` query and `num_key_value_heads`
//!   key/value heads of `head_dim`. The fast path stacks q|k|v into one `qkv`
//!   weight and k|v into one `kv` weight (for the taps).
//! - Context: TensorFold keeps the last `sliding_window - 1` committed taps a
//!   stream (`self.window`).
//! - Taps: the target's residual stream after `target_layer_ids`; `fc` maps
//!   their concatenation (`len(target_layer_ids) x hidden`) back to `hidden`.
//! - Head: a `selector_rank` `hidden_projection`, then the target's lm_head
//!   on a vocabulary subset and a `selector_top_k` selection.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

/// `dflash_config` of the drafter checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Dflash2Config {
    /// Training block (TensorFold's planner floor); GB10 drafts `max_rows`.
    pub block_size: u32,
    pub conv_group_size: u32,
    pub conv_kernel_size: u32,
    pub mask_token_id: u32,
    pub selector_rank: u32,
    pub selector_top_k: u32,
    pub target_layer_ids: Vec<u32>,
}

/// The drafter checkpoint's dimensions.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Qwen35Dflash2DraftCfg {
    pub hidden_size: u32,
    pub head_dim: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub sliding_window: u32,
    pub rms_norm_eps: f64,
    /// Whether a block row sees only the earlier block rows (default true, as
    /// TensorFold reads it).
    #[serde(default = "default_causal")]
    pub is_causal: bool,
    pub hidden_act: String,
    pub vocab_size: u32,
    pub dflash_config: Dflash2Config,
}

fn default_causal() -> bool {
    true
}

impl Qwen35Dflash2DraftCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading DFlash2 config {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing DFlash2 config {}", path.display()))?;
        cfg.check()?;
        Ok(cfg)
    }

    fn check(&self) -> Result<()> {
        ensure!(self.hidden_act == "silu", "DFlash2 models a SiLU MLP");
        ensure!(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "DFlash2: {} query heads do not group over {} kv heads",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        let group = self.dflash_config.conv_group_size;
        ensure!(
            group > 0 && self.hidden_size % group == 0,
            "DFlash2: conv_group_size {group} does not divide hidden {}",
            self.hidden_size
        );
        // `_dconv` mixes a row with the row before it only.
        ensure!(
            self.dflash_config.conv_kernel_size == 2,
            "DFlash2: TensorFold's _dconv is two-tap, got conv_kernel_size {}",
            self.dflash_config.conv_kernel_size
        );
        ensure!(
            self.sliding_window >= 2,
            "DFlash2: sliding_window {} leaves no context",
            self.sliding_window
        );
        ensure!(
            !self.dflash_config.target_layer_ids.is_empty(),
            "DFlash2: no target_layer_ids to tap"
        );
        Ok(())
    }

    /// Context rows TensorFold keeps a stream (`sliding_window - 1`).
    pub fn window(&self) -> u32 {
        self.sliding_window - 1
    }

    /// Target layers tapped.
    pub fn num_taps(&self) -> u32 {
        self.dflash_config.target_layer_ids.len() as u32
    }

    /// Width of one committed row's taps: the `fc` input.
    pub fn tap_width(&self) -> u32 {
        self.num_taps() * self.hidden_size
    }

    /// Query channels (`o_proj`'s input).
    pub fn q_width(&self) -> u32 {
        self.num_attention_heads * self.head_dim
    }

    /// One of the key or value projections' widths.
    pub fn kv_head_width(&self) -> u32 {
        self.num_key_value_heads * self.head_dim
    }

    /// The fast path's stacked `qkv` projection width.
    pub fn qkv_width(&self) -> u32 {
        self.q_width() + 2 * self.kv_head_width()
    }

    /// The taps' stacked `kv` projection width.
    pub fn kv_width(&self) -> u32 {
        2 * self.kv_head_width()
    }

    /// A conv's `kernel_projection` width: two branches x two taps x one
    /// value per channel group.
    pub fn conv_kernel_width(&self) -> u32 {
        2 * 2 * self.hidden_size / self.dflash_config.conv_group_size
    }

    /// The stacked gate|up projection width (one grouped launch).
    pub fn gate_up_width(&self) -> u32 {
        2 * self.intermediate_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Qwen35Dflash2DraftCfg {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_8_27b_dflash2.json");
        Qwen35Dflash2DraftCfg::from_json(&path).unwrap()
    }

    #[test]
    fn qwen38_27b_dflash2_reads_its_shape() {
        let cfg = cfg();
        assert_eq!(
            (
                cfg.num_hidden_layers,
                cfg.hidden_size,
                cfg.intermediate_size
            ),
            (5, 5120, 17408)
        );
        assert_eq!(
            (
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim
            ),
            (32, 8, 128)
        );
        assert_eq!((cfg.window(), cfg.is_causal), (2047, false));
        assert_eq!(cfg.dflash_config.target_layer_ids, [5, 19, 33, 47, 61]);
        // The drafter linear shapes of arch_notes.md.
        assert_eq!(cfg.tap_width(), 25600);
        assert_eq!(cfg.kv_width(), 2048);
        assert_eq!(cfg.conv_kernel_width(), 1280);
        assert_eq!(cfg.qkv_width(), 6144);
        assert_eq!(cfg.q_width(), 4096);
        assert_eq!(cfg.gate_up_width(), 34816);
        assert_eq!(cfg.dflash_config.selector_rank, 256);
        assert_eq!(cfg.dflash_config.selector_top_k, 16);
    }

    #[test]
    fn a_three_tap_conv_is_refused() {
        let mut bad = cfg();
        bad.dflash_config.conv_kernel_size = 3;
        assert!(bad.check().unwrap_err().to_string().contains("two-tap"));
    }
}
