//! The DFlash2 draft checkpoints the proposer worklets accept.
//!
//! A DFlash2 draft is a small block-diffusion model whose shapes come from its
//! own `config.json`, not from the target's. The worklets reject any config
//! that is not one of these checkpoints, so a typo cannot silently price a draft
//! nobody ships; adding a checkpoint is adding an entry here.

/// One draft checkpoint's dimensions, each named after its `config.json` key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dflash2Checkpoint {
    pub name: &'static str,
    /// `hidden_size`.
    pub hidden_dim: u32,
    /// `intermediate_size`.
    pub intermediate_dim: u32,
    /// `num_attention_heads`.
    pub num_qo_heads: u32,
    /// `num_key_value_heads`.
    pub num_kv_heads: u32,
    /// `head_dim`.
    pub head_dim: u32,
    /// `num_hidden_layers`.
    pub num_draft_layers: u32,
    /// `len(dflash_config.target_layer_ids)`: target hidden states `fc` reads.
    pub num_aux_layers: u32,
    /// `vocab_size`.
    pub vocab_size: u32,
    /// `dflash_config.conv_kernel_size`: the convolution reaches back one position.
    pub conv_taps: u32,
    /// `dflash_config.conv_group_size`.
    pub conv_group_size: u32,
    /// `dflash_config.selector_rank`.
    pub selector_rank: u32,
    /// `dflash_config.selector_top_k`.
    pub selector_top_k: u32,
    /// `dflash_config.block_size`: one bonus position plus `block_size - 1` drafts.
    pub block_size: u32,
    /// `sliding_window`, which every draft layer uses.
    pub sliding_window: u32,
}

impl Dflash2Checkpoint {
    /// Drafted tokens per pass: the block less the position it conditions on.
    pub const fn draft_tokens(&self) -> u32 {
        self.block_size - 1
    }
}

/// The GLM-5.3 proposer (`target_layer_ids` `[5, 19, 33, 47, 61, 75]`).
pub const GLM53_DFLASH2: Dflash2Checkpoint = Dflash2Checkpoint {
    name: "GLM-5.3 DFlash2",
    hidden_dim: 6144,
    intermediate_dim: 12288,
    num_qo_heads: 64,
    num_kv_heads: 8,
    head_dim: 128,
    num_draft_layers: 6,
    num_aux_layers: 6,
    vocab_size: 154880,
    conv_taps: 2,
    conv_group_size: 16,
    selector_rank: 256,
    selector_top_k: 16,
    block_size: 8,
    sliding_window: 2048,
};

/// `incoai/GLM-5.3-Flash-DFlash2` (`target_layer_ids` `[5, 14, 24, 33, 42]` of
/// the target's 45 layers).
pub const GLM53_FLASH_DFLASH2: Dflash2Checkpoint = Dflash2Checkpoint {
    name: "GLM-5.3-Flash DFlash2",
    hidden_dim: 4096,
    intermediate_dim: 12288,
    num_qo_heads: 32,
    num_kv_heads: 8,
    head_dim: 128,
    num_draft_layers: 5,
    num_aux_layers: 5,
    vocab_size: 154880,
    conv_taps: 2,
    conv_group_size: 16,
    selector_rank: 256,
    selector_top_k: 16,
    block_size: 8,
    sliding_window: 2048,
};

pub const DFLASH2_CHECKPOINTS: [Dflash2Checkpoint; 2] = [GLM53_DFLASH2, GLM53_FLASH_DFLASH2];

/// Accept `fields` (name, configured value, the checkpoint's value) when they
/// match one known checkpoint; otherwise name the closest one's mismatches.
pub(crate) fn validate_against_checkpoints(
    fields: impl Fn(&Dflash2Checkpoint) -> Vec<(&'static str, u32, u32)>,
) -> Result<(), String> {
    let mut best: Option<(usize, String)> = None;
    for checkpoint in &DFLASH2_CHECKPOINTS {
        let mismatches: Vec<String> = fields(checkpoint)
            .into_iter()
            .filter(|(_, actual, required)| actual != required)
            .map(|(name, actual, required)| format!("{name} must be {required}, got {actual}"))
            .collect();
        if mismatches.is_empty() {
            return Ok(());
        }
        if best
            .as_ref()
            .is_none_or(|(count, _)| mismatches.len() < *count)
        {
            best = Some((
                mismatches.len(),
                format!("closest is {}: {}", checkpoint.name, mismatches.join("; ")),
            ));
        }
    }
    Err(format!(
        "matches no DFlash2 checkpoint; {}",
        best.map(|(_, message)| message).unwrap_or_default()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_checkpoints_draft_seven_tokens_per_block() {
        assert_eq!(GLM53_DFLASH2.draft_tokens(), 7);
        assert_eq!(GLM53_FLASH_DFLASH2.draft_tokens(), 7);
    }

    #[test]
    fn a_mixed_config_matches_neither_checkpoint() {
        // Flash's hidden width with GLM-5.3's layer count is no shipped draft.
        let mixed = |c: &Dflash2Checkpoint| {
            vec![
                ("hidden_dim", 4096, c.hidden_dim),
                ("num_draft_layers", 6, c.num_draft_layers),
            ]
        };
        let err = validate_against_checkpoints(mixed).unwrap_err();
        assert!(err.contains("matches no DFlash2 checkpoint"), "{err}");
        let flash = |c: &Dflash2Checkpoint| vec![("hidden_dim", 4096, c.hidden_dim)];
        assert!(validate_against_checkpoints(flash).is_ok());
    }
}
