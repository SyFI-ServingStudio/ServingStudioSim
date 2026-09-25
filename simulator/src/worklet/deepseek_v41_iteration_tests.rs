//! Whole-iteration structure check for the DeepSeek-V4.1-Flash worklets:
//! prologue + Engram prefetch + 40 attention layers + 2 Engram blocks + 40 FFN
//! layers + head, built against the enumerate bridge (no DB, no GPU) and
//! compiled in the L4 order. Serial copies of gated side branches are the same
//! launch as their concurrent copy, so launch counts exclude `.serial` leaves.

use std::collections::BTreeMap;

use super::deepseek_v41_attention_tp::{production_layer, tests as attn};
use super::deepseek_v41_common::SERIAL_COPY_SUFFIX;
use super::deepseek_v41_engram_prefetch_local::tests as prefetch;
use super::deepseek_v41_engram_tp::tests as engram;
use super::deepseek_v41_head_tp::tests as head;
use super::deepseek_v41_moe_ffn_ep::tests as ffn;
use super::deepseek_v41_prologue_tp::tests as prologue;
use super::*;
use crate::timing::{CostNode, CostTree, CostTreeBuilder, PerfApiBridge};

fn compile_iteration() -> CostTree {
    let bridge = PerfApiBridge::new_uninit_for_test();
    bridge.enable_enumerate();
    let attention = |layer: u32| {
        DeepseekV41AttentionTpWorklet::build(
            format!("model.layers.{layer}.attn"),
            DeepseekV41AttentionTpWorklet::resolve_config(&attn::config(layer)),
            &bridge,
        )
        .unwrap()
    };
    let moe = |layer: u32| {
        DeepseekV41MoeFfnEpWorklet::build(
            format!("model.layers.{layer}.ffn"),
            DeepseekV41MoeFfnEpWorklet::resolve_config(&ffn::config()),
            &bridge,
        )
        .unwrap()
    };
    let engram_block = |layer: u32| {
        DeepseekV41EngramTpWorklet::build(
            format!("model.layers.{layer}.engram"),
            DeepseekV41EngramTpWorklet::resolve_config(&engram::config()),
            &bridge,
        )
        .unwrap()
    };
    let prologue = DeepseekV41PrologueTpWorklet::build(
        "model.prologue".into(),
        DeepseekV41PrologueTpWorklet::resolve_config(&prologue::config()),
        &bridge,
    )
    .unwrap();
    let lookups = DeepseekV41EngramPrefetchLocalWorklet::build(
        "model.engram_prefetch".into(),
        DeepseekV41EngramPrefetchLocalWorklet::resolve_config(&prefetch::config()),
        &bridge,
    )
    .unwrap();
    let head = DeepseekV41HeadTpWorklet::build(
        "model.head".into(),
        DeepseekV41HeadTpWorklet::resolve_config(&head::config()),
        &bridge,
    )
    .unwrap();

    let mut b = CostTreeBuilder::new();
    let mut sections = vec![prologue.compile(&mut b)];
    // Hash -> layer-1 consumer: layer 0 attention and FFN race the lookups.
    let main_path = CostNode::Sum(vec![attention(0).compile(&mut b), moe(0).compile(&mut b)]);
    sections.push(lookups.compile_joined(&mut b, main_path));
    for layer in 1..40 {
        if production_layer(layer).entry == DeepseekV41AttentionEntry::AfterEngram {
            sections.push(engram_block(layer).compile(&mut b));
        }
        sections.push(attention(layer).compile(&mut b));
        sections.push(moe(layer).compile(&mut b));
    }
    sections.push(head.compile(&mut b));
    b.finish(CostNode::Sum(sections))
}

fn n(slot: &crate::timing::LeafDesc) -> u64 {
    slot.kernel_config["n"]["value"].as_u64().unwrap()
}

#[test]
fn one_iteration_has_the_captured_launch_counts() {
    let tree = compile_iteration();
    let launches: Vec<_> = tree
        .slots
        .iter()
        .filter(|s| !s.name.ends_with(&format!(".{SERIAL_COPY_SUFFIX}")))
        .collect();
    let mut by_kind = BTreeMap::<&str, usize>::new();
    for slot in &launches {
        *by_kind.entry(slot.kind.as_str()).or_default() += 1;
    }
    let count = |kind: &str| by_kind.get(kind).copied().unwrap_or(0);
    let named = |suffix: &str| launches.iter().filter(|s| s.name.ends_with(suffix)).count();

    assert_eq!(count("mhc_fused_post_pre_rms_norm"), 77);
    assert_eq!(count("all_reduce_fusion"), 81);
    assert_eq!(count("dsa_paged_mqa_logits_decode"), 8);
    assert_eq!(count("dsa_mqa_logits_prefill"), 8);
    assert_eq!(named(".decode_topk"), 8);
    assert_eq!(named(".prefill_topk"), 8);
    assert_eq!(named("compressor.kv_score_proj"), 4);
    assert_eq!(named("compressor.save_compress_norm"), 4);
    assert_eq!(named("compressor.nvfp4_insert"), 4);
    assert_eq!(count("engram_lookup"), 2);
    assert_eq!(count("deepseek_v41_qnorm_rope_kv_insert"), 40);
    assert_eq!(named("mega_attn.decode"), 40);
    assert_eq!(named("mega_attn.prefill"), 40);
    assert_eq!(count("nvfp4_fused_moe"), 40);
    assert_eq!(count("batched_gemm"), 40);

    let fp32: Vec<u64> = launches
        .iter()
        .filter(|s| s.kind == "gemm_fp32_output")
        .map(|s| n(s))
        .collect();
    assert_eq!(fp32.iter().filter(|&&n| n == 384).count(), 40, "router gates");
    assert_eq!(fp32.iter().filter(|&&n| n == 1024).count(), 3, "ratio-2 compressors");
    assert_eq!(fp32.iter().filter(|&&n| n == 512).count(), 1, "ratio-1 compressor");
    assert_eq!(fp32.len(), 44);

    let mut mxfp8 = BTreeMap::<(u64, u64), usize>::new();
    for slot in launches
        .iter()
        .filter(|s| s.kind == "single_gemm" && s.kernel_config["dtype"] == "mxfp8_e4m3")
    {
        let k = slot.kernel_config["k"]["value"].as_u64().unwrap();
        *mxfp8.entry((k, n(slot))).or_default() += 1;
    }
    assert_eq!(mxfp8.values().sum::<usize>(), 210);
    assert_eq!(
        mxfp8,
        BTreeMap::from([
            ((5120, 1792), 40),
            ((1280, 8192), 40),
            ((2048, 5120), 40),
            ((576, 5120), 40),
            ((5120, 1152), 40),
            ((1280, 4096), 8),
            ((6144, 25600), 2),
        ])
    );
    // lm_head is the only bf16 dense GEMM.
    assert_eq!(
        launches
            .iter()
            .filter(|s| s.kind == "single_gemm" && s.kernel_config["dtype"] == "bf16")
            .count(),
        1
    );
    // The two NCCL AllGather proxies (Engram) plus the logits AllGather.
    assert_eq!(count("all_reduce"), 3);
}

#[test]
fn gated_side_branches_carry_exactly_one_serial_copy_each() {
    let tree = compile_iteration();
    let serial: Vec<_> = tree
        .slots
        .iter()
        .filter(|s| s.name.ends_with(&format!(".{SERIAL_COPY_SUFFIX}")))
        .collect();
    // Stage A: 4 compressor + 8 weights_proj; B1/B2: 4 + 4 compressor;
    // FFN: 3 shared-expert launches x 40.
    assert_eq!(serial.len(), 4 + 8 + 4 + 4 + 120);
    for copy in serial {
        let base = copy.name.strip_suffix(".serial").unwrap();
        let original = tree
            .slots
            .iter()
            .find(|s| s.name == base)
            .unwrap_or_else(|| panic!("serial copy {} has no concurrent copy", copy.name));
        assert_eq!(original.kind, copy.kind);
        assert_eq!(original.kernel_config, copy.kernel_config);
    }
}
