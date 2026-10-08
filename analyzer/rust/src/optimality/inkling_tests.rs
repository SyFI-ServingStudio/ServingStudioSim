//! CPU handoff test for Inkling-Small-NVFP4 at TP4 + EP4: per-request Arrow
//! geometry -> Python accountant (sliding-window / global relative attention,
//! short-conv states) -> semantic location map -> R6/R7. The simulator's
//! necessary_work_map_covers_the_compiled_locations test checks the same map
//! against the real compiled tree.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use datafusion::{arrow::json::ReaderBuilder, prelude::SessionContext};
use serde_json::{json, Value};

use super::{
    floors, ladder, levels::BaseRungs, location::LocationCatalog, prepare::KernelLocation, spec,
};

#[tokio::test]
async fn inkling_exact_iteration_handoff_reconciles_r6_r7() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    // The labeler is resolved from the run dir's owning checkout, so the run
    // must sit inside this one; `target/` is never a symlink to another tree.
    let scratch = repo.join("target");
    std::fs::create_dir_all(&scratch).unwrap();
    let dir = tempfile::tempdir_in(&scratch).unwrap();
    std::fs::create_dir(dir.path().join("raw")).unwrap();
    std::fs::write(
        dir.path().join("raw/params.json"),
        json!({"pools":{"main":{"groups":[{
            "gpu":"NVIDIA B200", "arch":{"type":"inkling_vllm_nvfp4_rel_attn_moe",
                "model_config":"model/config/inkling_small_nvfp4.json"}
        }]}}})
        .to_string(),
    )
    .unwrap();
    let list_u32 = || DataType::List(Arc::new(Field::new("item", DataType::UInt32, false)));
    let groups = DataType::Struct(
        vec![
            Field::new("batch_tokens", DataType::UInt32, false),
            Field::new("prefill_tokens", DataType::UInt32, false),
            Field::new("decode_request_count", DataType::UInt32, false),
            Field::new("decode_kv_total", DataType::UInt32, false),
            Field::new("prefill_prefix_lens", list_u32(), false),
            Field::new("prefill_append_lens", list_u32(), false),
            Field::new("decode_kv_lens", list_u32(), true),
        ]
        .into(),
    );
    let schema = Schema::new(vec![
        Field::new("pool_tag", DataType::Utf8, false),
        Field::new("worker_id", DataType::UInt16, false),
        Field::new("iter_id", DataType::UInt64, false),
        Field::new(
            "groups",
            DataType::List(Arc::new(Field::new("item", groups, false))),
            false,
        ),
    ]);
    // Timing-predict case 3: a 2020-token prefill plus 8 decodes, some inside
    // and some past the 512-key window.
    let encoded = json!({"pool_tag":"main","worker_id":0,"iter_id":3,"groups":[{
        "batch_tokens":2028,"prefill_tokens":2020,"decode_request_count":8,
        "decode_kv_total":15828,"prefill_prefix_lens":[0],"prefill_append_lens":[2020],
        "decode_kv_lens":[4145,342,1147,2208,4293,490,1071,2132]
    }]})
    .to_string();
    let batch = ReaderBuilder::new(Arc::new(schema))
        .build(encoded.as_bytes())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_batch("cost_log", batch).unwrap();
    let label = floors::compute_iteration_label(&ctx, dir.path(), "main", 0, 3, 1)
        .await
        .unwrap();
    let map: Value = serde_json::from_slice(
        &std::fs::read(
            repo.join("model/work/location_maps/inkling_vllm_nvfp4_tp4_ep4_unified.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let locations: Vec<_> = map["locations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| KernelLocation {
            name: row["location"].as_str().unwrap().into(),
            kind: "single_gemm".into(),
            is_communication: false,
        })
        .collect();
    let catalog = LocationCatalog::load(repo, dir.path()).unwrap();
    let (_, gpu) = spec::load_gpu_spec(repo, "NVIDIA B200").unwrap();
    let mut ladder = ladder::KernelLadder::worker("main", 0, &BaseRungs::default(), vec![]);
    let attribution = catalog
        .attribute_ladder(
            "main",
            &locations,
            &mut ladder,
            &label,
            gpu,
            4.0,
            ladder::NecessaryWorkPolicy::BatchLocked,
        )
        .unwrap();
    assert_eq!(
        attribution.mapping_id,
        "inkling-vllm-nvfp4-tp4-ep4-unified-v1"
    );
    assert!((ladder.rungs.segmented_necessary.unwrap() - label.floors.segmented).abs() < 1e-12);
    assert!((ladder.rungs.scope_fused_necessary.unwrap() - label.floors.fused).abs() < 1e-12);
    assert!(label.floors.segmented >= label.floors.fused && label.floors.fused > 0.0);
    // The Python accountant's totals for the same geometry.
    let flops: f64 = label.segments.iter().map(|segment| segment.flops).sum();
    let bytes: f64 = label.segments.iter().map(|segment| segment.bytes).sum();
    assert_eq!(label.segments.len(), 96);
    assert_eq!(flops, 42_791_756_742_656.0);
    assert!((bytes - 165_667_202_212.0).abs() < 1.0);
    let segment = |name: &str| {
        label
            .segments
            .iter()
            .find(|segment| segment.name == name)
            .unwrap()
            .flops
    };
    // Global decodes see every cached key; local ones at most 512 (32 heads x 128).
    assert_eq!(
        segment("global_nvfp4_moe.attn.decode"),
        4.0 * 32.0 * 128.0 * 15828.0 * 7.0
    );
    assert_eq!(
        segment("local_nvfp4_moe.attn.decode"),
        4.0 * 32.0 * 128.0 * 3904.0 * 32.0
    );
}
