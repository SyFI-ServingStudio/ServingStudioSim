//! CPU handoff test for GLM-5.3-Flash: per-request Arrow geometry -> Python
//! accountant (exact kpool DSA interactions) -> semantic location map -> R6/R7.
//! The simulator's necessary_work_map_covers_the_compiled_locations test checks
//! the same map against the real compiled tree.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use datafusion::{arrow::json::ReaderBuilder, prelude::SessionContext};
use serde_json::{json, Value};

use super::{
    floors, ladder, levels::BaseRungs, location::LocationCatalog, prepare::KernelLocation, spec,
};

#[tokio::test]
async fn glm53_exact_iteration_handoff_reconciles_r6_r7() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let logs = repo.join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    let dir = tempfile::tempdir_in(&logs).unwrap();
    std::fs::create_dir(dir.path().join("raw")).unwrap();
    std::fs::write(
        dir.path().join("raw/params.json"),
        json!({"pools":{"main":{"groups":[{
            "gpu":"NVIDIA B200", "arch":{"type":"glm53_flash_vllm_fp8_kda_dsa_moe",
                "model_config":"model/config/glm53_flash.json", "fp8":true}
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
    // The mixed timing-predict shape: a 29-token chunk at prefix 2019 plus decodes.
    let encoded = json!({"pool_tag":"main","worker_id":0,"iter_id":3,"groups":[{
        "batch_tokens":32,"prefill_tokens":29,"decode_request_count":3,"decode_kv_total":9000,
        "prefill_prefix_lens":[2019],"prefill_append_lens":[29],"decode_kv_lens":[3000,3000,3000]
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
            repo.join("model/work/location_maps/glm53_flash_vllm_fp8_kda_dsa_moe_unified.json"),
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
        "glm53-flash-vllm-fp8-kda-dsa-moe-unified-v1"
    );
    assert!((ladder.rungs.segmented_necessary.unwrap() - label.floors.segmented).abs() < 1e-12);
    assert!((ladder.rungs.scope_fused_necessary.unwrap() - label.floors.fused).abs() < 1e-12);
    assert!(label.floors.segmented >= label.floors.fused && label.floors.fused > 0.0);
    // The kpool cap is applied per request: decodes at 3000 select 2048 + 0 keys.
    let decode = label
        .segments
        .iter()
        .find(|segment| segment.name == "dsa_moe.attn.decode")
        .unwrap();
    assert_eq!(decode.flops, 3.0 * 2.0 * 64.0 * 1024.0 * 2048.0 * 11.0);
}
