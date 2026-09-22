//! CPU handoff test: Arrow request geometry -> Python accountant -> R6/R7.
//! Location sets are independently checked against real compiled trees by the
//! simulator's speculative_necessary_work_maps_cover_the_compiled_locations test.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use datafusion::{arrow::json::ReaderBuilder, prelude::SessionContext};
use serde_json::{json, Value};

use super::{
    floors, ladder, levels::BaseRungs, location::LocationCatalog, prepare::KernelLocation, spec,
};

#[tokio::test]
async fn speculative_exact_iteration_handoff_reconciles_r6_r7() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    // The run must sit under a logs root for discovery to find it, but `logs/`
    // is untracked working data — a fresh clone or worktree does not have one,
    // and `tempdir_in` on a missing directory fails before the test starts.
    let logs = repo.join("logs");
    std::fs::create_dir_all(&logs).unwrap();
    for (mode, depth, mapping) in [
        ("index_share", 5, "index_share"),
        ("full_index", 5, "full_index"),
        ("index_share", 1, "single_draft"),
    ] {
        let dir = tempfile::tempdir_in(&logs).unwrap();
        std::fs::create_dir(dir.path().join("raw")).unwrap();
        std::fs::write(dir.path().join("raw/params.json"), json!({"pools":{"main":{"groups":[{
            "gpu":"NVIDIA B200", "arch":{"type":"glm52_vllm_nvfp4_dsa_moe_speculative",
                "model_config":"model/config/glm52_nvfp4.json", "mtp_mode":mode,"draft_tokens":depth}
        }]}}}).to_string()).unwrap();
        let width = depth + 1;
        let geometry = json!({"draft_tokens":depth,"max_model_len":8192,
            "prefill":[[0,8]],"decode":[[100+width,width]]});
        let list_u32 = || DataType::List(Arc::new(Field::new("item", DataType::UInt32, false)));
        let groups = DataType::Struct(
            vec![
                Field::new("batch_tokens", DataType::UInt32, false),
                Field::new("prefill_tokens", DataType::UInt32, false),
                Field::new("decode_request_count", DataType::UInt32, false),
                Field::new("decode_kv_total", DataType::UInt32, false),
                Field::new("prefill_prefix_lens", list_u32(), false),
                Field::new("prefill_append_lens", list_u32(), false),
                Field::new("speculative_geometry", DataType::Utf8, true),
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
        let encoded = json!({"pool_tag":"main","worker_id":0,"iter_id":7,"groups":[{
            "batch_tokens":8+width,"prefill_tokens":8,"decode_request_count":1,"decode_kv_total":100,
            "prefill_prefix_lens":[0],"prefill_append_lens":[8],"speculative_geometry":geometry.to_string()
        }]}).to_string();
        let batch = ReaderBuilder::new(Arc::new(schema))
            .build(encoded.as_bytes())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_batch("cost_log", batch).unwrap();
        let label = floors::compute_iteration_label(&ctx, dir.path(), "main", 0, 7, 1)
            .await
            .unwrap();
        let map: Value = serde_json::from_slice(
            &std::fs::read(repo.join(format!(
                "model/work/location_maps/glm52_speculative_{mapping}.json"
            )))
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
        for gpu_count in [4.0, 8.0] {
            let mut ladder = ladder::KernelLadder::worker("main", 0, &BaseRungs::default(), vec![]);
            let attribution = catalog
                .attribute_ladder(
                    "main",
                    &locations,
                    &mut ladder,
                    &label,
                    gpu,
                    gpu_count,
                    ladder::NecessaryWorkPolicy::BatchLocked,
                )
                .unwrap();
            assert_eq!(attribution.mapping_id, map["mapping_id"].as_str().unwrap());
            assert!(
                (ladder.rungs.segmented_necessary.unwrap() - label.floors.segmented).abs() < 1e-12
            );
            assert!(
                (ladder.rungs.scope_fused_necessary.unwrap() - label.floors.fused).abs() < 1e-12
            );
            assert!(label.floors.segmented >= label.floors.fused && label.floors.fused > 0.0);
            assert!(ladder
                .kernels
                .iter()
                .all(|kernel| kernel.necessary_work.is_some()));
            ladder.to_json().unwrap();
        }
        let compositions = floors::compute_batch_locked_run_labels(
            &ctx,
            dir.path(),
            50_000,
            &std::collections::HashSet::new(),
        )
        .await
        .unwrap();
        assert!(compositions.errors.is_empty(), "{:?}", compositions.errors);
        assert_eq!(compositions.batch_locked_iterations, Some(1));
    }
}
