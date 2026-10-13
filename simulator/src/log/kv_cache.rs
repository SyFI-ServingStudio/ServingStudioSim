//! `kv_cache.json` — every KV pool's capacity breakdown (L7).
//!
//! One entry per registered KV pool (worker × group, the same key
//! `run_meta.json`'s `kv_pools` and the `kv_snapshot` rows use), each the
//! pool's [`KvCacheReport`]: the declared layout, the resolved options, what
//! each part is charged, and the concurrency at `max_model_len` under both
//! `kv_overhead` modes. Written once after the flow is built, next to
//! `run_meta.json`; `simulator kv-cache` prints the same document, or
//! [`kv_cache_text`] for reading.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use serde_json::json;

use crate::arch::kv_layout::KvGrowth;
use crate::worker::kv::{KvCacheReport, KvOverhead};
use crate::worker::GpuCluster;

const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
struct PoolEntry<'a> {
    pool_tag: &'static str,
    pool: u16,
    worker_id: u16,
    group_id: u16,
    gpu_name: &'a str,
    #[serde(flatten)]
    report: &'a KvCacheReport,
}

/// `{"schema_version": 1, "pools": [{pool_tag, pool, worker_id, group_id,
/// gpu_name, ...report}]}`, in registration order.
pub fn kv_cache_document(cluster: &GpuCluster) -> serde_json::Value {
    let pools: Vec<_> = cluster
        .kv_pools()
        .map(|(pool_tag, pool, worker_id, group_id, facts)| PoolEntry {
            pool_tag,
            pool,
            worker_id,
            group_id,
            gpu_name: gpu_name(cluster, pool, worker_id),
            report: &facts.report,
        })
        .collect();
    json!({
        "schema_version": SCHEMA_VERSION,
        "pools": pools,
    })
}

/// Write `<log_dir>/raw/kv_cache.json`.
pub fn write_kv_cache(log_dir: &Path, cluster: &GpuCluster) -> Result<()> {
    let raw = log_dir.join("raw");
    std::fs::create_dir_all(&raw)?;
    let document = kv_cache_document(cluster);
    std::fs::write(
        raw.join("kv_cache.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;
    Ok(())
}

/// The reports as text, one block per distinct report of a pool: a DP pool
/// whose workers and partitions all resolve alike prints once. Bytes are per
/// GPU; units stay in the JSON.
pub fn kv_cache_text(cluster: &GpuCluster) -> String {
    struct Block<'a> {
        pool_tag: &'static str,
        pool: u16,
        gpu_name: &'a str,
        report: &'a KvCacheReport,
        workers: Vec<u16>,
        groups: Vec<u16>,
    }
    let mut blocks: Vec<Block> = Vec::new();
    for (pool_tag, pool, worker_id, group_id, facts) in cluster.kv_pools() {
        let report = facts.report.as_ref();
        let index = match blocks.iter().position(|block| {
            block.pool_tag == pool_tag && block.pool == pool && block.report == report
        }) {
            Some(index) => index,
            None => {
                blocks.push(Block {
                    pool_tag,
                    pool,
                    gpu_name: gpu_name(cluster, pool, worker_id),
                    report,
                    workers: Vec::new(),
                    groups: Vec::new(),
                });
                blocks.len() - 1
            }
        };
        let block = &mut blocks[index];
        block.workers.push(worker_id);
        block.groups.push(group_id);
    }
    if blocks.is_empty() {
        return "no KV pools\n".to_string();
    }
    let mut out = String::new();
    for block in blocks {
        let workers = distinct(block.workers);
        let partitions = distinct(block.groups).len();
        let identical = if workers.len() > 1 {
            format!(" ({} identical)", workers.len())
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "KV cache · pool {} · {} · {} {}{identical}",
            block.pool_tag,
            block.gpu_name,
            if workers.len() == 1 {
                "worker"
            } else {
                "workers"
            },
            id_ranges(workers),
        );
        write_report(&mut out, block.report, partitions);
        out.push('\n');
    }
    out
}

fn write_report(out: &mut String, report: &KvCacheReport, partitions: usize) {
    let layout = &report.layout;
    let options = &report.options;
    let accounting = &report.accounting;
    let result = &report.result;
    let shards = f64::from(report.attention_shards.max(1));
    let per_gpu = |bytes: u64| bytes as f64 / shards;
    let vllm = options.kv_overhead == KvOverhead::Vllm;

    // Shape, budget and options.
    let mut budget = format!(
        "budget {} per GPU",
        human_bytes(per_gpu(accounting.budget_bytes))
    );
    if accounting.reserved_units > 0 {
        let reserved = accounting.reserved_units * accounting.bytes_per_unit;
        let _ = write!(
            budget,
            ", {} set aside by the worker",
            human_bytes(per_gpu(reserved))
        );
    }
    let _ = writeln!(
        out,
        "  {} × {} · {budget}",
        plural(partitions as u64, "partition"),
        plural(u64::from(report.attention_shards), "attention shard"),
    );
    let mut line = format!("  kv_overhead {}", options.kv_overhead.as_str());
    if options.prefix_caching {
        let _ = write!(
            line,
            " · prefix caching on, resumes at {}",
            options.prefix_resume_points.as_str()
        );
        if let Some(every) = options.prefix_resume_interval_tokens {
            let _ = write!(line, " and every {} tokens", grouped(u64::from(every)));
        }
    } else {
        line.push_str(" · prefix caching off");
    }
    let has_window = layout
        .parts
        .iter()
        .any(|part| matches!(part.growth, KvGrowth::SlidingWindow { .. }));
    if vllm && has_window {
        match options.max_in_flight_tokens {
            Some(tokens) => {
                let _ = write!(line, " · {} tokens in flight", grouped(u64::from(tokens)));
            }
            None => line.push_str(" · whole prompt in flight"),
        }
    }
    let _ = writeln!(out, "{line}");

    // What the arch declared.
    let _ = writeln!(out, "\n  Layout, per GPU");
    let mut rows = vec![row(&[
        "part",
        "growth",
        "bytes/entry",
        "padded to",
        "copies",
        "prefix reuse",
    ])];
    for part in &layout.parts {
        rows.push(vec![
            part.name.to_string(),
            growth(part.growth),
            grouped_f(per_gpu(part.bytes_per_entry)),
            part.overhead
                .padded_bytes_per_entry
                .map_or(DASH.to_string(), |bytes| grouped_f(per_gpu(bytes))),
            part.overhead
                .copies
                .map_or(DASH.to_string(), |copies| copies.to_string()),
            if part.prefix_reusable { "yes" } else { "no" }.to_string(),
        ]);
    }
    write_table(out, &rows, &[L, L, R, R, R, L]);
    let _ = writeln!(
        out,
        "    engine: block {}, fixed {}",
        layout
            .engine
            .block_tokens
            .map_or(DASH.to_string(), |tokens| format!(
                "{} tokens",
                grouped(u64::from(tokens))
            )),
        human_bytes(per_gpu(layout.engine.fixed_device_bytes)),
    );
    if !vllm {
        let _ = writeln!(
            out,
            "    kv_overhead none charges none of padding, copies, engine block or fixed bytes"
        );
    }

    // One request at max_model_len.
    let Some(max_model_len) = result.max_model_len else {
        let _ = writeln!(
            out,
            "\n  max_model_len unknown: no per-request breakdown or concurrency"
        );
        return;
    };
    let _ = writeln!(
        out,
        "\n  One request at max_model_len {} in vLLM's estimate, per GPU",
        grouped(u64::from(max_model_len))
    );
    let total: u64 = result.request_parts.iter().map(|part| part.bytes).sum();
    let mut rows = vec![row(&["part", "holds", "bytes", "share"])];
    for ((part, held), charge) in layout
        .parts
        .iter()
        .zip(&result.request_parts)
        .zip(&accounting.parts)
    {
        let holds = match (part.growth, held.tokens_held) {
            (KvGrowth::PerToken { tokens_per_entry }, Some(tokens)) if tokens_per_entry > 1 => {
                format!(
                    "{} tokens, {} entries",
                    grouped(tokens),
                    grouped(held.entries)
                )
            }
            (KvGrowth::SlidingWindow { window_tokens }, Some(tokens))
                if tokens < u64::from(max_model_len) =>
            {
                let window = u64::from(window_tokens);
                if vllm && tokens > window - 1 {
                    format!(
                        "{} tokens ({} + {} in flight)",
                        grouped(tokens),
                        grouped(window - 1),
                        grouped(tokens - (window - 1))
                    )
                } else {
                    format!("{} tokens (window)", grouped(tokens))
                }
            }
            (KvGrowth::PerToken { .. }, Some(tokens)) if tokens > u64::from(max_model_len) => {
                let block = accounting.charged_block_tokens;
                format!(
                    "{} tokens ({} of {})",
                    grouped(tokens),
                    plural(tokens / block, "block"),
                    grouped(block)
                )
            }
            (_, Some(tokens)) => format!("{} tokens", grouped(tokens)),
            (_, None) => {
                let copies = charge.held_copies;
                if charge.align_copy == Some(true) {
                    format!(
                        "{} ({} held + align)",
                        plural(u64::from(copies) + 1, "copy"),
                        copies
                    )
                } else {
                    plural(u64::from(copies), "copy")
                }
            }
        };
        rows.push(vec![
            held.name.to_string(),
            holds,
            human_bytes(per_gpu(held.bytes)),
            percent(held.bytes, total),
        ]);
    }
    if result.request_parts.len() > 1 {
        rows.push(vec![
            "total".to_string(),
            String::new(),
            human_bytes(per_gpu(total)),
            String::new(),
        ]);
    }
    write_table(out, &rows, &[L, L, R, R]);

    // The result, in vLLM's words.
    let concurrency = |value: Option<f64>| value.map_or(DASH.to_string(), |x| format!("{x:.2}x"));
    let _ = writeln!(
        out,
        "\n  Maximum concurrency for {} tokens per request: {}   (kv_overhead {}: {})",
        grouped(u64::from(max_model_len)),
        concurrency(result.max_concurrency_at_max_model_len),
        result.other_kv_overhead.as_str(),
        concurrency(result.other_max_concurrency_at_max_model_len),
    );
    if let Some(tokens) = result.capacity_tokens {
        let _ = writeln!(out, "  GPU KV cache size: {} tokens", grouped(tokens));
    }
    if let (Some(held), Some(estimated)) = (
        result.admission_units_at_max_model_len,
        result.units_per_request_at_max_model_len,
    ) {
        if held < estimated {
            let _ = writeln!(
                out,
                "  A running request holds {} of the estimate's {}: the align copy is not held",
                human_bytes(per_gpu(held * accounting.bytes_per_unit)),
                human_bytes(per_gpu(estimated * accounting.bytes_per_unit)),
            );
        }
    }
}

const DASH: &str = "—";
const L: bool = false;
const R: bool = true;

fn row(cells: &[&str]) -> Vec<String> {
    cells.iter().map(|cell| cell.to_string()).collect()
}

/// Rows under a header, each column as wide as its widest cell, `right`
/// columns right-aligned, two spaces apart.
fn write_table(out: &mut String, rows: &[Vec<String>], right: &[bool]) {
    let widths: Vec<usize> = (0..right.len())
        .map(|column| {
            rows.iter()
                .map(|row| row[column].chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    for row in rows {
        let mut line = String::from("  ");
        for (column, cell) in row.iter().enumerate() {
            let pad = widths[column] - cell.chars().count();
            line.push_str("  ");
            if right[column] {
                line.push_str(&" ".repeat(pad));
                line.push_str(cell);
            } else {
                line.push_str(cell);
                line.push_str(&" ".repeat(pad));
            }
        }
        let _ = writeln!(out, "{}", line.trim_end());
    }
}

fn growth(growth: KvGrowth) -> String {
    match growth {
        KvGrowth::PerToken {
            tokens_per_entry: 1,
        } => "per token".to_string(),
        KvGrowth::PerToken { tokens_per_entry } => format!("per {tokens_per_entry} tokens"),
        KvGrowth::SlidingWindow { window_tokens } => format!("window {window_tokens}"),
        KvGrowth::PerRequest => "per request".to_string(),
    }
}

fn gpu_name(cluster: &GpuCluster, pool: u16, worker_id: u16) -> &str {
    cluster
        .gpus
        .iter()
        .find(|gpu| gpu.pool == pool && gpu.worker_id == worker_id)
        .map_or("unknown GPU", |gpu| gpu.name.as_str())
}

fn distinct(mut ids: Vec<u16>) -> Vec<u16> {
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn plural(count: u64, noun: &str) -> String {
    match (count, noun) {
        (1, _) => format!("1 {noun}"),
        (_, "copy") => format!("{count} copies"),
        _ => format!("{} {noun}s", grouped(count)),
    }
}

fn percent(part: u64, total: u64) -> String {
    if total == 0 {
        return DASH.to_string();
    }
    format!("{:.0}%", 100.0 * part as f64 / total as f64)
}

/// Bytes in the largest binary unit that keeps them at or above one, to three
/// significant figures.
fn human_bytes(bytes: f64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024.0 {
        return format!("{bytes:.0} B");
    }
    let mut value = bytes;
    let mut unit = "B";
    for next in UNITS {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = next;
    }
    let decimals = if value >= 100.0 {
        0
    } else if value >= 10.0 {
        1
    } else {
        2
    };
    format!("{value:.decimals$} {unit}")
}

/// A per-GPU byte count, which is whole for every layout but printed whole
/// either way.
fn grouped_f(bytes: f64) -> String {
    grouped(bytes.round() as u64)
}

/// `1234567` as `1,234,567`.
fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// Distinct ids as ranges: `0–3, 7`.
fn id_ranges(mut ids: Vec<u16>) -> String {
    ids.sort_unstable();
    ids.dedup();
    let mut ranges: Vec<String> = Vec::new();
    let mut index = 0;
    while index < ids.len() {
        let start = ids[index];
        let mut end = start;
        while index + 1 < ids.len() && ids[index + 1] == end + 1 {
            index += 1;
            end = ids[index];
        }
        ranges.push(if start == end {
            start.to_string()
        } else {
            format!("{start}–{end}")
        });
        index += 1;
    }
    ranges.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::gpu_cluster::{CostSource, KvPoolFacts};
    use crate::worker::kv::KvAccounting;

    #[test]
    fn the_document_flattens_each_report_and_the_text_groups_alike_pools() {
        let mut cluster = GpuCluster::new(CostSource::analytic(1.0));
        let accounting = KvAccounting::tokens_for_test(1_000);
        let facts = KvPoolFacts::of(&accounting, Some(100));
        for worker_id in 0..4 {
            cluster.allocate(0, worker_id, 1, "NVIDIA H200", "main");
            cluster.register_kv_pool("main", 0, worker_id, 0, facts.clone());
        }
        let document = kv_cache_document(&cluster);
        assert_eq!(document["schema_version"], 1);
        let pools = document["pools"].as_array().unwrap();
        assert_eq!(pools.len(), 4);
        assert_eq!(pools[3]["worker_id"], 3);
        assert_eq!(
            pools[3]["layout"]["parts"][0]["growth"]["kind"],
            "per_token"
        );
        assert_eq!(pools[3]["accounting"]["capacity_units"], 1_000);
        assert_eq!(pools[3]["result"]["max_concurrency_at_max_model_len"], 10.0);

        let text = kv_cache_text(&cluster);
        assert_eq!(text.matches("KV cache · pool main").count(), 1, "{text}");
        assert!(text.contains("NVIDIA H200"), "{text}");
        assert_eq!(pools[3]["gpu_name"], "NVIDIA H200");
        assert!(text.contains("workers 0–3 (4 identical)"), "{text}");
        assert!(text.contains("GPU KV cache size: 1,000 tokens"), "{text}");
        assert!(!text.contains("units"), "units stay in the JSON: {text}");
    }

    #[test]
    fn numbers_and_ids_print_compactly() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(95_434_567_680), "95,434,567,680");
        assert_eq!(id_ranges(vec![3, 0, 1, 2, 7, 7]), "0–3, 7");
        assert_eq!(id_ranges(vec![5]), "5");
    }
}
