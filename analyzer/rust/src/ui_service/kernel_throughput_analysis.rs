//! Adapter from one exact CostTree leaf to the simulator's existing
//! `kernel-query grid` + `kernel-query eval` introspection protocol.
//!
//! The simulator remains the sole owner of kernel config decoding, sweep grids,
//! cache construction and interpolation. This module only selects the exact leaf,
//! expands the declared cache grid, and packages the two query responses for
//! viz-ui. Grid evaluation stays in cache-coordinate space; it must not invent
//! physical Inputs for ragged or re-axis kernels.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::kernel_query::{run_kernel_query, simulator_binary};

const MAX_GRID_POINTS: usize = 16_384;

#[derive(Deserialize)]
struct GridResponse {
    kind: String,
    describe_config: Value,
    input_fields: Vec<String>,
    grid_axes: Vec<Vec<f64>>,
}

pub(super) fn analyze_kernel_throughput(
    repo_root: &Path,
    cost_tree_detail: Value,
    leaf_id: usize,
) -> Result<Value> {
    let tree = cost_tree_detail
        .get("tree")
        .context("cost-tree detail has no tree")?;
    let leaf = find_preorder_node(tree, leaf_id)
        .with_context(|| format!("cost-tree node {leaf_id} is absent"))?;
    if leaf.get("kind").and_then(Value::as_str) != Some("leaf") {
        bail!("cost-tree node {leaf_id} is not a kernel leaf");
    }
    let slot = leaf
        .get("slot")
        .and_then(Value::as_object)
        .context("kernel leaf has no slot object")?;
    let kind = slot
        .get("kind")
        .and_then(Value::as_str)
        .context("kernel leaf slot has no kind")?;
    let config = slot
        .get("kernel_config")
        .filter(|value| value.is_object())
        .context("kernel leaf slot has no structured kernel_config")?;
    let simulator = simulator_binary(repo_root)?;

    let grid_value = run_kernel_query(
        repo_root,
        &simulator,
        json!({"op": "grid", "kind": kind, "config": config}),
    )?;
    let grid: GridResponse = serde_json::from_value(grid_value)
        .with_context(|| format!("decode {kind} grid response"))?;
    if grid.kind != kind {
        bail!(
            "kernel-query grid returned kind {:?}, expected {kind:?}",
            grid.kind
        );
    }
    let (display_points, coordinate_points) =
        expand_grid_points(&grid.input_fields, &grid.grid_axes)?;
    let eval = run_kernel_query(
        repo_root,
        &simulator,
        json!({
            "op": "eval_coords",
            "kind": kind,
            "config": config,
            "query_points": coordinate_points,
        }),
    )?;
    let mut results = eval
        .get("results")
        .and_then(Value::as_array)
        .cloned()
        .context("kernel-query eval_coords response has no results array")?;
    if results.len() != display_points.len() {
        bail!(
            "kernel-query eval_coords returned {} points for a {}-point grid",
            results.len(),
            display_points.len()
        );
    }
    for (result, display_input) in results.iter_mut().zip(display_points) {
        result
            .as_object_mut()
            .context("kernel-query eval_coords result is not an object")?
            .insert("input".into(), display_input);
    }

    Ok(json!({
        "schema_version": 1,
        "identity": cost_tree_detail.get("identity"),
        "leaf_id": leaf_id,
        "slot": slot,
        "exact_input": leaf.pointer("/stats/input"),
        "describe_config": grid.describe_config,
        "input_fields": grid.input_fields,
        "grid_axes": grid.grid_axes,
        "points": results,
        "semantics": "cache_eval_at_declared_grid",
    }))
}

fn find_preorder_node(node: &Value, target: usize) -> Option<&Value> {
    fn walk<'a>(node: &'a Value, target: usize, next: &mut usize) -> Option<&'a Value> {
        let current = *next;
        *next += 1;
        if current == target {
            return Some(node);
        }
        for child in node
            .get("children")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(found) = walk(child, target, next) {
                return Some(found);
            }
        }
        None
    }
    walk(node, target, &mut 0)
}

fn expand_grid_points(
    input_fields: &[String],
    axes: &[Vec<f64>],
) -> Result<(Vec<Value>, Vec<Vec<f64>>)> {
    if input_fields.is_empty() || input_fields.len() != axes.len() {
        bail!(
            "kernel grid has {} input fields for {} axes",
            input_fields.len(),
            axes.len()
        );
    }
    if axes.iter().any(Vec::is_empty) {
        bail!("kernel grid contains an empty axis");
    }
    let point_count = axes.iter().try_fold(1usize, |count, axis| {
        count
            .checked_mul(axis.len())
            .context("kernel grid size overflow")
    })?;
    if point_count > MAX_GRID_POINTS {
        bail!("kernel grid has {point_count} points; limit is {MAX_GRID_POINTS}");
    }

    let mut display_points = Vec::with_capacity(point_count);
    let mut coordinate_points = Vec::with_capacity(point_count);
    let mut values = vec![0.0; axes.len()];
    fn expand(
        fields: &[String],
        axes: &[Vec<f64>],
        depth: usize,
        values: &mut [f64],
        display_points: &mut Vec<Value>,
        coordinate_points: &mut Vec<Vec<f64>>,
    ) {
        if depth == axes.len() {
            let object: Map<String, Value> = fields
                .iter()
                .cloned()
                .zip(values.iter().copied().map(json_coord))
                .collect();
            display_points.push(Value::Object(object));
            coordinate_points.push(values.to_vec());
            return;
        }
        for &value in &axes[depth] {
            values[depth] = value;
            expand(
                fields,
                axes,
                depth + 1,
                values,
                display_points,
                coordinate_points,
            );
        }
    }
    expand(
        input_fields,
        axes,
        0,
        &mut values,
        &mut display_points,
        &mut coordinate_points,
    );
    Ok((display_points, coordinate_points))
}

/// Sweep axes use `f64` internally for interpolation, while every current
/// physical kernel Input uses integer counts/bytes/tokens. Preserve integral
/// coordinates as JSON integers so serde can decode the kernel's own `u32`/
/// `u64` input type; genuinely fractional axes remain JSON floats.
fn json_coord(value: f64) -> Value {
    if value >= 0.0 && value.fract() == 0.0 && value <= u64::MAX as f64 {
        Value::from(value as u64)
    } else {
        Value::from(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_grid_in_row_major_order() {
        let (display_points, coordinate_points) = expand_grid_points(
            &["m".into(), "n".into()],
            &[vec![1.0, 2.0], vec![8.0, 16.0]],
        )
        .unwrap();
        assert_eq!(
            display_points,
            vec![
                json!({"m": 1, "n": 8}),
                json!({"m": 1, "n": 16}),
                json!({"m": 2, "n": 8}),
                json!({"m": 2, "n": 16}),
            ]
        );
        assert_eq!(
            coordinate_points,
            vec![
                vec![1.0, 8.0],
                vec![1.0, 16.0],
                vec![2.0, 8.0],
                vec![2.0, 16.0],
            ]
        );
    }

    #[test]
    fn selects_the_same_preorder_id_as_ui_annotation() {
        let tree = json!({
            "kind": "sum",
            "children": [
                {"kind": "leaf", "slot": {"name": "a"}},
                {"kind": "scale", "children": [
                    {"kind": "leaf", "slot": {"name": "b"}}
                ]}
            ]
        });
        assert_eq!(
            find_preorder_node(&tree, 3).unwrap().pointer("/slot/name"),
            Some(&json!("b"))
        );
    }
}
