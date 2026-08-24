//! `analyze gen-iter-breakdown` — human-readable cost tree (time + percentage).
//!
//! Reads the standard `cost_log` (per-iter `slot_time_ms` + `groups` inputs) and
//! the matching `cost_manifest` (tree `nodes`/`slots`/`node_labels`), then renders
//! one ASCII cost tree per iteration into `<log_dir>/reports/iter_breakdown.ans`.
//! Each line is `<▸-gutter><tree-prefix><label>  <time> us (<pct>%)`, with the time
//! column aligned to a single right edge. The critical path is flagged with a `▸`
//! gutter (plain text, so it shows in any viewer). Colored by default — the timing
//! cell tinted by node type (non-leaf blue, leaf white), bold critical path, yellow
//! header title bar; `--no-color` drops the ANSI for a clean editor view. The
//! per-line ANSI overhead is uniform, so columns stay aligned even where the editor
//! shows the escape codes literally. This is the human-readable twin of `analyze
//! trace` (which
//! emits the same tree as a Perfetto binary) — a verb, not a subject: its output
//! contract is colored text, not report/payload JSON.
//!
//! Rendering choices (clean tree, per user):
//! - **No kernel config/shapes**: leaves show `short_name (kind)`, not the
//!   `backends=… hidden=… dtype=…` config blob; worklet nodes strip the trailing
//!   ` [tp=4; …]` shape bracket off their `node_labels` entry.
//! - **No `Leaf#N` indices**.
//! - **Scale passthrough**: a `Scale{n}` stays as one line (`layer ×94`) but the
//!   ×n is pushed into its subtree, so `disp(idx) = node_time(idx) × scale_above`
//!   and per-layer percentages are meaningful against the whole iteration (not
//!   ~1/94 fragments).
//! - **Repeated parallel siblings collapse** to one representative + `×N`; for a
//!   `Max` parent the representative's time column shows the max (bottleneck) member
//!   and the `×N` line trails `avg … us` — the gap is the expert-load skew signal
//!   (cf. the MoE per-rank-load v1 floor).
//! - **Critical path** = greedy max-`node_time` child descent from the root, flagged
//!   with a `▸` gutter (and bold when colored); other rows get a blank gutter. When
//!   colored, the timing cell is tinted by node type: non-leaf blue, leaf white.

pub mod kernel_time_share;

use std::fs;
use std::ops::Range;
use std::path::Path;

use anyhow::{bail, Context, Result};
use datafusion::prelude::SessionContext;

use crate::io::{read_cost_manifests, report_path};
use crate::session::{
    col, collect, column_f64, register_cost_log, require_columns, value_f32_list, value_f64,
    value_groups, value_string, GroupInput, COST_LOG_TABLE,
};
use crate::trace::manifest::{node_time, FlatCostNode, Manifest};

/// Columns the breakdown reads from `cost_log` (drift-guarded). `groups` (the
/// per-HP-group arch input) is the one column `analyze trace` doesn't need.
const COLUMNS: &[&str] = &[
    "pool_tag",
    "worker_id",
    "iter_id",
    "batch_id",
    "section",
    "layer",
    "total_time_ms",
    "slot_time_ms",
    "groups",
];

/// One iteration's (or one AFD building block's) row, materialized from the parquet.
struct BreakRow {
    pool_tag: String,
    worker_id: u16,
    iter_id: u64,
    /// Micro-batch / pipeline slot (AFD) — distinguishes rows that share
    /// `(iter_id, section, layer)` but belong to different slots. `0` for iter-wise.
    batch_id: u64,
    /// Building-block section (`iter` for iter-wise; `attn` / `prologue` /
    /// `pre_attn` / `post_attn` / `post_attn_last` / `epilogue` for AFD). Selects
    /// which sub-manifest interprets `slot_ns`.
    section: String,
    /// Layer index this block belongs to (`-1` for iter-wise / iteration-level
    /// prologue & epilogue). Shown in the AFD header only.
    layer: i16,
    total_time_ms: f64,
    /// Per-slot leaf duration, pre-rounded to ns (index = manifest slot).
    slot_ns: Vec<i64>,
    /// Per-HP-group arch input (drives the header before the tree).
    groups: Vec<GroupInput>,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "worker_id/iter_id/batch_id/layer/slot_ns are small non-negative integer ids/durations written by our own simulator, widened to f64 only by the Arrow/Parquet cost_log schema, so the cast back is exact and in range"
)]
pub async fn run(
    ctx: &SessionContext,
    log_dir: &Path,
    iter: Option<u64>,
    max_iters: usize,
    color: bool,
) -> Result<()> {
    let manifests = read_cost_manifests(log_dir)?;

    if !register_cost_log(ctx, log_dir).await? {
        bail!("cost_log/ dir not found under {}", log_dir.display());
    }
    require_columns(ctx, COST_LOG_TABLE, COLUMNS).await?;

    // Pick the render window in SQL *before* touching the heavy `groups` /
    // `slot_time_ms` list columns: a 2000s AFD run has ~50M cost_log rows, but we
    // only ever print `max_iters` of them (default 32). The old path pulled + decoded
    // every row's list columns and then truncated in Rust — decoding 50M rows to keep
    // 32. `plan_window` instead uses scalar-only passes (a COUNT and a TopK on
    // iter_id) to derive a predicate selecting just the window, so the one heavy scan
    // below is filtered + `LIMIT`ed to <= max_iters rows.
    let (filter, total_matching) = plan_window(ctx, iter, max_iters).await?;
    if total_matching == 0 {
        match iter {
            Some(want) => bail!("no cost_log row with iter_id={want}"),
            None => bail!("cost_log has no rows"),
        }
    }
    // Rows beyond the window, counted pre-truncation exactly like the old path
    // (`rows.len() - max_iters` over the `--iter`-filtered set) so the footer is
    // unchanged.
    let omitted = total_matching.saturating_sub(max_iters);

    // Heavy pass, windowed. Cast the low-cardinality string columns (pool_tag,
    // section) → VARCHAR for the same reason as `analyze trace`: parquet hands them
    // back as a DictionaryArray (RLE_DICTIONARY), which `value_string`'s StringArray
    // downcast rejects. `WHERE {filter}` + `ORDER BY … LIMIT` reproduce the old
    // "sort the whole table, take the first max_iters rows" set exactly (every
    // windowed row sorts ahead of every excluded one), but decode only the kept rows.
    let other_cols = COLUMNS
        .iter()
        .filter(|c| **c != "pool_tag" && **c != "section")
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT CAST(pool_tag AS VARCHAR) AS pool_tag, CAST(section AS VARCHAR) AS section, \
         {other_cols} FROM cost_log WHERE {filter} \
         ORDER BY pool_tag, worker_id, iter_id LIMIT {max_iters}"
    );
    let batches = collect(ctx, &sql).await?;

    let mut rows: Vec<BreakRow> = Vec::new();
    for b in &batches {
        let (pt, wid, iid) = (
            col(b, "pool_tag")?,
            col(b, "worker_id")?,
            col(b, "iter_id")?,
        );
        let bid = col(b, "batch_id")?;
        let (sec, lay) = (col(b, "section")?, col(b, "layer")?);
        let (tt, st, gr) = (
            col(b, "total_time_ms")?,
            col(b, "slot_time_ms")?,
            col(b, "groups")?,
        );
        for r in 0..b.num_rows() {
            let slot_ns = value_f32_list(st, r)?
                .iter()
                .map(|ms| (ms * 1e6).round() as i64)
                .collect();
            rows.push(BreakRow {
                pool_tag: value_string(pt, r)?,
                worker_id: value_f64(wid, r)? as u16,
                iter_id: value_f64(iid, r)? as u64,
                batch_id: value_f64(bid, r)? as u64,
                section: value_string(sec, r)?,
                layer: value_f64(lay, r)? as i16,
                total_time_ms: value_f64(tt, r)?,
                slot_ns,
                groups: value_groups(gr, r)?,
            });
        }
    }

    let mut out = String::new();
    for row in &rows {
        let key = (row.pool_tag.clone(), row.worker_id);
        let doc = manifests
            .get(&key)
            .with_context(|| format!("missing cost manifest for {}/{}", key.0, key.1))?;
        let manifest = doc.section(&row.section).with_context(|| {
            format!(
                "cost manifest for {}/{} has no section {:?}",
                key.0, key.1, row.section
            )
        })?;
        out.push_str(&render_header(row, manifest, color));
        out.push('\n');
        out.push_str(&render_tree(manifest, &row.slot_ns, color));
        out.push('\n');
    }
    if omitted > 0 {
        out.push_str(&format!(
            "… {omitted} more iteration(s) omitted (--max-iters {max_iters}); \
             use --iter <id> to pick one\n"
        ));
    }

    let out_path = report_path(log_dir, "iter_breakdown.ans");
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(&out_path, out.as_bytes())
        .with_context(|| format!("write {}", out_path.display()))?;
    println!(
        "wrote {} ({} iteration(s){})",
        out_path.display(),
        rows.len(),
        if omitted > 0 {
            format!(", {omitted} omitted")
        } else {
            String::new()
        }
    );
    Ok(())
}

/// Choose the SQL predicate selecting the rows `gen-iter-breakdown` renders, using
/// only scalar columns so the heavy `groups` / `slot_time_ms` list columns are never
/// decoded here. Also returns the count of rows matching the `--iter` filter
/// (pre-truncation) for the "N omitted" footer.
///
/// - `--iter want`: predicate `iter_id = want` (parquet row-group pruning on iter_id
///   keeps the heavy pass to the row groups holding `want`).
/// - no `--iter`: the window is the first `max_iters` rows in
///   (pool_tag, worker_id, iter_id) order. A scalar-only TopK finds the largest
///   iter_id among them (`M`); `iter_id <= M` is then a superset the heavy pass
///   re-orders + `LIMIT`s down to exactly those rows — equivalent because every one
///   of the first max_iters rows has iter_id <= M and no excluded row can sort ahead
///   of them, and a clean range predicate prunes row groups well.
async fn plan_window(
    ctx: &SessionContext,
    iter: Option<u64>,
    max_iters: usize,
) -> Result<(String, usize)> {
    if let Some(want) = iter {
        let n = scalar_count(ctx, &format!("WHERE iter_id = {want}")).await?;
        return Ok((format!("iter_id = {want}"), n));
    }
    let total = scalar_count(ctx, "").await?;
    if total == 0 {
        // Caller bails; the predicate is never used (`total_matching == 0`).
        return Ok(("iter_id >= 0".to_string(), 0));
    }
    // Order by the VARCHAR-cast pool_tag so this matches the heavy pass's ordering
    // exactly (both sort the same string values), making `M` the true max iter_id of
    // the rendered set.
    let sql = format!(
        "SELECT iter_id FROM cost_log \
         ORDER BY CAST(pool_tag AS VARCHAR), worker_id, iter_id LIMIT {max_iters}"
    );
    let batches = collect(ctx, &sql).await?;
    let mut max_iter_id = 0u64;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "iter_id is a non-negative sequential id written by our own simulator, widened to f64 only by the Arrow/Parquet cost_log schema, so the cast back is exact and in range"
    )]
    for b in &batches {
        for id in column_f64(col(b, "iter_id")?)? {
            max_iter_id = max_iter_id.max(id as u64);
        }
    }
    Ok((format!("iter_id <= {max_iter_id}"), total))
}

/// `COUNT(*)` over cost_log with an optional `WHERE …` clause (scalar-only, so it
/// reads row-group metadata / the iter_id column, never the list columns). Empty
/// table → 0.
async fn scalar_count(ctx: &SessionContext, where_clause: &str) -> Result<usize> {
    let sql = format!("SELECT COUNT(*) AS n FROM cost_log {where_clause}");
    let batches = collect(ctx, &sql).await?;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "n is a SQL COUNT(*), a non-negative row count bounded by the table size, widened to f64 only by the query result type"
    )]
    match batches.first() {
        Some(b) if b.num_rows() > 0 => Ok(value_f64(col(b, "n")?, 0)? as usize),
        _ => Ok(0),
    }
}

// ── time fold ──────────────────────────────────────────────────────────────

// `node_time` (the local cost fold: Leaf=slot, Sum=Σ, Max=max/overlap,
// Scale=n×child) now lives in `crate::trace::manifest` — shared with
// `trace::place`'s critical-path collapse — and is imported at the top.

// ── labels ─────────────────────────────────────────────────────────────────

/// Last dotted segment of a slot name (`unified.pre_attn.qkv_proj` → `qkv_proj`).
fn leaf_short(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// Strip a trailing ` [..]` shape bracket off a worklet label
/// (`unified.attn_block (AttnBlockTpWorklet) [tp=4; …]` → `… (AttnBlockTpWorklet)`).
fn strip_brackets(label: &str) -> &str {
    label
        .split_once(" [")
        .map(|(head, _)| head)
        .unwrap_or(label)
}

/// Strip a trailing ` (Xxx)` parenthetical (the worklet struct name) off a label
/// (`unified.attn_block (AttnBlockTpWorklet)` → `unified.attn_block`).
fn strip_worklet(label: &str) -> &str {
    if label.ends_with(')') {
        if let Some((head, _)) = label.rsplit_once(" (") {
            return head;
        }
    }
    label
}

/// Display-clean a worklet node label: drop both the ` [shape]` bracket and the
/// ` (WorkletStructName)` parenthetical, leaving just the dotted name.
fn clean_label(label: &str) -> &str {
    strip_worklet(strip_brackets(label))
}

fn node_label(m: &Manifest, idx: usize) -> Option<&str> {
    m.node_labels.get(idx).and_then(|o| o.as_deref())
}

fn fmt_overlap(o: f32) -> String {
    if (o - o.round()).abs() < 1e-6 {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "o is a small repeat-group overlap count/multiplier, far under i64::MAX"
        )]
        let rounded = o.round() as i64;
        format!("{rounded}")
    } else {
        format!("{o}")
    }
}

/// us from ns, rounded.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "ns is one iteration's kernel duration, far under both f64's 2^53 exact-integer range and i64::MAX after dividing to us"
)]
fn us(ns: i64) -> i64 {
    (ns as f64 / 1000.0).round() as i64
}

/// The clean display label for node `idx`. Root (idx 0) → `total` (the arch
/// header already names the model). Leaf → `short (kind)` (no config). Sum/Max →
/// the bracket-stripped worklet label, else the operator. Scale → `{label} ×{n}`.
fn base_label(m: &Manifest, idx: usize) -> String {
    if idx == 0 {
        return "total".to_string();
    }
    let lbl = node_label(m, idx).map(clean_label);
    match &m.nodes[idx] {
        FlatCostNode::Leaf(slot) => {
            let d = &m.slots[*slot];
            format!("{} ({})", leaf_short(&d.name), d.kind)
        }
        FlatCostNode::Sum { .. } => lbl.unwrap_or("Sum").to_string(),
        FlatCostNode::Max { overlap, .. } => lbl
            .map(str::to_string)
            .unwrap_or_else(|| format!("Max{{overlap={}}}", fmt_overlap(*overlap))),
        FlatCostNode::Scale { n, .. } => format!("{} ×{}", lbl.unwrap_or("scale"), n),
    }
}

// ── sibling collapse ───────────────────────────────────────────────────────

/// A run of consecutive structurally-identical siblings, collapsed to one
/// representative + a `×members.len()` count.
struct Group {
    members: Vec<usize>,
}

/// Recursive structural signature: operator + stripped node-label + leaf
/// (kind, short-name), ignoring slot *index* and config. Two MoE `expert_compute`
/// subtrees over different slots hash identically and collapse into one ×N.
fn signature(m: &Manifest, idx: usize) -> String {
    let lbl = node_label(m, idx).map(strip_brackets).unwrap_or("");
    match &m.nodes[idx] {
        FlatCostNode::Leaf(slot) => {
            let d = &m.slots[*slot];
            format!("L[{lbl}|{}|{}]", d.kind, leaf_short(&d.name))
        }
        FlatCostNode::Sum { children } => format!("S[{lbl}]({})", sig_children(m, children)),
        FlatCostNode::Max { overlap, children } => {
            format!(
                "M[{lbl}|{}]({})",
                fmt_overlap(*overlap),
                sig_children(m, children)
            )
        }
        FlatCostNode::Scale { n, children } => {
            format!("X[{lbl}|{n}]({})", signature(m, children.start))
        }
    }
}

fn sig_children(m: &Manifest, children: &Range<usize>) -> String {
    children
        .clone()
        .map(|c| signature(m, c))
        .collect::<Vec<_>>()
        .join(",")
}

/// Collapse a child range into runs of consecutive identical-signature siblings.
fn collapse(m: &Manifest, children: Range<usize>) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    let mut last_sig: Option<String> = None;
    for c in children {
        let sig = signature(m, c);
        if last_sig.as_deref() == Some(sig.as_str()) {
            groups.last_mut().unwrap().members.push(c);
        } else {
            groups.push(Group { members: vec![c] });
            last_sig = Some(sig);
        }
    }
    groups
}

/// The representative member of a group: for a `Max` parent the slowest member
/// (the bottleneck instance, whose subtree we show); otherwise the first.
fn rep_of(m: &Manifest, g: &Group, is_max: bool, slot_ns: &[i64]) -> usize {
    if is_max {
        *g.members
            .iter()
            .max_by_key(|&&c| node_time(m, c, slot_ns))
            .unwrap()
    } else {
        g.members[0]
    }
}

// ── rendering ──────────────────────────────────────────────────────────────

/// One rendered tree row, pre-alignment. `plain` is the prefix+label (ANSI-free,
/// so its char count is its visible width); `time_ns` is the scale-applied
/// "whole-iteration contribution". `is_leaf` tints the row (non-leaf blue, leaf
/// white); `critical` adds bold + the `▸` gutter.
struct Line {
    plain: String,
    is_leaf: bool,
    time_ns: i64,
    pct: f64,
    trailing: String,
    critical: bool,
}

/// `×N` annotation carried onto a collapsed representative's line.
struct Repeat {
    n: usize,
    /// For a `Max` parent: the average member contribution in ns. The row's time
    /// column shows the max (bottleneck) member, so the two together read as skew.
    avg: Option<i64>,
}

/// Box-drawing prefixes for one node: `own` precedes the node's own label;
/// `child` is the base every child connector extends.
struct Indent {
    own: String,
    child: String,
}

struct Renderer<'a> {
    m: &'a Manifest,
    slot_ns: &'a [i64],
    root_ns: i64,
}

impl<'a> Renderer<'a> {
    fn render(
        &self,
        idx: usize,
        scale: i64,
        indent: &Indent,
        on_crit: bool,
        repeat: Option<Repeat>,
        out: &mut Vec<Line>,
    ) {
        let disp = node_time(self.m, idx, self.slot_ns).saturating_mul(scale);
        #[allow(
            clippy::cast_precision_loss,
            reason = "disp/root_ns are one iteration's ns durations, far under f64's 2^53 exact-integer range"
        )]
        let pct = if self.root_ns > 0 {
            disp as f64 / self.root_ns as f64 * 100.0
        } else {
            0.0
        };
        let mut label = base_label(self.m, idx);
        let mut trailing = String::new();
        if let Some(rep) = repeat {
            label = format!("{label} ×{}", rep.n);
            // The row's time column already shows the max (bottleneck) member; the
            // trailing adds the average, so the gap between them reads as load skew.
            if let Some(avg) = rep.avg {
                trailing = format!("avg {} us", us(avg));
            }
        }
        out.push(Line {
            plain: format!("{}{label}", indent.own),
            is_leaf: matches!(&self.m.nodes[idx], FlatCostNode::Leaf(_)),
            time_ns: disp,
            pct,
            trailing,
            critical: on_crit,
        });

        match &self.m.nodes[idx] {
            FlatCostNode::Leaf(_) => {}
            FlatCostNode::Scale { n, children } => {
                let ci = Indent {
                    own: format!("{}└─ ", indent.child),
                    child: format!("{}   ", indent.child),
                };
                // Scale passthrough: push ×n into the subtree so per-layer nodes
                // show their whole-iteration contribution.
                self.render(
                    children.start,
                    scale.saturating_mul(*n as i64),
                    &ci,
                    on_crit,
                    None,
                    out,
                );
            }
            FlatCostNode::Sum { children } | FlatCostNode::Max { children, .. } => {
                let is_max = matches!(&self.m.nodes[idx], FlatCostNode::Max { .. });
                let groups = collapse(self.m, children.clone());
                // Critical child = the group whose representative has the max
                // node_time (siblings share scale, so node_time orders them).
                let crit = groups
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, g)| {
                        node_time(
                            self.m,
                            rep_of(self.m, g, is_max, self.slot_ns),
                            self.slot_ns,
                        )
                    })
                    .map(|(gi, _)| gi);
                let n_groups = groups.len();
                for (gi, g) in groups.iter().enumerate() {
                    let last = gi + 1 == n_groups;
                    let ci = Indent {
                        own: format!("{}{}", indent.child, if last { "└─ " } else { "├─ " }),
                        child: format!("{}{}", indent.child, if last { "   " } else { "│  " }),
                    };
                    let child_crit = on_crit && Some(gi) == crit;
                    let rep_idx = rep_of(self.m, g, is_max, self.slot_ns);
                    let repeat = if g.members.len() > 1 {
                        let avg = if is_max {
                            Some(self.group_avg(g, scale))
                        } else {
                            None
                        };
                        Some(Repeat {
                            n: g.members.len(),
                            avg,
                        })
                    } else {
                        None
                    };
                    self.render(rep_idx, scale, &ci, child_crit, repeat, out);
                }
            }
        }
    }

    /// Average member contribution (scale-applied ns) across a collapsed group.
    fn group_avg(&self, g: &Group, scale: i64) -> i64 {
        let times: Vec<i64> = g
            .members
            .iter()
            .map(|&c| node_time(self.m, c, self.slot_ns).saturating_mul(scale))
            .collect();
        #[allow(
            clippy::cast_possible_wrap,
            reason = "times.len() is a small repeat-group member count, far under i64::MAX"
        )]
        let count = times.len() as i64;
        times.iter().sum::<i64>() / count
    }
}

/// Render one iteration's tree (header excluded) to text, aligning the time
/// column and (when `color`) wrapping critical-path rows yellow.
fn render_tree(m: &Manifest, slot_ns: &[i64], color: bool) -> String {
    let root_ns = node_time(m, 0, slot_ns);
    let r = Renderer {
        m,
        slot_ns,
        root_ns,
    };
    let mut lines = Vec::new();
    let root_indent = Indent {
        own: String::new(),
        child: String::new(),
    };
    r.render(0, 1, &root_indent, true, None, &mut lines);
    format_lines(&lines, color)
}

const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const WHITE: &str = "\x1b[37m";
const BOLD: &str = "\x1b[1m";
const DEFAULT_FG: &str = "\x1b[39m";
const RESET: &str = "\x1b[0m";

fn format_lines(lines: &[Line], color: bool) -> String {
    // `▸ ` (critical) / `  ` (other) left gutter marks the critical path in plain
    // text too (so `--no-color` still shows it).
    let lead = |l: &Line| format!("{}{}", if l.critical { "▸ " } else { "  " }, l.plain);
    // Single aligned time column: pad every lead to the widest, then the time cell.
    let width = lines
        .iter()
        .map(|l| lead(l).chars().count())
        .max()
        .unwrap_or(0);
    let mut out = String::new();
    for l in lines {
        let head = lead(l);
        let pad = " ".repeat(width - head.chars().count());
        let time = format!("{:>9} us ({:>5.1}%)", us(l.time_ns), l.pct);
        let trail = if l.trailing.is_empty() {
            String::new()
        } else {
            format!("   {}", l.trailing)
        };
        // Color: only the timing/percentage cell is tinted by node type — non-leaf
        // blue, leaf white — while the label stays plain; the critical path is bold
        // across the whole row. Every colored line carries the SAME ANSI codes at the
        // same positions and lengths: a 4-char lead (`BOLD` on critical, else a no-op
        // `RESET`), a 5-char `BLUE`/`WHITE` tint before the time cell, `DEFAULT_FG`
        // after (ends only the tint, keeping bold for trailing), `RESET` at the end.
        // So the per-line escape overhead is constant and the columns stay aligned
        // even in a viewer that prints the codes literally (e.g. the editor).
        // `--no-color` emits no codes at all (the `▸` gutter still flags critical).
        let row = if !color {
            format!("{head}{pad}  {time}{trail}")
        } else {
            let lead_code = if l.critical { BOLD } else { RESET };
            let tint = if l.is_leaf { WHITE } else { BLUE };
            format!("{lead_code}{head}{pad}  {tint}{time}{DEFAULT_FG}{trail}{RESET}")
        };
        out.push_str(&row);
        out.push('\n');
    }
    out
}

/// The arch-input header printed before each iteration's tree: a yellow title
/// bar, the model label (`node_labels[0]`), one aligned row per HP group's batch
/// shape, and the iter total.
fn render_header(row: &BreakRow, m: &Manifest, color: bool) -> String {
    let arch = node_label(m, 0).unwrap_or("(unknown arch)");
    let mut s = String::new();

    // Title bar — yellow when colored, a strong visual separator between iters.
    // Iter-wise rows keep the bare `iter N` title (non-regressing); AFD layer-wise
    // rows append the building-block section (+ its layer when ≥ 0, + the pipeline
    // slot) so rows sharing `(iter, section, layer)` across slots are distinguishable.
    let section_tag = if row.section == "iter" {
        String::new()
    } else if row.layer >= 0 {
        format!("· {} L{} · slot {} ", row.section, row.layer, row.batch_id)
    } else {
        format!("· {} · slot {} ", row.section, row.batch_id)
    };
    let title = format!(
        "═══ worker {}/{} · iter {} {section_tag}",
        row.pool_tag, row.worker_id, row.iter_id
    );
    let bar = 64usize.saturating_sub(title.chars().count());
    let title_line = format!("{title}{}", "═".repeat(bar));
    if color {
        s.push_str(&format!("{YELLOW}{title_line}{RESET}\n"));
    } else {
        s.push_str(&title_line);
        s.push('\n');
    }

    s.push_str(&format!("arch:  {arch}\n"));

    // Input: a faithful dump of the per-group input record (the sim's
    // `GroupInputLog`), one compact JSON object per group — every field, by its real
    // name, with its real value, no renaming or zero-suppression. So it shows exactly
    // what the cost_log row carries: iter/attn groups carry the full prefill/decode/KV
    // shape; the ffn side carries only `batch_tokens` (the rest stay zero/empty). One
    // line per group keeps it readable even when a group has many prefill requests.
    if row.groups.is_empty() {
        s.push_str("input: (none)\n");
    } else {
        s.push_str("input:\n");
        for g in &row.groups {
            let json = serde_json::to_string(g).unwrap_or_else(|_| "{}".into());
            s.push_str("  ");
            s.push_str(&json);
            s.push('\n');
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "total_time_ms is one iteration's total in ms; converted to ns it stays far under i64::MAX"
    )]
    let total_ns = (row.total_time_ms * 1e6).round() as i64;
    s.push_str(&format!("total: {} us\n", us(total_ns)));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty box-drawing indent for the root in tests.
    fn root_indent() -> Indent {
        Indent {
            own: String::new(),
            child: String::new(),
        }
    }

    /// Same mixed tree as `place.rs`: Sum( Leaf0, Scale{3}( Max{2}[Leaf1,Leaf2] ), Leaf3 ).
    /// BFS-flat: 0 Sum{1..4} 1 Leaf0 2 Scale{3,4..5} 3 Leaf3 4 Max{2,5..7} 5 Leaf1 6 Leaf2.
    fn mixed() -> Manifest {
        serde_json::from_str(
            r#"{
              "slots": [
                {"name": "m.a", "kind": "k", "kernel_config": {}},
                {"name": "m.b", "kind": "k", "kernel_config": {}},
                {"name": "m.c", "kind": "k", "kernel_config": {}},
                {"name": "m.d", "kind": "k", "kernel_config": {}}
              ],
              "nodes": [
                {"Sum": {"children": {"start": 1, "end": 4}}},
                {"Leaf": 0},
                {"Scale": {"n": 3, "children": {"start": 4, "end": 5}}},
                {"Leaf": 3},
                {"Max": {"overlap": 2.0, "children": {"start": 5, "end": 7}}},
                {"Leaf": 1},
                {"Leaf": 2}
              ],
              "node_labels": ["model [x]", null, "layer", null, "attn", null, null]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn node_time_matches_place_fold() {
        let m = mixed();
        // a=10, b=8, c=4, d=5. Max{2}[8,4]=4; Scale{3}(4)=12; Sum(10,12,5)=27.
        let slot_ns = [10i64, 8, 4, 5];
        assert_eq!(node_time(&m, 0, &slot_ns), 27);
        assert_eq!(node_time(&m, 4, &slot_ns), 4, "Max{{2}}[8,4]");
        assert_eq!(node_time(&m, 2, &slot_ns), 12, "Scale{{3}}(4)");
    }

    #[test]
    fn scale_passthrough_makes_layer_full_contribution() {
        let m = mixed();
        let slot_ns = [10i64, 8, 4, 5];
        let root_ns = node_time(&m, 0, &slot_ns);
        let r = Renderer {
            m: &m,
            slot_ns: &slot_ns,
            root_ns,
        };
        let mut lines = Vec::new();
        r.render(0, 1, &root_indent(), true, None, &mut lines);
        // The Scale line and the Max child under it both show 12ns (the ×3
        // whole-iteration contribution), not the 4ns single-layer time.
        let scale_line = lines.iter().find(|l| l.plain.contains("layer ×3")).unwrap();
        assert_eq!(scale_line.time_ns, 12);
        let attn_line = lines.iter().find(|l| l.plain.contains("attn")).unwrap();
        assert_eq!(
            attn_line.time_ns, 12,
            "Max child ×3 = 12, scale passed through"
        );
    }

    #[test]
    fn strip_brackets_drops_shape() {
        assert_eq!(strip_brackets("x (T) [a=1; b=2]"), "x (T)");
        assert_eq!(strip_brackets("no bracket"), "no bracket");
    }

    #[test]
    fn clean_label_drops_shape_and_worklet_name() {
        // Full worklet label → just the dotted name.
        assert_eq!(
            clean_label("unified.attn_block (AttnBlockTpWorklet) [tp=4; qo 64→16]"),
            "unified.attn_block"
        );
        assert_eq!(
            clean_label("unified.moe_router (MoeRouterLocalWorklet)"),
            "unified.moe_router"
        );
        // No parenthetical / no bracket → untouched.
        assert_eq!(clean_label("layer"), "layer");
    }

    /// Two structurally-identical Max siblings collapse to one ×2 with max/avg;
    /// distinct siblings stay separate.
    fn two_experts() -> Manifest {
        // Max{1}[ Sum(Leaf0), Sum(Leaf1) ] — two identical-shape expert subtrees.
        // BFS: 0 Max{1,1..3} 1 Sum{3..4} 2 Sum{4..5} 3 Leaf0 4 Leaf1.
        serde_json::from_str(
            r#"{
              "slots": [
                {"name": "m.expert", "kind": "gg", "kernel_config": {"expert": 0}},
                {"name": "m.expert", "kind": "gg", "kernel_config": {"expert": 1}}
              ],
              "nodes": [
                {"Max": {"overlap": 1.0, "children": {"start": 1, "end": 3}}},
                {"Sum": {"children": {"start": 3, "end": 4}}},
                {"Sum": {"children": {"start": 4, "end": 5}}},
                {"Leaf": 0},
                {"Leaf": 1}
              ],
              "node_labels": ["model [x]", "expert_compute", "expert_compute", null, null]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn collapses_identical_siblings_with_avg() {
        let m = two_experts();
        let slot_ns = [10i64, 6]; // expert times differ (skew)
        let root_ns = node_time(&m, 0, &slot_ns);
        let r = Renderer {
            m: &m,
            slot_ns: &slot_ns,
            root_ns,
        };
        let mut lines = Vec::new();
        r.render(0, 1, &root_indent(), true, None, &mut lines);
        // root Max + collapsed ×2 representative Sum + that Sum's leaf = 3 lines
        // (only the two sibling Sums collapse; the rep still renders its child).
        assert_eq!(lines.len(), 3);
        let rep = &lines[1];
        assert!(
            rep.plain.contains("expert_compute ×2"),
            "got {:?}",
            rep.plain
        );
        // Trailing shows avg only = (10+6)/2 = 8; the max is the row's time column.
        assert_eq!(rep.trailing, format!("avg {} us", us(8)));
        // Representative shows the slow (max) member's subtree → 10ns.
        assert_eq!(rep.time_ns, 10);
    }

    #[test]
    fn critical_path_is_the_max_child_chain() {
        let m = mixed();
        let slot_ns = [10i64, 8, 4, 5];
        let root_ns = node_time(&m, 0, &slot_ns);
        let r = Renderer {
            m: &m,
            slot_ns: &slot_ns,
            root_ns,
        };
        let mut lines = Vec::new();
        r.render(0, 1, &root_indent(), true, None, &mut lines);
        // root (Sum=27) critical; its max child is Scale (12) > Leaf0 (10) > Leaf3 (5).
        let total = lines.iter().find(|l| l.plain == "total").unwrap();
        assert!(total.critical);
        let scale = lines.iter().find(|l| l.plain.contains("layer ×3")).unwrap();
        assert!(scale.critical, "Scale(12) is the max child of root");
        let leaf_a = lines.iter().find(|l| l.plain.contains("a (k)")).unwrap();
        assert!(!leaf_a.critical, "Leaf0(10) < Scale(12), off critical path");
    }

    #[test]
    fn renders_clean_tree_snapshot() {
        let m = mixed();
        let slot_ns = [10_000i64, 8_000, 4_000, 5_000]; // ns → readable us
        let text = render_tree(&m, &slot_ns, false);
        // No Leaf#N, no config blob; clean leaf labels; us units everywhere.
        assert!(text.contains("total"));
        assert!(text.contains("a (k)"), "leaf shows short_name (kind)");
        assert!(!text.contains("Leaf#"), "no leaf indices");
        assert!(!text.contains("config"), "no config blob");
        // No ANSI when color is off.
        assert!(!text.contains('\u{1b}'), "no escapes with --no-color");
        // Every rendered row carries a `us (` time cell.
        for line in text.lines() {
            assert!(line.contains("us ("), "row missing time cell: {line:?}");
        }
    }
}
