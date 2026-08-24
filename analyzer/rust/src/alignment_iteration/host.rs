//! The host (CPU) side of a measured iteration, anchored onto the GPU axis.
//!
//! `alignment/nsys/parse.py --host-timeline-output` writes a deliberately
//! un-interpreted sidecar: every NVTX range and every CUDA runtime call, on
//! every thread that carries either, in absolute nsys nanoseconds. Nothing in
//! it says which iteration an event belongs to, how deeply a range is nested,
//! or what kind of call it is. All three are anchoring decisions, and this is
//! where the anchor rule already lives — deciding them at extraction time would
//! fork that rule across two languages.
//!
//! What the lanes are for: a decode iteration on this capture spends 2.13 ms in
//! `preprocess` with 0.003 ms of it on the GPU. The device lane can only show
//! that as a bubble. The host lane names it.

use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

use super::read_json;

pub const WINDOW_RULE: &str = "an iteration's host window is its reference-rank GPU span unioned \
     with every worker's `vllm_iteration(N)` NVTX phase ranges. Consecutive \
     windows overlap — that overlap IS the pipelining — but never nest.";

pub const OWNERSHIP_RULE: &str = "a host event belongs to every iteration whose window it \
     overlaps, on every thread that carries NVTX marks or CUDA API calls. A \
     per-thread rule would be tighter for the four worker main threads and \
     undefined for the scheduler and helper threads, which carry no iteration \
     marks at all. Consecutive iterations therefore repeat each other's edge \
     events; consumers drawing several on one axis deduplicate by (thread, \
     start_ns).";

/// One bucket per kind of host call, so a lane of hundreds of ticks still reads
/// as a shape. Matched longest-prefix-first against the CUDA API name with its
/// `_v7000`-style suffix already stripped.
const API_CLASSES: &[(&str, &str)] = &[
    ("cudaGraphLaunch", "graph launch"),
    ("cuLaunchKernel", "kernel launch"),
    ("cudaLaunchCooperativeKernel", "kernel launch"),
    ("cudaLaunchKernel", "kernel launch"),
    ("cudaMemcpy", "memcpy"),
    ("cudaMemset", "memcpy"),
    ("cudaEventSynchronize", "synchronize"),
    ("cudaStreamSynchronize", "synchronize"),
    ("cudaDeviceSynchronize", "synchronize"),
    ("cudaEventQuery", "event bookkeeping"),
    ("cudaEventRecord", "event bookkeeping"),
    ("cudaEventCreate", "event bookkeeping"),
    ("cudaEventDestroy", "event bookkeeping"),
    ("cudaStreamWaitEvent", "event bookkeeping"),
    ("cudaMalloc", "allocate"),
    ("cudaFree", "allocate"),
];

const API_OTHER: &str = "other runtime call";

/// Display order of the buckets, and the index space `api` rows encode.
pub const CLASS_ORDER: &[&str] = &[
    "kernel launch",
    "graph launch",
    "memcpy",
    "synchronize",
    "event bookkeeping",
    "allocate",
    API_OTHER,
];

fn api_class_index(name: &str) -> usize {
    let bare = name
        .rsplit_once("_v")
        .filter(|(_, suffix)| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
        .map_or(name, |(head, _)| head);
    let label = API_CLASSES
        .iter()
        .find(|(prefix, _)| bare.starts_with(prefix))
        .map_or(API_OTHER, |(_, label)| *label);
    CLASS_ORDER
        .iter()
        .position(|candidate| *candidate == label)
        .expect("every API class label is in CLASS_ORDER")
}

#[derive(Deserialize)]
struct Sidecar {
    schema_version: u32,
    threads: Vec<SidecarThread>,
    strings: Vec<String>,
    /// `[thread_index, start_ns, end_ns, string_id]`, absolute nanoseconds.
    nvtx_ranges: Vec<[i64; 4]>,
    /// New sidecars append the NSYS runtime `correlationId`; the four-field
    /// form remains readable so existing captures can still be analyzed.
    api_calls: Vec<SidecarApiCall>,
    unclosed_nvtx_marks: u64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SidecarApiCall {
    Legacy([i64; 4]),
    Correlated([Option<i64>; 5]),
}

impl SidecarApiCall {
    fn fields(&self) -> Result<(i64, i64, i64, i64, Option<i64>)> {
        match self {
            Self::Legacy([thread_index, start, end, string_id]) => {
                Ok((*thread_index, *start, *end, *string_id, None))
            }
            Self::Correlated([thread_index, start, end, string_id, correlation_id]) => Ok((
                (*thread_index).context("host api row has no thread index")?,
                (*start).context("host api row has no start")?,
                (*end).context("host api row has no end")?,
                (*string_id).context("host api row has no string id")?,
                *correlation_id,
            )),
        }
    }
}

#[derive(Clone, Deserialize)]
struct SidecarThread {
    global_tid: i64,
    device_id: Option<i64>,
    process: String,
    role: String,
    main: bool,
}

/// One iteration's host window in absolute nanoseconds, paired with the anchor
/// its rows are drawn from.
pub struct HostWindow {
    pub iteration_id: u64,
    pub start_ns: u64,
    pub end_ns: u64,
    pub anchor_ns: u64,
}

/// The host events of one iteration, already anchor-relative.
#[derive(Default)]
pub struct IterationHost {
    window_ns: (i64, i64),
    /// thread index → `[start_ns, duration_ns, string_id, depth]`
    nvtx: BTreeMap<usize, Vec<[i64; 4]>>,
    /// thread index → `[start_ns, duration_ns, string_id, class_index, correlation_id]`.
    /// The last field is null for legacy sidecars that predate correlation IDs.
    api: BTreeMap<usize, Vec<ApiEvent>>,
}

#[derive(Clone, Copy)]
struct ApiEvent {
    start: i64,
    duration: i64,
    string_id: i64,
    class: i64,
    correlation_id: Option<i64>,
}

impl IterationHost {
    pub fn value(&self) -> Value {
        let lanes = |rows: &BTreeMap<usize, Vec<[i64; 4]>>| {
            rows.iter()
                .map(|(index, items)| (index.to_string(), json!(items)))
                .collect::<serde_json::Map<_, _>>()
        };
        let api = self
            .api
            .iter()
            .map(|(index, items)| {
                let rows = items
                    .iter()
                    .map(|item| {
                        json!([
                            item.start,
                            item.duration,
                            item.string_id,
                            item.class,
                            item.correlation_id,
                        ])
                    })
                    .collect::<Vec<_>>();
                (index.to_string(), json!(rows))
            })
            .collect::<serde_json::Map<_, _>>();
        json!({
            "window_ns": [self.window_ns.0, self.window_ns.1],
            "nvtx": lanes(&self.nvtx),
            "api": api,
        })
    }
}

/// The host sidecar, attributed to the windows it was asked about.
pub struct HostTimeline {
    source: String,
    threads: Vec<SidecarThread>,
    strings: Vec<String>,
    unclosed_nvtx_marks: u64,
    by_iteration: BTreeMap<u64, IterationHost>,
}

impl HostTimeline {
    /// Read the sidecar and attribute every event to the windows it overlaps.
    ///
    /// One pass over the events, not one pass per window: a full capture is a
    /// million API rows against two thousand windows, and the quadratic form of
    /// that is the difference between seconds and an hour.
    pub fn load(path: &Path, windows: &[HostWindow]) -> Result<Self> {
        let sidecar: Sidecar = read_json(path)?;
        ensure!(
            sidecar.schema_version == 1,
            "host timeline schema_version must be 1, found {}",
            sidecar.schema_version
        );

        let mut ordered: Vec<&HostWindow> = windows.iter().collect();
        ordered.sort_by_key(|window| (window.start_ns, window.end_ns));
        // Stabbing by binary search below is only correct while no window is
        // contained in another. Adjacent iterations overlap; they must not nest.
        for pair in ordered.windows(2) {
            ensure!(
                pair[1].end_ns >= pair[0].end_ns,
                "iteration {} nests inside iteration {}; host attribution would miss it",
                pair[1].iteration_id,
                pair[0].iteration_id
            );
        }
        let lows: Vec<u64> = ordered.iter().map(|window| window.start_ns).collect();
        let highs: Vec<u64> = ordered.iter().map(|window| window.end_ns).collect();

        #[allow(
            clippy::cast_possible_wrap,
            reason = "start_ns/end_ns/anchor_ns are nsys capture-relative ns timestamps, far under i64::MAX (~292 years of ns)"
        )]
        let mut by_iteration: BTreeMap<u64, IterationHost> = ordered
            .iter()
            .map(|window| {
                (
                    window.iteration_id,
                    IterationHost {
                        window_ns: (
                            window.start_ns as i64 - window.anchor_ns as i64,
                            window.end_ns as i64 - window.anchor_ns as i64,
                        ),
                        ..IterationHost::default()
                    },
                )
            })
            .collect();

        // Both bounds ascending (asserted above), so the windows overlapping an
        // event are one contiguous index range.
        //
        // Intervals are half-open here as everywhere else in this crate: an
        // event that ends exactly when a window opens has not entered it, and
        // counting it would put a zero-width row in that iteration's lane. The
        // `max(start + 1)` keeps a zero-duration call — nsys does emit them —
        // from belonging to no window at all.
        let overlapping = |start: u64, end: u64| {
            let end = end.max(start + 1);
            let first = highs.partition_point(|high| *high <= start);
            let last = lows.partition_point(|low| *low < end);
            first..last
        };

        for [thread_index, start, end, string_id] in &sidecar.nvtx_ranges {
            #[allow(
                clippy::cast_sign_loss,
                reason = "nsys nvtx timestamps are absolute non-negative nanoseconds, though the sidecar schema stores them as i64"
            )]
            let (start, end) = (*start as u64, *end as u64);
            for slot in overlapping(start, end) {
                let window = ordered[slot];
                let entry = by_iteration
                    .get_mut(&window.iteration_id)
                    .expect("every window has an entry");
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_wrap,
                    reason = "thread_index is a small non-negative index into the sidecar's own thread table, and the ns offsets are within one iteration window; both are far inside usize/i64 bounds"
                )]
                entry
                    .nvtx
                    .entry(*thread_index as usize)
                    .or_default()
                    // Depth is filled in below, once a thread's rows are known.
                    .push([
                        start as i64 - window.anchor_ns as i64,
                        (end - start) as i64,
                        *string_id,
                        0,
                    ]);
            }
        }

        for call in &sidecar.api_calls {
            let (thread_index, start, end, string_id, correlation_id) = call.fields()?;
            #[allow(
                clippy::cast_sign_loss,
                reason = "nsys api-call timestamps are absolute non-negative nanoseconds, though the sidecar schema stores them as i64"
            )]
            let (start, end) = (start as u64, end as u64);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                clippy::cast_sign_loss,
                reason = "string_id indexes the sidecar's own string pool and is non-negative by construction of the host-timeline extractor (a bad index panics on the indexing rather than miscomputing); api_class_index returns a tiny index into the fixed CLASS_ORDER table, well inside i64 range"
            )]
            let class = api_class_index(&sidecar.strings[string_id as usize]) as i64;
            for slot in overlapping(start, end) {
                let window = ordered[slot];
                let entry = by_iteration
                    .get_mut(&window.iteration_id)
                    .expect("every window has an entry");
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_wrap,
                    reason = "thread_index is a small non-negative index into the sidecar's own thread table, and the ns offsets are within one iteration window; both are far inside usize/i64 bounds"
                )]
                entry
                    .api
                    .entry(thread_index as usize)
                    .or_default()
                    .push(ApiEvent {
                        start: start as i64 - window.anchor_ns as i64,
                        duration: (end - start) as i64,
                        string_id,
                        class,
                        correlation_id,
                    });
            }
        }

        for iteration in by_iteration.values_mut() {
            for rows in iteration.nvtx.values_mut() {
                assign_depth(rows);
            }
            for rows in iteration.api.values_mut() {
                rows.sort_by_key(|row| {
                    (
                        row.start,
                        row.duration,
                        row.string_id,
                        row.class,
                        row.correlation_id,
                    )
                });
            }
        }

        Ok(Self {
            source: path.display().to_string(),
            threads: sidecar.threads,
            strings: sidecar.strings,
            unclosed_nvtx_marks: sidecar.unclosed_nvtx_marks,
            by_iteration,
        })
    }

    pub fn iteration(&self, iteration_id: u64) -> Option<&IterationHost> {
        self.by_iteration.get(&iteration_id)
    }

    /// The roster, the rules and the string pool — everything a lane needs that
    /// is not per-iteration. Lives in `meta` so a shard carries only its rows.
    pub fn meta(&self) -> Value {
        json!({
            "source": self.source,
            "window_rule": WINDOW_RULE,
            "ownership_rule": OWNERSHIP_RULE,
            "api_classes": CLASS_ORDER,
            "unclosed_nvtx_marks": self.unclosed_nvtx_marks,
            "threads": self.threads
                .iter()
                .map(|thread| json!({
                    "global_tid": thread.global_tid,
                    "device_id": thread.device_id,
                    "process": thread.process,
                    "role": thread.role,
                    "main": thread.main,
                }))
                .collect::<Vec<_>>(),
            "strings": self.strings,
        })
    }
}

/// Count each range's nesting depth against its own thread's other ranges.
///
/// Counted, not assumed: `preprocess` sits inside the harness's
/// `execute_context` range and `ncclAllGather` sits inside `postprocess`, so a
/// fixed outer/inner split would draw them all flat.
fn assign_depth(rows: &mut [[i64; 4]]) {
    // Outermost first at equal starts, so a linear sweep with a stack of open
    // ends is enough — no O(n^2) pairwise containment test.
    rows.sort_by_key(|row| (row[0], -row[1]));
    let mut open_ends: Vec<i64> = Vec::new();
    for row in rows.iter_mut() {
        let end = row[0] + row[1];
        while open_ends.last().is_some_and(|last| *last < end) {
            open_ends.pop();
        }
        #[allow(
            clippy::cast_possible_wrap,
            reason = "open_ends is a nesting-depth stack bounded by realistic NVTX range nesting, far under i64::MAX"
        )]
        {
            row[3] = open_ends.len() as i64;
        }
        open_ends.push(end);
    }
}

/// Resolve a lane's kernel-side context to a nameable window.
pub fn window(
    iteration_id: u64,
    anchor_ns: u64,
    gpu_span: (u64, u64),
    nvtx_bounds: impl IntoIterator<Item = (u64, u64)>,
) -> HostWindow {
    let (mut start, mut end) = gpu_span;
    for (range_start, range_end) in nvtx_bounds {
        start = start.min(range_start);
        end = end.max(range_end);
    }
    HostWindow {
        iteration_id,
        start_ns: start,
        end_ns: end,
        anchor_ns,
    }
}

/// Read the sidecar only when the manifest names one, so a capture parsed before
/// it existed degrades to the device-only lane instead of failing.
pub fn load_if_configured(
    path: Option<&Path>,
    windows: &[HostWindow],
) -> Result<Option<HostTimeline>> {
    let Some(path) = path else { return Ok(None) };
    HostTimeline::load(path, windows)
        .with_context(|| format!("read host timeline {}", path.display()))
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_classes_strip_the_version_suffix_before_matching() {
        let index = |name: &str| CLASS_ORDER[api_class_index(name)];

        assert_eq!(index("cudaLaunchKernel_v7000"), "kernel launch");
        assert_eq!(index("cudaStreamSynchronize_v3020"), "synchronize");
        assert_eq!(index("cudaGraphLaunch"), "graph launch");
        assert_eq!(index("cudaMemcpyAsync_v3020"), "memcpy");
        assert_eq!(index("cuCtxGetDevice"), "other runtime call");
    }

    #[test]
    fn a_name_ending_in_v_and_letters_is_not_a_version_suffix() {
        // `_view` is part of the name; stripping it would change the prefix the
        // class table matches against.
        assert_eq!(CLASS_ORDER[api_class_index("cudaMalloc_view")], "allocate");
    }

    #[test]
    fn depth_is_counted_from_real_containment_not_assumed() {
        // `execute_context` [0, 100) contains `preprocess` [10, 40), which
        // contains `ncclAllGather` [15, 20). A sibling at [50, 60) is back at
        // depth 1, not 2.
        let mut rows = [
            [15, 5, 2, 0],
            [0, 100, 0, 0],
            [50, 10, 3, 0],
            [10, 30, 1, 0],
        ];

        assign_depth(&mut rows);

        assert_eq!(
            rows,
            [
                [0, 100, 0, 0],
                [10, 30, 1, 1],
                [15, 5, 2, 2],
                [50, 10, 3, 1]
            ]
        );
    }

    /// The attribution predicate, lifted out of `load` so the boundary rule is
    /// testable without a sidecar file on disk.
    fn overlapped_windows(windows: &[(u64, u64)], start: u64, end: u64) -> Vec<usize> {
        let highs: Vec<u64> = windows.iter().map(|(_, high)| *high).collect();
        let lows: Vec<u64> = windows.iter().map(|(low, _)| *low).collect();
        let end = end.max(start + 1);
        (highs.partition_point(|high| *high <= start)..lows.partition_point(|low| *low < end))
            .collect()
    }

    #[test]
    fn an_event_ending_exactly_when_a_window_opens_is_not_in_it() {
        // Half-open, as every other interval in this crate is. Zero-width
        // contact would put an invisible row in the next iteration's lane.
        let windows = [(0, 100), (100, 200)];

        assert_eq!(overlapped_windows(&windows, 40, 100), vec![0]);
        assert_eq!(overlapped_windows(&windows, 100, 140), vec![1]);
        // A genuine straddle belongs to both — that overlap is the pipelining.
        assert_eq!(overlapped_windows(&windows, 90, 110), vec![0, 1]);
    }

    #[test]
    fn a_zero_duration_call_still_belongs_to_the_window_containing_it() {
        // nsys does emit start == end rows; a strict half-open test alone would
        // file them under no iteration at all.
        let windows = [(0, 100), (100, 200)];

        assert_eq!(overlapped_windows(&windows, 50, 50), vec![0]);
        assert_eq!(overlapped_windows(&windows, 100, 100), vec![1]);
    }

    #[test]
    fn the_window_is_the_gpu_span_widened_by_every_worker_range() {
        // The GPU span outlives the NVTX ranges on the right (CUDA graph replay)
        // and starts after them on the left (host work before the first launch).
        let resolved = window(7, 300, (300, 900), [(200, 500), (250, 800)]);

        assert_eq!((resolved.start_ns, resolved.end_ns), (200, 900));
        assert_eq!(resolved.anchor_ns, 300);
    }
}
