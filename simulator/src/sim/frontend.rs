//! L7-γ trace frontend — loads single-round workload CSV(s) once at startup
//! into an ordered arrival queue, then emits `Request`s at their scheduled
//! arrival time during the tick loop.
//!
//! Single-round shape (mirrors `ref/moesim-rs/src/trace/mod.rs`):
//! `id,input_len,output_len,arrival_time`. Multi-round (a `round_idx` column)
//! is rejected — per-round `Request` allocation + continuation linkage is
//! deferred (see L7 design §3.2/§3.3).

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::common::{Request, RequestId, Time};

/// One single-round trace row.
#[derive(Clone, Debug, Deserialize)]
pub struct TraceEntry {
    pub id: u32,
    pub input_len: u32,
    pub output_len: u32,
    pub arrival_time: f64,
}

/// Immutable arrival queue + a cursor into it. `drain_due` advances the cursor
/// in arrival-time order; the trace is never mutated after `load`. CSV
/// `arrival_time` is normalized to rate=1, so the effective arrival time is
/// `arrival_time / request_rate` (a higher rate compresses the timeline).
#[derive(Debug)]
pub struct TraceFrontend {
    entries: Vec<TraceEntry>,
    cursor: usize,
    request_rate: f64,
}

/// Effective arrival time: the rate-1-normalized `arrival_time` remapped to the
/// target `request_rate` (req/s). Higher rate ⇒ arrivals sooner.
fn effective_arrival(arrival_time: f64, request_rate: f64) -> Time {
    Time::from_ms(arrival_time / request_rate)
}

impl TraceFrontend {
    /// Load + validate one or more single-round CSV files (concatenated in the
    /// given order). Each file must carry the 4-column single-round header.
    /// `request_rate` (req/s, > 0) remaps the rate-1-normalized arrival times.
    pub fn load(files: &[PathBuf], request_rate: f64) -> Result<Self> {
        if files.is_empty() {
            bail!("no trace files given (--trace-files)");
        }
        if !(request_rate.is_finite() && request_rate > 0.0) {
            bail!("request_rate must be finite and > 0 (got {request_rate})");
        }
        let mut entries = Vec::new();
        for file in files {
            load_one(file, &mut entries)?;
        }
        if entries.is_empty() {
            bail!("trace files contained no rows");
        }
        validate_single_round(&entries)?;
        Ok(Self {
            entries,
            cursor: 0,
            request_rate,
        })
    }

    /// Total number of requests in the trace (capacity hint for `RequestStore`).
    pub fn expected_count(&self) -> usize {
        self.entries.len()
    }

    /// All arrivals have been emitted.
    pub fn exhausted(&self) -> bool {
        self.cursor >= self.entries.len()
    }

    /// Emit every arrival due by `now` (effective `arrival_time <= now`), in
    /// arrival order, to `emit`. The drain loop lives here so callers can't
    /// under-drain by polling once per tick — a single call empties the tick.
    pub fn drain_due(&mut self, now: Time, mut emit: impl FnMut(Request)) {
        while self.cursor < self.entries.len() {
            let e = &self.entries[self.cursor];
            let t = effective_arrival(e.arrival_time, self.request_rate);
            if t > now {
                break;
            }
            let req = Request::new(RequestId(e.id), e.input_len, e.output_len, t);
            self.cursor += 1;
            emit(req);
        }
    }
}

fn load_one(path: &Path, out: &mut Vec<TraceEntry>) -> Result<()> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("opening trace file {}", path.display()))?;

    // Reject multi-round by header presence (matches ref's `round_idx` discriminator).
    let headers = rdr
        .headers()
        .with_context(|| format!("reading header of {}", path.display()))?;
    if headers.iter().any(|h| h == "round_idx") {
        bail!(
            "{}: multi-round traces (round_idx column) are not supported yet",
            path.display()
        );
    }

    for (i, result) in rdr.deserialize().enumerate() {
        let entry: TraceEntry =
            result.with_context(|| format!("{}: parsing row {i}", path.display()))?;
        if entry.input_len == 0 {
            bail!("{}: row {i} has input_len=0", path.display());
        }
        if entry.output_len == 0 {
            bail!("{}: row {i} has output_len=0", path.display());
        }
        if !entry.arrival_time.is_finite() || entry.arrival_time < 0.0 {
            bail!(
                "{}: row {i} has invalid arrival_time={} (must be finite, non-negative)",
                path.display(),
                entry.arrival_time
            );
        }
        out.push(entry);
    }
    Ok(())
}

/// Ids must be sequential `0..N`; `arrival_time` non-decreasing across rows.
fn validate_single_round(entries: &[TraceEntry]) -> Result<()> {
    for (i, e) in entries.iter().enumerate() {
        if e.id != i as u32 {
            bail!("trace row {i} has id={}, expected sequential id={i}", e.id);
        }
        if i > 0 && e.arrival_time < entries[i - 1].arrival_time {
            bail!(
                "trace row {i} arrival_time={} < previous {}",
                e.arrival_time,
                entries[i - 1].arrival_time
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    fn drain_at(fe: &mut TraceFrontend, now_ms: f64) -> Vec<RequestId> {
        let mut ids = Vec::new();
        fe.drain_due(Time::from_ms(now_ms), |r| ids.push(r.id));
        ids
    }

    #[test]
    fn drains_due_in_arrival_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "t.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             1,16,4,5.0\n\
             2,4,1,5.0\n",
        );
        // rate=1 → effective arrival == CSV arrival (ms).
        let mut fe = TraceFrontend::load(&[path], 1.0).unwrap();
        assert_eq!(fe.expected_count(), 3);

        // At t=0 only req 0 is due; a second drain at t=0 yields nothing.
        assert_eq!(drain_at(&mut fe, 0.0), vec![RequestId(0)]);
        assert_eq!(drain_at(&mut fe, 0.0), vec![]);

        // At t=5 both remaining drain in one call, in row order.
        assert_eq!(drain_at(&mut fe, 5.0), vec![RequestId(1), RequestId(2)]);
        assert!(fe.exhausted());
    }

    #[test]
    fn request_rate_compresses_arrivals() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "rate.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             1,8,2,10.0\n",
        );
        // rate=2 halves the rate-1 timeline: req 1 at 10.0/2 = 5.0ms.
        let mut fe = TraceFrontend::load(&[path], 2.0).unwrap();
        assert_eq!(drain_at(&mut fe, 4.9), vec![RequestId(0)]); // req 1 not due yet
        assert_eq!(drain_at(&mut fe, 5.0), vec![RequestId(1)]);
    }

    #[test]
    fn rejects_nonpositive_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "r.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        assert!(TraceFrontend::load(&[path], 0.0).is_err());
    }

    #[test]
    fn rejects_multi_round() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "mr.csv",
            "id,input_len,output_len,arrival_time,round_idx,tool_wait_after_ms,prefix_len\n\
             0,100,20,0.0,0,0.0,0\n",
        );
        let err = TraceFrontend::load(&[path], 1.0).unwrap_err();
        assert!(err.to_string().contains("multi-round"));
    }

    #[test]
    fn rejects_non_sequential_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "bad.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             5,8,2,1.0\n",
        );
        assert!(TraceFrontend::load(&[path], 1.0).is_err());
    }

    #[test]
    fn rejects_decreasing_arrival() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "bad2.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,5.0\n\
             1,8,2,1.0\n",
        );
        assert!(TraceFrontend::load(&[path], 1.0).is_err());
    }
}
