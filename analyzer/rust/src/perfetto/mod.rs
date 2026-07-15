//! Subject-agnostic Perfetto trace writer (TrackEvent model). Knows nothing
//! about kernels or cost trees — it only lays tracks and BEGIN/END/INSTANT
//! slices on a timeline and serializes to a gzipped `.pftrace`. The trace
//! builder (`crate::trace`) drives it; this module could back any timeline.
//!
//! Model: a single packet sequence (`SEQ_ID`) of `TracePacket`s. Tracks are
//! declared by `TrackDescriptor` packets (process → thread → child lanes);
//! slices are BEGIN/END `TrackEvent` pairs on a track, nested implicitly by the
//! order they're emitted (a BEGIN before its parent's END nests inside it).
//! Names are inlined (no interning) for v1 — simpler, slightly larger output.
//!
//! Track UUIDs are a pure function of their identity (pid / name / seed), so the
//! same logical trace serializes to byte-stable output run to run.

#[allow(dead_code)]
pub(crate) mod proto;

use std::io::Write;

use anyhow::Result;
use flate2::write::GzEncoder;
use flate2::Compression;
use prost::Message;

use proto::track_event::Type as EventType;
use proto::{
    DebugAnnotation, ProcessDescriptor, ThreadDescriptor, TracePacket, TrackDescriptor, TrackEvent,
};

/// One synthetic packet sequence for the whole trace. The UI drops track_event
/// packets that lack a `trusted_packet_sequence_id`.
const SEQ_ID: u32 = 1;
const SEQ_INCREMENTAL_STATE_CLEARED: u32 = 1;

/// A debug annotation value attached to a slice (shown in the UI Arguments pane).
pub enum Annotation {
    Str(String, String),
    Int(String, i64),
    Uint(String, u64),
    Dbl(String, f64),
}

impl Annotation {
    pub fn str(name: impl Into<String>, value: impl Into<String>) -> Self {
        Annotation::Str(name.into(), value.into())
    }
    pub fn int(name: impl Into<String>, value: i64) -> Self {
        Annotation::Int(name.into(), value)
    }
    pub fn uint(name: impl Into<String>, value: u64) -> Self {
        Annotation::Uint(name.into(), value)
    }
    pub fn dbl(name: impl Into<String>, value: f64) -> Self {
        Annotation::Dbl(name.into(), value)
    }

    fn to_proto(&self) -> DebugAnnotation {
        let mut a = DebugAnnotation::default();
        match self {
            Annotation::Str(n, v) => {
                a.name = Some(n.clone());
                a.string_value = Some(v.clone());
            }
            Annotation::Int(n, v) => {
                a.name = Some(n.clone());
                a.int_value = Some(*v);
            }
            Annotation::Uint(n, v) => {
                a.name = Some(n.clone());
                a.uint_value = Some(*v);
            }
            Annotation::Dbl(n, v) => {
                a.name = Some(n.clone());
                a.double_value = Some(*v);
            }
        }
        a
    }
}

/// 64-bit FNV-1a — a small, dependency-free hash for deterministic track UUIDs.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub struct TraceWriter {
    packets: Vec<TracePacket>,
    first: bool,
}

impl Default for TraceWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceWriter {
    pub fn new() -> Self {
        Self {
            packets: Vec::new(),
            first: true,
        }
    }

    /// First packet of the sequence carries SEQ_INCREMENTAL_STATE_CLEARED so the
    /// UI starts from clean sequence state.
    fn new_packet(&mut self) -> TracePacket {
        let mut p = TracePacket {
            trusted_packet_sequence_id: Some(SEQ_ID),
            ..Default::default()
        };
        if self.first {
            p.sequence_flags = Some(SEQ_INCREMENTAL_STATE_CLEARED);
            self.first = false;
        }
        p
    }

    /// Declare a process track (a top-level group row). UUID is `pid << 32`,
    /// matching the convention that child tracks reference via `parent_uuid`.
    pub fn process_track(&mut self, pid: i32, name: &str) -> u64 {
        let uuid = (pid as u64) << 32;
        let mut p = self.new_packet();
        p.track_descriptor = Some(TrackDescriptor {
            uuid: Some(uuid),
            name: Some(name.to_string()),
            process: Some(ProcessDescriptor {
                pid: Some(pid),
                process_name: Some(name.to_string()),
            }),
            ..Default::default()
        });
        self.packets.push(p);
        uuid
    }

    /// Declare a thread track under a process (the lane slices are emitted on).
    pub fn thread_track(&mut self, parent: u64, pid: i32, tid: i32, name: &str) -> u64 {
        let uuid = fnv1a(format!("thread:{pid}:{tid}:{name}").as_bytes()) | 1;
        let mut p = self.new_packet();
        p.track_descriptor = Some(TrackDescriptor {
            uuid: Some(uuid),
            parent_uuid: Some(parent),
            name: Some(name.to_string()),
            thread: Some(ThreadDescriptor {
                pid: Some(pid),
                tid: Some(tid),
                thread_name: Some(name.to_string()),
            }),
            ..Default::default()
        });
        self.packets.push(p);
        uuid
    }

    /// Declare a child track (a sub-lane under another track) — used for parallel
    /// `Max` branches so overlapping work renders on separate rows. `seed`
    /// disambiguates lanes that share a name under the same parent.
    pub fn child_track(&mut self, parent: u64, name: &str, seed: u64) -> u64 {
        let uuid = fnv1a(format!("child:{parent}:{seed}:{name}").as_bytes()) | 1;
        let mut p = self.new_packet();
        p.track_descriptor = Some(TrackDescriptor {
            uuid: Some(uuid),
            parent_uuid: Some(parent),
            name: Some(name.to_string()),
            ..Default::default()
        });
        self.packets.push(p);
        uuid
    }

    pub fn begin(&mut self, track: u64, ts_ns: i64, name: &str, anns: &[Annotation]) {
        let mut p = self.new_packet();
        p.timestamp = Some(ts_ns as u64);
        p.track_event = Some(TrackEvent {
            r#type: Some(EventType::SliceBegin as i32),
            track_uuid: Some(track),
            name: Some(name.to_string()),
            debug_annotations: anns.iter().map(Annotation::to_proto).collect(),
            flow_ids: Vec::new(),
        });
        self.packets.push(p);
    }

    /// A slice-begin carrying `flow_ids`: any two slices sharing an id are joined
    /// by a directed arrow in the Perfetto UI (used to link a send to its recv).
    pub fn begin_flow(
        &mut self,
        track: u64,
        ts_ns: i64,
        name: &str,
        anns: &[Annotation],
        flow_ids: &[u64],
    ) {
        let mut p = self.new_packet();
        p.timestamp = Some(ts_ns as u64);
        p.track_event = Some(TrackEvent {
            r#type: Some(EventType::SliceBegin as i32),
            track_uuid: Some(track),
            name: Some(name.to_string()),
            debug_annotations: anns.iter().map(Annotation::to_proto).collect(),
            flow_ids: flow_ids.to_vec(),
        });
        self.packets.push(p);
    }

    pub fn end(&mut self, track: u64, ts_ns: i64) {
        let mut p = self.new_packet();
        p.timestamp = Some(ts_ns as u64);
        p.track_event = Some(TrackEvent {
            r#type: Some(EventType::SliceEnd as i32),
            track_uuid: Some(track),
            ..Default::default()
        });
        self.packets.push(p);
    }

    pub fn instant(&mut self, track: u64, ts_ns: i64, name: &str, anns: &[Annotation]) {
        let mut p = self.new_packet();
        p.timestamp = Some(ts_ns as u64);
        p.track_event = Some(TrackEvent {
            r#type: Some(EventType::Instant as i32),
            track_uuid: Some(track),
            name: Some(name.to_string()),
            debug_annotations: anns.iter().map(Annotation::to_proto).collect(),
            flow_ids: Vec::new(),
        });
        self.packets.push(p);
    }

    /// Stream the protobuf packet fields through gzip into ``output``.
    ///
    /// A protobuf repeated message may be encoded as consecutive field-1
    /// length-delimited values. Encoding one packet at a time avoids holding a
    /// full raw protobuf buffer alongside the final gzip buffer for large runs.
    pub fn write_gzip<W: Write>(self, output: W) -> Result<W> {
        let mut encoder = GzEncoder::new(output, Compression::default());
        for packet in self.packets {
            encoder.write_all(&[0x0a])?; // Trace.packet: field 1, wire type 2.
            encoder.write_all(&packet.encode_length_delimited_to_vec())?;
        }
        Ok(encoder.finish()?)
    }

    #[cfg(test)]
    pub fn into_gzip(self) -> Result<Vec<u8>> {
        self.write_gzip(Vec::new())
    }
}
