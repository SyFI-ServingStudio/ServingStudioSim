//! Lossless run encoding for per-row `u32` vectors in serialized kernel inputs.
//!
//! Some leaf inputs carry one value per query row (`dsa_sparse_index_remap`'s
//! `local_span_lengths` / `valid_counts`). Written out element by element, one
//! 8k-token long-context chunk is ~100 KB of JSON per slot, and a 1M-context
//! replay's `cost_log` reached gigabytes of `slot_input` — past what one Arrow
//! `Utf8` read batch can index. Those vectors are causal ramps, caps and kpool
//! sawtooths, so they collapse to a few segments per request.
//!
//! The JSON is an array whose elements are either a bare value or a run
//! `[start, step, len]` / `[start, step, len, repeat]`: the block
//! `start + step * i` for `i in 0..len`, emitted `repeat` times (default 1).
//! A plain array of numbers is therefore also a valid encoding, so inputs
//! written before this format deserialize unchanged. Use it as
//! `#[serde(with = "crate::timing::run_encoded")]` on a `Vec<u32>` field.

use serde::de::Error as _;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A run must expand to at least this many values; shorter stretches are
/// written as bare values, so unstructured vectors (decode rows) never grow.
const MIN_RUN_VALUES: usize = 4;

/// Upper bound on a decoded vector, so a short document cannot request an
/// arbitrarily large allocation. Far above any real per-iteration row count.
const MAX_DECODED_VALUES: u64 = 1 << 24;

#[derive(Debug, PartialEq)]
enum Segment {
    Value(u32),
    Run {
        start: u32,
        step: i64,
        len: u32,
        repeat: u32,
    },
}

impl Serialize for Segment {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match *self {
            Segment::Value(value) => serializer.serialize_u32(value),
            Segment::Run {
                start,
                step,
                len,
                repeat: 1,
            } => (start, step, len).serialize(serializer),
            Segment::Run {
                start,
                step,
                len,
                repeat,
            } => (start, step, len, repeat).serialize(serializer),
        }
    }
}

/// Greedy: the longest arithmetic run from the cursor, then how many times
/// that whole block repeats right after it.
fn encode(values: &[u32]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut at = 0;
    while at < values.len() {
        let rest = &values[at..];
        if rest.len() < MIN_RUN_VALUES {
            segments.extend(rest.iter().map(|&value| Segment::Value(value)));
            break;
        }
        let step = i64::from(rest[1]) - i64::from(rest[0]);
        let len = 2 + rest
            .windows(2)
            .skip(1)
            .take_while(|pair| i64::from(pair[1]) - i64::from(pair[0]) == step)
            .count();
        let block = &rest[..len];
        let repeat = 1 + rest[len..]
            .chunks_exact(len)
            .take_while(|chunk| *chunk == block)
            .count();
        if len * repeat < MIN_RUN_VALUES {
            segments.push(Segment::Value(rest[0]));
            at += 1;
            continue;
        }
        segments.push(Segment::Run {
            start: rest[0],
            step,
            len: len as u32,
            repeat: repeat as u32,
        });
        at += len * repeat;
    }
    segments
}

pub fn serialize<S: Serializer>(values: &[u32], serializer: S) -> Result<S::Ok, S::Error> {
    let segments = encode(values);
    let mut seq = serializer.serialize_seq(Some(segments.len()))?;
    for segment in &segments {
        seq.serialize_element(segment)?;
    }
    seq.end()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Element {
    Value(u32),
    Run(u32, i64, u32),
    RepeatedRun(u32, i64, u32, u32),
}

pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u32>, D::Error> {
    let mut values = Vec::new();
    for element in Vec::<Element>::deserialize(deserializer)? {
        let (start, step, len, repeat) = match element {
            Element::Value(value) => (value, 0, 1, 1),
            Element::Run(start, step, len) => (start, step, len, 1),
            Element::RepeatedRun(start, step, len, repeat) => (start, step, len, repeat),
        };
        if len == 0 || repeat == 0 {
            return Err(D::Error::custom("a run needs len >= 1 and repeat >= 1"));
        }
        if values.len() as u64 + u64::from(len) * u64::from(repeat) > MAX_DECODED_VALUES {
            return Err(D::Error::custom(format!(
                "run-encoded vector exceeds {MAX_DECODED_VALUES} values"
            )));
        }
        let last = i64::from(start) + step * i64::from(len - 1);
        if u32::try_from(last).is_err() {
            return Err(D::Error::custom(format!(
                "run [{start}, {step}, {len}] leaves the u32 range"
            )));
        }
        let block_start = values.len();
        values.extend((0..i64::from(len)).map(|i| (i64::from(start) + step * i) as u32));
        for _ in 1..repeat {
            values.extend_from_within(block_start..block_start + len as usize);
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Holder(#[serde(with = "super")] Vec<u32>);

    fn encoded(values: &[u32]) -> Value {
        serde_json::to_value(Holder(values.to_vec())).unwrap()
    }

    fn round_trip(values: Vec<u32>) {
        let text = serde_json::to_string(&Holder(values.clone())).unwrap();
        assert_eq!(serde_json::from_str::<Holder>(&text).unwrap().0, values);
    }

    #[test]
    fn causal_ramps_and_caps_collapse_to_one_run_each() {
        let spans: Vec<u32> = (319_223..319_223 + 8192).collect();
        assert_eq!(encoded(&spans), json!([[319_223, 1, 8192]]));
        let valid: Vec<u32> = (1..=6000).map(|span: u32| span.min(2048)).collect();
        assert_eq!(encoded(&valid), json!([[1, 1, 2048], [2048, 0, 3952]]));
        round_trip(spans);
        round_trip(valid);
    }

    #[test]
    fn kpool_sawtooth_is_one_repeated_block() {
        // pooled_valid_count: min(span, 2048 + span % 4).
        let valid: Vec<u32> = (2052..2052 + 4096)
            .map(|span: u32| span.min(2048 + span % 4))
            .collect();
        assert_eq!(encoded(&valid), json!([[2048, 1, 4, 1024]]));
        let warm: Vec<u32> = (1..=4099)
            .map(|span: u32| span.min(2048 + span % 4))
            .collect();
        assert_eq!(encoded(&warm).as_array().unwrap().len(), 2);
        round_trip(warm);
    }

    #[test]
    fn per_request_segments_and_unstructured_values_round_trip() {
        let mut batch: Vec<u32> = vec![70_001, 9, 131_072, 5]; // decode rows
        batch.extend(4000..4100);
        batch.extend(100..103);
        let value = encoded(&batch);
        assert_eq!(
            value,
            json!([70_001, 9, 131_072, 5, [4000, 1, 100], 100, 101, 102])
        );
        round_trip(batch);
        round_trip(vec![]);
        round_trip(vec![7]);
        round_trip(vec![9, 7, 5, 3, 1]);
        round_trip(vec![0, u32::MAX, 0, u32::MAX, 0, u32::MAX]);
    }

    #[test]
    fn plain_arrays_from_older_logs_still_decode() {
        let holder: Holder = serde_json::from_str("[3, 4, 5, 5, 5]").unwrap();
        assert_eq!(holder.0, vec![3, 4, 5, 5, 5]);
    }

    #[test]
    fn invalid_runs_are_rejected() {
        for text in [
            "[[1, 1, 0]]",
            "[[1, 1, 3, 0]]",
            "[[1, -1, 3]]",
            "[[4294967295, 1, 2]]",
            "[[0, 0, 65536, 65536]]",
        ] {
            assert!(serde_json::from_str::<Holder>(text).is_err(), "{text}");
        }
    }
}
