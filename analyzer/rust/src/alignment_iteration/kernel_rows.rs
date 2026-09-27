//! Schema-6 kernel rows: `parsed.kernels.parquet` beside `parsed.json`.
//!
//! Kernel rows are 95% of a normalized capture. As inline JSON, a 32.6M-kernel
//! GLM-5.3 TP4 capture was an 11 GB document whose single-threaded parse took
//! ~19 s before any analysis could start. The parser now writes them as one
//! parquet row per kernel (`alignment/nsys/parsed_io.py` owns the layout), and
//! this module decodes the row groups in parallel and hangs each row back on
//! the range it came from, in file order — range order, then ordinal order,
//! exactly the inline order.

use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};
use arrow_array::{Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::Deserialize;

use super::{MeasuredKernel, ParsedTrace};

/// `parsed.json`'s pointer to its kernel rows.
#[derive(Deserialize)]
pub(super) struct KernelRowsRef {
    file: String,
    format: String,
    rows: usize,
}

/// One decoded row: its range's position in `iteration_details[..].ranges[..]`.
type LocatedKernel = (usize, usize, MeasuredKernel);

/// Attach the rows `rows` names to `trace`, whose ranges must carry none inline.
pub(super) fn attach(
    trace: &mut ParsedTrace,
    parsed_path: &Path,
    rows: &KernelRowsRef,
) -> Result<()> {
    ensure!(
        rows.format == "parquet",
        "unsupported kernel_rows format {:?} in {}",
        rows.format,
        parsed_path.display()
    );
    ensure!(
        trace
            .iteration_details
            .iter()
            .all(|detail| detail.ranges.iter().all(|range| range.kernels.is_empty())),
        "{} names kernel_rows but also carries inline kernels",
        parsed_path.display()
    );
    let path = parsed_path.with_file_name(&rows.file);
    let open = || File::open(&path).with_context(|| format!("open {}", path.display()));
    let row_groups = ParquetRecordBatchReaderBuilder::try_new(open()?)
        .with_context(|| format!("read parquet metadata of {}", path.display()))?
        .metadata()
        .num_row_groups();
    let decoded = (0..row_groups)
        .into_par_iter()
        .map(|row_group| -> Result<Vec<LocatedKernel>> {
            let reader = ParquetRecordBatchReaderBuilder::try_new(open()?)?
                .with_row_groups(vec![row_group])
                .build()?;
            let mut kernels = Vec::new();
            for batch in reader {
                decode_batch(&batch?, &mut kernels)
                    .with_context(|| format!("{} row group {row_group}", path.display()))?;
            }
            Ok(kernels)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut attached = 0;
    for (detail_index, range_index, kernel) in decoded.into_iter().flatten() {
        trace
            .iteration_details
            .get_mut(detail_index)
            .and_then(|detail| detail.ranges.get_mut(range_index))
            .ok_or_else(|| {
                anyhow!(
                    "{}: kernel row names range {detail_index}/{range_index}, which {} lacks",
                    path.display(),
                    parsed_path.display()
                )
            })?
            .kernels
            .push(kernel);
        attached += 1;
    }
    ensure!(
        attached == rows.rows,
        "{} has {attached} kernel rows, {} declares {}",
        path.display(),
        parsed_path.display(),
        rows.rows
    );
    Ok(())
}

fn decode_batch(batch: &RecordBatch, out: &mut Vec<LocatedKernel>) -> Result<()> {
    let detail_index = typed::<UInt32Array>(batch, "detail_index")?;
    let range_index = typed::<UInt32Array>(batch, "range_index")?;
    let name_id = typed::<UInt32Array>(batch, "name_id")?;
    let category = typed::<StringArray>(batch, "category")?;
    let start_ns = typed::<UInt64Array>(batch, "start_ns")?;
    let end_ns = typed::<UInt64Array>(batch, "end_ns")?;
    let stream_id = typed::<UInt64Array>(batch, "stream_id")?;
    let correlation_id = typed::<UInt64Array>(batch, "correlation_id")?;
    let track_index = typed::<UInt32Array>(batch, "track_index")?;
    for (name, column) in [
        ("detail_index", detail_index as &dyn Array),
        ("range_index", range_index),
        ("name_id", name_id),
        ("category", category),
        ("start_ns", start_ns),
        ("end_ns", end_ns),
        ("track_index", track_index),
    ] {
        ensure!(column.null_count() == 0, "column {name:?} has nulls");
    }
    let optional =
        |column: &UInt64Array, row: usize| column.is_valid(row).then(|| column.value(row));
    out.reserve(batch.num_rows());
    for row in 0..batch.num_rows() {
        out.push((
            detail_index.value(row) as usize,
            range_index.value(row) as usize,
            MeasuredKernel {
                name_id: u64::from(name_id.value(row)),
                category: category.value(row).to_owned(),
                start_ns: start_ns.value(row),
                end_ns: end_ns.value(row),
                correlation_id: optional(correlation_id, row),
                track_index: track_index.value(row) as usize,
                stream_id: optional(stream_id, row),
            },
        ));
    }
    Ok(())
}

fn typed<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow!("kernel rows lack column {name:?}"))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| anyhow!("kernel rows column {name:?} has an unexpected type"))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch, StringArray, UInt32Array, UInt64Array};
    use parquet::arrow::ArrowWriter;

    use super::super::parsed_trace;

    /// Two ranges' kernels in file order, as `alignment/nsys/parsed_io.py` writes them.
    fn write_rows(path: &std::path::Path, ranges: &[(u32, u32)], stream: Option<u64>) {
        let n = ranges.len();
        let u32s = |values: Vec<u32>| Arc::new(UInt32Array::from(values)) as ArrayRef;
        let u64s = |values: Vec<Option<u64>>| Arc::new(UInt64Array::from(values)) as ArrayRef;
        let batch = RecordBatch::try_from_iter([
            ("detail_index", u32s(ranges.iter().map(|r| r.0).collect())),
            ("range_index", u32s(ranges.iter().map(|r| r.1).collect())),
            ("ordinal", u32s((1..=n as u32).collect())),
            ("name_id", u32s((0..n as u32).collect())),
            (
                "category",
                Arc::new(StringArray::from(vec!["gemm"; n])) as ArrayRef,
            ),
            (
                "start_ns",
                u64s((0..n as u64).map(|i| Some(10 * i)).collect()),
            ),
            (
                "end_ns",
                u64s((0..n as u64).map(|i| Some(10 * i + 5)).collect()),
            ),
            ("stream_id", u64s(vec![stream; n])),
            ("correlation_id", u64s(vec![None; n])),
            ("track_index", u32s(vec![0; n])),
        ])
        .unwrap();
        let mut writer =
            ArrowWriter::try_new(fs::File::create(path).unwrap(), batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn write_parsed(path: &std::path::Path, rows: usize) {
        let range = r#"{"device_id":0,"phase":"forward","start_ns":0,"end_ns":100}"#;
        fs::write(
            path,
            format!(
                r#"{{"kernel_names":{{}},"kernel_rows":{{"file":"parsed.kernels.parquet","format":"parquet","rows":{rows}}},
                "iteration_details":[{{"iteration":7,"iteration_type":"decode","ranges":[{range},{range}]}}]}}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn schema_six_rows_return_to_their_ranges_in_file_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("parsed.json");
        write_rows(
            &dir.path().join("parsed.kernels.parquet"),
            &[(0, 0), (0, 1), (0, 1)],
            Some(19),
        );
        write_parsed(&path, 3);

        let trace = parsed_trace(&path).unwrap();
        let ranges = &trace.iteration_details[0].ranges;
        let starts = |index: usize| {
            ranges[index]
                .kernels
                .iter()
                .map(|k| k.start_ns)
                .collect::<Vec<_>>()
        };
        assert_eq!(starts(0), vec![0]);
        assert_eq!(starts(1), vec![10, 20]);
        let kernel = &ranges[1].kernels[1];
        assert_eq!((kernel.name_id, kernel.category.as_str()), (2, "gemm"));
        assert_eq!((kernel.stream_id, kernel.correlation_id), (Some(19), None));
    }

    #[test]
    fn schema_six_rejects_a_row_count_or_range_it_does_not_declare() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("parsed.json");
        write_rows(
            &dir.path().join("parsed.kernels.parquet"),
            &[(0, 0), (0, 1)],
            None,
        );
        write_parsed(&path, 3);
        let error = parsed_trace(&path).err().unwrap();
        assert!(format!("{error:#}").contains("declares 3"), "{error:#}");

        let other = tempfile::tempdir().unwrap();
        let path = other.path().join("parsed.json");
        write_rows(
            &other.path().join("parsed.kernels.parquet"),
            &[(0, 2)],
            None,
        );
        write_parsed(&path, 1);
        let error = parsed_trace(&path).err().unwrap();
        assert!(format!("{error:#}").contains("range 0/2"), "{error:#}");
    }
}
