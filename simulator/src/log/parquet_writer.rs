//! Generic streaming Parquet writer — appends multiple Arrow `RecordBatch`es to
//! one file, opening the file lazily on first non-empty write. Ported from
//! `ref/moesim-rs/src/logging/parquet_writer.rs` (ZSTD-L3 compression).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

fn writer_properties() -> Result<WriterProperties> {
    Ok(WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build())
}

/// Appends record batches to one parquet file. The file (and its parent dir)
/// is created on the first non-empty `write`, so a stream that never produces a
/// row leaves no file behind.
pub struct StreamingParquetWriter {
    path: PathBuf,
    schema: Arc<Schema>,
    writer: Option<ArrowWriter<File>>,
    rows_written: usize,
}

impl StreamingParquetWriter {
    pub fn new(path: PathBuf, schema: Arc<Schema>) -> Self {
        Self {
            path,
            schema,
            writer: None,
            rows_written: 0,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    fn writer_mut(&mut self) -> Result<&mut ArrowWriter<File>> {
        if self.writer.is_none() {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = File::create(&self.path)?;
            let writer =
                ArrowWriter::try_new(file, self.schema.clone(), Some(writer_properties()?))?;
            self.writer = Some(writer);
        }
        Ok(self.writer.as_mut().unwrap())
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<usize> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(0);
        }
        self.writer_mut()?.write(batch)?;
        self.rows_written += num_rows;
        Ok(num_rows)
    }

    pub fn close(&mut self) -> Result<usize> {
        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }
        Ok(self.rows_written)
    }
}
