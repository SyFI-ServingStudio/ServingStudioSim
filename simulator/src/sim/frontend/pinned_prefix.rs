//! The optional `prefix_len` column of `text-generation-independent`.
//!
//! `prefix_len` declares how many tokens before a request's fresh `input_len`
//! prompt are already resident in KV when it arrives, which the request carries
//! as [`SessionInput::PinnedPrefix`](crate::common::SessionInput::PinnedPrefix).
//! `input_len` keeps meaning the fresh tokens to compute, as in the session
//! format's `prefix_len,input_len` pair.
//!
//! req-frontend verifies an exact header and does not declare this column, so
//! its loader refuses a file that carries it. Until it does, this module is the
//! one place ServingStudio Sim reads that format's CSV itself:
//!
//! - a file without the column goes straight through req-frontend's loader;
//! - a file with it is checked against the same declared schema minus this one
//!   column, then each row is decoded into req-frontend's own row and tag types
//!   and validated by the same rules.
//!
//! Delete this module once req-frontend owns the column.

use anyhow::{anyhow, bail, Context, Result};
use req_frontend::schema::format::text_generation::independent::{self, TextGenerationRow};
use req_frontend::schema::format::ParsedIndependentRow;
use req_frontend::schema::{
    InputFileSchema, RequestPriority, RequestSession, RequestSlo, RequestSpeculative, TraceTag,
};
use serde::de::DeserializeOwned;

/// The optional column's header name.
pub(super) const PINNED_PREFIX_COLUMN: &str = "prefix_len";

/// One `text-generation-independent` row and its pinned prefix (0 when the
/// file has no `prefix_len` column or the cell is blank).
#[derive(Debug)]
pub(super) struct TextGenerationIndependentRow {
    pub(super) row: TextGenerationRow,
    pub(super) pinned_prefix_tokens: u32,
}

/// Load a `text-generation-independent` file, reading `prefix_len` if present.
pub(super) fn load(
    path: &str,
    input_file_schema: &InputFileSchema,
) -> Result<Vec<ParsedIndependentRow<TextGenerationIndependentRow>>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open input file: {path}"))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read the header of {path}"))?
        .clone();
    let Some(prefix_column) = headers
        .iter()
        .position(|column| column == PINNED_PREFIX_COLUMN)
    else {
        return Ok(independent::load(path, input_file_schema)?
            .into_iter()
            .map(|parsed| with_row(parsed, 0))
            .collect());
    };
    input_file_schema
        .verify_header(
            headers
                .iter()
                .filter(|column| *column != PINNED_PREFIX_COLUMN),
        )
        .map_err(|mismatch| anyhow!("{path}: {mismatch}"))?;

    let mut rows = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let at = format!("{path} line {}", index + 2);
        let record = record.with_context(|| format!("{at}: failed to read row"))?;
        let row: TextGenerationRow = record
            .deserialize(Some(&headers))
            .with_context(|| format!("{at}: failed to parse base columns"))?;
        validate_row(&row, &at)?;
        let pinned_prefix_tokens = parse_prefix_tokens(&record[prefix_column], &row, &at)?;

        let session: RequestSession =
            decode_tag(&record, &headers, input_file_schema, TraceTag::Session, &at)?;
        session.validate(&at)?;
        if pinned_prefix_tokens != 0
            && session
                .session_id
                .as_deref()
                .is_some_and(|id| !id.is_empty())
        {
            bail!(
                "{at}: prefix_len pins a standalone request's prefix; a session row declares \
                 its reusable prefix with prefix_kv"
            );
        }
        let slo: RequestSlo = decode_tag(&record, &headers, input_file_schema, TraceTag::Slo, &at)?;
        slo.validate(&at)?;
        let priority: RequestPriority = decode_tag(
            &record,
            &headers,
            input_file_schema,
            TraceTag::Priority,
            &at,
        )?;
        priority.validate(&at)?;
        let speculative: RequestSpeculative = decode_tag(
            &record,
            &headers,
            input_file_schema,
            TraceTag::Speculative,
            &at,
        )?;
        speculative.validate(&at)?;

        rows.push(ParsedIndependentRow {
            row: TextGenerationIndependentRow {
                row,
                pinned_prefix_tokens,
            },
            session,
            slo,
            priority,
            speculative,
        });
    }
    Ok(rows)
}

fn with_row(
    parsed: ParsedIndependentRow<TextGenerationRow>,
    pinned_prefix_tokens: u32,
) -> ParsedIndependentRow<TextGenerationIndependentRow> {
    ParsedIndependentRow {
        row: TextGenerationIndependentRow {
            row: parsed.row,
            pinned_prefix_tokens,
        },
        session: parsed.session,
        slo: parsed.slo,
        priority: parsed.priority,
        speculative: parsed.speculative,
    }
}

/// The checks req-frontend's `TextGenerationRow::validate` applies; its trait
/// is not exported, so they are restated here.
fn validate_row(row: &TextGenerationRow, at: &str) -> Result<()> {
    if row.id.is_empty() {
        bail!("{at}: id is empty");
    }
    if !row.arrival_time.is_finite() || row.arrival_time < 0.0 {
        bail!("{at}: arrival_time must be finite and non-negative");
    }
    if row.input_len == 0 {
        bail!("{at}: input_len must be greater than zero");
    }
    if row.output_len == 0 {
        bail!("{at}: output_len must be greater than zero");
    }
    Ok(())
}

/// A blank cell means no pinned prefix.
fn parse_prefix_tokens(cell: &str, row: &TextGenerationRow, at: &str) -> Result<u32> {
    let cell = cell.trim();
    if cell.is_empty() {
        return Ok(0);
    }
    let prefix_tokens: u32 = cell
        .parse()
        .with_context(|| format!("{at}: prefix_len={cell:?} is not a non-negative u32"))?;
    let context_tokens = u64::from(prefix_tokens) + row.input_len as u64;
    if context_tokens > u64::from(u32::MAX) {
        bail!(
            "{at}: prefix_len + input_len = {context_tokens} exceeds the u32 context-length range"
        );
    }
    Ok(prefix_tokens)
}

fn decode_tag<Tag: Default + DeserializeOwned>(
    record: &csv::StringRecord,
    headers: &csv::StringRecord,
    input_file_schema: &InputFileSchema,
    tag: TraceTag,
    at: &str,
) -> Result<Tag> {
    if !input_file_schema.carries(tag) {
        return Ok(Tag::default());
    }
    record
        .deserialize(Some(headers))
        .with_context(|| format!("{at}: failed to parse declared tag columns"))
}
