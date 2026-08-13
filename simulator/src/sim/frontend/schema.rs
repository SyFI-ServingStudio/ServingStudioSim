//! Declared, exact trace schemas that produce statically typed request rows.
//!
//! A file declares one [`TraceKind`]. The declaration selects one concrete
//! `TraceDefinition` implementation before any row is parsed; headers never
//! infer the request family and rows never carry a request-kind union.

use std::collections::HashMap;
use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;

use anyhow::{anyhow, bail, Context, Result};
use serde::de::DeserializeOwned;

use super::arrival::{
    ReleaseMetadata, ScheduledRequest, SchedulingDeclaration, SessionReleaseMetadata,
};
use crate::common::{
    AudioExtent, AudioTextGenerationDefinition, DecodingStrategy, ImageExtent,
    ImageGenerationDefinition, ImageTextGenerationDefinition, ImageToVideoDefinition,
    OmniGenerationDefinition, OmniInputSegment, OmniOutputSpec, RequestDefinition, RequestId,
    SessionInput, SpeechGenerationDefinition, TextGenerationDefinition, Time, VideoExtent,
    VideoGenerationDefinition, VideoTextGenerationDefinition,
};

const DEFAULT_PRIORITY: i32 = 0;

/// The taxonomy itself is defined once, in the crate this simulator and the
/// replay client share: which kinds exist, which tags may be declared on them,
/// and which columns each combination obliges a file to carry. A trace this
/// simulator accepts is then exactly a trace the client accepts, because there
/// is only one rule and neither side wrote its own copy of it.
///
/// What stays here is how *this* program reads such a file: the CSV plumbing
/// below, and the typed parse targets those columns produce.
pub use req_frontend::schema::{SourceSchema, TraceDeclaration, TraceKind, TraceTag};

/// Check a file's header against its declaration, naming the file.
///
/// The path is formatted into the message rather than attached as context: an
/// anyhow context becomes the outermost message, which would leave a caller
/// reading `to_string()` with a filename and no reason.
fn verify_header(declaration: &TraceDeclaration, index: &HeaderIndex, path: &Path) -> Result<()> {
    declaration
        .verify_header(index.names())
        .map_err(|mismatch| anyhow!("{}: {mismatch}", path.display()))
}

/// Parse whichever tag columns the declaration says are present.
fn parse_declared_tags(declaration: &TraceDeclaration, row: &Row<'_>) -> Result<ParsedTags> {
    let session = if declaration.carries(TraceTag::Session) {
        parse_session(row)?
    } else {
        None
    };
    let scheduling = if declaration.carries(TraceTag::Slo) {
        parse_scheduling(row)?
    } else {
        SchedulingDeclaration::default()
    };
    let decoding = if declaration.carries(TraceTag::Speculative) {
        parse_decoding(row)?
    } else {
        DecodingStrategy::Standard
    };
    Ok(ParsedTags {
        session,
        scheduling,
        decoding,
    })
}

#[derive(Clone, Copy, Debug)]
struct ParsedSession {
    session_id: u32,
    declared_prefix_tokens: u32,
    tool_wait_after: Time,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ParsedTags {
    session: Option<ParsedSession>,
    scheduling: SchedulingDeclaration,
    decoding: DecodingStrategy,
}

impl ParsedTags {
    fn session_input(self, session_start_time: Option<Time>) -> SessionInput {
        match (self.session, session_start_time) {
            (None, None) => SessionInput::Standalone,
            (Some(session), Some(session_start_time)) => SessionInput::Session {
                session_id: session.session_id,
                session_start_time,
                declared_prefix_tokens: session.declared_prefix_tokens,
            },
            _ => unreachable!("parsed session and resolved session start must agree"),
        }
    }

    fn release_session(self) -> Option<SessionReleaseMetadata> {
        self.session.map(|session| SessionReleaseMetadata {
            session_id: session.session_id,
            tool_wait_after: session.tool_wait_after,
        })
    }

    fn reject_autoregressive_only_tags(self, row: &Row<'_>) -> Result<()> {
        if let Some(session) = self.session {
            if session.declared_prefix_tokens != 0 {
                bail!(
                    "{}: prefix_kv={} is not meaningful for a generated-media request",
                    row.at(),
                    session.declared_prefix_tokens
                );
            }
        }
        if self.decoding != DecodingStrategy::Standard {
            bail!(
                "{}: speculative decoding is only valid for autoregressive output",
                row.at()
            );
        }
        Ok(())
    }
}

/// Private parser contract: choosing `Definition` fixes the accepted kind.
pub(super) trait TraceDefinition: RequestDefinition + Sized {
    const KIND: TraceKind;

    fn parse_definition(
        row: &Row<'_>,
        tags: ParsedTags,
        session_start_time: Option<Time>,
    ) -> Result<Self>;

    /// Build this definition from a canonical `session-execution-v2` row.
    ///
    /// Defaults to a refusal rather than a best-effort mapping: the canonical
    /// format describes a text session, and quietly accepting it for another
    /// request family would invent fields the file never carried.
    fn parse_execution_v2(_row: &Row<'_>, _session: SessionInput) -> Result<Self> {
        bail!(
            "{:?} traces cannot be read from a session-execution-v2 file",
            Self::KIND
        )
    }
}

impl TraceDefinition for TextGenerationDefinition {
    const KIND: TraceKind = TraceKind::TextGeneration;

    fn parse_definition(
        row: &Row<'_>,
        tags: ParsedTags,
        session_start_time: Option<Time>,
    ) -> Result<Self> {
        Ok(Self {
            prompt_tokens: positive_u32(row, "input_len")?,
            target_output_tokens: positive_u32(row, "output_len")?,
            session: tags.session_input(session_start_time),
            decoding: tags.decoding,
        })
    }

    fn parse_execution_v2(row: &Row<'_>, session: SessionInput) -> Result<Self> {
        // `input_len` may be zero when a prefix exists: a round that appends
        // nothing still re-sends its whole conversation, and still prefills
        // whatever part of that prefix is no longer resident.
        let prompt_tokens: u32 = row.cell("input_len")?;
        if prompt_tokens == 0 && session.declared_prefix_tokens() == 0 {
            bail!(
                "{}: input_len=0 with no prefix is an empty prompt",
                row.at()
            );
        }
        Ok(Self {
            prompt_tokens,
            target_output_tokens: positive_u32(row, "output_len")?,
            session,
            decoding: DecodingStrategy::Standard,
        })
    }
}

macro_rules! impl_encoded_definition {
    ($definition:ty, $kind:expr, $extent:expr) => {
        impl TraceDefinition for $definition {
            const KIND: TraceKind = $kind;

            fn parse_definition(
                row: &Row<'_>,
                tags: ParsedTags,
                session_start_time: Option<Time>,
            ) -> Result<Self> {
                Ok(Self {
                    text_prompt_tokens: positive_u32(row, "input_len")?,
                    encoded_input_tokens: positive_u32(row, "encoded_tokens")?,
                    target_output_tokens: positive_u32(row, "output_len")?,
                    extent: $extent(row)?,
                    session: tags.session_input(session_start_time),
                    decoding: tags.decoding,
                })
            }
        }
    };
}

impl_encoded_definition!(
    ImageTextGenerationDefinition,
    TraceKind::ImageToText,
    parse_image_extent
);
impl_encoded_definition!(
    VideoTextGenerationDefinition,
    TraceKind::VideoToText,
    parse_video_extent
);
impl_encoded_definition!(
    AudioTextGenerationDefinition,
    TraceKind::AudioToText,
    parse_audio_extent
);

macro_rules! impl_generated_definition {
    ($definition:ty, $kind:expr, $extent:expr) => {
        impl TraceDefinition for $definition {
            const KIND: TraceKind = $kind;

            fn parse_definition(
                row: &Row<'_>,
                tags: ParsedTags,
                _session_start_time: Option<Time>,
            ) -> Result<Self> {
                tags.reject_autoregressive_only_tags(row)?;
                Ok(Self {
                    text_prompt_tokens: positive_u32(row, "input_len")?,
                    target_generation_steps: positive_u32(row, "denoise_steps")?,
                    extent: $extent(row)?,
                })
            }
        }
    };
}

impl_generated_definition!(
    ImageGenerationDefinition,
    TraceKind::TextToImage,
    parse_image_extent
);
impl_generated_definition!(
    VideoGenerationDefinition,
    TraceKind::TextToVideo,
    parse_video_extent
);
impl_generated_definition!(
    SpeechGenerationDefinition,
    TraceKind::TextToSpeech,
    parse_audio_extent
);

impl TraceDefinition for ImageToVideoDefinition {
    const KIND: TraceKind = TraceKind::ImageToVideo;

    fn parse_definition(
        row: &Row<'_>,
        tags: ParsedTags,
        _session_start_time: Option<Time>,
    ) -> Result<Self> {
        tags.reject_autoregressive_only_tags(row)?;
        Ok(Self {
            text_prompt_tokens: positive_u32(row, "input_len")?,
            encoded_input_tokens: positive_u32(row, "encoded_tokens")?,
            input_extent: parse_input_image_extent(row)?,
            output_extent: parse_video_extent(row)?,
            target_generation_steps: positive_u32(row, "denoise_steps")?,
        })
    }
}

impl TraceDefinition for OmniGenerationDefinition {
    const KIND: TraceKind = TraceKind::OmniGeneration;

    fn parse_definition(
        row: &Row<'_>,
        tags: ParsedTags,
        session_start_time: Option<Time>,
    ) -> Result<Self> {
        let input: Vec<OmniInputSegment> = json_cell(row, "input_segments")?;
        let output: Vec<OmniOutputSpec> = json_cell(row, "output_segments")?;
        validate_omni_input(row, &input)?;
        validate_omni_output(row, &output)?;
        Ok(Self {
            input,
            output,
            session: tags.session_input(session_start_time),
            decoding: tags.decoding,
        })
    }
}

fn parse_session(row: &Row<'_>) -> Result<Option<ParsedSession>> {
    let Some(session_id) = row.optional_cell("session_id").transpose()? else {
        let prefix_tokens: u32 = row.cell_or("prefix_kv", 0)?;
        let tool_wait_ms: f64 = row.cell_or("tool_wait_after_ms", 0.0)?;
        if prefix_tokens != 0 || tool_wait_ms != 0.0 {
            bail!(
                "{}: prefix_kv/tool_wait_after_ms requires a non-empty session_id",
                row.at()
            );
        }
        return Ok(None);
    };
    let tool_wait_ms: f64 = row.cell_or("tool_wait_after_ms", 0.0)?;
    require_nonnegative_finite(row, "tool_wait_after_ms", tool_wait_ms)?;
    Ok(Some(ParsedSession {
        session_id,
        declared_prefix_tokens: row.cell_or("prefix_kv", 0)?,
        tool_wait_after: Time::from_ms(tool_wait_ms),
    }))
}

fn parse_scheduling(row: &Row<'_>) -> Result<SchedulingDeclaration> {
    let relative_completion_deadline = row
        .optional_cell::<f64>("deadline_ms")
        .transpose()?
        .map(|deadline_ms| {
            require_nonnegative_finite(row, "deadline_ms", deadline_ms)?;
            Ok::<Time, anyhow::Error>(Time::from_ms(deadline_ms))
        })
        .transpose()?;
    Ok(SchedulingDeclaration {
        priority: row.cell_or("priority", DEFAULT_PRIORITY)?,
        relative_completion_deadline,
    })
}

fn parse_decoding(row: &Row<'_>) -> Result<DecodingStrategy> {
    let Some(accept_rate) = row.optional_cell::<f32>("accept_rate").transpose()? else {
        return Ok(DecodingStrategy::Standard);
    };
    if !accept_rate.is_finite() || !(0.0..=1.0).contains(&accept_rate) {
        bail!(
            "{}: accept_rate={} must be finite and in 0.0..=1.0",
            row.at(),
            accept_rate
        );
    }
    Ok(DecodingStrategy::Speculative { accept_rate })
}

fn parse_release(row: &Row<'_>, tags: ParsedTags) -> Result<ReleaseMetadata> {
    let trace_arrival_time_ms: f64 = row.cell("arrival_time")?;
    require_nonnegative_finite(row, "arrival_time", trace_arrival_time_ms)?;
    Ok(ReleaseMetadata {
        request_id: RequestId(row.cell("id")?),
        trace_arrival_time_ms,
        session: tags.release_session(),
    })
}

/// Turn one canonical row into a scheduled request.
///
/// Deliberately its own path rather than a remapping onto the native columns:
/// the canonical format differs from the native one in identity *type*, not
/// just in column names, and hiding that behind an alias table would be the
/// kind of silent reinterpretation this format exists to prevent.
fn parse_execution_v2_row<Definition: TraceDefinition>(
    row: &Row<'_>,
    session_start_times: &mut HashMap<u32, Time>,
    identities: &mut SourceIdentities,
) -> Result<ScheduledRequest<Definition>> {
    let source_request_id = row.raw("request_id")?;
    let source_session_id = row.raw("session_id")?;
    if source_request_id.is_empty() || source_session_id.is_empty() {
        bail!("{}: request_id and session_id must be non-empty", row.at());
    }

    let trace_arrival_time_ms: f64 = row.cell("arrival_time_ms")?;
    require_nonnegative_finite(row, "arrival_time_ms", trace_arrival_time_ms)?;
    let tool_wait_ms: f64 = row.cell("tool_wait_after_ms")?;
    require_nonnegative_finite(row, "tool_wait_after_ms", tool_wait_ms)?;

    let session_id = identities.intern_session(source_session_id);
    let request_id = identities.intern_request(source_request_id);
    let declared_prefix_tokens: u32 = row.cell("prefix_len")?;
    let round_idx: u32 = row.cell("round_idx")?;
    if round_idx == 0 && declared_prefix_tokens != 0 {
        bail!(
            "{}: round_idx=0 declares prefix_len={declared_prefix_tokens}, but a session's \
             first round has no previous context to reuse",
            row.at()
        );
    }

    let session_start_time = *session_start_times
        .entry(session_id)
        .or_insert_with(|| Time::from_ms(trace_arrival_time_ms));

    let release = ReleaseMetadata {
        request_id,
        trace_arrival_time_ms,
        session: Some(SessionReleaseMetadata {
            session_id,
            tool_wait_after: Time::from_ms(tool_wait_ms),
        }),
    };
    let session = SessionInput::Session {
        session_id,
        session_start_time,
        declared_prefix_tokens,
    };
    Ok(ScheduledRequest {
        release,
        scheduling: SchedulingDeclaration::default(),
        definition: Definition::parse_execution_v2(row, session)?,
    })
}

fn positive_u32(row: &Row<'_>, column: &str) -> Result<u32> {
    let value: u32 = row.cell(column)?;
    if value == 0 {
        bail!("{}: {column}=0", row.at());
    }
    Ok(value)
}

fn json_cell<T: DeserializeOwned>(row: &Row<'_>, column: &str) -> Result<T> {
    let text = row.raw(column)?;
    serde_json::from_str(text).with_context(|| {
        format!(
            "{}: column `{column}` must contain valid JSON for its declared schema",
            row.at()
        )
    })
}

fn validate_omni_input(row: &Row<'_>, input: &[OmniInputSegment]) -> Result<()> {
    if input.is_empty() {
        bail!(
            "{}: input_segments must contain at least one segment",
            row.at()
        );
    }
    for (index, segment) in input.iter().enumerate() {
        match segment {
            OmniInputSegment::Text { tokens } => {
                require_positive_fact(row, &format!("input_segments[{index}].tokens"), *tokens)?;
            }
            OmniInputSegment::Image {
                extent,
                encoded_tokens,
            } => {
                validate_image_extent(row, &format!("input_segments[{index}].extent"), extent)?;
                require_positive_fact(
                    row,
                    &format!("input_segments[{index}].encoded_tokens"),
                    *encoded_tokens,
                )?;
            }
            OmniInputSegment::Audio {
                extent,
                encoded_tokens,
            } => {
                validate_audio_extent(row, &format!("input_segments[{index}].extent"), extent)?;
                require_positive_fact(
                    row,
                    &format!("input_segments[{index}].encoded_tokens"),
                    *encoded_tokens,
                )?;
            }
            OmniInputSegment::Video {
                extent,
                encoded_tokens,
            } => {
                validate_video_extent(row, &format!("input_segments[{index}].extent"), extent)?;
                require_positive_fact(
                    row,
                    &format!("input_segments[{index}].encoded_tokens"),
                    *encoded_tokens,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_omni_output(row: &Row<'_>, output: &[OmniOutputSpec]) -> Result<()> {
    if output.is_empty() {
        bail!(
            "{}: output_segments must contain at least one segment",
            row.at()
        );
    }
    for (index, spec) in output.iter().enumerate() {
        let label = format!("output_segments[{index}]");
        match spec {
            OmniOutputSpec::Text { target_tokens } => {
                require_positive_fact(row, &format!("{label}.target_tokens"), *target_tokens)?;
            }
            OmniOutputSpec::Image {
                extent,
                target_tokens,
            } => {
                validate_image_extent(row, &format!("{label}.extent"), extent)?;
                require_positive_fact(row, &format!("{label}.target_tokens"), *target_tokens)?;
            }
            OmniOutputSpec::Audio {
                extent,
                target_tokens,
            } => {
                validate_audio_extent(row, &format!("{label}.extent"), extent)?;
                require_positive_fact(row, &format!("{label}.target_tokens"), *target_tokens)?;
            }
            OmniOutputSpec::Video {
                extent,
                target_tokens,
            } => {
                validate_video_extent(row, &format!("{label}.extent"), extent)?;
                require_positive_fact(row, &format!("{label}.target_tokens"), *target_tokens)?;
            }
        }
    }
    Ok(())
}

fn require_positive_fact(row: &Row<'_>, label: &str, value: u32) -> Result<()> {
    if value == 0 {
        bail!("{}: {label}=0", row.at());
    }
    Ok(())
}

fn validate_image_extent(row: &Row<'_>, label: &str, extent: &ImageExtent) -> Result<()> {
    require_positive_fact(row, &format!("{label}.width"), extent.width)?;
    require_positive_fact(row, &format!("{label}.height"), extent.height)
}

fn validate_video_extent(row: &Row<'_>, label: &str, extent: &VideoExtent) -> Result<()> {
    require_positive_fact(row, &format!("{label}.width"), extent.width)?;
    require_positive_fact(row, &format!("{label}.height"), extent.height)?;
    require_positive_fact(row, &format!("{label}.frames"), extent.frames)
}

fn validate_audio_extent(row: &Row<'_>, label: &str, extent: &AudioExtent) -> Result<()> {
    if extent.samples == 0 {
        bail!("{}: {label}.samples=0", row.at());
    }
    Ok(())
}

fn parse_image_extent(row: &Row<'_>) -> Result<ImageExtent> {
    Ok(ImageExtent {
        width: positive_u32(row, "media_width")?,
        height: positive_u32(row, "media_height")?,
    })
}

fn parse_input_image_extent(row: &Row<'_>) -> Result<ImageExtent> {
    Ok(ImageExtent {
        width: positive_u32(row, "input_media_width")?,
        height: positive_u32(row, "input_media_height")?,
    })
}

fn parse_video_extent(row: &Row<'_>) -> Result<VideoExtent> {
    let frames = resolve_count(row, "media_duration_s", "media_fps")?;
    Ok(VideoExtent {
        width: positive_u32(row, "media_width")?,
        height: positive_u32(row, "media_height")?,
        frames: u32::try_from(frames)
            .with_context(|| format!("{}: video frame count exceeds u32", row.at()))?,
    })
}

fn parse_audio_extent(row: &Row<'_>) -> Result<AudioExtent> {
    Ok(AudioExtent {
        samples: resolve_count(row, "media_duration_s", "media_sample_rate_hz")?,
    })
}

fn resolve_count(row: &Row<'_>, duration_column: &str, rate_column: &str) -> Result<u64> {
    let duration: f64 = row.cell(duration_column)?;
    let rate: f64 = row.cell(rate_column)?;
    let count = (duration * rate).round();
    if !count.is_finite() || count < 1.0 || count > u64::MAX as f64 {
        bail!(
            "{}: {duration_column}={duration} * {rate_column}={rate} resolves to \
             {count} units (must be in 1..=u64::MAX)",
            row.at()
        );
    }
    Ok(count as u64)
}

fn require_nonnegative_finite(row: &Row<'_>, column: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!(
            "{}: invalid {column}={value} (must be finite and non-negative)",
            row.at()
        );
    }
    Ok(())
}

struct HeaderIndex(HashMap<String, usize>);

impl HeaderIndex {
    fn build(headers: &csv::StringRecord, path: &Path) -> Result<Self> {
        let mut positions = HashMap::new();
        for (position, name) in headers.iter().enumerate() {
            if positions.insert(name.to_string(), position).is_some() {
                bail!("{}: duplicate trace column `{name}`", path.display());
            }
        }
        Ok(Self(positions))
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

pub(super) struct Row<'a> {
    record: &'a csv::StringRecord,
    index: &'a HeaderIndex,
    path: &'a Path,
    line: usize,
}

impl Row<'_> {
    fn at(&self) -> String {
        format!("{}: row {}", self.path.display(), self.line)
    }

    fn raw(&self, column: &str) -> Result<&str> {
        let position = *self
            .index
            .0
            .get(column)
            .with_context(|| format!("{}: column `{column}` not in header", self.at()))?;
        Ok(self.record.get(position).unwrap_or("").trim())
    }

    fn cell<T: FromStr>(&self, column: &str) -> Result<T>
    where
        T::Err: Display,
    {
        let text = self.raw(column)?;
        text.parse().map_err(|error| {
            anyhow::anyhow!("{}: column `{column}` value {text:?}: {error}", self.at())
        })
    }

    fn cell_or<T: FromStr>(&self, column: &str, neutral: T) -> Result<T>
    where
        T::Err: Display,
    {
        Ok(self.optional_cell(column).transpose()?.unwrap_or(neutral))
    }

    fn optional_cell<T: FromStr>(&self, column: &str) -> Option<Result<T>>
    where
        T::Err: Display,
    {
        match self.raw(column) {
            Err(error) => Some(Err(error)),
            Ok("") => None,
            Ok(_) => Some(self.cell(column)),
        }
    }
}

/// Opaque source identifiers mapped to this simulator's dense internal ids.
///
/// Dense ids are an internal storage choice, so the canonical format is not
/// asked to carry them. Assignment follows canonical row order — the order the
/// file already fixes — because tie-breaking downstream is by dense id, and any
/// other assignment order would make the two systems agree only by coincidence.
/// The mapping is kept so a run can be explained in the source's own names.
#[derive(Debug, Default)]
pub struct SourceIdentities {
    sessions: HashMap<String, u32>,
    session_order: Vec<String>,
    requests: Vec<String>,
}

impl SourceIdentities {
    fn intern_session(&mut self, source_id: &str) -> u32 {
        if let Some(dense) = self.sessions.get(source_id) {
            return *dense;
        }
        let dense = self.session_order.len() as u32;
        self.sessions.insert(source_id.to_string(), dense);
        self.session_order.push(source_id.to_string());
        dense
    }

    fn intern_request(&mut self, source_id: &str) -> RequestId {
        let dense = self.requests.len() as u32;
        self.requests.push(source_id.to_string());
        RequestId(dense)
    }

    /// Source session ids in dense-id order.
    pub fn session_source_ids(&self) -> &[String] {
        &self.session_order
    }

    /// Source request ids in dense-id order.
    pub fn request_source_ids(&self) -> &[String] {
        &self.requests
    }
}

pub(super) fn load_file<Definition: TraceDefinition>(
    path: &Path,
    declaration: &TraceDeclaration,
    session_start_times: &mut HashMap<u32, Time>,
    identities: &mut SourceIdentities,
    output: &mut Vec<ScheduledRequest<Definition>>,
) -> Result<()> {
    if declaration.kind != Definition::KIND {
        bail!(
            "typed frontend for {:?} cannot load a {:?} trace",
            Definition::KIND,
            declaration.kind
        );
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("opening trace file {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("reading header of {}", path.display()))?
        .clone();
    let index = HeaderIndex::build(&headers, path)?;
    verify_header(declaration, &index, path)?;

    for (record_index, record) in reader.records().enumerate() {
        let record = record
            .with_context(|| format!("{}: parsing row {}", path.display(), record_index + 2))?;
        let row = Row {
            record: &record,
            index: &index,
            path,
            line: record_index + 2,
        };
        if declaration.source_schema == SourceSchema::SessionExecutionV2 {
            output.push(parse_execution_v2_row::<Definition>(
                &row,
                session_start_times,
                identities,
            )?);
            continue;
        }
        let tags = parse_declared_tags(declaration, &row)?;
        let release = parse_release(&row, tags)?;
        // Global arrival-order validation runs after all files are concatenated,
        // so the first occurrence is the session's first declared arrival.
        let session_start_time = release.session.map(|session| {
            *session_start_times
                .entry(session.session_id)
                .or_insert_with(|| Time::from_ms(release.trace_arrival_time_ms))
        });
        output.push(ScheduledRequest {
            release,
            scheduling: tags.scheduling,
            definition: Definition::parse_definition(&row, tags, session_start_time)?,
        });
    }
    Ok(())
}
