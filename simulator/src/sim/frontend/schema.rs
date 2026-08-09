//! Declared, exact trace schemas that produce statically typed request rows.
//!
//! A file declares one [`TraceKind`]. The declaration selects one concrete
//! `TraceDefinition` implementation before any row is parsed; headers never
//! infer the request family and rows never carry a request-kind union.

use std::collections::HashMap;
use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;

use anyhow::{bail, Context, Result};
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

const RELEASE_COLUMNS: &[&str] = &["id", "arrival_time"];
const AUTOREGRESSIVE_COLUMNS: &[&str] = &["input_len", "output_len"];
const GENERATED_MEDIA_COLUMNS: &[&str] = &["input_len", "denoise_steps"];
const ENCODED_INPUT_COLUMNS: &[&str] = &["encoded_tokens"];
const IMAGE_COLUMNS: &[&str] = &["media_width", "media_height"];
const VIDEO_COLUMNS: &[&str] = &[
    "media_width",
    "media_height",
    "media_duration_s",
    "media_fps",
];
const AUDIO_COLUMNS: &[&str] = &["media_duration_s", "media_sample_rate_hz"];
const INPUT_IMAGE_COLUMNS: &[&str] = &["input_media_width", "input_media_height"];
const OMNI_COLUMNS: &[&str] = &["input_segments", "output_segments"];
const FOREIGN_MULTI_ROUND: &str = "round_idx";
const DEFAULT_PRIORITY: i32 = 0;

/// One statically selected request schema per trace file family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceKind {
    TextGeneration,
    ImageToText,
    VideoToText,
    AudioToText,
    TextToImage,
    TextToVideo,
    TextToSpeech,
    ImageToVideo,
    OmniGeneration,
}

impl TraceKind {
    pub const CHOICES: &'static [&'static str] = &[
        "text_generation",
        "image_to_text",
        "video_to_text",
        "audio_to_text",
        "text_to_image",
        "text_to_video",
        "text_to_speech",
        "image_to_video",
        "omni_generation",
    ];

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "text_generation" => Self::TextGeneration,
            "image_to_text" => Self::ImageToText,
            "video_to_text" => Self::VideoToText,
            "audio_to_text" => Self::AudioToText,
            "text_to_image" => Self::TextToImage,
            "text_to_video" => Self::TextToVideo,
            "text_to_speech" => Self::TextToSpeech,
            "image_to_video" => Self::ImageToVideo,
            "omni_generation" => Self::OmniGeneration,
            other => bail!(
                "unknown trace_kind {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
    }

    fn definition_columns(self) -> Vec<&'static str> {
        let mut columns = match self {
            Self::TextGeneration | Self::ImageToText | Self::VideoToText | Self::AudioToText => {
                AUTOREGRESSIVE_COLUMNS.to_vec()
            }
            Self::TextToImage | Self::TextToVideo | Self::TextToSpeech => {
                GENERATED_MEDIA_COLUMNS.to_vec()
            }
            Self::ImageToVideo => {
                let mut columns = GENERATED_MEDIA_COLUMNS.to_vec();
                columns.extend_from_slice(ENCODED_INPUT_COLUMNS);
                columns.extend_from_slice(INPUT_IMAGE_COLUMNS);
                columns
            }
            Self::OmniGeneration => OMNI_COLUMNS.to_vec(),
        };
        match self {
            Self::TextGeneration | Self::OmniGeneration => {}
            Self::ImageToText => {
                columns.extend_from_slice(ENCODED_INPUT_COLUMNS);
                columns.extend_from_slice(IMAGE_COLUMNS);
            }
            Self::VideoToText => {
                columns.extend_from_slice(ENCODED_INPUT_COLUMNS);
                columns.extend_from_slice(VIDEO_COLUMNS);
            }
            Self::AudioToText => {
                columns.extend_from_slice(ENCODED_INPUT_COLUMNS);
                columns.extend_from_slice(AUDIO_COLUMNS);
            }
            Self::TextToImage => columns.extend_from_slice(IMAGE_COLUMNS),
            Self::TextToVideo | Self::ImageToVideo => columns.extend_from_slice(VIDEO_COLUMNS),
            Self::TextToSpeech => columns.extend_from_slice(AUDIO_COLUMNS),
        }
        columns
    }
}

/// Orthogonal declarations whose values are routed into their actual owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceTag {
    Session,
    Slo,
    Speculative,
}

impl TraceTag {
    pub const CHOICES: &'static [&'static str] = &["session", "slo", "speculative"];

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "session" => Self::Session,
            "slo" => Self::Slo,
            "speculative" => Self::Speculative,
            other => bail!(
                "unknown trace_tag {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
    }

    fn columns(self) -> &'static [&'static str] {
        match self {
            Self::Session => &["session_id", "prefix_kv", "tool_wait_after_ms"],
            Self::Slo => &["deadline_ms", "priority"],
            Self::Speculative => &["accept_rate"],
        }
    }
}

#[derive(Clone, Debug)]
pub struct TraceDeclaration {
    pub kind: TraceKind,
    pub tags: Vec<TraceTag>,
}

impl TraceDeclaration {
    pub fn parse(kind: &str, tags: &[String]) -> Result<Self> {
        let kind = TraceKind::parse(kind)?;
        let mut parsed_tags = Vec::with_capacity(tags.len());
        for tag_name in tags {
            let tag = TraceTag::parse(tag_name)?;
            if parsed_tags.contains(&tag) {
                bail!("trace_tags lists {tag:?} more than once");
            }
            parsed_tags.push(tag);
        }
        Ok(Self {
            kind,
            tags: parsed_tags,
        })
    }

    pub fn text() -> Self {
        Self {
            kind: TraceKind::TextGeneration,
            tags: Vec::new(),
        }
    }

    fn carries(&self, tag: TraceTag) -> bool {
        self.tags.contains(&tag)
    }

    fn expected_columns(&self) -> Vec<&'static str> {
        let mut columns = RELEASE_COLUMNS.to_vec();
        columns.extend(self.kind.definition_columns());
        for tag in &self.tags {
            columns.extend_from_slice(tag.columns());
        }
        columns
    }

    fn verify_header(&self, index: &HeaderIndex, path: &Path) -> Result<()> {
        if index.has(FOREIGN_MULTI_ROUND) {
            bail!(
                "{}: multi-round traces (round_idx column) are not supported yet; \
                 declare the session trace tag and group rows with session_id",
                path.display()
            );
        }
        let expected = self.expected_columns();
        let missing: Vec<&str> = expected
            .iter()
            .copied()
            .filter(|column| !index.has(column))
            .collect();
        let unexpected: Vec<&str> = index
            .names()
            .filter(|column| !expected.contains(column))
            .collect();
        if missing.is_empty() && unexpected.is_empty() {
            return Ok(());
        }
        bail!(
            "{}: header does not match the declared trace (kind={:?}, tags={:?})\n  \
             missing: {missing:?}\n  unexpected: {unexpected:?}\n  \
             expected exactly: {expected:?}",
            path.display(),
            self.kind,
            self.tags,
        )
    }

    fn parse_tags(&self, row: &Row<'_>) -> Result<ParsedTags> {
        let session = if self.carries(TraceTag::Session) {
            parse_session(row)?
        } else {
            None
        };
        let scheduling = if self.carries(TraceTag::Slo) {
            parse_scheduling(row)?
        } else {
            SchedulingDeclaration::default()
        };
        let decoding = if self.carries(TraceTag::Speculative) {
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

    fn has(&self, column: &str) -> bool {
        self.0.contains_key(column)
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

pub(super) fn load_file<Definition: TraceDefinition>(
    path: &Path,
    declaration: &TraceDeclaration,
    session_start_times: &mut HashMap<u32, Time>,
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
    declaration.verify_header(&index, path)?;

    for (record_index, record) in reader.records().enumerate() {
        let record = record
            .with_context(|| format!("{}: parsing row {}", path.display(), record_index + 2))?;
        let row = Row {
            record: &record,
            index: &index,
            path,
            line: record_index + 2,
        };
        let tags = declaration.parse_tags(&row)?;
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
