//! Adapt req-frontend's validated input rows into statically typed ServingStudioSim requests.
//!
//! The shared crate owns complete input-file formats, header matching, CSV
//! decoding, tag decoding, and format-specific structural validation. This
//! module starts after that boundary: it assigns ServingStudioSim's dense ids and turns
//! each shared row into one concrete [`RequestDefinition`].

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use req_frontend::schema::format::{
    audio_to_text, image_to_text, image_to_video, omni_generation, text_generation, text_to_image,
    text_to_speech, text_to_video, video_to_text, ParsedIndependentRow,
};
use req_frontend::schema::{
    RequestPlacement, RequestPriority, RequestSession, RequestSlo, RequestSpeculative,
};

use super::arrival::{
    PlacementDeclaration, ReleaseMetadata, ScheduledRequest, SchedulingDeclaration,
    SessionReleaseMetadata,
};
use crate::common::{
    AudioExtent, AudioTextGenerationDefinition, DecodingStrategy, ImageExtent,
    ImageGenerationDefinition, ImageTextGenerationDefinition, ImageToVideoDefinition,
    OmniGenerationDefinition, RequestDefinition, RequestId, SessionInput, SloContract,
    SpeechGenerationDefinition, TextGenerationDefinition, Time, VideoExtent,
    VideoGenerationDefinition, VideoTextGenerationDefinition, WorkerId,
};

pub use req_frontend::schema::{InputFileFormat, InputFileSchema, RequestFamily, TraceTag};

/// Private adapter contract: choosing `Definition` fixes the accepted family.
pub(super) trait TraceDefinition: RequestDefinition + Sized {
    const FAMILY: RequestFamily;

    fn load_file(
        path: &Path,
        input_file_schema: &InputFileSchema,
        session_start_times: &mut HashMap<u32, Time>,
        source_identities: &mut SourceIdentities,
        output: &mut Vec<ScheduledRequest<Self>>,
    ) -> Result<()>;
}

#[derive(Debug)]
struct ParsedMetadata {
    release: ReleaseMetadata,
    slo: SloContract,
    scheduling: SchedulingDeclaration,
    placement: PlacementDeclaration,
    session_input: SessionInput,
    decoding: DecodingStrategy,
}

#[allow(clippy::too_many_arguments)]
fn parse_independent_metadata(
    source_request_id: &str,
    trace_arrival_time_ms: f64,
    session: RequestSession,
    slo: RequestSlo,
    priority: RequestPriority,
    speculative: RequestSpeculative,
    placement: RequestPlacement,
    session_start_times: &mut HashMap<u32, Time>,
    source_identities: &mut SourceIdentities,
) -> Result<ParsedMetadata> {
    let request_id = source_identities.intern_request(source_request_id)?;
    let (release_session, session_input) = match session.session_id.as_deref() {
        Some(source_session_id) if !source_session_id.is_empty() => {
            let session_id = source_identities.intern_session(source_session_id);
            let session_start_time = *session_start_times
                .entry(session_id)
                .or_insert_with(|| Time::from_ms(trace_arrival_time_ms));
            let declared_prefix_tokens = session.prefix_kv.unwrap_or(0);
            let tool_wait_after = Time::from_ms(session.tool_wait_after_ms.unwrap_or(0.0));
            (
                Some(SessionReleaseMetadata {
                    session_id,
                    tool_wait_after,
                }),
                SessionInput::Session {
                    session_id,
                    session_start_time,
                    declared_prefix_tokens,
                },
            )
        }
        _ => (None, SessionInput::Standalone),
    };
    Ok(ParsedMetadata {
        release: ReleaseMetadata {
            request_id,
            trace_arrival_time_ms,
            session: release_session,
        },
        slo: slo_contract(slo),
        scheduling: SchedulingDeclaration {
            priority: i32::try_from(priority.priority_or_default()).with_context(|| {
                format!(
                    "request {source_request_id:?}: priority does not fit ServingStudioSim's i32 contract"
                )
            })?,
        },
        placement: PlacementDeclaration {
            target_worker: placement.target_worker.map(WorkerId),
        },
        session_input,
        decoding: speculative.strategy(),
    })
}

trait IndependentSourceRow {
    fn source_request_id(&self) -> &str;
    fn trace_arrival_time_ms(&self) -> f64;
}

macro_rules! impl_independent_source_row {
    ($row:ty) => {
        impl IndependentSourceRow for $row {
            fn source_request_id(&self) -> &str {
                &self.id
            }

            fn trace_arrival_time_ms(&self) -> f64 {
                self.arrival_time
            }
        }
    };
}

impl_independent_source_row!(text_generation::independent::TextGenerationRow);
impl_independent_source_row!(image_to_text::Row);
impl_independent_source_row!(video_to_text::Row);
impl_independent_source_row!(audio_to_text::Row);
impl_independent_source_row!(text_to_image::Row);
impl_independent_source_row!(text_to_video::Row);
impl_independent_source_row!(text_to_speech::Row);
impl_independent_source_row!(image_to_video::Row);
impl_independent_source_row!(omni_generation::Row);

fn append_independent<Row, Definition>(
    rows: Vec<ParsedIndependentRow<Row>>,
    session_start_times: &mut HashMap<u32, Time>,
    source_identities: &mut SourceIdentities,
    output: &mut Vec<ScheduledRequest<Definition>>,
    mut build_definition: impl FnMut(Row, SessionInput, DecodingStrategy) -> Result<Definition>,
) -> Result<()>
where
    Row: IndependentSourceRow,
    Definition: RequestDefinition,
{
    for parsed in rows {
        let metadata = parse_independent_metadata(
            parsed.row.source_request_id(),
            parsed.row.trace_arrival_time_ms(),
            parsed.session,
            parsed.slo,
            parsed.priority,
            parsed.speculative,
            parsed.placement,
            session_start_times,
            source_identities,
        )?;
        let definition = build_definition(parsed.row, metadata.session_input, metadata.decoding)?;
        output.push(ScheduledRequest {
            release: metadata.release,
            slo: metadata.slo,
            scheduling: metadata.scheduling,
            placement: metadata.placement,
            definition,
        });
    }
    Ok(())
}

impl TraceDefinition for TextGenerationDefinition {
    const FAMILY: RequestFamily = RequestFamily::TextGeneration;

    fn load_file(
        path: &Path,
        input_file_schema: &InputFileSchema,
        session_start_times: &mut HashMap<u32, Time>,
        source_identities: &mut SourceIdentities,
        output: &mut Vec<ScheduledRequest<Self>>,
    ) -> Result<()> {
        let path = path_text(path)?;
        match input_file_schema.input_file_format {
            InputFileFormat::TextGenerationIndependent => append_independent(
                text_generation::independent::load(path, input_file_schema)?,
                session_start_times,
                source_identities,
                output,
                |row, session, decoding| {
                    Ok(Self {
                        prompt_tokens: usize_to_u32(row.input_len, "input_len")?,
                        target_output_tokens: usize_to_u32(row.output_len, "output_len")?,
                        session,
                        decoding,
                    })
                },
            ),
            InputFileFormat::TextGenerationSessionExecutionV2 => {
                let sessions = text_generation::session::load(path, input_file_schema)?;
                for (source_session_id, rounds) in sessions {
                    let session_id = source_identities.intern_session(&source_session_id);
                    let declared_session_start_time = Time::from_ms(
                        rounds
                            .first()
                            .expect("req-frontend never returns an empty session")
                            .arrival_time,
                    );
                    let session_start_time = *session_start_times
                        .entry(session_id)
                        .or_insert(declared_session_start_time);
                    for round in rounds {
                        let request_id = source_identities.intern_request(&round.request_id)?;
                        let declared_prefix_tokens = usize_to_u32(round.prefix_len, "prefix_len")?;
                        output.push(ScheduledRequest {
                            release: ReleaseMetadata {
                                request_id,
                                trace_arrival_time_ms: round.arrival_time,
                                session: Some(SessionReleaseMetadata {
                                    session_id,
                                    tool_wait_after: Time::from_ms(round.tool_wait_after_ms),
                                }),
                            },
                            slo: slo_contract(round.slo),
                            scheduling: SchedulingDeclaration {
                                priority: i32::try_from(round.priority.priority_or_default())
                                    .context("session request priority does not fit i32")?,
                            },
                            placement: PlacementDeclaration {
                                target_worker: round.placement.target_worker.map(WorkerId),
                            },
                            definition: Self {
                                prompt_tokens: usize_to_u32(round.input_len, "input_len")?,
                                target_output_tokens: usize_to_u32(round.output_len, "output_len")?,
                                session: SessionInput::Session {
                                    session_id,
                                    session_start_time,
                                    declared_prefix_tokens,
                                },
                                decoding: round.speculative.strategy(),
                            },
                        });
                    }
                }
                Ok(())
            }
            other => wrong_format(Self::FAMILY, other),
        }
    }
}

impl TraceDefinition for ImageTextGenerationDefinition {
    const FAMILY: RequestFamily = RequestFamily::ImageToText;

    fn load_file(
        path: &Path,
        input_file_schema: &InputFileSchema,
        session_start_times: &mut HashMap<u32, Time>,
        source_identities: &mut SourceIdentities,
        output: &mut Vec<ScheduledRequest<Self>>,
    ) -> Result<()> {
        ensure_format(input_file_schema, InputFileFormat::ImageToTextIndependent)?;
        append_independent(
            image_to_text::load(path_text(path)?, input_file_schema)?,
            session_start_times,
            source_identities,
            output,
            |row, session, decoding| {
                Ok(Self {
                    text_prompt_tokens: usize_to_u32(row.input_len, "input_len")?,
                    encoded_input_tokens: usize_to_u32(row.encoded_tokens, "encoded_tokens")?,
                    target_output_tokens: usize_to_u32(row.output_len, "output_len")?,
                    extent: ImageExtent {
                        width: row.media_width,
                        height: row.media_height,
                    },
                    session,
                    decoding,
                })
            },
        )
    }
}

impl TraceDefinition for VideoTextGenerationDefinition {
    const FAMILY: RequestFamily = RequestFamily::VideoToText;

    fn load_file(
        path: &Path,
        input_file_schema: &InputFileSchema,
        session_start_times: &mut HashMap<u32, Time>,
        source_identities: &mut SourceIdentities,
        output: &mut Vec<ScheduledRequest<Self>>,
    ) -> Result<()> {
        ensure_format(input_file_schema, InputFileFormat::VideoToTextIndependent)?;
        append_independent(
            video_to_text::load(path_text(path)?, input_file_schema)?,
            session_start_times,
            source_identities,
            output,
            |row, session, decoding| {
                Ok(Self {
                    text_prompt_tokens: usize_to_u32(row.input_len, "input_len")?,
                    encoded_input_tokens: usize_to_u32(row.encoded_tokens, "encoded_tokens")?,
                    target_output_tokens: usize_to_u32(row.output_len, "output_len")?,
                    extent: video_extent(
                        row.media_width,
                        row.media_height,
                        row.media_duration_s,
                        row.media_fps,
                    )?,
                    session,
                    decoding,
                })
            },
        )
    }
}

impl TraceDefinition for AudioTextGenerationDefinition {
    const FAMILY: RequestFamily = RequestFamily::AudioToText;

    fn load_file(
        path: &Path,
        input_file_schema: &InputFileSchema,
        session_start_times: &mut HashMap<u32, Time>,
        source_identities: &mut SourceIdentities,
        output: &mut Vec<ScheduledRequest<Self>>,
    ) -> Result<()> {
        ensure_format(input_file_schema, InputFileFormat::AudioToTextIndependent)?;
        append_independent(
            audio_to_text::load(path_text(path)?, input_file_schema)?,
            session_start_times,
            source_identities,
            output,
            |row, session, decoding| {
                Ok(Self {
                    text_prompt_tokens: usize_to_u32(row.input_len, "input_len")?,
                    encoded_input_tokens: usize_to_u32(row.encoded_tokens, "encoded_tokens")?,
                    target_output_tokens: usize_to_u32(row.output_len, "output_len")?,
                    extent: audio_extent(row.media_duration_s, row.media_sample_rate_hz)?,
                    session,
                    decoding,
                })
            },
        )
    }
}

macro_rules! impl_generated_definition {
    ($definition:ty, $family:expr, $format:expr, $module:ident, $extent:expr) => {
        impl TraceDefinition for $definition {
            const FAMILY: RequestFamily = $family;

            fn load_file(
                path: &Path,
                input_file_schema: &InputFileSchema,
                session_start_times: &mut HashMap<u32, Time>,
                source_identities: &mut SourceIdentities,
                output: &mut Vec<ScheduledRequest<Self>>,
            ) -> Result<()> {
                ensure_format(input_file_schema, $format)?;
                append_independent(
                    $module::load(path_text(path)?, input_file_schema)?,
                    session_start_times,
                    source_identities,
                    output,
                    |row, session, decoding| {
                        reject_generated_only_tags(session, decoding)?;
                        Ok(Self {
                            text_prompt_tokens: usize_to_u32(row.input_len, "input_len")?,
                            target_generation_steps: usize_to_u32(
                                row.denoise_steps,
                                "denoise_steps",
                            )?,
                            extent: $extent(row)?,
                        })
                    },
                )
            }
        }
    };
}

impl_generated_definition!(
    ImageGenerationDefinition,
    RequestFamily::TextToImage,
    InputFileFormat::TextToImageIndependent,
    text_to_image,
    |row: text_to_image::Row| -> Result<ImageExtent> {
        Ok(ImageExtent {
            width: row.media_width,
            height: row.media_height,
        })
    }
);
impl_generated_definition!(
    VideoGenerationDefinition,
    RequestFamily::TextToVideo,
    InputFileFormat::TextToVideoIndependent,
    text_to_video,
    |row: text_to_video::Row| -> Result<VideoExtent> {
        video_extent(
            row.media_width,
            row.media_height,
            row.media_duration_s,
            row.media_fps,
        )
    }
);
impl_generated_definition!(
    SpeechGenerationDefinition,
    RequestFamily::TextToSpeech,
    InputFileFormat::TextToSpeechIndependent,
    text_to_speech,
    |row: text_to_speech::Row| -> Result<AudioExtent> {
        audio_extent(row.media_duration_s, row.media_sample_rate_hz)
    }
);

impl TraceDefinition for ImageToVideoDefinition {
    const FAMILY: RequestFamily = RequestFamily::ImageToVideo;

    fn load_file(
        path: &Path,
        input_file_schema: &InputFileSchema,
        session_start_times: &mut HashMap<u32, Time>,
        source_identities: &mut SourceIdentities,
        output: &mut Vec<ScheduledRequest<Self>>,
    ) -> Result<()> {
        ensure_format(input_file_schema, InputFileFormat::ImageToVideoIndependent)?;
        append_independent(
            image_to_video::load(path_text(path)?, input_file_schema)?,
            session_start_times,
            source_identities,
            output,
            |row, session, decoding| {
                reject_generated_only_tags(session, decoding)?;
                Ok(Self {
                    text_prompt_tokens: usize_to_u32(row.input_len, "input_len")?,
                    encoded_input_tokens: usize_to_u32(row.encoded_tokens, "encoded_tokens")?,
                    input_extent: ImageExtent {
                        width: row.input_media_width,
                        height: row.input_media_height,
                    },
                    output_extent: video_extent(
                        row.media_width,
                        row.media_height,
                        row.media_duration_s,
                        row.media_fps,
                    )?,
                    target_generation_steps: usize_to_u32(row.denoise_steps, "denoise_steps")?,
                })
            },
        )
    }
}

impl TraceDefinition for OmniGenerationDefinition {
    const FAMILY: RequestFamily = RequestFamily::OmniGeneration;

    fn load_file(
        path: &Path,
        input_file_schema: &InputFileSchema,
        session_start_times: &mut HashMap<u32, Time>,
        source_identities: &mut SourceIdentities,
        output: &mut Vec<ScheduledRequest<Self>>,
    ) -> Result<()> {
        ensure_format(
            input_file_schema,
            InputFileFormat::OmniGenerationIndependent,
        )?;
        append_independent(
            omni_generation::load(path_text(path)?, input_file_schema)?,
            session_start_times,
            source_identities,
            output,
            |row, session, decoding| {
                Ok(Self {
                    input: row.input_segments,
                    output: row.output_segments,
                    session,
                    decoding,
                })
            },
        )
    }
}

fn wrong_format<Definition>(
    family: RequestFamily,
    input_file_format: InputFileFormat,
) -> Result<Definition> {
    bail!(
        "typed frontend for {family:?} cannot load input file format {:?}",
        input_file_format.name()
    )
}

fn ensure_format(input_file_schema: &InputFileSchema, expected: InputFileFormat) -> Result<()> {
    if input_file_schema.input_file_format != expected {
        return wrong_format(
            expected.request_family(),
            input_file_schema.input_file_format,
        );
    }
    Ok(())
}

fn path_text(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("trace path is not valid UTF-8: {}", path.display()))
}

fn usize_to_u32(value: usize, column: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{column}={value} exceeds u32"))
}

fn slo_contract(slo: RequestSlo) -> SloContract {
    SloContract {
        ttft_slo: slo.ttft_slo_ms.map(Time::from_ms),
        tpot_slo: slo.tpot_slo_ms.map(Time::from_ms),
        e2e_slo: slo.e2e_slo_ms.map(Time::from_ms),
    }
}

fn reject_generated_only_tags(session: SessionInput, decoding: DecodingStrategy) -> Result<()> {
    if session.declared_prefix_tokens() != 0 {
        bail!("prefix_kv is not meaningful for a generated-media request");
    }
    if decoding != DecodingStrategy::Standard {
        bail!("speculative decoding is only valid for autoregressive output");
    }
    Ok(())
}

fn video_extent(width: u32, height: u32, duration_seconds: f64, fps: f64) -> Result<VideoExtent> {
    let frames = resolve_count(duration_seconds, fps, "media_duration_s", "media_fps")?;
    Ok(VideoExtent {
        width,
        height,
        frames: u32::try_from(frames).context("video frame count exceeds u32")?,
    })
}

fn audio_extent(duration_seconds: f64, sample_rate_hz: u32) -> Result<AudioExtent> {
    Ok(AudioExtent {
        samples: resolve_count(
            duration_seconds,
            f64::from(sample_rate_hz),
            "media_duration_s",
            "media_sample_rate_hz",
        )?,
    })
}

fn resolve_count(duration: f64, rate: f64, duration_name: &str, rate_name: &str) -> Result<u64> {
    let count = (duration * rate).round();
    if !count.is_finite() || count < 1.0 || count > u64::MAX as f64 {
        bail!(
            "{duration_name}={duration} * {rate_name}={rate} resolves to {count} units \
             (must be in 1..=u64::MAX)"
        );
    }
    Ok(count as u64)
}

/// Opaque source identifiers mapped to this simulator's dense internal ids.
///
/// Assignment follows validated row order. The mapping is retained so reports
/// can use source identifiers while `RequestStore` keeps dense O(1) indexing.
#[derive(Debug, Default)]
pub struct SourceIdentities {
    sessions: HashMap<String, u32>,
    session_order: Vec<String>,
    request_ids: HashMap<String, RequestId>,
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

    fn intern_request(&mut self, source_id: &str) -> Result<RequestId> {
        if self.request_ids.contains_key(source_id) {
            bail!("duplicate request id {source_id:?} across input files");
        }
        let dense = RequestId(self.requests.len() as u32);
        self.request_ids.insert(source_id.to_string(), dense);
        self.requests.push(source_id.to_string());
        Ok(dense)
    }

    pub fn session_source_ids(&self) -> &[String] {
        &self.session_order
    }

    pub fn request_source_ids(&self) -> &[String] {
        &self.requests
    }
}

pub(super) fn load_file<Definition: TraceDefinition>(
    path: &Path,
    input_file_schema: &InputFileSchema,
    session_start_times: &mut HashMap<u32, Time>,
    source_identities: &mut SourceIdentities,
    output: &mut Vec<ScheduledRequest<Definition>>,
) -> Result<()> {
    if input_file_schema.request_family() != Definition::FAMILY {
        bail!(
            "typed frontend for {:?} cannot load request family {:?}",
            Definition::FAMILY,
            input_file_schema.request_family()
        );
    }
    Definition::load_file(
        path,
        input_file_schema,
        session_start_times,
        source_identities,
        output,
    )
}
