//! L7-γ typed trace frontend.
//!
//! A [`TraceFrontend<Definition>`] owns exactly one request family. Runtime
//! dispatch exists only in [`LoadedTrace`], the startup boundary that converts
//! a config string into a concrete frontend. The production tick loop accepts
//! only `TraceFrontend<TextGenerationDefinition>`, so unsupported request
//! families cannot leak into the current text-only L5 workers.

mod arrival;
mod release;
mod schema;

use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::common::{
    AudioTextGenerationDefinition, ImageGenerationDefinition, ImageTextGenerationDefinition,
    ImageToVideoDefinition, OmniGenerationDefinition, Request, RequestDefinition, RequestId,
    SpeechGenerationDefinition, TextGenerationDefinition, Time, VideoGenerationDefinition,
    VideoTextGenerationDefinition,
};
use release::ReplayScheduler;
use schema::TraceDefinition;

pub use arrival::{
    ReleaseMetadata, ScheduledRequest, SchedulingDeclaration, SessionReleaseMetadata,
};
pub use release::{ArrivalMode, ArrivalSchedule, CapacityLimit, SessionDependency};
pub use schema::{InputFileFormat, InputFileSchema, RequestFamily, SourceIdentities, TraceTag};

/// Typed immutable requests plus a definition-blind replay scheduler.
#[derive(Debug)]
pub struct TraceFrontend<Definition: RequestDefinition = TextGenerationDefinition> {
    scheduled_requests: Vec<ScheduledRequest<Definition>>,
    releases: Vec<ReleaseMetadata>,
    emitted: usize,
    completed: u64,
    replay_scheduler: ReplayScheduler,
    /// Source-to-dense identifier mapping, kept so a run can be reported in the
    /// trace's own names rather than in this simulator's storage ordinals.
    source_identities: SourceIdentities,
}

impl<Definition: RequestDefinition> TraceFrontend<Definition> {
    pub fn source_identities(&self) -> &SourceIdentities {
        &self.source_identities
    }
}

/// One row of the normalized plan, in the source's own identifiers.
///
/// This is the artifact a differential test compares against TraceLab's export.
/// It is deliberately stated in source ids and resolved causal links rather than
/// in dense storage ordinals: dense ids are this simulator's private choice,
/// while the plan is the thing both systems must agree on.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PlanRow {
    pub request_id: String,
    pub session_id: String,
    pub round_idx: usize,
    pub session_arrival_time_ms: String,
    pub predecessor_request_id: Option<String>,
    pub prefix_len: u32,
    pub input_len: u32,
    pub output_len: u32,
    pub tool_wait_after_ms: String,
}

impl TraceFrontend<TextGenerationDefinition> {
    /// Project the loaded trace into its normalized plan, preserving row order.
    ///
    /// Round indices and predecessors are recovered from session membership in
    /// row order rather than read back from a column, so the plan reflects the
    /// chain this simulator will actually execute.
    pub fn plan_rows(&self) -> Vec<PlanRow> {
        let mut rounds_seen: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        let mut previous_request_by_session: std::collections::HashMap<u32, String> =
            std::collections::HashMap::new();
        let mut rows = Vec::with_capacity(self.scheduled_requests.len());
        for request in &self.scheduled_requests {
            let release = &request.release;
            let session_id = release
                .session
                .map(|session| session.session_id)
                .unwrap_or(u32::MAX);
            let round_idx = *rounds_seen
                .entry(session_id)
                .and_modify(|count| *count += 1)
                .or_insert(0);
            let source_session_id = self
                .source_identities
                .session_source_ids()
                .get(session_id as usize)
                .cloned()
                .unwrap_or_else(|| session_id.to_string());
            let source_request_id = self
                .source_identities
                .request_source_ids()
                .get(release.request_id.0 as usize)
                .cloned()
                .unwrap_or_else(|| release.request_id.0.to_string());
            let predecessor_request_id =
                previous_request_by_session.insert(session_id, source_request_id.clone());
            rows.push(PlanRow {
                request_id: source_request_id,
                session_id: source_session_id,
                round_idx,
                session_arrival_time_ms: format!("{:.6}", release.trace_arrival_time_ms),
                predecessor_request_id,
                prefix_len: request.definition.session.declared_prefix_tokens(),
                input_len: request.definition.prompt_tokens,
                output_len: request.definition.target_output_tokens,
                tool_wait_after_ms: format!(
                    "{:.6}",
                    release
                        .session
                        .map(|session| session.tool_wait_after.as_ms())
                        .unwrap_or(0.0)
                ),
            });
        }
        rows
    }
}

impl TraceFrontend<TextGenerationDefinition> {
    /// Public typed loader for the production text family.
    pub fn load(
        files: &[PathBuf],
        input_file_schema: &InputFileSchema,
        arrival: ArrivalSchedule,
        capacity: CapacityLimit,
        session_dependency: SessionDependency,
    ) -> Result<Self> {
        load_typed(
            files,
            input_file_schema,
            arrival,
            capacity,
            session_dependency,
        )
    }
}

impl<Definition: RequestDefinition> TraceFrontend<Definition> {
    pub fn expected_count(&self) -> usize {
        self.scheduled_requests.len()
    }

    pub fn exhausted(&self) -> bool {
        self.emitted >= self.scheduled_requests.len()
    }

    pub fn drain_due(&mut self, now: Time, mut emit: impl FnMut(Request<Definition>)) {
        loop {
            let in_flight = self.in_flight();
            let Some((index, arrival_time)) =
                self.replay_scheduler
                    .next_ready(&self.releases, in_flight, now)
            else {
                break;
            };
            self.emitted += 1;
            emit(self.scheduled_requests[index].realize_at(arrival_time));
        }
    }

    pub fn record_completion(&mut self, request: RequestId, now: Time) {
        self.completed += 1;
        self.replay_scheduler
            .on_completion(&self.releases, request, now);
    }

    pub fn in_flight(&self) -> u64 {
        self.emitted as u64 - self.completed
    }

    pub fn submitted(&self) -> u64 {
        self.emitted as u64
    }

    pub fn num_completed(&self) -> u64 {
        self.completed
    }
}

/// Startup-only dispatch. Each variant already contains a homogeneous,
/// statically typed trace; this enum never crosses into Flow or workers.
#[derive(Debug)]
pub enum LoadedTrace {
    TextGeneration(TraceFrontend<TextGenerationDefinition>),
    ImageToText(TraceFrontend<ImageTextGenerationDefinition>),
    VideoToText(TraceFrontend<VideoTextGenerationDefinition>),
    AudioToText(TraceFrontend<AudioTextGenerationDefinition>),
    TextToImage(TraceFrontend<ImageGenerationDefinition>),
    TextToVideo(TraceFrontend<VideoGenerationDefinition>),
    TextToSpeech(TraceFrontend<SpeechGenerationDefinition>),
    ImageToVideo(TraceFrontend<ImageToVideoDefinition>),
    OmniGeneration(TraceFrontend<OmniGenerationDefinition>),
}

impl LoadedTrace {
    pub fn load(
        files: &[PathBuf],
        input_file_schema: &InputFileSchema,
        arrival: ArrivalSchedule,
        capacity: CapacityLimit,
        session_dependency: SessionDependency,
    ) -> Result<Self> {
        Ok(match input_file_schema.request_family() {
            RequestFamily::TextGeneration => Self::TextGeneration(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
            RequestFamily::ImageToText => Self::ImageToText(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
            RequestFamily::VideoToText => Self::VideoToText(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
            RequestFamily::AudioToText => Self::AudioToText(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
            RequestFamily::TextToImage => Self::TextToImage(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
            RequestFamily::TextToVideo => Self::TextToVideo(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
            RequestFamily::TextToSpeech => Self::TextToSpeech(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
            RequestFamily::ImageToVideo => Self::ImageToVideo(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
            RequestFamily::OmniGeneration => Self::OmniGeneration(load_typed(
                files,
                input_file_schema,
                arrival,
                capacity,
                session_dependency,
            )?),
        })
    }

    /// Narrow startup dispatch to the only request family current L5 workers
    /// implement. The returned type is the proof consumed by `run_sim`.
    pub fn into_current_text_frontend(self) -> Result<TraceFrontend> {
        Ok(match self {
            Self::TextGeneration(frontend) => frontend,
            Self::ImageToText(_) => return unsupported_family("image_to_text"),
            Self::VideoToText(_) => return unsupported_family("video_to_text"),
            Self::AudioToText(_) => return unsupported_family("audio_to_text"),
            Self::TextToImage(_) => return unsupported_family("text_to_image"),
            Self::TextToVideo(_) => return unsupported_family("text_to_video"),
            Self::TextToSpeech(_) => return unsupported_family("text_to_speech"),
            Self::ImageToVideo(_) => return unsupported_family("image_to_video"),
            Self::OmniGeneration(_) => return unsupported_family("omni_generation"),
        })
    }
}

fn unsupported_family<Definition>(kind: &str) -> Result<Definition> {
    bail!(
        "input file format for {kind:?} parsed as its own request family, but current \
         deployments are text_generation-only; add the matching encoder/media \
         worker family before executing this trace"
    )
}

fn load_typed<Definition: TraceDefinition>(
    files: &[PathBuf],
    input_file_schema: &InputFileSchema,
    arrival: ArrivalSchedule,
    capacity: CapacityLimit,
    session_dependency: SessionDependency,
) -> Result<TraceFrontend<Definition>> {
    if files.is_empty() {
        bail!("no trace files given (--trace-files)");
    }
    validate_replay(session_dependency, input_file_schema)?;
    let mut scheduled_requests = Vec::new();
    let mut session_start_times = std::collections::HashMap::new();
    let mut source_identities = schema::SourceIdentities::default();
    for file in files {
        schema::load_file(
            file,
            input_file_schema,
            &mut session_start_times,
            &mut source_identities,
            &mut scheduled_requests,
        )?;
    }
    if scheduled_requests.is_empty() {
        bail!("trace files contained no rows");
    }
    validate_arrival_order(&scheduled_requests)?;
    let releases = scheduled_requests
        .iter()
        .map(|request| request.release)
        .collect::<Vec<_>>();
    let replay_scheduler = ReplayScheduler::new(arrival, capacity, session_dependency, &releases);
    Ok(TraceFrontend {
        scheduled_requests,
        releases,
        emitted: 0,
        completed: 0,
        replay_scheduler,
        source_identities,
    })
}

/// Reject a release configuration that cannot run, before any file is read.
///
/// Takes the declaration too, because one axis's precondition is about the data
/// rather than its own payload: chaining rounds is meaningless on a trace that
/// declares no sessions. Checking it here rather than at the config call site
/// means every caller is covered, tests included.
fn validate_replay(
    session_dependency: SessionDependency,
    input_file_schema: &InputFileSchema,
) -> Result<()> {
    if session_dependency == SessionDependency::Chained
        && !input_file_schema.carries(TraceTag::Session)
        && !input_file_schema.input_file_format.has_session_topology()
    {
        bail!(
            "session_dependency: chained needs the `session` trace tag — without \
             session columns there are no rounds to chain"
        );
    }
    Ok(())
}

/// Arrival times must be non-decreasing across rows.
///
/// This is a property of the concatenated *list*, not of any one schema, so it
/// lives here rather than in [`schema`]: ids are the `RequestStore` index and
/// the order is what open-loop replay walks, whatever kind the rows are.
fn validate_arrival_order<Definition: RequestDefinition>(
    scheduled_requests: &[ScheduledRequest<Definition>],
) -> Result<()> {
    for (index, scheduled_request) in scheduled_requests.iter().enumerate() {
        if index > 0
            && scheduled_request.release.trace_arrival_time_ms
                < scheduled_requests[index - 1].release.trace_arrival_time_ms
        {
            bail!(
                "trace row {index} arrival_time={} < previous {}",
                scheduled_request.release.trace_arrival_time_ms,
                scheduled_requests[index - 1].release.trace_arrival_time_ms
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        AcceptanceProfile, AudioExtent, DecodingStrategy, ImageExtent, OmniInputSegment,
        OmniOutputSpec, RequestId, SessionInput, SloContract, VideoExtent,
    };
    use std::io::Write;
    use std::path::Path;

    fn write_csv(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    fn write_omni_csv(
        directory: &Path,
        name: &str,
        input: &[OmniInputSegment],
        output: &[OmniOutputSpec],
    ) -> PathBuf {
        let path = directory.join(name);
        let mut writer = csv::Writer::from_path(&path).unwrap();
        writer
            .write_record(["id", "arrival_time", "input_segments", "output_segments"])
            .unwrap();
        let input_json = serde_json::to_string(input).unwrap();
        let output_json = serde_json::to_string(output).unwrap();
        writer
            .write_record(["0", "0.0", input_json.as_str(), output_json.as_str()])
            .unwrap();
        writer.flush().unwrap();
        path
    }

    /// Write the canonical TraceLab execution trace used by the v2 tests.
    ///
    /// Session `b` arrives first and has two rounds; session `a` arrives later.
    /// The identifiers are opaque strings whose lexicographic order disagrees
    /// with their arrival order, so a loader that sorted by id would be caught.
    fn write_execution_v2(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("session-execution-v2.csv");
        std::fs::write(
            &path,
            "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             session_b_round_000000,b,0,0.000000,0,512,64,100.000000\n\
             session_b_round_000001,b,1,0.000000,576,0,64,0.000000\n\
             session_a_round_000000,a,0,250.000000,0,400,48,0.000000\n",
        )
        .unwrap();
        path
    }

    fn declare_execution_v2() -> InputFileSchema {
        InputFileSchema::new(
            InputFileFormat::TextGenerationSessionExecutionV2,
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn execution_v2_maps_opaque_ids_to_dense_ids_in_row_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_execution_v2(dir.path());

        let frontend = TraceFrontend::load(
            &[path],
            &declare_execution_v2(),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap();

        // Row order, not identifier order: `b` arrives first and gets dense 0.
        assert_eq!(
            frontend.source_identities().session_source_ids(),
            &["b".to_string(), "a".to_string()]
        );
        assert_eq!(
            frontend.source_identities().request_source_ids(),
            &[
                "session_b_round_000000".to_string(),
                "session_b_round_000001".to_string(),
                "session_a_round_000000".to_string(),
            ]
        );
    }

    #[test]
    fn execution_v2_feeds_prefix_and_fresh_input_straight_into_the_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_execution_v2(dir.path());

        let frontend = TraceFrontend::load(
            &[path],
            &declare_execution_v2(),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap();
        let plan = frontend.plan_rows();

        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].prefix_len, 0);
        assert_eq!(plan[0].input_len, 512);
        assert_eq!(plan[0].tool_wait_after_ms, "100.000000");
        // A round that appends nothing still re-sends its whole conversation.
        assert_eq!(plan[1].prefix_len, 576);
        assert_eq!(plan[1].input_len, 0);
        assert_eq!(
            plan[1].predecessor_request_id.as_deref(),
            Some("session_b_round_000000")
        );
        assert_eq!(plan[2].session_id, "a");
        assert_eq!(plan[2].session_arrival_time_ms, "250.000000");
        assert_eq!(plan[2].predecessor_request_id, None);
    }

    #[test]
    fn execution_v2_rejects_a_prefix_on_a_first_round() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.csv");
        std::fs::write(
            &path,
            "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms\n\
             session_a_round_000000,a,0,0.000000,128,512,64,0.000000\n",
        )
        .unwrap();

        let error = TraceFrontend::load(
            &[path],
            &declare_execution_v2(),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("no previous context"), "{error}");
    }

    /// A canonical trace may carry the orthogonal tags, and they must land.
    ///
    /// The canonical format spells its own session columns, which is why it
    /// implies that tag — but a tag it knows nothing about is the file's own
    /// declaration, and a v2 path that read the column and dropped it would let
    /// a trace set service bounds that the replay client it is compared against
    /// honoured and the simulator did not.
    #[test]
    fn execution_v2_carries_the_orthogonal_tags_it_declares() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tagged.csv");
        std::fs::write(
            &path,
            "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms,ttft_slo_ms,tpot_slo_ms,e2e_slo_ms,priority\n\
             session_a_round_000000,a,0,0.000000,0,512,64,0.000000,500,20,2000,3\n\
             session_a_round_000001,a,1,0.000000,576,0,64,0.000000,,,,0\n",
        )
        .unwrap();
        let declaration = InputFileSchema::new(
            InputFileFormat::TextGenerationSessionExecutionV2,
            vec![TraceTag::Slo, TraceTag::Priority],
        )
        .unwrap();

        let frontend = TraceFrontend::load(
            std::slice::from_ref(&path),
            &declaration,
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap();

        let scheduled = &frontend.scheduled_requests;
        assert_eq!(scheduled[0].slo.ttft_slo, Some(Time::from_ms(500.0)));
        assert_eq!(scheduled[0].slo.tpot_slo, Some(Time::from_ms(20.0)));
        assert_eq!(scheduled[0].slo.e2e_slo, Some(Time::from_ms(2000.0)));
        assert_eq!(scheduled[0].scheduling.priority, 3);
        // Blank cells declare no metric bounds; they are not zero-valued SLOs.
        assert_eq!(scheduled[1].slo, SloContract::default());

        // And the same file without the declaration is refused rather than
        // parsed with two columns nobody reads.
        let error = TraceFrontend::load(
            &[path],
            &declare_execution_v2(),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("ttft_slo_ms"), "{error}");
    }

    #[test]
    fn native_traces_still_reject_a_round_index_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("native.csv");
        std::fs::write(
            &path,
            "id,input_len,output_len,arrival_time,round_idx\n0,4,4,0.0,0\n",
        )
        .unwrap();

        let error = load_text(&[path], 1.0).unwrap_err().to_string();

        assert!(error.contains("round_idx"), "{error}");
    }

    fn declare(kind: &str, tags: &[&str]) -> InputFileSchema {
        let input_file_format = match kind {
            "text_generation" => InputFileFormat::TextGenerationIndependent,
            "image_to_text" => InputFileFormat::ImageToTextIndependent,
            "video_to_text" => InputFileFormat::VideoToTextIndependent,
            "audio_to_text" => InputFileFormat::AudioToTextIndependent,
            "text_to_image" => InputFileFormat::TextToImageIndependent,
            "text_to_video" => InputFileFormat::TextToVideoIndependent,
            "text_to_speech" => InputFileFormat::TextToSpeechIndependent,
            "image_to_video" => InputFileFormat::ImageToVideoIndependent,
            "omni_generation" => InputFileFormat::OmniGenerationIndependent,
            other => panic!("unsupported test family {other}"),
        };
        let tags = tags
            .iter()
            .map(|tag| TraceTag::parse(tag).unwrap())
            .collect();
        InputFileSchema::new(input_file_format, tags).unwrap()
    }

    fn open_loop(rate: f64) -> ArrivalSchedule {
        ArrivalSchedule::trace_timed(rate).unwrap()
    }

    /// No cap — the shape most tests want, where only arrival paces the run.
    fn uncapped() -> CapacityLimit {
        CapacityLimit::unlimited()
    }

    fn capped(max_active_units: usize) -> CapacityLimit {
        CapacityLimit::parse(Some(max_active_units)).unwrap()
    }

    /// Load a plain text trace open-loop — the shape most tests want.
    fn load_text(paths: &[PathBuf], rate: f64) -> Result<TraceFrontend> {
        let arrival = ArrivalSchedule::trace_timed(rate)?;
        TraceFrontend::load(
            paths,
            &InputFileSchema::text_generation_independent(),
            arrival,
            uncapped(),
            SessionDependency::Independent,
        )
    }

    fn drain_at(fe: &mut TraceFrontend, now_ms: f64) -> Vec<RequestId> {
        let mut ids = Vec::new();
        fe.drain_due(Time::from_ms(now_ms), |request| ids.push(request.core.id));
        ids
    }

    /// Drain returning `(id, arrival_ms)` so closed-loop tests can assert the
    /// admission-clock stamping, not just ordering.
    fn drain_pairs(fe: &mut TraceFrontend, now_ms: f64) -> Vec<(RequestId, f64)> {
        let mut out = Vec::new();
        fe.drain_due(Time::from_ms(now_ms), |request| {
            out.push((request.core.id, request.core.arrival_time.as_ms()))
        });
        out
    }

    fn drain_session_facts(
        frontend: &mut TraceFrontend,
        now_ms: f64,
    ) -> Vec<(RequestId, f64, Option<f64>)> {
        let mut released = Vec::new();
        frontend.drain_due(Time::from_ms(now_ms), |request| {
            released.push((
                request.core.id,
                request.core.arrival_time.as_ms(),
                request
                    .definition
                    .session
                    .session_start_time()
                    .map(Time::as_ms),
            ));
        });
        released
    }

    // ---- open / closed loop replay -----------------------------------------

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
        let mut fe = load_text(&[path], 1.0).unwrap();
        assert_eq!(fe.expected_count(), 3);

        // At t=0 only req 0 is due; a second drain at t=0 yields nothing.
        assert_eq!(
            drain_session_facts(&mut fe, 0.0),
            vec![(RequestId(0), 0.0, None)]
        );
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
        let mut fe = load_text(&[path], 2.0).unwrap();
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
        assert!(load_text(&[path], 0.0).is_err());
    }

    #[test]
    fn maps_opaque_source_ids_to_dense_request_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "bad.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             5,8,2,1.0\n",
        );
        let frontend = load_text(&[path], 1.0).unwrap();
        assert_eq!(
            frontend.source_identities().request_source_ids(),
            &["0".to_string(), "5".to_string()]
        );
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
        assert!(load_text(&[path], 1.0).is_err());
    }

    #[test]
    fn closed_loop_caps_in_flight_and_stamps_admission_clock() {
        let dir = tempfile::tempdir().unwrap();
        // Spread-out CSV arrival times — all IGNORED in closed-loop.
        let path = write_csv(
            dir.path(),
            "cl.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             1,8,2,100.0\n\
             2,8,2,200.0\n\
             3,8,2,300.0\n",
        );
        // cap=2: at t=10 only two admit despite all four being past their CSV
        // arrival, and both are stamped with the admission clock (10ms).
        let mut fe = TraceFrontend::load(
            &[path],
            &InputFileSchema::text_generation_independent(),
            ArrivalSchedule::saturated(),
            capped(2),
            SessionDependency::Independent,
        )
        .unwrap();
        assert_eq!(
            drain_pairs(&mut fe, 10.0),
            vec![(RequestId(0), 10.0), (RequestId(1), 10.0)]
        );
        assert_eq!(fe.in_flight(), 2);
        assert_eq!(fe.submitted(), 2);

        // Full: a second drain admits nothing until a slot frees.
        assert_eq!(drain_pairs(&mut fe, 20.0), vec![]);

        // One completes → exactly one slot opens; the next admits at 30ms.
        fe.record_completion(RequestId(0), Time::from_ms(20.0));
        assert_eq!(fe.in_flight(), 1);
        assert_eq!(drain_pairs(&mut fe, 30.0), vec![(RequestId(2), 30.0)]);
        assert_eq!(fe.in_flight(), 2);

        // Complete the rest; the tail request drains and the ledger zeroes out.
        fe.record_completion(RequestId(1), Time::from_ms(35.0));
        fe.record_completion(RequestId(2), Time::from_ms(38.0));
        assert_eq!(drain_pairs(&mut fe, 40.0), vec![(RequestId(3), 40.0)]);
        assert!(fe.exhausted());
        fe.record_completion(RequestId(3), Time::from_ms(50.0));
        assert_eq!(fe.in_flight(), 0);
        assert_eq!(fe.num_completed(), 4);
    }

    #[test]
    fn rejects_zero_max_concurrency() {
        let err = CapacityLimit::parse(Some(0)).unwrap_err();
        assert!(err.to_string().contains("max_concurrency"), "{err}");
    }

    // ---- session chaining ---------------------------------------------------

    const SESSION_HEADER: &str =
        "id,input_len,output_len,arrival_time,session_id,prefix_kv,tool_wait_after_ms\n";

    /// The point of chaining: a later round waits for its predecessor, so it can
    /// be released AFTER a head that sits later in the trace — the out-of-order
    /// emission the pre-sized `RequestStore` exists for.
    #[test]
    fn session_rounds_wait_for_their_predecessor() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "chain.csv",
            &format!(
                "{SESSION_HEADER}\
                 0,8,2,0.0,7,0,0.0\n\
                 1,8,2,0.0,9,0,0.0\n\
                 2,8,2,0.0,7,64,0.0\n"
            ),
        );
        let mut fe = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["session"]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap();

        // Both session heads go at t=0; row 2 is session 7's second round and
        // stays back even though its own trace arrival has passed.
        assert_eq!(drain_at(&mut fe, 0.0), vec![RequestId(0), RequestId(1)]);
        assert!(!fe.exhausted());
        assert_eq!(drain_at(&mut fe, 100.0), vec![]);

        // Session 7's first round finishes → its successor becomes releasable.
        fe.record_completion(RequestId(0), Time::from_ms(120.0));
        assert_eq!(drain_at(&mut fe, 120.0), vec![RequestId(2)]);
        assert!(fe.exhausted());
    }

    /// The successor is stamped with its release time, not its trace time: the
    /// trace's own timestamp records how fast the *recording* machine served the
    /// previous round, which is the thing being simulated.
    #[test]
    fn tool_wait_delays_the_successor_and_restamps_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "wait.csv",
            &format!(
                "{SESSION_HEADER}\
                 0,8,2,0.0,3,0,250.0\n\
                 1,8,2,0.0,3,64,0.0\n"
            ),
        );
        let mut fe = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["session"]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap();
        assert_eq!(
            drain_session_facts(&mut fe, 0.0),
            vec![(RequestId(0), 0.0, Some(0.0))]
        );

        // Round 0 completes at 100ms, but declares a 250ms tool wait after it.
        fe.record_completion(RequestId(0), Time::from_ms(100.0));
        assert_eq!(drain_at(&mut fe, 349.0), vec![]);
        assert_eq!(
            drain_session_facts(&mut fe, 350.0),
            // Arrival is the successor's release clock; session start stays at round 0.
            vec![(RequestId(1), 350.0, Some(0.0))]
        );
    }

    /// Pacing and causality compose: the global cap still applies while each
    /// session waits for its own predecessor and tool wait. A not-yet-due
    /// successor does not prevent an independent session head from using a
    /// free closed-loop slot.
    #[test]
    fn a_session_holds_its_slot_across_tool_waits() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "closed-chain.csv",
            &format!(
                "{SESSION_HEADER}\
                 0,8,2,100.0,7,0,5.0\n\
                 1,8,2,200.0,9,0,0.0\n\
                 2,8,2,300.0,7,64,0.0\n\
                 3,8,2,400.0,,,\n\
                 4,8,2,500.0,9,64,0.0\n"
            ),
        );
        let mut frontend = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["session"]),
            ArrivalSchedule::saturated(),
            capped(2),
            SessionDependency::Chained,
        )
        .unwrap();

        // Saturated arrival ignores every trace timestamp and fills both slots.
        assert_eq!(
            drain_pairs(&mut frontend, 0.0),
            vec![(RequestId(0), 0.0), (RequestId(1), 0.0)]
        );

        // Session 7's round completes at t=10 and it enters a 5 ms tool wait.
        // It has no request in flight, but the conversation is not over, so it
        // keeps its slot: the standalone row must NOT slip in. This is the whole
        // point of counting sessions rather than requests — the measured runner
        // holds its permit for the lifetime of the session task.
        frontend.record_completion(RequestId(0), Time::from_ms(10.0));
        assert_eq!(drain_pairs(&mut frontend, 10.0), vec![]);
        assert_eq!(frontend.in_flight(), 1);

        // At t=15 the wait elapses and the same session continues, still on the
        // slot it never gave up.
        assert_eq!(drain_pairs(&mut frontend, 15.0), vec![(RequestId(2), 15.0)]);

        // Only when a whole session ends does a new unit start.
        frontend.record_completion(RequestId(2), Time::from_ms(20.0));
        assert_eq!(drain_pairs(&mut frontend, 20.0), vec![(RequestId(3), 20.0)]);

        frontend.record_completion(RequestId(1), Time::from_ms(25.0));
        assert_eq!(drain_pairs(&mut frontend, 25.0), vec![(RequestId(4), 25.0)]);
        assert!(frontend.exhausted());
    }

    /// The exit evidence named in the alignment plan: with a cap of two,
    /// sessions A and B hold both slots across their tool waits, and session C
    /// cannot start until one of them terminates entirely.
    #[test]
    fn a_third_session_waits_for_a_whole_session_to_terminate() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "three-sessions.csv",
            &format!(
                "{SESSION_HEADER}\
                 0,8,2,0.0,1,0,50.0\n\
                 1,8,2,0.0,2,0,50.0\n\
                 2,8,2,0.0,3,0,0.0\n\
                 3,8,2,0.0,1,64,0.0\n\
                 4,8,2,0.0,2,64,0.0\n"
            ),
        );
        let mut frontend = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["session"]),
            open_loop(1.0),
            capped(2),
            SessionDependency::Chained,
        )
        .unwrap();

        // Sessions 1 and 2 take the two slots. Session 3's head is eligible by
        // arrival — every row arrives at 0 — but there is no capacity for it.
        assert_eq!(
            drain_pairs(&mut frontend, 0.0),
            vec![(RequestId(0), 0.0), (RequestId(1), 0.0)]
        );

        // Both finish their first round and sit in a 50 ms tool wait. Nothing is
        // in flight at all, yet session 3 still cannot start.
        frontend.record_completion(RequestId(0), Time::from_ms(10.0));
        frontend.record_completion(RequestId(1), Time::from_ms(10.0));
        assert_eq!(frontend.in_flight(), 0);
        assert_eq!(drain_pairs(&mut frontend, 10.0), vec![]);

        // Their second rounds resume on the slots they held throughout.
        assert_eq!(
            drain_pairs(&mut frontend, 60.0),
            vec![(RequestId(3), 60.0), (RequestId(4), 60.0)]
        );
        assert_eq!(drain_pairs(&mut frontend, 60.0), vec![]);

        // Session 1 ends. Its slot — not merely its in-flight request — frees,
        // and session 3 finally starts.
        frontend.record_completion(RequestId(3), Time::from_ms(70.0));
        assert_eq!(drain_pairs(&mut frontend, 70.0), vec![(RequestId(2), 70.0)]);
        assert!(frontend.exhausted());
    }

    /// The combination the old single axis could not express: replay the trace's
    /// timeline *and* cap how many conversations run at once.
    #[test]
    fn trace_timed_arrival_composes_with_a_capacity_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "timed-capped.csv",
            &format!(
                "{SESSION_HEADER}\
                 0,8,2,0.0,1,0,0.0\n\
                 1,8,2,100.0,2,0,0.0\n\
                 2,8,2,200.0,3,0,0.0\n"
            ),
        );
        let mut frontend = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["session"]),
            open_loop(1.0),
            capped(1),
            SessionDependency::Chained,
        )
        .unwrap();

        // Arrival still gates: session 2 has not arrived at t=0.
        assert_eq!(drain_pairs(&mut frontend, 0.0), vec![(RequestId(0), 0.0)]);
        // Capacity also gates: session 2 has arrived by t=100 but session 1 is
        // still running. It is stamped with the instant the slot opened, not its
        // trace arrival — the measured runner starts its clock at the permit.
        assert_eq!(drain_pairs(&mut frontend, 100.0), vec![]);
        frontend.record_completion(RequestId(0), Time::from_ms(150.0));
        assert_eq!(
            drain_pairs(&mut frontend, 150.0),
            vec![(RequestId(1), 150.0)]
        );
        // A head the cap never touched keeps its own arrival, so tick
        // granularity stays out of arrival times.
        frontend.record_completion(RequestId(1), Time::from_ms(160.0));
        assert_eq!(
            drain_pairs(&mut frontend, 250.0),
            vec![(RequestId(2), 200.0)]
        );
    }

    #[test]
    fn trace_timed_capacity_stamps_every_slot_opened_in_one_drain() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "timed-capped-multiple-slots.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             1,8,2,0.0\n\
             2,8,2,0.0\n\
             3,8,2,0.0\n",
        );
        let mut frontend = TraceFrontend::load(
            &[path],
            &InputFileSchema::text_generation_independent(),
            open_loop(1.0),
            capped(2),
            SessionDependency::Independent,
        )
        .unwrap();

        assert_eq!(
            drain_pairs(&mut frontend, 0.0),
            vec![(RequestId(0), 0.0), (RequestId(1), 0.0)]
        );
        assert_eq!(drain_pairs(&mut frontend, 10.0), vec![]);

        frontend.record_completion(RequestId(0), Time::from_ms(20.0));
        frontend.record_completion(RequestId(1), Time::from_ms(20.0));
        assert_eq!(
            drain_pairs(&mut frontend, 20.0),
            vec![(RequestId(2), 20.0), (RequestId(3), 20.0)]
        );
    }

    /// A trace that declares sessions but whose rows all opt out is *data* with
    /// no conversations, not a misconfiguration: every row is a head, so
    /// chaining degenerates to open-loop replay exactly.
    #[test]
    fn chained_dependency_with_no_session_rows_matches_open_loop() {
        let dir = tempfile::tempdir().unwrap();
        let chained = write_csv(
            dir.path(),
            "a.csv",
            &format!(
                "{SESSION_HEADER}\
                 0,8,2,0.0,,,\n\
                 1,16,4,5.0,,,\n\
                 2,4,1,5.0,,,\n"
            ),
        );
        let open = write_csv(
            dir.path(),
            "b.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             1,16,4,5.0\n\
             2,4,1,5.0\n",
        );

        let mut chained = TraceFrontend::load(
            &[chained],
            &declare("text_generation", &["session"]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap();
        let mut open = load_text(&[open], 1.0).unwrap();
        for now in [0.0, 5.0] {
            assert_eq!(drain_at(&mut chained, now), drain_at(&mut open, now));
        }
        assert!(chained.exhausted() && open.exhausted());
    }

    /// Asking for chaining on a trace that declares no sessions is a hard error,
    /// not a silent degrade — the run would look like it chained and would not.
    #[test]
    fn rejects_chained_dependency_without_the_session_tag() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "nosession.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let err = TraceFrontend::load(
            &[path],
            &InputFileSchema::text_generation_independent(),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Chained,
        )
        .unwrap_err();
        assert!(err.to_string().contains("`session` trace tag"));
    }

    // ---- declared replay axes ----------------------------------------------

    /// The retired names must not silently mean something new: `open_loop`
    /// forbade a cap and `closed_loop` discarded the timeline, so mapping them
    /// onto the new axes by guesswork would change a run's meaning without
    /// changing its config.
    #[test]
    fn retired_pacing_names_are_rejected_with_their_replacement() {
        for retired in ["open_loop", "closed_loop"] {
            let err = ArrivalSchedule::parse(retired, 1.0)
                .unwrap_err()
                .to_string();
            assert!(err.contains("arrival_mode"), "{err}");
            assert!(err.contains("separate axes"), "{err}");
        }
    }

    #[test]
    fn every_advertised_replay_axis_value_parses() {
        for name in ArrivalSchedule::CONFIG_CHOICES {
            ArrivalSchedule::parse(name, 1.0).unwrap();
        }
        for name in SessionDependency::CHOICES {
            SessionDependency::parse(name).unwrap();
        }
        // A cap is optional under either arrival mode, and composes with both.
        assert_eq!(
            CapacityLimit::parse(None).unwrap(),
            CapacityLimit::unlimited()
        );
        assert!(CapacityLimit::parse(Some(4)).is_ok());
        assert!(ArrivalSchedule::parse("teleport", 1.0).is_err());
        assert!(ArrivalSchedule::parse("session_chain", 1.0).is_err());
        assert!(SessionDependency::parse("causal-ish").is_err());
        assert!(SessionDependency::parse("session_chain").is_err());
    }

    // ---- declared schema ----------------------------------------------------

    /// The four-column header under a plain text declaration parses to a tagless
    /// text arrival — the baseline every other case builds on.
    #[test]
    fn text_declaration_yields_a_tagless_text_arrival() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "legacy.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let fe = load_text(&[path], 1.0).unwrap();
        let scheduled_request = &fe.scheduled_requests[0];
        assert_eq!(scheduled_request.release.request_id, RequestId(0));
        assert_eq!(scheduled_request.definition.prompt_tokens, 8);
        assert_eq!(scheduled_request.definition.target_output_tokens, 2);
        assert_eq!(
            scheduled_request.definition.session,
            SessionInput::Standalone
        );
        assert_eq!(
            scheduled_request.definition.decoding,
            DecodingStrategy::Standard
        );
        assert_eq!(
            scheduled_request.scheduling,
            SchedulingDeclaration::default()
        );
        assert_eq!(scheduled_request.slo, SloContract::default());
    }

    /// Tags are declared as a set and stack: three disciplines, one file.
    #[test]
    fn declared_tags_stack_independently() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "stacked.csv",
            "id,input_len,output_len,arrival_time,session_id,prefix_kv,tool_wait_after_ms,\
             ttft_slo_ms,tpot_slo_ms,e2e_slo_ms,priority,accept_rate\n\
             0,8,2,4.0,7,512,250.0,300.0,25.0,1200.0,3,0.75\n",
        );
        let fe = TraceFrontend::load(
            &[path],
            &declare(
                "text_generation",
                &["session", "slo", "priority", "speculative"],
            ),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap();
        let scheduled_request = &fe.scheduled_requests[0];
        assert_eq!(
            scheduled_request.release.session,
            Some(SessionReleaseMetadata {
                session_id: 0,
                tool_wait_after: Time::from_ms(250.0),
            })
        );
        assert_eq!(
            scheduled_request.definition.session,
            SessionInput::Session {
                session_id: 0,
                session_start_time: Time::from_ms(4.0),
                declared_prefix_tokens: 512,
            }
        );
        assert_eq!(
            scheduled_request.scheduling,
            SchedulingDeclaration { priority: 3 }
        );
        assert_eq!(
            scheduled_request.slo,
            SloContract {
                ttft_slo: Some(Time::from_ms(300.0)),
                tpot_slo: Some(Time::from_ms(25.0)),
                e2e_slo: Some(Time::from_ms(1200.0)),
            }
        );
        assert_eq!(
            scheduled_request.definition.decoding,
            DecodingStrategy::Speculative {
                accept_rate: AcceptanceProfile::Uniform(0.75)
            }
        );
        let realized_request = scheduled_request.realize_at(Time::from_ms(10.0));
        assert_eq!(realized_request.core.slo, scheduled_request.slo);
        assert_eq!(
            realized_request.definition.session.session_start_time(),
            Some(Time::from_ms(4.0)),
            "replay release time must not rewrite the trace-declared session start"
        );
    }

    #[test]
    fn startup_narrowing_rejects_non_text_families() {
        let dir = tempfile::tempdir().unwrap();
        let image_path = write_csv(
            dir.path(),
            "image.csv",
            "id,input_len,output_len,arrival_time,encoded_tokens,media_width,media_height\n\
             0,8,2,0.0,256,512,512\n",
        );
        let loaded = LoadedTrace::load(
            &[image_path],
            &declare("image_to_text", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap();
        assert!(loaded
            .into_current_text_frontend()
            .unwrap_err()
            .to_string()
            .contains("text_generation-only"));
    }

    /// A declared tag is still per-row optional: a blank `session_id` means this
    /// one request belongs to no conversation, and blank SLO cells mean that
    /// request declares no metric-specific service bound.
    #[test]
    fn blank_cell_opts_one_row_out_of_a_declared_tag() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "sparse.csv",
            "id,input_len,output_len,arrival_time,session_id,prefix_kv,tool_wait_after_ms,\
             ttft_slo_ms,tpot_slo_ms,e2e_slo_ms,priority\n\
             0,8,2,0.0,,,,,,,\n\
             1,8,2,1.0,4,64,0.0,,,900.0,1\n",
        );
        let fe = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["session", "slo", "priority"]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap();
        assert!(fe.scheduled_requests[0].release.session.is_none());
        assert_eq!(fe.scheduled_requests[0].slo, SloContract::default());
        assert_eq!(fe.scheduled_requests[0].scheduling.priority, 0);
        assert_eq!(
            fe.scheduled_requests[1].release.session.unwrap().session_id,
            0
        );
    }

    #[test]
    fn metric_specific_slo_must_be_positive_when_declared() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "invalid-slo.csv",
            "id,input_len,output_len,arrival_time,ttft_slo_ms,tpot_slo_ms,e2e_slo_ms\n\
             0,8,2,0.0,0,,\n",
        );

        let error = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["slo"]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("ttft_slo_ms"), "{error}");
        assert!(error.contains("greater than zero"), "{error}");
    }

    /// Encoded media stays explicit in its typed definition instead of being
    /// collapsed into a plain-text prompt.
    #[test]
    fn image_to_text_keeps_encoded_tokens_explicit() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "vlm.csv",
            "id,input_len,output_len,arrival_time,media_width,media_height,encoded_tokens\n\
             0,40,16,0.0,1024,1024,1024\n\
             1,40,16,1.0,512,512,256\n",
        );
        let LoadedTrace::ImageToText(mut frontend) = LoadedTrace::load(
            &[path],
            &declare("image_to_text", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap() else {
            panic!("expected image-to-text frontend");
        };
        assert_eq!(
            frontend.scheduled_requests[0].definition.extent,
            ImageExtent {
                width: 1024,
                height: 1024,
            }
        );
        let mut prompt_parts = Vec::new();
        frontend.drain_due(Time::from_ms(1.0), |request| {
            prompt_parts.push((
                request.definition.text_prompt_tokens,
                request.definition.encoded_input_tokens,
            ));
        });
        assert_eq!(prompt_parts, vec![(40, 1024), (40, 256)]);
    }

    /// Speech-to-text: same direction, a different extent, chosen by the
    /// declaration alone.
    #[test]
    fn audio_to_text_resolves_duration_and_rate_to_samples() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "asr.csv",
            "id,input_len,output_len,arrival_time,media_duration_s,media_sample_rate_hz,\
             encoded_tokens\n\
             0,8,32,0.0,12.5,16000,625\n",
        );
        let LoadedTrace::AudioToText(frontend) = LoadedTrace::load(
            &[path],
            &declare("audio_to_text", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap() else {
            panic!("expected audio-to-text frontend");
        };
        assert_eq!(
            frontend.scheduled_requests[0].definition.extent,
            AudioExtent { samples: 200_000 }
        );
        assert_eq!(
            frontend.scheduled_requests[0]
                .definition
                .encoded_input_tokens,
            625
        );
    }

    /// Generated media carries a step count instead of a token count, and its
    /// duration × fps resolves to the frame count temporal attention scales on.
    /// Nothing of it leaks into the prompt.
    #[test]
    fn text_to_video_resolves_frames_and_stays_out_of_the_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "t2v.csv",
            "id,input_len,arrival_time,media_width,media_height,media_duration_s,\
             media_fps,denoise_steps\n\
             0,64,0.0,1280,720,5.0,24,50\n",
        );
        let LoadedTrace::TextToVideo(mut frontend) = LoadedTrace::load(
            &[path],
            &declare("text_to_video", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap() else {
            panic!("expected text-to-video frontend");
        };
        assert_eq!(
            frontend.scheduled_requests[0].definition.extent,
            VideoExtent {
                width: 1280,
                height: 720,
                frames: 120,
            }
        );
        let mut generated_requests = Vec::new();
        frontend.drain_due(Time::from_ms(0.0), |request| {
            generated_requests.push((
                request.definition.text_prompt_tokens,
                request.definition.target_generation_steps,
            ));
        });
        assert_eq!(generated_requests, vec![(64, 50)]);
    }

    /// Image-to-video keeps the encoded image on the input side and the
    /// diffusion work/shape on the output side of one concrete family.
    #[test]
    fn image_to_video_parses_both_directional_extents() {
        let directory = tempfile::tempdir().unwrap();
        let path = write_csv(
            directory.path(),
            "i2v.csv",
            "id,input_len,arrival_time,denoise_steps,encoded_tokens,input_media_width,\
             input_media_height,media_width,media_height,media_duration_s,media_fps\n\
             0,32,0.0,48,256,1024,768,1280,720,5.0,24\n",
        );
        let LoadedTrace::ImageToVideo(frontend) = LoadedTrace::load(
            &[path],
            &declare("image_to_video", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap() else {
            panic!("expected image-to-video frontend");
        };
        let definition = &frontend.scheduled_requests[0].definition;
        assert_eq!(
            definition.input_extent,
            ImageExtent {
                width: 1024,
                height: 768,
            }
        );
        assert_eq!(definition.encoded_input_tokens, 256);
        assert_eq!(
            definition.output_extent,
            VideoExtent {
                width: 1280,
                height: 720,
                frames: 120,
            }
        );
        assert_eq!(definition.target_generation_steps, 48);
    }

    /// One omni request may repeat and mix modalities on both sides. The
    /// concrete segment vectors preserve that ordered model contract.
    #[test]
    fn omni_generation_parses_mixed_input_and_output_segments() {
        let directory = tempfile::tempdir().unwrap();
        let input = vec![
            OmniInputSegment::Text { tokens: 12 },
            OmniInputSegment::Image {
                extent: ImageExtent {
                    width: 640,
                    height: 480,
                },
                encoded_tokens: 256,
            },
            OmniInputSegment::Audio {
                extent: AudioExtent { samples: 16_000 },
                encoded_tokens: 50,
            },
            OmniInputSegment::Video {
                extent: VideoExtent {
                    width: 320,
                    height: 180,
                    frames: 24,
                },
                encoded_tokens: 128,
            },
        ];
        let output = vec![
            OmniOutputSpec::Text { target_tokens: 8 },
            OmniOutputSpec::Image {
                extent: ImageExtent {
                    width: 512,
                    height: 512,
                },
                target_tokens: 64,
            },
            OmniOutputSpec::Audio {
                extent: AudioExtent { samples: 24_000 },
                target_tokens: 75,
            },
            OmniOutputSpec::Video {
                extent: VideoExtent {
                    width: 640,
                    height: 360,
                    frames: 48,
                },
                target_tokens: 96,
            },
        ];
        let path = write_omni_csv(directory.path(), "omni.csv", &input, &output);

        let LoadedTrace::OmniGeneration(frontend) = LoadedTrace::load(
            &[path],
            &declare("omni_generation", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap() else {
            panic!("expected omni-generation frontend");
        };
        let definition = &frontend.scheduled_requests[0].definition;
        assert_eq!(definition.input, input);
        assert_eq!(definition.output, output);
        assert_eq!(
            definition.initial_progress().output_tokens_emitted,
            vec![0, 0, 0, 0]
        );
    }

    #[test]
    fn omni_generation_rejects_empty_segment_plans() {
        let directory = tempfile::tempdir().unwrap();
        let output = vec![OmniOutputSpec::Text { target_tokens: 1 }];
        let path = write_omni_csv(directory.path(), "empty-input.csv", &[], &output);
        let error = LoadedTrace::load(
            &[path],
            &declare("omni_generation", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap_err();
        assert!(error.to_string().contains("input_segments"));
    }

    /// Multiple files concatenate — all under the one declaration the run made.
    #[test]
    fn files_concatenate_under_one_declaration() {
        let dir = tempfile::tempdir().unwrap();
        let first = write_csv(
            dir.path(),
            "a.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let second = write_csv(
            dir.path(),
            "b.csv",
            "id,input_len,output_len,arrival_time\n1,64,1,1.0\n",
        );
        let fe = load_text(&[first, second], 1.0).unwrap();
        assert_eq!(fe.expected_count(), 2);
        assert_eq!(fe.scheduled_requests[1].definition.prompt_tokens, 64);
    }

    // ---- header verification (hard fail) ------------------------------------

    /// A file that carries more than the run declared is an error, not a file
    /// with spare columns: far more often a stale trace or a typo'd declaration.
    #[test]
    fn rejects_column_the_declaration_does_not_cover() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "extra.csv",
            "id,input_len,output_len,arrival_time,session_id,prefix_kv,tool_wait_after_ms\n\
             0,8,2,0.0,1,0,0.0\n",
        );
        // Session columns present but the run declared no session tag.
        let err = load_text(&[path], 1.0).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("unexpected"), "{message}");
        assert!(message.contains("session_id"), "{message}");
    }

    /// The mirror case: the run declares a discipline the file cannot feed.
    #[test]
    fn rejects_declared_tag_with_no_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "missing.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let err = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["slo"]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("missing"), "{message}");
        assert!(message.contains("ttft_slo_ms"), "{message}");
    }

    /// Declaring the wrong media kind fails on the columns, so a video trace can
    /// never be silently read as stills.
    #[test]
    fn rejects_wrong_media_kind() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "clip.csv",
            "id,input_len,output_len,arrival_time,media_width,media_height,media_duration_s,\
             media_fps,denoise_steps\n\
             0,64,1,0.0,1280,720,5.0,24,50\n",
        );
        let err = LoadedTrace::load(
            &[path],
            &declare("text_to_image", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("unexpected"), "{message}");
        assert!(message.contains("media_fps"), "{message}");
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
        let err = load_text(&[path], 1.0).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("unexpected"), "{message}");
        assert!(message.contains("round_idx"), "{message}");
    }

    /// A clip must have at least one frame; `duration × fps` rounding to zero is
    /// a malformed row, not an empty job.
    #[test]
    fn rejects_media_resolving_to_zero_units() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "empty_clip.csv",
            "id,input_len,arrival_time,media_width,media_height,media_duration_s,\
             media_fps,denoise_steps\n\
             0,64,0.0,1280,720,0.0,24,50\n",
        );
        let err = LoadedTrace::load(
            &[path],
            &declare("text_to_video", &[]),
            open_loop(1.0),
            uncapped(),
            SessionDependency::Independent,
        )
        .unwrap_err();
        assert!(err.to_string().contains("greater than zero"));
    }

    // ---- declaration parsing ------------------------------------------------

    #[test]
    fn rejects_unknown_declaration_names() {
        assert!(InputFileFormat::parse("hologram-output").is_err());
        assert!(TraceTag::parse("vibes").is_err());
    }

    #[test]
    fn rejects_repeated_tag() {
        let tags = vec![TraceTag::Session, TraceTag::Session];
        let err =
            InputFileSchema::new(InputFileFormat::TextGenerationIndependent, tags).unwrap_err();
        assert!(err.to_string().contains("more than once"));
    }

    /// Every advertised choice must actually parse — the launcher hands these
    /// exact strings to config validation.
    #[test]
    fn every_advertised_choice_parses() {
        for input_file_format in InputFileFormat::CHOICES {
            InputFileFormat::parse(input_file_format).unwrap();
        }
        for tag in TraceTag::CHOICES {
            TraceTag::parse(tag).unwrap();
        }
    }
}
