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
pub use release::ReplayMode;
pub use schema::{TraceDeclaration, TraceKind, TraceTag};

/// Typed immutable requests plus a definition-blind replay scheduler.
#[derive(Debug)]
pub struct TraceFrontend<Definition: RequestDefinition = TextGenerationDefinition> {
    scheduled_requests: Vec<ScheduledRequest<Definition>>,
    releases: Vec<ReleaseMetadata>,
    emitted: usize,
    completed: u64,
    replay_scheduler: ReplayScheduler,
}

impl TraceFrontend<TextGenerationDefinition> {
    /// Public typed loader for the production text family.
    pub fn load(
        files: &[PathBuf],
        declaration: &TraceDeclaration,
        mode: ReplayMode,
    ) -> Result<Self> {
        load_typed(files, declaration, mode)
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
        declaration: &TraceDeclaration,
        mode: ReplayMode,
    ) -> Result<Self> {
        Ok(match declaration.kind {
            TraceKind::TextGeneration => {
                Self::TextGeneration(load_typed(files, declaration, mode)?)
            }
            TraceKind::ImageToText => Self::ImageToText(load_typed(files, declaration, mode)?),
            TraceKind::VideoToText => Self::VideoToText(load_typed(files, declaration, mode)?),
            TraceKind::AudioToText => Self::AudioToText(load_typed(files, declaration, mode)?),
            TraceKind::TextToImage => Self::TextToImage(load_typed(files, declaration, mode)?),
            TraceKind::TextToVideo => Self::TextToVideo(load_typed(files, declaration, mode)?),
            TraceKind::TextToSpeech => Self::TextToSpeech(load_typed(files, declaration, mode)?),
            TraceKind::ImageToVideo => Self::ImageToVideo(load_typed(files, declaration, mode)?),
            TraceKind::OmniGeneration => {
                Self::OmniGeneration(load_typed(files, declaration, mode)?)
            }
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
        "trace_kind {kind:?} parsed as its own request family, but current \
         deployments are text_generation-only; add the matching encoder/media \
         worker family before executing this trace"
    )
}

fn load_typed<Definition: TraceDefinition>(
    files: &[PathBuf],
    declaration: &TraceDeclaration,
    mode: ReplayMode,
) -> Result<TraceFrontend<Definition>> {
    if files.is_empty() {
        bail!("no trace files given (--trace-files)");
    }
    validate_mode(mode, declaration)?;
    let mut scheduled_requests = Vec::new();
    for file in files {
        schema::load_file(file, declaration, &mut scheduled_requests)?;
    }
    if scheduled_requests.is_empty() {
        bail!("trace files contained no rows");
    }
    validate_arrival_order(&scheduled_requests)?;
    let releases = scheduled_requests
        .iter()
        .map(|request| request.release)
        .collect::<Vec<_>>();
    let replay_scheduler = ReplayScheduler::new(mode, &releases);
    Ok(TraceFrontend {
        scheduled_requests,
        releases,
        emitted: 0,
        completed: 0,
        replay_scheduler,
    })
}

/// Reject a pacing request that cannot run, before any file is read.
///
/// Takes the declaration too, because one mode's precondition is about the data
/// rather than its own payload: chaining rounds is meaningless on a trace that
/// declares no sessions. Checking it here rather than at the config call site
/// means every caller is covered, tests included.
fn validate_mode(mode: ReplayMode, declaration: &TraceDeclaration) -> Result<()> {
    match mode {
        ReplayMode::OpenLoop { request_rate } => require_rate(request_rate)?,
        ReplayMode::SessionChain { request_rate } => {
            require_rate(request_rate)?;
            if !declaration.tags.contains(&TraceTag::Session) {
                bail!(
                    "replay_mode: session_chain needs the `session` trace tag — \
                     without session columns there are no rounds to chain, and \
                     chaining would silently degrade to open-loop replay"
                );
            }
        }
        ReplayMode::ClosedLoop { cap } => {
            if cap == 0 {
                bail!("max_concurrency must be greater than 0 (got 0)");
            }
        }
    }
    Ok(())
}

/// Shared by the two modes that replay the trace's own timeline.
fn require_rate(request_rate: f64) -> Result<()> {
    if !(request_rate.is_finite() && request_rate > 0.0) {
        bail!("request_rate must be finite and > 0 (got {request_rate})");
    }
    Ok(())
}

/// Ids must be sequential `0..N`; `arrival_time` non-decreasing across rows.
///
/// This is a property of the concatenated *list*, not of any one schema, so it
/// lives here rather than in [`schema`]: ids are the `RequestStore` index and
/// the order is what open-loop replay walks, whatever kind the rows are.
fn validate_arrival_order<Definition: RequestDefinition>(
    scheduled_requests: &[ScheduledRequest<Definition>],
) -> Result<()> {
    for (index, scheduled_request) in scheduled_requests.iter().enumerate() {
        if scheduled_request.release.request_id.0 != index as u32 {
            bail!(
                "trace row {index} has id={}, expected sequential id={index}",
                scheduled_request.release.request_id.0
            );
        }
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
        AudioExtent, DecodingStrategy, ImageExtent, OmniInputSegment, OmniOutputSpec, PrefixInput,
        RequestId, VideoExtent,
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

    fn declare(kind: &str, tags: &[&str]) -> TraceDeclaration {
        let tags: Vec<String> = tags.iter().map(|t| (*t).to_string()).collect();
        TraceDeclaration::parse(kind, &tags).unwrap()
    }

    fn open_loop(rate: f64) -> ReplayMode {
        ReplayMode::OpenLoop { request_rate: rate }
    }

    /// Load a plain text trace open-loop — the shape most tests want.
    fn load_text(paths: &[PathBuf], rate: f64) -> Result<TraceFrontend> {
        TraceFrontend::load(paths, &TraceDeclaration::text(), open_loop(rate))
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
    fn rejects_non_sequential_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "bad.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             5,8,2,1.0\n",
        );
        assert!(load_text(&[path], 1.0).is_err());
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
            &TraceDeclaration::text(),
            ReplayMode::ClosedLoop { cap: 2 },
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
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "z.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let err = TraceFrontend::load(
            &[path],
            &TraceDeclaration::text(),
            ReplayMode::ClosedLoop { cap: 0 },
        )
        .unwrap_err();
        assert!(err.to_string().contains("max_concurrency"));
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
            ReplayMode::SessionChain { request_rate: 1.0 },
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
            ReplayMode::SessionChain { request_rate: 1.0 },
        )
        .unwrap();
        assert_eq!(drain_at(&mut fe, 0.0), vec![RequestId(0)]);

        // Round 0 completes at 100ms, but declares a 250ms tool wait after it.
        fe.record_completion(RequestId(0), Time::from_ms(100.0));
        assert_eq!(drain_at(&mut fe, 349.0), vec![]);
        assert_eq!(
            drain_pairs(&mut fe, 350.0),
            vec![(RequestId(1), 350.0)] // release clock, not the CSV's 0.0
        );
    }

    /// A trace that declares sessions but whose rows all opt out is *data* with
    /// no conversations, not a misconfiguration: every row is a head, so
    /// chaining degenerates to open-loop replay exactly.
    #[test]
    fn session_chain_with_no_session_rows_is_open_loop() {
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
            ReplayMode::SessionChain { request_rate: 1.0 },
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
    fn rejects_session_chain_without_the_session_tag() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "nosession.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let err = TraceFrontend::load(
            &[path],
            &TraceDeclaration::text(),
            ReplayMode::SessionChain { request_rate: 1.0 },
        )
        .unwrap_err();
        assert!(err.to_string().contains("`session` trace tag"));
    }

    // ---- declared replay mode ----------------------------------------------

    /// The flat config puts `max_concurrency` beside the mode name, so a payload
    /// the declared mode would ignore is rejected rather than left dead.
    #[test]
    fn replay_mode_rejects_a_payload_its_mode_ignores() {
        for name in ["open_loop", "session_chain"] {
            let err = ReplayMode::parse(name, 1.0, Some(4)).unwrap_err();
            assert!(err.to_string().contains("max_concurrency"), "{name}");
        }
        let err = ReplayMode::parse("closed_loop", 1.0, None).unwrap_err();
        assert!(err.to_string().contains("needs workload.max_concurrency"));
    }

    #[test]
    fn every_advertised_replay_mode_parses() {
        for name in ReplayMode::CHOICES {
            let cap = (*name == "closed_loop").then_some(4);
            ReplayMode::parse(name, 1.0, cap).unwrap();
        }
        assert!(ReplayMode::parse("teleport", 1.0, None).is_err());
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
        assert_eq!(scheduled_request.definition.prefix, PrefixInput::None);
        assert_eq!(
            scheduled_request.definition.decoding,
            DecodingStrategy::Standard
        );
        assert_eq!(
            scheduled_request.scheduling,
            SchedulingDeclaration::default()
        );
    }

    /// Tags are declared as a set and stack: three disciplines, one file.
    #[test]
    fn declared_tags_stack_independently() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "stacked.csv",
            "id,input_len,output_len,arrival_time,session_id,prefix_kv,tool_wait_after_ms,\
             deadline_ms,priority,accept_rate\n\
             0,8,2,0.0,7,512,250.0,1200.0,3,0.75\n",
        );
        let fe = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["session", "slo", "speculative"]),
            open_loop(1.0),
        )
        .unwrap();
        let scheduled_request = &fe.scheduled_requests[0];
        assert_eq!(
            scheduled_request.release.session,
            Some(SessionReleaseMetadata {
                session_id: 7,
                tool_wait_after: Time::from_ms(250.0),
            })
        );
        assert_eq!(
            scheduled_request.definition.prefix,
            PrefixInput::Session {
                session_id: 7,
                declared_prefix_tokens: 512,
            }
        );
        assert_eq!(
            scheduled_request.scheduling,
            SchedulingDeclaration {
                priority: 3,
                relative_completion_deadline: Some(Time::from_ms(1200.0)),
            }
        );
        assert_eq!(
            scheduled_request.definition.decoding,
            DecodingStrategy::Speculative { accept_rate: 0.75 }
        );
        let realized_request = scheduled_request.realize_at(Time::from_ms(10.0));
        assert_eq!(
            realized_request.core.scheduling.completion_deadline,
            Some(Time::from_ms(1210.0))
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
        )
        .unwrap();
        assert!(loaded
            .into_current_text_frontend()
            .unwrap_err()
            .to_string()
            .contains("text_generation-only"));
    }

    /// A declared tag is still per-row optional: a blank `session_id` means this
    /// one request belongs to no conversation, and a blank `deadline_ms` means
    /// the workload ranks by priority without an absolute time target.
    #[test]
    fn blank_cell_opts_one_row_out_of_a_declared_tag() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "sparse.csv",
            "id,input_len,output_len,arrival_time,session_id,prefix_kv,tool_wait_after_ms,\
             deadline_ms,priority\n\
             0,8,2,0.0,,,,,\n\
             1,8,2,1.0,4,64,0.0,900.0,1\n",
        );
        let fe = TraceFrontend::load(
            &[path],
            &declare("text_generation", &["session", "slo"]),
            open_loop(1.0),
        )
        .unwrap();
        assert!(fe.scheduled_requests[0].release.session.is_none());
        assert_eq!(
            fe.scheduled_requests[0]
                .scheduling
                .relative_completion_deadline,
            None
        );
        assert_eq!(fe.scheduled_requests[0].scheduling.priority, 0);
        assert_eq!(
            fe.scheduled_requests[1].release.session.unwrap().session_id,
            4
        );
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
        let LoadedTrace::ImageToText(mut frontend) =
            LoadedTrace::load(&[path], &declare("image_to_text", &[]), open_loop(1.0)).unwrap()
        else {
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
        let LoadedTrace::AudioToText(frontend) =
            LoadedTrace::load(&[path], &declare("audio_to_text", &[]), open_loop(1.0)).unwrap()
        else {
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
        let LoadedTrace::TextToVideo(mut frontend) =
            LoadedTrace::load(&[path], &declare("text_to_video", &[]), open_loop(1.0)).unwrap()
        else {
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
        let LoadedTrace::ImageToVideo(frontend) =
            LoadedTrace::load(&[path], &declare("image_to_video", &[]), open_loop(1.0)).unwrap()
        else {
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

        let LoadedTrace::OmniGeneration(frontend) =
            LoadedTrace::load(&[path], &declare("omni_generation", &[]), open_loop(1.0)).unwrap()
        else {
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
        let error = LoadedTrace::load(&[path], &declare("omni_generation", &[]), open_loop(1.0))
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
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("missing"), "{message}");
        assert!(message.contains("deadline_ms"), "{message}");
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
        let err =
            LoadedTrace::load(&[path], &declare("text_to_image", &[]), open_loop(1.0)).unwrap_err();
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
        assert!(err.to_string().contains("multi-round"));
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
        let err =
            LoadedTrace::load(&[path], &declare("text_to_video", &[]), open_loop(1.0)).unwrap_err();
        assert!(err.to_string().contains("must be in 1..=u64::MAX"));
    }

    // ---- declaration parsing ------------------------------------------------

    #[test]
    fn rejects_unknown_declaration_names() {
        assert!(TraceDeclaration::parse("hologram_output", &[]).is_err());
        assert!(TraceDeclaration::parse("text_generation", &["vibes".to_string()]).is_err());
    }

    #[test]
    fn rejects_repeated_tag() {
        let tags = ["session".to_string(), "session".to_string()];
        let err = TraceDeclaration::parse("text_generation", &tags).unwrap_err();
        assert!(err.to_string().contains("more than once"));
    }

    /// Every advertised choice must actually parse — the launcher hands these
    /// exact strings to config validation.
    #[test]
    fn every_advertised_choice_parses() {
        for kind in TraceKind::CHOICES {
            TraceKind::parse(kind).unwrap();
        }
        for tag in TraceTag::CHOICES {
            TraceTag::parse(tag).unwrap();
        }
    }
}
