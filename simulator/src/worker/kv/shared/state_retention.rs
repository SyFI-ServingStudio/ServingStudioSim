//! Which round-end recurrent states a retained context keeps.
//!
//! A context resumable only where a state was saved (`context_end`) leaves one
//! state at its end per round. [`StateRetention`] decides how many of the
//! session's earlier round-end states stay with it, so that a later prompt
//! that ends inside the context resumes from the latest kept state at or
//! before its end instead of from nothing.
//!
//! A context and its states are one unit: charged together (the context plus
//! one reusable state each), kept or evicted whole. A request that takes the
//! context carries the states that survive its round, still charged, until its
//! context returns with them below its new end.
//!
//! A round whose decode runs elsewhere leaves two states: its pass's end at
//! the prompt's end, and the handed-back decode's end at the context's end
//! ([`StateTag::handback`]).
//!
//! The rules take no tuned parameter: [`StateRetention::Latest`] keeps a count,
//! [`StateRetention::Anchors`] a count and a round stride, both the user's.

/// What a round-end state is, beside where it sits: the round that computed
/// it, counted along the session's lineage (one past the round of the state
/// it resumed from, 0 when it resumed from none), and whether it is a
/// handed-back decode's end rather than the end of a pass this engine ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StateTag {
    pub(crate) round: u32,
    pub(crate) handback: bool,
}

impl StateTag {
    /// The tag of the state a pass resuming from `resume` leaves at its end.
    pub(crate) fn after(resume: Option<StateTag>) -> Self {
        Self {
            round: resume.map_or(0, |tag| tag.round + 1),
            handback: false,
        }
    }

    /// The tag of a handed-back decode's end, given `latest`, the latest
    /// state of the context at or before the prompt's end: the decode
    /// belongs to that state's round (0 without one).
    pub(crate) fn handed_back(latest: Option<RoundState>) -> Self {
        Self {
            round: latest.map_or(0, |state| state.tag.round),
            handback: true,
        }
    }
}

/// One round-end state: where it sits in the context, and what it is.
/// Ordered by position first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct RoundState {
    pub(crate) at: u64,
    pub(crate) tag: StateTag,
}

/// The states a running request carries from the entry it took, and the tag
/// of the state it resumed from (`None`: it resumed from none).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Carried {
    pub(crate) states: Vec<RoundState>,
    pub(crate) resume: Option<StateTag>,
}

/// Which round-end states of a context stay with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateRetention {
    /// The latest `k` round-end states, the end's included
    /// (`recurrent_state_checkpoints`). `k = 1` keeps only the end.
    Latest { k: u32 },
    /// The latest handed-back decode's end, the latest pass's end, and
    /// `count` anchors: walking back from that pass, the earlier passes' ends
    /// of rounds before it that are a multiple of `stride`
    /// (`recurrent_state_anchors`, `recurrent_state_anchor_stride`). Stride 2
    /// records an anchor every other round, so the kept anchor is one to two
    /// rounds back at a constant state count; stride 1 keeps the latest
    /// earlier pass ends.
    Anchors { count: u32, stride: u32 },
}

impl Default for StateRetention {
    fn default() -> Self {
        Self::Latest { k: 1 }
    }
}

impl StateRetention {
    /// Whether an entry keeps states below its end.
    pub(crate) fn keeps_earlier(self) -> bool {
        match self {
            Self::Latest { k } => k > 1,
            Self::Anchors { .. } => true,
        }
    }

    /// The states of `states` (ascending, one per position, the entry's end
    /// last) that stay.
    pub(crate) fn keep(self, mut states: Vec<RoundState>) -> Vec<RoundState> {
        match self {
            Self::Latest { k } => {
                states.drain(..states.len().saturating_sub(k as usize));
                states
            }
            Self::Anchors { count, stride } => {
                let handback = states.iter().rposition(|state| state.tag.handback);
                let pass = states.iter().rposition(|state| !state.tag.handback);
                let mut keep = vec![false; states.len()];
                for index in handback.into_iter().chain(pass) {
                    keep[index] = true;
                }
                if let Some(pass) = pass {
                    let current = states[pass].tag.round;
                    let anchors = (0..pass)
                        .rev()
                        .filter(|&index| {
                            let tag = states[index].tag;
                            !tag.handback && tag.round < current && tag.round % stride.max(1) == 0
                        })
                        .take(count as usize);
                    for index in anchors.collect::<Vec<_>>() {
                        keep[index] = true;
                    }
                }
                let mut kept = keep.into_iter();
                states.retain(|_| kept.next().unwrap_or(false));
                states
            }
        }
    }

    /// Of `states` (ascending, at or before the resume point of a round that
    /// resumed from a `resume` state), those still kept once that round adds
    /// its `added` states: its pass's end, and with 2 a handed-back decode's
    /// end.
    pub(crate) fn survivors(
        self,
        states: &[RoundState],
        resume: StateTag,
        added: u32,
    ) -> Vec<RoundState> {
        let pass = StateTag::after(Some(resume));
        let mut all = states.to_vec();
        all.push(RoundState {
            at: u64::MAX - 1,
            tag: pass,
        });
        if added >= 2 {
            all.push(RoundState {
                at: u64::MAX,
                tag: StateTag {
                    handback: true,
                    ..pass
                },
            });
        }
        let mut kept = self.keep(all);
        kept.retain(|state| state.at < u64::MAX - 1);
        kept
    }
}

/// Sort `states` by position and keep one per position, the latest round's.
pub(crate) fn normalize(states: &mut Vec<RoundState>) {
    states.sort_unstable_by(|a, b| a.at.cmp(&b.at).then(b.tag.cmp(&a.tag)));
    states.dedup_by_key(|state| state.at);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(positions: &[u64]) -> Vec<RoundState> {
        positions
            .iter()
            .map(|&at| RoundState {
                at,
                tag: StateTag::default(),
            })
            .collect()
    }

    #[test]
    fn latest_keeps_the_last_k_states_and_survivors_leave_room_for_the_new_end() {
        let latest = StateRetention::Latest { k: 3 };
        assert_eq!(latest.keep(at(&[1, 2, 3, 4])), at(&[2, 3, 4]));
        assert_eq!(latest.keep(at(&[4])), at(&[4]));
        assert_eq!(
            latest.survivors(&at(&[1, 2, 3]), StateTag::default(), 1),
            at(&[2, 3])
        );
        assert!(StateRetention::default()
            .survivors(&at(&[1, 2]), StateTag::default(), 1)
            .is_empty());
        // A round that also hands a decode back leaves room for both.
        assert_eq!(
            latest.survivors(&at(&[1, 2, 3]), StateTag::default(), 2),
            at(&[3])
        );
    }

    fn tagged(at: u64, round: u32, handback: bool) -> RoundState {
        RoundState {
            at,
            tag: StateTag { round, handback },
        }
    }

    #[test]
    fn anchors_keep_the_latest_hand_back_and_pass_and_strided_earlier_passes() {
        let anchors = StateRetention::Anchors {
            count: 1,
            stride: 2,
        };
        // Rounds 0..=3, each a pass end then a hand-back.
        let states: Vec<RoundState> = (0..4)
            .flat_map(|round| {
                let at = 100 * u64::from(round + 1);
                [tagged(at, round, false), tagged(at + 50, round, true)]
            })
            .collect();
        assert_eq!(
            anchors.keep(states.clone()),
            [
                tagged(300, 2, false),
                tagged(400, 3, false),
                tagged(450, 3, true)
            ],
            "round 2 is the latest even round before round 3"
        );
        let stride_one = StateRetention::Anchors {
            count: 2,
            stride: 1,
        };
        assert_eq!(
            stride_one.keep(states.clone()),
            [
                tagged(200, 1, false),
                tagged(300, 2, false),
                tagged(400, 3, false),
                tagged(450, 3, true)
            ]
        );
        // A round resuming from round 3's hand-back adds a round-4 pass end
        // and hand-back: of the states up to the resume point, only round
        // 2's pass end stays, as the anchor.
        assert_eq!(
            anchors.survivors(&states, StateTag::handed_back(Some(states[6])), 2),
            [tagged(300, 2, false)]
        );
        assert!(anchors.keeps_earlier());
    }

    #[test]
    fn normalize_keeps_the_latest_round_at_each_position() {
        let tagged = |at, round| RoundState {
            at,
            tag: StateTag {
                round,
                handback: false,
            },
        };
        let mut states = vec![tagged(20, 1), tagged(10, 0), tagged(20, 3)];
        normalize(&mut states);
        assert_eq!(states, [tagged(10, 0), tagged(20, 3)]);
    }
}
