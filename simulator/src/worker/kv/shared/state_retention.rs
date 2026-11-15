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

/// What a round-end state is, beside where it sits: the round that computed
/// it, counted along the session's lineage (one past the round of the state
/// it resumed from, 0 when it resumed from none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StateTag {
    pub(crate) round: u32,
}

impl StateTag {
    /// The tag of the state a round resuming from `resume` leaves at its end.
    pub(crate) fn after(resume: Option<StateTag>) -> Self {
        Self {
            round: resume.map_or(0, |tag| tag.round + 1),
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
        }
    }

    /// Of `states` (ascending, at or before the resume point of a round that
    /// resumed from a `resume` state), those still kept once that round adds
    /// the state at its own end.
    pub(crate) fn survivors(self, states: &[RoundState], resume: StateTag) -> Vec<RoundState> {
        let mut all = states.to_vec();
        all.push(RoundState {
            at: u64::MAX,
            tag: StateTag::after(Some(resume)),
        });
        let mut kept = self.keep(all);
        kept.pop();
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
            latest.survivors(&at(&[1, 2, 3]), StateTag::default()),
            at(&[2, 3])
        );
        assert!(StateRetention::default()
            .survivors(&at(&[1, 2]), StateTag::default())
            .is_empty());
    }

    #[test]
    fn normalize_keeps_the_latest_round_at_each_position() {
        let tagged = |at, round| RoundState {
            at,
            tag: StateTag { round },
        };
        let mut states = vec![tagged(20, 1), tagged(10, 0), tagged(20, 3)];
        normalize(&mut states);
        assert_eq!(states, [tagged(10, 0), tagged(20, 3)]);
    }
}
