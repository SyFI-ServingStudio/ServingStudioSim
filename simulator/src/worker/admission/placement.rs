//! Partition placement policy used by local admission lifecycles.

#[derive(Clone, Copy, Debug)]
pub enum LoadBalance {
    Single,
    RoundRobin {
        next: u16,
    },
    /// vLLM's data-parallel load balancer (`DPLBAsyncMPClient.
    /// get_core_engine_for_request`, vllm/v1/engine/core_client.py:1546-1597):
    /// the engine with the lowest `waiting + running` wins, waiting requests
    /// count extra as KV usage climbs past one half, and the scan start rotates
    /// after every placement so ties spread round-robin.
    LeastLoaded {
        next: u16,
    },
    /// Not vLLM: the partition with the least KV in use plus queued KV
    /// footprint (prompt, declared prefix and output tokens), as a fraction of
    /// its capacity. Request counts ignore context length, so with long-tailed
    /// contexts one rank fills while others idle; this balances the tokens
    /// that bound both capacity and decode attention time. Ties rotate.
    KvLeastLoaded {
        next: u16,
    },
    /// Not vLLM: the fewest `waiting + running` among the partitions where the
    /// new request's footprint fits on top of their KV in use and queued
    /// footprint; when none fits, the lowest such KV (as `KvLeastLoaded`).
    /// Attention-DP ranks step in lockstep, so the busiest rank's batch sets
    /// the step time: balance the request count, and use KV only to keep a
    /// rank from overfilling. Ties rotate.
    KvFitLeastLoaded {
        next: u16,
    },
}

/// One partition's load as vLLM's DP coordinator reports it per engine.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PartitionLoad {
    /// Requests placed on the partition and not yet started.
    pub waiting: u32,
    /// Requests the partition is prefilling or decoding.
    pub running: u32,
    /// Fraction of the partition's KV capacity in use, in `[0, 1]`.
    pub kv_usage: f64,
    /// The partition's queued requests' full KV footprint as a fraction of its
    /// capacity (may exceed 1).
    pub queued_kv_usage: f64,
    /// The request being placed's full KV footprint as a fraction of the
    /// partition's capacity.
    pub incoming_kv_usage: f64,
}

impl PartitionLoad {
    /// vLLM's engine score (core_client.py:1571-1579). With one API client the
    /// exact in-flight count equals `waiting + running`; waiting requests are
    /// penalised by up to 3x as KV usage rises from 50% to 100%.
    fn vllm_score(self) -> f64 {
        let mut score = f64::from(self.waiting) + f64::from(self.running);
        if self.waiting > 0 {
            score += f64::from(self.waiting) * 6.0 * (self.kv_usage - 0.5).max(0.0);
        }
        score
    }

    fn kv_score(self) -> f64 {
        self.kv_usage + self.queued_kv_usage
    }

    /// Request count where the incoming request fits; otherwise past every
    /// fitting partition, ordered by KV.
    fn fit_score(self) -> f64 {
        if self.kv_score() + self.incoming_kv_usage <= 1.0 {
            f64::from(self.waiting) + f64::from(self.running)
        } else {
            f64::from(u32::MAX) + self.kv_score()
        }
    }
}

impl LoadBalance {
    /// Choose a partition without applying its capacity gate. Load-aware
    /// policies must use [`Self::choose_by_load`].
    pub(crate) fn choose(&mut self, num_partitions: usize) -> usize {
        match self {
            Self::Single => 0,
            Self::RoundRobin { next } => {
                let partition = (*next as usize) % num_partitions;
                *next = ((*next as usize + 1) % num_partitions) as u16;
                partition
            }
            Self::LeastLoaded { .. } | Self::KvLeastLoaded { .. } | Self::KvFitLeastLoaded { .. } => {
                panic!("load-aware placement needs partition loads; use choose_by_load")
            }
        }
    }

    /// Whether [`Self::choose_by_load`] reads the loads it is given.
    pub(crate) fn needs_load(&self) -> bool {
        matches!(
            self,
            Self::LeastLoaded { .. } | Self::KvLeastLoaded { .. } | Self::KvFitLeastLoaded { .. }
        )
    }

    /// Choose a partition from per-partition loads (`loads[i]` is partition
    /// `i`). Policies that ignore load fall back to [`Self::choose`].
    pub(crate) fn choose_by_load(&mut self, loads: &[PartitionLoad]) -> usize {
        let (next, score): (&mut u16, fn(PartitionLoad) -> f64) = match self {
            Self::LeastLoaded { next } => (next, PartitionLoad::vllm_score),
            Self::KvLeastLoaded { next } => (next, PartitionLoad::kv_score),
            Self::KvFitLeastLoaded { next } => (next, PartitionLoad::fit_score),
            _ => return self.choose(loads.len()),
        };
        let num_partitions = loads.len();
        let start = (*next as usize) % num_partitions;
        let mut best = start;
        let mut best_score = f64::INFINITY;
        for offset in 0..num_partitions {
            let partition = (start + offset) % num_partitions;
            let partition_score = score(loads[partition]);
            if partition_score < best_score {
                best_score = partition_score;
                best = partition;
            }
        }
        *next = ((start + 1) % num_partitions) as u16;
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(waiting: u32, running: u32, kv_usage: f64) -> PartitionLoad {
        PartitionLoad {
            waiting,
            running,
            kv_usage,
            queued_kv_usage: 0.0,
            incoming_kv_usage: 0.0,
        }
    }

    #[test]
    fn round_robin_wraps_across_partitions() {
        let mut placement = LoadBalance::RoundRobin { next: 0 };
        assert_eq!(placement.choose(2), 0);
        assert_eq!(placement.choose(2), 1);
        assert_eq!(placement.choose(2), 0);
    }

    #[test]
    fn least_loaded_rotates_ties_like_round_robin() {
        let mut placement = LoadBalance::LeastLoaded { next: 0 };
        let idle = [PartitionLoad::default(); 3];
        assert_eq!(placement.choose_by_load(&idle), 0);
        assert_eq!(placement.choose_by_load(&idle), 1);
        assert_eq!(placement.choose_by_load(&idle), 2);
        assert_eq!(placement.choose_by_load(&idle), 0);
    }

    #[test]
    fn least_loaded_picks_fewest_outstanding_requests() {
        let mut placement = LoadBalance::LeastLoaded { next: 0 };
        let loads = [load(0, 5, 0.1), load(1, 1, 0.1), load(0, 3, 0.1)];
        assert_eq!(placement.choose_by_load(&loads), 1);
        // The scan start moved to partition 1 but the minimum is unchanged.
        assert_eq!(placement.choose_by_load(&loads), 1);
    }

    #[test]
    fn least_loaded_penalises_waiting_under_kv_pressure() {
        let mut placement = LoadBalance::LeastLoaded { next: 0 };
        // Partition 0: 2 waiting at 100% KV -> 2 + 2 + 2*6*0.5 = 10.
        // Partition 1: 0 waiting, 8 running -> 8.
        let loads = [load(2, 2, 1.0), load(0, 8, 1.0)];
        assert_eq!(placement.choose_by_load(&loads), 1);
        // Below 50% usage the penalty is off: 4 < 8.
        let mut placement = LoadBalance::LeastLoaded { next: 0 };
        let loads = [load(2, 2, 0.5), load(0, 8, 0.5)];
        assert_eq!(placement.choose_by_load(&loads), 0);
    }

    #[test]
    fn kv_least_loaded_balances_tokens_not_requests() {
        let mut placement = LoadBalance::KvLeastLoaded { next: 0 };
        // Partition 0 runs one long request; partition 1 runs four short ones.
        let loads = [load(0, 1, 0.8), load(0, 4, 0.3)];
        assert_eq!(placement.choose_by_load(&loads), 1);
        // A queued footprint counts as if it were already in use.
        let queued = PartitionLoad {
            queued_kv_usage: 0.6,
            ..load(1, 4, 0.3)
        };
        assert_eq!(placement.choose_by_load(&[load(0, 1, 0.8), queued]), 0);
        assert!(placement.needs_load());
    }

    #[test]
    fn kv_least_loaded_rotates_ties() {
        let mut placement = LoadBalance::KvLeastLoaded { next: 0 };
        let idle = [PartitionLoad::default(); 2];
        assert_eq!(placement.choose_by_load(&idle), 0);
        assert_eq!(placement.choose_by_load(&idle), 1);
    }

    #[test]
    fn kv_fit_least_loaded_balances_count_while_the_request_fits() {
        let mut placement = LoadBalance::KvFitLeastLoaded { next: 0 };
        let incoming = |l: PartitionLoad| PartitionLoad {
            incoming_kv_usage: 0.15,
            ..l
        };
        // Both fit: the fewer requests wins even with more KV in use.
        let loads = [incoming(load(0, 1, 0.8)), incoming(load(0, 4, 0.3))];
        assert_eq!(placement.choose_by_load(&loads), 0);
        // 0.9 + 0.15 does not fit: the busier partition takes it.
        let loads = [incoming(load(0, 1, 0.9)), incoming(load(0, 4, 0.3))];
        assert_eq!(placement.choose_by_load(&loads), 1);
        // Nothing fits: the lowest KV.
        let loads = [incoming(load(0, 1, 0.95)), incoming(load(0, 4, 0.9))];
        assert_eq!(placement.choose_by_load(&loads), 1);
    }

    #[test]
    fn non_load_policies_ignore_loads() {
        let mut placement = LoadBalance::RoundRobin { next: 0 };
        let loads = [load(9, 9, 1.0), PartitionLoad::default()];
        assert_eq!(placement.choose_by_load(&loads), 0);
        assert_eq!(placement.choose_by_load(&loads), 1);
        assert!(!placement.needs_load());
        assert!(LoadBalance::LeastLoaded { next: 0 }.needs_load());
    }
}
