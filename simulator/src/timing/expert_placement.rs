//! Where an expert-parallel group keeps each logical expert: which physical
//! slots, on which rank.
//!
//! A kernel that runs the whole EP layer (DeepGEMM's mega MoE) is billed by the
//! rows every rank's slots receive, so the placement decides the busiest rank
//! as much as the routing does. Placements here are functions of the layer's
//! long-run expert load only, never of the batch being priced: a deployment
//! rebalances from history, and the batch then lands on that layout.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotPlacement {
    /// One slot per expert, in expert order: rank `r` holds the contiguous
    /// experts `[r·E/ep, (r+1)·E/ep)` (no redundancy, no rebalancing).
    Contiguous,
    /// DeepInfer's EPLB (`model_executor/eplb/policy.py`
    /// `rebalance_experts_global`, a679bb8) with `num_redundant` extra slots,
    /// from the layer's per-step expert counts over a history window:
    /// 1. each extra replica goes to the expert whose copy minimizes the sum
    ///    over steps of the largest per-replica load;
    /// 2. replicas, heaviest total per-replica load first (ties: slot order),
    ///    each go to the non-full rank that minimizes the sum over steps of
    ///    `max(rank load + replica, peak)` (ties: the lower rank).
    ///
    /// An expert's rows split evenly over its replicas. DeepGEMM's mega gate
    /// picks replica `(rank + token · 23333) mod replicas` per token; over the
    /// 40 DSv4.1 layers that rule and the even split give the same mean
    /// busiest-rank ratio and differ by at most 0.018 on any layer
    /// (`wt-dsv41-deepinfer/logs/20261115_8_megamoe_oos/stats_all_layers.txt`).
    Eplb { num_redundant: u32 },
    /// DeepInfer's benchmark mode (`DEEPINFER_FORCE_LOAD_BALANCE=1`, set
    /// unconditionally by `examples/benchmark.py`): the router's choice is
    /// replaced by `top_k` distinct slots drawn uniformly over all
    /// `E + num_redundant` slots, so the demand's routing does not matter.
    ForcedBalance { num_redundant: u32 },
}

impl SlotPlacement {
    /// Physical slots over the whole group for `num_experts` logical experts.
    pub fn num_slots(self, num_experts: usize) -> usize {
        match self {
            Self::Contiguous => num_experts,
            Self::Eplb { num_redundant } | Self::ForcedBalance { num_redundant } => {
                num_experts + num_redundant as usize
            }
        }
    }

    /// The slots of each logical expert, from that layer's history:
    /// `history[step][expert]` routed rows. Not defined for
    /// [`Self::ForcedBalance`], which ignores the experts.
    pub fn layout(self, history: &[Vec<f64>], ep_size: usize) -> LayerSlots {
        assert!(!history.is_empty(), "a placement needs at least one history step");
        let num_experts = history[0].len();
        let num_slots = self.num_slots(num_experts);
        assert!(
            ep_size > 0 && num_slots % ep_size == 0,
            "slots must split over ranks"
        );
        match self {
            Self::Contiguous => LayerSlots {
                slots_of: (0..num_experts).map(|expert| vec![expert as u32]).collect(),
                num_slots,
            },
            Self::Eplb { num_redundant } => eplb(history, num_redundant, ep_size),
            Self::ForcedBalance { .. } => panic!("forced balance has no expert layout"),
        }
    }
}

/// One layer's placement: `slots_of[expert]` lists that expert's replica slots
/// in replica order; slot `s` lives on rank `s / (num_slots / ep_size)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerSlots {
    pub slots_of: Vec<Vec<u32>>,
    pub num_slots: usize,
}

impl LayerSlots {
    /// Per-slot rows for one layer's per-expert rows: an expert's rows split
    /// evenly over its replicas, the remainder to the first replicas.
    pub fn place(&self, expert_rows: &[u32]) -> Vec<u32> {
        assert_eq!(
            expert_rows.len(),
            self.slots_of.len(),
            "one count per expert"
        );
        let mut slots = vec![0u32; self.num_slots];
        for (&rows, replicas) in expert_rows.iter().zip(&self.slots_of) {
            let n = replicas.len() as u32;
            for (index, &slot) in replicas.iter().enumerate() {
                slots[slot as usize] = rows / n + u32::from((index as u32) < rows % n);
            }
        }
        slots
    }
}

fn eplb(history: &[Vec<f64>], num_redundant: u32, ep_size: usize) -> LayerSlots {
    let num_experts = history[0].len();
    let num_slots = num_experts + num_redundant as usize;
    let per_rank = num_slots / ep_size;
    // Replication. Per step, replicating `j` changes only that step's maximum
    // and only when `j` holds it, so the step-summed objective is smallest for
    // the expert whose own steps' maxima drop the most; experts holding no
    // step's maximum tie at the unchanged sum (argmin: the lowest id, which
    // can only win when no maximum drops at all).
    let mut replicas = vec![1u32; num_experts];
    let mut physical: Vec<usize> = (0..num_experts).collect();
    for _ in 0..num_redundant {
        let mut drop = vec![0.0f64; num_experts];
        for step in history {
            // The source breaks ties between equal scores with a 1e-4 * id
            // offset, so the maximum is unique.
            let score = |e: usize| (step[e] + 1e-4 * e as f64) / f64::from(replicas[e]);
            let hot = (0..num_experts)
                .max_by(|&a, &b| score(a).total_cmp(&score(b)))
                .expect("at least one expert");
            let replaced = step[hot] / f64::from(replicas[hot] + 1);
            let rest = (0..num_experts)
                .filter(|&e| e != hot)
                .map(score)
                .fold(f64::NEG_INFINITY, f64::max);
            drop[hot] += score(hot) - replaced.max(rest);
        }
        let chosen = (0..num_experts)
            .max_by(|&a, &b| drop[a].total_cmp(&drop[b]).then(b.cmp(&a)))
            .expect("at least one expert");
        replicas[chosen] += 1;
        physical.push(chosen);
    }
    // Packing (`_batch_greedy_indices`), in f32 as the source computes it:
    // near-equal candidate scores then break the same way.
    let per_step = |slot: usize, step: usize| {
        history[step][physical[slot]] as f32 / replicas[physical[slot]] as f32
    };
    let weight = |slot: usize| {
        (0..history.len()).map(|s| history[s][physical[slot]] as f32).sum::<f32>()
            / replicas[physical[slot]] as f32
    };
    let mut order: Vec<usize> = (0..num_slots).collect();
    order.sort_by(|&a, &b| weight(b).total_cmp(&weight(a))); // stable: slot order on ties
    let steps = history.len();
    let mut loads = vec![vec![0.0f32; steps]; ep_size];
    let mut peaks = vec![0.0f32; steps];
    let mut rank_slots: Vec<Vec<usize>> = vec![Vec::with_capacity(per_rank); ep_size];
    for slot in order {
        let mut best: Option<(f32, usize)> = None;
        for rank in 0..ep_size {
            if rank_slots[rank].len() == per_rank {
                continue;
            }
            let score: f32 = (0..steps)
                .map(|s| (loads[rank][s] + per_step(slot, s)).max(peaks[s]))
                .sum();
            if best.is_none_or(|(b, _)| score < b) {
                best = Some((score, rank));
            }
        }
        let (_, rank) = best.expect("as many slots as replicas");
        for s in 0..steps {
            loads[rank][s] += per_step(slot, s);
            peaks[s] = peaks[s].max(loads[rank][s]);
        }
        rank_slots[rank].push(physical[slot]);
    }
    let mut slots_of = vec![Vec::new(); num_experts];
    for (rank, experts) in rank_slots.iter().enumerate() {
        for (index, &expert) in experts.iter().enumerate() {
            slots_of[expert].push((rank * per_rank + index) as u32);
        }
    }
    LayerSlots {
        slots_of,
        num_slots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_is_identity() {
        let slots = SlotPlacement::Contiguous.layout(&[vec![1.0; 8]], 4);
        assert_eq!(
            slots.place(&[3, 0, 1, 2, 5, 6, 7, 8]),
            vec![3, 0, 1, 2, 5, 6, 7, 8]
        );
    }

    /// Equal loads: the id offset makes the last experts the hottest, so they
    /// get the replicas, and every rank ends with the same load.
    #[test]
    fn eplb_on_uniform_load_replicates_the_last_experts_and_balances() {
        let slots = SlotPlacement::Eplb { num_redundant: 32 }.layout(&[vec![1.0; 384]], 4);
        assert_eq!(slots.num_slots, 416);
        let doubled: Vec<usize> = (0..384).filter(|&e| slots.slots_of[e].len() == 2).collect();
        assert_eq!(doubled, (352..384).collect::<Vec<_>>());
        let placed = slots.place(&[100; 384]);
        assert_eq!(placed.iter().sum::<u32>(), 38400);
        let ranks: Vec<u32> = placed.chunks(104).map(|r| r.iter().sum()).collect();
        assert_eq!(ranks, vec![9600; 4]);
    }

    /// A hot expert is split over replicas on different ranks, and the
    /// heaviest rank carries less than with the contiguous layout.
    #[test]
    fn eplb_spreads_a_hot_expert() {
        let mut load = vec![1.0; 16];
        load[0] = 9.0;
        let slots = SlotPlacement::Eplb { num_redundant: 4 }.layout(std::slice::from_ref(&load), 4);
        assert!(slots.slots_of[0].len() >= 3);
        let rows: Vec<u32> = load.iter().map(|&l| (l * 10.0) as u32).collect();
        let max_rank = |placed: Vec<u32>, per: usize| {
            placed
                .chunks(per)
                .map(|r| r.iter().sum::<u32>())
                .max()
                .unwrap()
        };
        let contiguous = max_rank(SlotPlacement::Contiguous.layout(std::slice::from_ref(&load), 4).place(&rows), 4);
        assert!(max_rank(slots.place(&rows), 5) < contiguous);
        assert_eq!(
            slots.place(&rows).iter().sum::<u32>(),
            rows.iter().sum::<u32>()
        );
    }
}
