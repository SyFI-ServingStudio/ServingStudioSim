//! Partition placement policy used by local admission lifecycles.

#[derive(Clone, Copy, Debug)]
pub enum LoadBalance {
    Single,
    RoundRobin { next: u16 },
}

impl LoadBalance {
    /// Choose a partition without applying its capacity gate.
    pub(crate) fn choose(&mut self, num_partitions: usize) -> usize {
        match self {
            Self::Single => 0,
            Self::RoundRobin { next } => {
                let partition = (*next as usize) % num_partitions;
                *next = ((*next as usize + 1) % num_partitions) as u16;
                partition
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_wraps_across_partitions() {
        let mut placement = LoadBalance::RoundRobin { next: 0 };
        assert_eq!(placement.choose(2), 0);
        assert_eq!(placement.choose(2), 1);
        assert_eq!(placement.choose(2), 0);
    }
}
