use std::sync::Arc;

use crate::op::Op;
use crate::timing::kernels::{AllReduceKernelInput, MoeAlltoallKernelInput};
use crate::timing::{BuildError, Evaluator, LeafMetrics, PerfApiBridge, Probe, SlotInput};

/// A communication placeholder for K3 v1.
///
/// The production recipe names the collective, but intentionally assigns it
/// zero time until the B200 communication curves are available. Keeping it as
/// a communication-kind leaf lets the analyzer exclude it and lets a later
/// profile fill the same manifest location without changing the recipe.
pub struct ZeroAllReduceProbe {
    pub config: serde_json::Value,
}

impl Probe for ZeroAllReduceProbe {
    type Input = AllReduceKernelInput;

    fn eval(&self, _input: &Self::Input) -> LeafMetrics {
        LeafMetrics::ZERO
    }

    fn kind(&self) -> &'static str {
        "all_reduce"
    }

    fn describe_config(&self) -> serde_json::Value {
        self.config.clone()
    }
}

/// A communication placeholder for K3 v1's expert-parallel exchange.
pub struct ZeroMoeAlltoallProbe {
    pub config: serde_json::Value,
}

impl Probe for ZeroMoeAlltoallProbe {
    type Input = MoeAlltoallKernelInput;

    fn eval(&self, _input: &Self::Input) -> LeafMetrics {
        LeafMetrics::ZERO
    }

    fn kind(&self) -> &'static str {
        "moe_alltoall"
    }

    fn describe_config(&self) -> serde_json::Value {
        self.config.clone()
    }
}

pub fn zero_all_reduce(name: String, config: serde_json::Value) -> Op<ZeroAllReduceProbe> {
    Op::new(name, Arc::new(ZeroAllReduceProbe { config }))
}

pub fn zero_moe_alltoall(name: String, config: serde_json::Value) -> Op<ZeroMoeAlltoallProbe> {
    Op::new(name, Arc::new(ZeroMoeAlltoallProbe { config }))
}

pub fn build_atomic<K, C, F>(
    name: &str,
    suffix: &str,
    config: C,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let full_name = format!("{name}.{suffix}");
    Ok(Op::new(
        full_name.clone(),
        Arc::new(build(full_name, config, bridge)?),
    ))
}

pub fn eval_atomic_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, evaluator: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    if zero {
        evaluator.push(LeafMetrics::ZERO, || input.clone().into());
    } else {
        op.eval(&input, evaluator);
    }
}
