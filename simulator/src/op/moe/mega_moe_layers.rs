//! The DeepGEMM mega MoE of every layer a folded body stands for (L2).
//!
//! A mega MoE launch is billed by the rows each physical slot receives, and
//! every layer routes differently and is placed from its own history, so each
//! layer is its own kernel config ([`MegaMoeKernelConfig`] with that layer's
//! demand). A folded body is evaluated once per layer it stands for, so its one
//! leaf carries the mean over those layers: the step then sums every layer's
//! own busiest-rank time exactly, with no layer priced from another's routing.

use std::sync::Arc;

use crate::timing::kernels::{MegaMoeKernel, MegaMoeKernelInput};
use crate::timing::kernels::MegaMoeKernelConfig;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe};

pub struct MegaMoeLayersOp {
    pub name: String,
    /// One kernel per distinct layer config, with the number of layers using it.
    kernels: Vec<(Arc<MegaMoeKernel>, u32)>,
    num_layers: u32,
}

impl MegaMoeLayersOp {
    pub fn build(
        name: String,
        layer_configs: &[MegaMoeKernelConfig],
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        assert!(!layer_configs.is_empty(), "a body stands for at least one layer");
        let mut distinct: Vec<(MegaMoeKernelConfig, u32)> = Vec::new();
        for config in layer_configs {
            match distinct.iter_mut().find(|(seen, _)| seen == config) {
                Some((_, count)) => *count += 1,
                None => distinct.push((config.clone(), 1)),
            }
        }
        let kernels = distinct
            .into_iter()
            .map(|(config, count)| {
                // Every layer's kernel takes the op's role name, so a backend
                // override for the role reaches all of them.
                Ok((Arc::new(MegaMoeKernel::build(name.clone(), config, bridge)?), count))
            })
            .collect::<Result<_, BuildError>>()?;
        Ok(Self {
            name,
            kernels,
            num_layers: layer_configs.len() as u32,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let first = &self.kernels[0].0;
        let mut config = first.describe_config();
        if let serde_json::Value::Object(map) = &mut config {
            map.insert("layers".into(), self.num_layers.into());
            map.insert("distinct_layer_configs".into(), self.kernels.len().into());
        }
        builder.leaf(self.name.clone(), first.kind(), config)
    }

    /// The layers' mean metrics at `input`, as one leaf.
    pub fn eval(&self, input: &MegaMoeKernelInput, ev: &mut Evaluator) {
        let mut mean = LeafMetrics::ZERO;
        let scale = 1.0 / self.num_layers as f32;
        for (kernel, count) in &self.kernels {
            let leaf = kernel.eval(input);
            let w = *count as f32 * scale;
            mean.m.time_ms += leaf.m.time_ms * w;
            mean.m.flops += leaf.m.flops * w;
            mean.m.bytes += leaf.m.bytes * w;
            mean.m.energy_j += leaf.m.energy_j * w;
            mean.coverage |= leaf.coverage;
            if mean.backend_index == LeafMetrics::NO_BACKEND {
                mean.backend_index = leaf.backend_index;
            }
        }
        ev.push(mean, || input.clone().into());
    }

    /// An idle step: the leaf is pushed at zero to keep the slot walk aligned.
    pub fn eval_zero(&self, input: &MegaMoeKernelInput, ev: &mut Evaluator) {
        ev.push(LeafMetrics::ZERO, || input.clone().into());
    }
}
