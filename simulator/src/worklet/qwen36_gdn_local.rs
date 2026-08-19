//! Qwen3.6 TP1 Gated DeltaNet attention section in vLLM launch granularity.
//!
//! This local, self-synchronizing section starts at the decoder's delayed
//! residual-add + input RMSNorm boundary and ends at the post-attention
//! residual-add + RMSNorm boundary. The following MoE router therefore consumes
//! normalized hidden states and owns no norm. There are no TP, EP, collective,
//! or network children.

use std::sync::Arc;

use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::ssm::{GdnDecodeOp, GdnDecodeOpConfig, GdnDecodeOpInput, GdnPrefillOp, GdnPrefillOpConfig, GdnPrefillOpInput};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp8PerTokenGroupQuantKernelConfig, GdnCausalConvDecodeKernelConfig,
    GdnCausalConvPrefillKernelConfig, GdnChunkDeltaRuleKernelConfig, GdnGatedRmsNormKernel,
    GdnGatedRmsNormKernelConfig, GdnGatedRmsNormKernelInput,
    GdnPrefillPostConvKernelConfig, GdnRecurrentDecodeKernelConfig,
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN: u32 = 2048;
const NUM_KEY_HEADS: u32 = 16;
const NUM_VALUE_HEADS: u32 = 32;
const KEY_HEAD_DIM: u32 = 128;
const VALUE_HEAD_DIM: u32 = 128;
const CONV_KERNEL_SIZE: u32 = 4;
const QKVZ_WIDTH: u32 = 12_288;
const BA_WIDTH: u32 = 64;
const CONV_CHANNELS: u32 = 8192;
const CORE_BYTES_PER_TOKEN: u32 = 8192;
const BA_HALF_BYTES_PER_TOKEN: u32 = 64;
const FP8_GROUP_SIZE: u32 = 128;
const SCALE_FORMAT: &str = "ue8m0_column_major";
/// The generic elementwise cache sweeps `num_tokens` through 65,536. Encoding
/// one entire recurrent state as a single token would therefore request a
/// 137-GiB input and output row. State movement is instead expressed in fixed
/// 4-KiB byte-transfer units while preserving the exact logical traffic.
const STATE_TRANSFER_UNIT_BYTES: u32 = 4096;
const STATE_TRANSFER_UNITS_PER_SEQUENCE: u32 = 512;

#[derive(Clone, Debug)]
pub struct Qwen36GdnLocalWorkletConfig {
    pub hidden: Dim,
    pub num_key_heads: Dim,
    pub num_value_heads: Dim,
    pub key_head_dim: Dim,
    pub value_head_dim: Dim,
    pub conv_kernel_size: Dim,
    pub activation_dtype: DType,
    pub conv_state_dtype: DType,
    pub ssm_state_dtype: DType,
    pub gpu_name: String,
    pub residual_rms_norm_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
    pub fp8_gemm_backends: Vec<&'static str>,
    pub bf16_gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub causal_conv_prefill_backends: Vec<&'static str>,
    pub prefill_post_conv_backends: Vec<&'static str>,
    pub chunk_delta_rule_backends: Vec<&'static str>,
    pub causal_conv_decode_backends: Vec<&'static str>,
    pub recurrent_decode_backends: Vec<&'static str>,
    pub gated_rms_norm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Qwen36GdnLocalWorkletResolved {
    pub raw_cfg: Qwen36GdnLocalWorkletConfig,
    pub input_add_rms_norm: ResidualRmsNormKernelConfig,
    pub qkvz: SingleFp8GemmWithQuantConfig,
    pub ba: SingleGemmKernelConfig,
    pub split_b: ElementwiseKernelConfig,
    pub split_a: ElementwiseKernelConfig,
    pub core_output_zero: ElementwiseKernelConfig,
    pub state_gather: ElementwiseKernelConfig,
    pub state_zero: ElementwiseKernelConfig,
    pub prefill: GdnPrefillOpConfig,
    pub state_scatter: ElementwiseKernelConfig,
    pub decode: GdnDecodeOpConfig,
    pub core_output_copy: ElementwiseKernelConfig,
    pub gated_norm: GdnGatedRmsNormKernelConfig,
    pub out_proj: SingleFp8GemmWithQuantConfig,
    pub post_attention_add_rms_norm: ResidualRmsNormKernelConfig,
    pub recurrent_state_bytes_per_sequence: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen36GdnLocalWorkletInput {
    pub prefill_sequence_lengths: Vec<u32>,
    pub prefill_has_initial_state: Vec<bool>,
    pub decode_batch_size: u32,
}

pub struct Qwen36GdnLocalWorklet {
    pub name: String,
    pub input_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub qkvz: SingleFp8GemmWithQuantOp,
    pub ba: Op<SingleGemmKernel>,
    pub split_b: Op<ElementwiseKernel>,
    pub split_a: Op<ElementwiseKernel>,
    pub core_output_zero: Op<ElementwiseKernel>,
    pub state_gather: Op<ElementwiseKernel>,
    pub state_zero: Op<ElementwiseKernel>,
    pub prefill: GdnPrefillOp,
    pub state_scatter: Op<ElementwiseKernel>,
    pub decode: GdnDecodeOp,
    pub core_output_copy: Op<ElementwiseKernel>,
    pub gated_norm: Op<GdnGatedRmsNormKernel>,
    pub out_proj: SingleFp8GemmWithQuantOp,
    pub post_attention_add_rms_norm: Op<ResidualRmsNormKernel>,
    resolved: Qwen36GdnLocalWorkletResolved,
}

impl Qwen36GdnLocalWorklet {
    pub fn resolve_config(
        cfg: &Qwen36GdnLocalWorkletConfig,
    ) -> Qwen36GdnLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Qwen36GdnLocalWorkletConfig: {reason}"));

        let state_bytes = checked_product(
            "recurrent state bytes",
            &[
                cfg.num_value_heads.get(),
                cfg.value_head_dim.get(),
                cfg.key_head_dim.get(),
                cfg.ssm_state_dtype.size_bytes(),
            ],
        )
        .expect("validated recurrent-state byte count must fit u32");
        let encoded_state_bytes = STATE_TRANSFER_UNIT_BYTES
            .checked_mul(STATE_TRANSFER_UNITS_PER_SEQUENCE)
            .expect("state transfer-unit byte count must fit u32");
        assert_eq!(
            state_bytes, encoded_state_bytes,
            "recurrent state must be exactly representable in transfer units"
        );
        let quant = |hidden_size: Dim| Fp8PerTokenGroupQuantKernelConfig {
            backends: cfg.fp8_quant_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size,
            group_size: FP8_GROUP_SIZE,
            input_dtype: cfg.activation_dtype,
            scale_format: SCALE_FORMAT.to_string(),
        };
        let elementwise = |input: u32, output: u32| ElementwiseKernelConfig {
            backends: cfg.elementwise_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            input_bytes_per_token: input.into(),
            output_bytes_per_token: output.into(),
        };
        let norm = || ResidualRmsNormKernelConfig {
            backends: cfg.residual_rms_norm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden: cfg.hidden.clone(),
            dtype: cfg.activation_dtype,
        };

        let prefill = GdnPrefillOpConfig {
            causal_conv: GdnCausalConvPrefillKernelConfig {
                backends: cfg.causal_conv_prefill_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                channels: CONV_CHANNELS.into(),
                kernel_size: cfg.conv_kernel_size.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.conv_state_dtype,
            },
            post_conv: GdnPrefillPostConvKernelConfig {
                backends: cfg.prefill_post_conv_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qk_heads: cfg.num_key_heads.clone(),
                num_value_heads: cfg.num_value_heads.clone(),
                key_head_dim: cfg.key_head_dim.clone(),
                value_head_dim: cfg.value_head_dim.clone(),
                dtype: cfg.activation_dtype,
            },
            chunk_delta_rule: GdnChunkDeltaRuleKernelConfig {
                backends: cfg.chunk_delta_rule_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_key_heads: cfg.num_key_heads.clone(),
                num_heads: cfg.num_value_heads.clone(),
                key_head_dim: cfg.key_head_dim.clone(),
                value_head_dim: cfg.value_head_dim.clone(),
                dtype: cfg.activation_dtype,
            },
        };
        let decode = GdnDecodeOpConfig {
            causal_conv: GdnCausalConvDecodeKernelConfig {
                backends: cfg.causal_conv_decode_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                channels: CONV_CHANNELS.into(),
                kernel_size: cfg.conv_kernel_size.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.conv_state_dtype,
            },
            recurrent: GdnRecurrentDecodeKernelConfig {
                backends: cfg.recurrent_decode_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qk_heads: cfg.num_key_heads.clone(),
                num_value_heads: cfg.num_value_heads.clone(),
                key_head_dim: cfg.key_head_dim.clone(),
                value_head_dim: cfg.value_head_dim.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.ssm_state_dtype,
            },
        };

        Qwen36GdnLocalWorkletResolved {
            input_add_rms_norm: norm(),
            qkvz: SingleFp8GemmWithQuantConfig {
                quant: quant(cfg.hidden.clone()),
                gemm: SingleGemmKernelConfig {
                    backends: cfg.fp8_gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: QKVZ_WIDTH.into(),
                    k: cfg.hidden.clone(),
                    dtype: DType::Fp8E4m3,
                },
            },
            ba: SingleGemmKernelConfig {
                backends: cfg.bf16_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: BA_WIDTH.into(),
                k: cfg.hidden.clone(),
                dtype: cfg.activation_dtype,
            },
            split_b: elementwise(BA_HALF_BYTES_PER_TOKEN, BA_HALF_BYTES_PER_TOKEN),
            split_a: elementwise(BA_HALF_BYTES_PER_TOKEN, BA_HALF_BYTES_PER_TOKEN),
            core_output_zero: elementwise(0, CORE_BYTES_PER_TOKEN),
            state_gather: elementwise(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            state_zero: elementwise(0, STATE_TRANSFER_UNIT_BYTES),
            prefill,
            state_scatter: elementwise(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            decode,
            core_output_copy: elementwise(CORE_BYTES_PER_TOKEN, CORE_BYTES_PER_TOKEN),
            gated_norm: GdnGatedRmsNormKernelConfig {
                backends: cfg.gated_rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.value_head_dim.clone(),
                dtype: cfg.activation_dtype,
            },
            out_proj: SingleFp8GemmWithQuantConfig {
                quant: quant((NUM_VALUE_HEADS * VALUE_HEAD_DIM).into()),
                gemm: SingleGemmKernelConfig {
                    backends: cfg.fp8_gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: cfg.hidden.clone(),
                    k: (NUM_VALUE_HEADS * VALUE_HEAD_DIM).into(),
                    dtype: DType::Fp8E4m3,
                },
            },
            post_attention_add_rms_norm: norm(),
            recurrent_state_bytes_per_sequence: state_bytes,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Qwen36GdnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let input_add_rms_norm = build_atomic(&name, "input_add_rms_norm", resolved.input_add_rms_norm.clone(), ResidualRmsNormKernel::build, bridge)?;
        let qkvz = SingleFp8GemmWithQuantOp::build(format!("{name}.qkvz"), resolved.qkvz.clone(), bridge)?;
        let ba = build_atomic(&name, "ba", resolved.ba.clone(), SingleGemmKernel::build, bridge)?;
        let split_b = build_atomic(&name, "split_b", resolved.split_b.clone(), ElementwiseKernel::build, bridge)?;
        let split_a = build_atomic(&name, "split_a", resolved.split_a.clone(), ElementwiseKernel::build, bridge)?;
        let core_output_zero = build_atomic(&name, "core_output_zero", resolved.core_output_zero.clone(), ElementwiseKernel::build, bridge)?;
        let state_gather = build_atomic(&name, "state_gather", resolved.state_gather.clone(), ElementwiseKernel::build, bridge)?;
        let state_zero = build_atomic(&name, "state_zero", resolved.state_zero.clone(), ElementwiseKernel::build, bridge)?;
        let prefill = GdnPrefillOp::build(format!("{name}.prefill"), resolved.prefill.clone(), bridge)?;
        let state_scatter = build_atomic(&name, "state_scatter", resolved.state_scatter.clone(), ElementwiseKernel::build, bridge)?;
        let decode = GdnDecodeOp::build(format!("{name}.decode"), resolved.decode.clone(), bridge)?;
        let core_output_copy = build_atomic(&name, "core_output_copy", resolved.core_output_copy.clone(), ElementwiseKernel::build, bridge)?;
        let gated_norm = build_atomic(&name, "gated_norm", resolved.gated_norm.clone(), GdnGatedRmsNormKernel::build, bridge)?;
        let out_proj = SingleFp8GemmWithQuantOp::build(format!("{name}.out_proj"), resolved.out_proj.clone(), bridge)?;
        let post_attention_add_rms_norm = build_atomic(&name, "post_attention_add_rms_norm", resolved.post_attention_add_rms_norm.clone(), ResidualRmsNormKernel::build, bridge)?;
        Ok(Self { name, input_add_rms_norm, qkvz, ba, split_b, split_a, core_output_zero, state_gather, state_zero, prefill, state_scatter, decode, core_output_copy, gated_norm, out_proj, post_attention_add_rms_norm, resolved })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Qwen36GdnLocalWorklet) [local (1 GPU); Hg={}, H={}, K={}, V={}]",
                self.name,
                cfg.num_key_heads,
                cfg.num_value_heads,
                cfg.key_head_dim,
                cfg.value_head_dim,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_add_rms_norm.compile(builder), self.qkvz.compile(builder), self.ba.compile(builder),
                self.split_b.compile(builder), self.split_a.compile(builder), self.core_output_zero.compile(builder),
                self.state_gather.compile(builder), self.state_zero.compile(builder), self.prefill.compile(builder),
                self.state_scatter.compile(builder), self.decode.compile(builder), self.core_output_copy.compile(builder),
                self.gated_norm.compile(builder), self.out_proj.compile(builder), self.post_attention_add_rms_norm.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Qwen36GdnLocalWorkletInput, ev: &mut Evaluator) {
        let work = derive_work(input).unwrap_or_else(|reason| panic!("invalid Qwen36GdnLocalWorkletInput: {reason}"));
        eval_atomic_or_zero(&self.input_add_rms_norm, work.input_add_rms_norm, false, ev);
        self.qkvz.eval(&work.qkvz, ev);
        eval_atomic_or_zero(&self.ba, work.ba, false, ev);
        eval_atomic_or_zero(&self.split_b, work.split_b, false, ev);
        eval_atomic_or_zero(&self.split_a, work.split_a, false, ev);
        eval_atomic_or_zero(&self.core_output_zero, work.core_output_zero, false, ev);
        eval_atomic_or_zero(&self.state_gather, work.state_gather, work.num_prefill == 0, ev);
        eval_atomic_or_zero(&self.state_zero, work.state_zero, work.num_fresh == 0, ev);
        self.prefill.eval(&work.prefill, ev);
        eval_atomic_or_zero(&self.state_scatter, work.state_scatter, work.num_prefill == 0, ev);
        self.decode.eval(&work.decode, ev);
        eval_atomic_or_zero(&self.core_output_copy, work.core_output_copy, false, ev);
        eval_atomic_or_zero(&self.gated_norm, work.gated_norm, false, ev);
        self.out_proj.eval(&work.out_proj, ev);
        eval_atomic_or_zero(&self.post_attention_add_rms_norm, work.post_attention_add_rms_norm, false, ev);
    }
}

struct WorkInputs {
    input_add_rms_norm: ResidualRmsNormKernelInput,
    qkvz: SingleFp8GemmWithQuantInput,
    ba: SingleGemmKernelInput,
    split_b: ElementwiseKernelInput,
    split_a: ElementwiseKernelInput,
    core_output_zero: ElementwiseKernelInput,
    state_gather: ElementwiseKernelInput,
    state_zero: ElementwiseKernelInput,
    prefill: GdnPrefillOpInput,
    state_scatter: ElementwiseKernelInput,
    decode: GdnDecodeOpInput,
    core_output_copy: ElementwiseKernelInput,
    gated_norm: GdnGatedRmsNormKernelInput,
    out_proj: SingleFp8GemmWithQuantInput,
    post_attention_add_rms_norm: ResidualRmsNormKernelInput,
    num_prefill: u32,
    num_fresh: u32,
    #[cfg(test)] num_chunks: u32,
    #[cfg(test)] max_chunks: u32,
}

fn derive_work(input: &Qwen36GdnLocalWorkletInput) -> Result<WorkInputs, String> {
    if input.prefill_sequence_lengths.len() != input.prefill_has_initial_state.len() {
        return Err("prefill lengths and initial-state flags must have equal length".into());
    }
    let mut prefill_tokens = 0_u64;
    let mut chunks = 0_u64;
    let mut max_chunks = 0_u64;
    for &length in &input.prefill_sequence_lengths {
        if length == 0 { return Err("prefill sequence lengths must be positive".into()); }
        prefill_tokens = prefill_tokens.checked_add(u64::from(length)).ok_or("prefill token sum overflow")?;
        let count = u64::from(length).checked_add(63).ok_or("chunk rounding overflow")? / 64;
        chunks = chunks.checked_add(count).ok_or("chunk-count sum overflow")?;
        max_chunks = max_chunks.max(count);
    }
    let total = prefill_tokens.checked_add(u64::from(input.decode_batch_size)).ok_or("total-token sum overflow")?;
    if total == 0 { return Err("total token count must be positive".into()); }
    let total = u32::try_from(total).map_err(|_| "total token count exceeds u32")?;
    let num_prefill = u32::try_from(input.prefill_sequence_lengths.len()).map_err(|_| "prefill count exceeds u32")?;
    let num_fresh = u32::try_from(input.prefill_has_initial_state.iter().filter(|&&has| !has).count()).map_err(|_| "fresh-prefill count exceeds u32")?;
    // For these three generic elementwise leaves, `num_tokens` counts 4-KiB
    // byte-transfer units, not model tokens or requests. This preserves exactly
    // 2 MiB of recurrent-state traffic per sequence while keeping the cache's
    // standard token sweep allocation-safe on one H200.
    let prefill_state_units = state_transfer_units(num_prefill)?;
    let fresh_state_units = state_transfer_units(num_fresh)?;
    let norm_rows = total.checked_mul(NUM_VALUE_HEADS).ok_or("gated norm row count overflow")?;
    let mixed = prefill_tokens > 0 && input.decode_batch_size > 0;
    let copy_tokens = if mixed { total.checked_mul(2).ok_or("mixed core-copy count overflow")? } else { total };
    Ok(WorkInputs {
        input_add_rms_norm: ResidualRmsNormKernelInput { m: total },
        qkvz: SingleFp8GemmWithQuantInput { num_tokens: total },
        ba: SingleGemmKernelInput { m: total },
        split_b: ElementwiseKernelInput { num_tokens: total },
        split_a: ElementwiseKernelInput { num_tokens: total },
        core_output_zero: ElementwiseKernelInput { num_tokens: total },
        state_gather: ElementwiseKernelInput { num_tokens: prefill_state_units },
        state_zero: ElementwiseKernelInput { num_tokens: fresh_state_units },
        prefill: GdnPrefillOpInput { sequence_lengths: input.prefill_sequence_lengths.clone() },
        state_scatter: ElementwiseKernelInput { num_tokens: prefill_state_units },
        decode: GdnDecodeOpInput { batch_size: input.decode_batch_size },
        core_output_copy: ElementwiseKernelInput { num_tokens: copy_tokens },
        gated_norm: GdnGatedRmsNormKernelInput { m: norm_rows },
        out_proj: SingleFp8GemmWithQuantInput { num_tokens: total },
        post_attention_add_rms_norm: ResidualRmsNormKernelInput { m: total },
        num_prefill, num_fresh,
        #[cfg(test)] num_chunks: u32::try_from(chunks).map_err(|_| "chunk count exceeds u32")?,
        #[cfg(test)] max_chunks: u32::try_from(max_chunks).map_err(|_| "maximum chunk count exceeds u32")?,
    })
}

fn validate_config(cfg: &Qwen36GdnLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, expected) in [
        ("hidden", cfg.hidden.get(), HIDDEN), ("num_key_heads", cfg.num_key_heads.get(), NUM_KEY_HEADS),
        ("num_value_heads", cfg.num_value_heads.get(), NUM_VALUE_HEADS), ("key_head_dim", cfg.key_head_dim.get(), KEY_HEAD_DIM),
        ("value_head_dim", cfg.value_head_dim.get(), VALUE_HEAD_DIM), ("conv_kernel_size", cfg.conv_kernel_size.get(), CONV_KERNEL_SIZE),
    ] { if actual != expected { return Err(format!("{name} must be {expected}, got {actual}")); } }
    if cfg.activation_dtype != DType::Bf16 { return Err("activation_dtype must be BF16".into()); }
    if cfg.conv_state_dtype != DType::Bf16 { return Err("conv_state_dtype must be BF16".into()); }
    if cfg.ssm_state_dtype != DType::Fp32 { return Err("ssm_state_dtype must be FP32".into()); }
    Ok(())
}

fn checked_product(label: &str, values: &[u32]) -> Result<u32, String> {
    values.iter().try_fold(1_u32, |product, &value| product.checked_mul(value).ok_or_else(|| format!("{label} overflow")))
}

fn state_transfer_units(sequence_count: u32) -> Result<u32, String> {
    sequence_count
        .checked_mul(STATE_TRANSFER_UNITS_PER_SEQUENCE)
        .ok_or_else(|| "state transfer-unit count overflow".to_string())
}

fn build_atomic<K, C, F>(parent: &str, suffix: &str, config: C, build: F, bridge: &PerfApiBridge) -> Result<Op<K>, BuildError>
where K: Probe, F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError> {
    let name = format!("{parent}.{suffix}");
    Ok(Op::new(name.clone(), Arc::new(build(name, config, bridge)?)))
}

fn eval_atomic_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where K: Probe, K::Input: Clone + Into<SlotInput> {
    let metrics = if zero { LeafMetrics::ZERO } else { op.kernel.eval(&input) };
    ev.push(metrics, || input.clone().into());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::{CostTreeBuilder, PerfApiBridge};

    fn cfg() -> Qwen36GdnLocalWorkletConfig {
        Qwen36GdnLocalWorkletConfig {
            hidden: HIDDEN.into(), num_key_heads: NUM_KEY_HEADS.into(), num_value_heads: NUM_VALUE_HEADS.into(),
            key_head_dim: KEY_HEAD_DIM.into(), value_head_dim: VALUE_HEAD_DIM.into(), conv_kernel_size: CONV_KERNEL_SIZE.into(),
            activation_dtype: DType::Bf16, conv_state_dtype: DType::Bf16, ssm_state_dtype: DType::Fp32,
            gpu_name: "NVIDIA H200".into(), residual_rms_norm_backends: vec!["vllm_cuda"],
            fp8_quant_backends: vec!["vllm_cuda"], fp8_gemm_backends: vec!["deepgemm"], bf16_gemm_backends: vec!["torch_linear"],
            elementwise_backends: vec!["triton"], causal_conv_prefill_backends: vec!["vllm_triton"],
            prefill_post_conv_backends: vec!["vllm_triton"],
            chunk_delta_rule_backends: vec!["flashinfer"],
            causal_conv_decode_backends: vec!["vllm_triton"], recurrent_decode_backends: vec!["vllm_triton"], gated_rms_norm_backends: vec!["vllm_triton"],
        }
    }

    fn enumerate_op() -> Qwen36GdnLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test(); bridge.enable_enumerate();
        Qwen36GdnLocalWorklet::build("model.gdn".into(), Qwen36GdnLocalWorklet::resolve_config(&cfg()), &bridge).unwrap()
    }

    #[test]
    fn resolution_freezes_projection_shapes_dtypes_and_backends() {
        let r = Qwen36GdnLocalWorklet::resolve_config(&cfg());
        assert_eq!((r.qkvz.quant.hidden_size.get(), r.qkvz.gemm.k.get(), r.qkvz.gemm.n.get()), (2048, 2048, 12288));
        assert_eq!(r.qkvz.quant.group_size, 128); assert_eq!(r.qkvz.quant.input_dtype, DType::Bf16); assert_eq!(r.qkvz.quant.scale_format, SCALE_FORMAT);
        assert_eq!(r.qkvz.gemm.dtype, DType::Fp8E4m3); assert_eq!(r.qkvz.gemm.backends, ["deepgemm"]);
        assert_eq!((r.ba.k.get(), r.ba.n.get(), r.ba.dtype), (2048, 64, DType::Bf16)); assert_eq!(r.ba.backends, ["torch_linear"]);
        assert_eq!((r.out_proj.quant.hidden_size.get(), r.out_proj.gemm.k.get(), r.out_proj.gemm.n.get()), (4096, 4096, 2048));
        assert_eq!(r.out_proj.gemm.dtype, DType::Fp8E4m3);
        assert_eq!((r.input_add_rms_norm.hidden.get(), r.post_attention_add_rms_norm.hidden.get()), (2048, 2048));
    }

    #[test]
    fn resolution_freezes_bytes_and_state_dtypes() {
        let r = Qwen36GdnLocalWorklet::resolve_config(&cfg());
        let rates = |c: &ElementwiseKernelConfig| (c.input_bytes_per_token.get(), c.output_bytes_per_token.get());
        assert_eq!(rates(&r.split_b), (64,64)); assert_eq!(rates(&r.split_a), (64,64)); assert_eq!(rates(&r.core_output_zero), (0,8192));
        assert_eq!(STATE_TRANSFER_UNIT_BYTES * STATE_TRANSFER_UNITS_PER_SEQUENCE, 2_097_152);
        assert_eq!(r.recurrent_state_bytes_per_sequence, 2_097_152); assert_eq!(rates(&r.state_gather), (4096,4096));
        assert_eq!(rates(&r.state_zero), (0,4096)); assert_eq!(rates(&r.state_scatter), (4096,4096)); assert_eq!(rates(&r.core_output_copy), (8192,8192));
        assert_eq!(r.prefill.causal_conv.state_dtype, DType::Bf16); assert_eq!(r.decode.causal_conv.state_dtype, DType::Bf16);
        assert_eq!(r.prefill.chunk_delta_rule.dtype, DType::Bf16); assert_eq!(r.prefill.chunk_delta_rule.backends, ["flashinfer"]); assert_eq!(r.prefill.chunk_delta_rule.gpu_name, "NVIDIA H200");
        assert_eq!(r.decode.recurrent.state_dtype, DType::Fp32); assert_eq!(r.gated_norm.hidden.get(), 128);
    }

    #[test]
    fn invalid_identity_and_state_relationships_are_rejected() {
        let mut bad = cfg(); bad.hidden = 4096.into(); assert!(std::panic::catch_unwind(|| Qwen36GdnLocalWorklet::resolve_config(&bad)).is_err());
        let mut bad = cfg(); bad.conv_state_dtype = DType::Fp32; assert!(std::panic::catch_unwind(|| Qwen36GdnLocalWorklet::resolve_config(&bad)).is_err());
        let mut bad = cfg(); bad.ssm_state_dtype = DType::Bf16; assert!(std::panic::catch_unwind(|| Qwen36GdnLocalWorklet::resolve_config(&bad)).is_err());
    }

    #[test]
    fn runtime_geometry_covers_decode_prefill_mixed_and_ragged() {
        let decode = derive_work(&Qwen36GdnLocalWorkletInput { decode_batch_size: 7, ..Default::default() }).unwrap();
        assert_eq!((decode.qkvz.num_tokens, decode.num_prefill, decode.num_fresh, decode.core_output_copy.num_tokens), (7,0,0,7)); assert_eq!(decode.gated_norm.m, 224);
        let prefill = derive_work(&Qwen36GdnLocalWorkletInput { prefill_sequence_lengths: vec![128], prefill_has_initial_state: vec![true], decode_batch_size: 0 }).unwrap();
        assert_eq!((prefill.qkvz.num_tokens, prefill.num_chunks, prefill.max_chunks, prefill.core_output_copy.num_tokens), (128,2,2,128));
        assert_eq!((prefill.state_gather.num_tokens, prefill.state_zero.num_tokens, prefill.state_scatter.num_tokens), (512,0,512));
        let mixed = derive_work(&Qwen36GdnLocalWorkletInput { prefill_sequence_lengths: vec![3,65,2], prefill_has_initial_state: vec![false,true,false], decode_batch_size: 5 }).unwrap();
        assert_eq!((mixed.prefill.sequence_lengths.iter().sum::<u32>(), mixed.num_prefill, mixed.num_fresh, mixed.num_chunks, mixed.max_chunks), (70,3,2,4,2));
        assert_eq!((mixed.qkvz.num_tokens, mixed.core_output_copy.num_tokens, mixed.gated_norm.m), (75,150,2400));
        assert_eq!((mixed.state_gather.num_tokens, mixed.state_zero.num_tokens, mixed.state_scatter.num_tokens), (1536,1024,1536));
    }

    #[test]
    fn state_transfer_units_preserve_exact_semantic_bytes_and_check_overflow() {
        let resolved = Qwen36GdnLocalWorklet::resolve_config(&cfg());
        let rates = |config: &ElementwiseKernelConfig| {
            (
                u64::from(config.input_bytes_per_token.get()),
                u64::from(config.output_bytes_per_token.get()),
            )
        };
        for sequence_count in [0, 1, 3] {
            let units = state_transfer_units(sequence_count).unwrap();
            assert_eq!(units, sequence_count * STATE_TRANSFER_UNITS_PER_SEQUENCE);
            let semantic_bytes = u64::from(sequence_count) * 2_097_152;
            assert_eq!(u64::from(units) * u64::from(STATE_TRANSFER_UNIT_BYTES), semantic_bytes);
            let (gather_input, gather_output) = rates(&resolved.state_gather);
            let (zero_input, zero_output) = rates(&resolved.state_zero);
            let (scatter_input, scatter_output) = rates(&resolved.state_scatter);
            assert_eq!((u64::from(units) * gather_input, u64::from(units) * gather_output), (semantic_bytes, semantic_bytes));
            assert_eq!((u64::from(units) * zero_input, u64::from(units) * zero_output), (0, semantic_bytes));
            assert_eq!((u64::from(units) * scatter_input, u64::from(units) * scatter_output), (semantic_bytes, semantic_bytes));
        }
        assert_eq!(state_transfer_units(0).unwrap(), 0);
        assert_eq!(state_transfer_units(1).unwrap(), 512);
        assert_eq!(state_transfer_units(3).unwrap(), 1536);
        assert!(state_transfer_units(u32::MAX).is_err());
    }

    #[test]
    fn runtime_validation_rejects_malformed_and_overflowing_inputs() {
        assert!(derive_work(&Qwen36GdnLocalWorkletInput::default()).is_err());
        assert!(derive_work(&Qwen36GdnLocalWorkletInput { prefill_sequence_lengths: vec![1], prefill_has_initial_state: vec![], decode_batch_size: 0 }).is_err());
        assert!(derive_work(&Qwen36GdnLocalWorkletInput { prefill_sequence_lengths: vec![0], prefill_has_initial_state: vec![false], decode_batch_size: 1 }).is_err());
        assert!(derive_work(&Qwen36GdnLocalWorkletInput { prefill_sequence_lengths: vec![u32::MAX], prefill_has_initial_state: vec![true], decode_batch_size: 1 }).is_err());
        assert!(derive_work(&Qwen36GdnLocalWorkletInput { decode_batch_size: u32::MAX, ..Default::default() }).is_err());
    }

    #[test]
    fn compile_has_exact_fifteen_children_and_twenty_flattened_leaves() {
        let op = enumerate_op(); let mut builder = CostTreeBuilder::new(); let root = op.compile(&mut builder); let tree = builder.finish(root);
        let expected = [
            "input_add_rms_norm","qkvz.input_quant","qkvz.gemm","ba","split_b","split_a","core_output_zero","state_gather","state_zero",
            "prefill.causal_conv","prefill.post_conv","prefill.chunk_delta_rule",
            "state_scatter","decode.causal_conv","decode.recurrent","core_output_copy","gated_norm","out_proj.input_quant","out_proj.gemm","post_attention_add_rms_norm"
        ];
        assert_eq!(tree.n_slots(), 20);
        assert_eq!(tree.slots.iter().map(|s| s.name.strip_prefix("model.gdn.").unwrap()).collect::<Vec<_>>(), expected);
        assert_eq!(tree.slots.first().unwrap().kind, "residual_rms_norm"); assert_eq!(tree.slots.last().unwrap().kind, "residual_rms_norm");
        assert!(!tree.slots.iter().any(|slot| matches!(slot.kind.as_str(), "all_reduce" | "all_to_all" | "send_recv")));
        if let CostNode::Labeled { child, .. } = tree.root { if let CostNode::Sum(children) = *child { assert_eq!(children.len(), 15); } else { panic!("expected Sum"); } } else { panic!("expected Labeled"); }
    }

    #[test]
    fn compiled_configs_preserve_all_backend_roles() {
        let op = enumerate_op();
        assert_eq!(op.input_add_rms_norm.kernel.config.backends, ["vllm_cuda"]); assert_eq!(op.ba.kernel.config.backends, ["torch_linear"]);
        assert_eq!(op.split_b.kernel.config.backends, ["triton"]); assert_eq!(op.prefill.causal_conv.config.backends, ["vllm_triton"]);
        assert_eq!(op.prefill.chunk_delta_rule.config.dtype, DType::Bf16); assert_eq!(op.prefill.chunk_delta_rule.config.backends, ["flashinfer"]); assert_eq!(op.prefill.chunk_delta_rule.config.gpu_name, "NVIDIA H200");
        assert_eq!(op.decode.recurrent.config.backends, ["vllm_triton"]); assert_eq!(op.gated_norm.kernel.config.backends, ["vllm_triton"]);
    }

    struct MustNotEval;

    impl Probe for MustNotEval {
        type Input = ElementwiseKernelInput;

        fn eval(&self, _input: &Self::Input) -> LeafMetrics {
            panic!("zero-count placeholder must not evaluate its kernel")
        }

        fn kind(&self) -> &'static str { "elementwise" }

        fn describe_config(&self) -> serde_json::Value { serde_json::json!({}) }
    }

    #[test]
    fn conditional_atomic_zero_records_faithful_typed_input_without_evaluation() {
        let op = Op::new("zero".into(), Arc::new(MustNotEval));
        let mut metrics = [LeafMetrics::MISS];
        let mut inputs = Vec::new();
        let mut evaluator = Evaluator::with_inputs(&mut metrics, &mut inputs);
        eval_atomic_or_zero(
            &op,
            ElementwiseKernelInput { num_tokens: 0 },
            true,
            &mut evaluator,
        );
        assert_eq!(evaluator.filled(), 1);
        assert_eq!(metrics[0].m.time_ms, 0.0);
        assert_eq!(metrics[0].m.flops, 0.0);
        assert_eq!(metrics[0].m.bytes, 0.0);
        assert_eq!(metrics[0].m.energy_j, 0.0);
        assert_eq!(metrics[0].coverage, CoverageFlags::EMPTY);
        assert_eq!(
            serde_json::to_value(inputs).unwrap(),
            serde_json::json!([{"num_tokens": 0}])
        );
    }
}
