//! FFT overlap-add and overlap-save convolution plans.

use crate::{
    BoundaryPolicy, ConvolutionAlgorithm, ConvolutionCostPlan, ConvolutionMode, ConvolutionPlan,
    ConvolutionResult, SignalError,
    convolution::{fft_product, finish_convolution, validate_real_signal},
    convolution_plan::{linear_full_len, retained_span},
};

/// Frequency-domain blocking method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockConvolutionMethod {
    /// Transform independent input spans and add their overlapping tails.
    OverlapAdd,
    /// Transform overlapping windows and discard their aliased prefixes.
    OverlapSave,
}

/// Explicit plan for a bounded FFT convolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockConvolutionPlan {
    /// Ordinary convolution geometry and normalization.
    pub convolution: ConvolutionPlan,
    /// Overlap-add or overlap-save execution.
    pub method: BlockConvolutionMethod,
    /// Transform length allocated for every block.
    pub fft_len: usize,
}

/// Exact boundary synthesis and final retained span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockBoundaryReport {
    /// Boundary policy inherited from the convolution plan.
    pub policy: BoundaryPolicy,
    /// Zeros materialized before the first input for overlap-save.
    pub left_padding: usize,
    /// Zeros available after the last input to flush the linear tail.
    pub right_padding: usize,
    /// Aliased prefix discarded from every overlap-save transform.
    pub discarded_prefix_per_block: usize,
    /// First full-convolution sample retained by full/same/valid mode.
    pub retained_start: usize,
    /// Exact number of final output samples.
    pub retained_len: usize,
}

/// Inspectable blocked-convolution execution facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockConvolutionReport {
    /// Blocking method used.
    pub method: BlockConvolutionMethod,
    /// Transform length of each block.
    pub fft_len: usize,
    /// New input samples advanced by each block.
    pub input_span_per_block: usize,
    /// Samples retained from each transformed block before the final crop.
    pub retained_span_per_block: usize,
    /// Maximum new input samples buffered before a block can emit.
    pub latency_samples: usize,
    /// Number of transformed blocks.
    pub blocks: usize,
    /// Kernel tail shared by neighboring blocks.
    pub overlap_samples: usize,
    /// Boundary and crop evidence.
    pub boundary: BlockBoundaryReport,
}

/// Canonical samples with both ordinary and blocked execution reports.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockConvolutionResult {
    /// Convolution samples and ordinary cost/span report.
    pub convolution: ConvolutionResult,
    /// Blocking, latency, and boundary report.
    pub blocked: BlockConvolutionReport,
}

/// Executes overlap-add or overlap-save convolution under a fixed FFT ceiling.
pub fn convolve_blocked(
    signal: &[f64],
    kernel: &[f64],
    plan: &BlockConvolutionPlan,
) -> Result<BlockConvolutionResult, SignalError> {
    validate_real_signal(signal)?;
    validate_real_signal(kernel)?;
    let base_cost = plan.convolution.inspect(signal.len(), kernel.len())?;
    if !matches!(plan.convolution.mode, ConvolutionMode::Linear(_)) {
        return Err(SignalError::InvalidPolicy {
            policy: "blocked convolution mode",
            reason: "overlap-add and overlap-save are linear convolution plans",
        });
    }
    if plan.fft_len < kernel.len() {
        return Err(SignalError::InvalidLength {
            len: plan.fft_len,
            reason: "blocked FFT length must be at least the kernel length",
        });
    }
    let retained_per_block = plan.fft_len - kernel.len() + 1;
    let full_len = linear_full_len(signal.len(), kernel.len())?;
    let (raw, blocks) = match plan.method {
        BlockConvolutionMethod::OverlapAdd => {
            overlap_add(signal, kernel, plan.fft_len, retained_per_block, full_len)?
        }
        BlockConvolutionMethod::OverlapSave => {
            overlap_save(signal, kernel, plan.fft_len, retained_per_block, full_len)?
        }
    };
    let cost = blocked_cost(base_cost, plan.fft_len, blocks)?;
    let convolution = finish_convolution(signal.len(), kernel, &plan.convolution, raw, cost)?;
    let (retained_start, retained_len) =
        retained_span(plan.convolution.mode, signal.len(), kernel.len())?;
    let overlap = kernel.len() - 1;
    let (left_padding, discarded_prefix_per_block) = match plan.method {
        BlockConvolutionMethod::OverlapAdd => (0, 0),
        BlockConvolutionMethod::OverlapSave => (overlap, overlap),
    };
    Ok(BlockConvolutionResult {
        convolution,
        blocked: BlockConvolutionReport {
            method: plan.method,
            fft_len: plan.fft_len,
            input_span_per_block: retained_per_block,
            retained_span_per_block: retained_per_block,
            latency_samples: retained_per_block,
            blocks,
            overlap_samples: overlap,
            boundary: BlockBoundaryReport {
                policy: plan.convolution.boundary,
                left_padding,
                right_padding: overlap,
                discarded_prefix_per_block,
                retained_start,
                retained_len,
            },
        },
    })
}

fn overlap_add(
    signal: &[f64],
    kernel: &[f64],
    fft_len: usize,
    input_span: usize,
    full_len: usize,
) -> Result<(Vec<f64>, usize), SignalError> {
    let mut output = vec![0.0; full_len];
    let mut blocks = 0;
    for (block, chunk) in signal.chunks(input_span).enumerate() {
        let transformed = fft_product(chunk, kernel, fft_len)?;
        let start = block * input_span;
        let contribution_len = chunk.len() + kernel.len() - 1;
        for (offset, value) in transformed.into_iter().take(contribution_len).enumerate() {
            output[start + offset] += value;
        }
        blocks += 1;
    }
    validate_real_signal(&output)?;
    Ok((output, blocks))
}

fn overlap_save(
    signal: &[f64],
    kernel: &[f64],
    fft_len: usize,
    retained_span: usize,
    full_len: usize,
) -> Result<(Vec<f64>, usize), SignalError> {
    let overlap = kernel.len() - 1;
    let blocks = full_len.div_ceil(retained_span);
    let mut output = Vec::with_capacity(blocks * retained_span);
    for block in 0..blocks {
        let padded_start = block * retained_span;
        let window = (0..fft_len)
            .map(|offset| {
                let padded_index = padded_start + offset;
                padded_index
                    .checked_sub(overlap)
                    .and_then(|index| signal.get(index))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        let transformed = fft_product(&window, kernel, fft_len)?;
        output.extend(transformed.into_iter().skip(overlap).take(retained_span));
    }
    output.truncate(full_len);
    validate_real_signal(&output)?;
    Ok((output, blocks))
}

fn blocked_cost(
    mut cost: ConvolutionCostPlan,
    fft_len: usize,
    blocks: usize,
) -> Result<ConvolutionCostPlan, SignalError> {
    let stages = usize::try_from(usize::BITS - fft_len.leading_zeros()).unwrap_or(usize::MAX);
    cost.selected = ConvolutionAlgorithm::Fft;
    cost.fft_len = fft_len;
    cost.fft_cost_units = fft_len
        .checked_mul(stages)
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| value.checked_add(fft_len))
        .and_then(|value| value.checked_mul(blocks))
        .ok_or(SignalError::InvalidLength {
            len: fft_len,
            reason: "blocked convolution cost overflowed",
        })?;
    cost.fft_scratch_bytes = fft_len
        .checked_mul(3)
        .and_then(|cells| cells.checked_mul(2 * size_of::<f64>()))
        .ok_or(SignalError::InvalidLength {
            len: fft_len,
            reason: "blocked convolution scratch size overflowed",
        })?;
    Ok(cost)
}
