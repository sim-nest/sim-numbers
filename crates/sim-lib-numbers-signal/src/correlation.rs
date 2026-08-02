//! Cross-correlation with explicit lag and normalization conventions.

use sim_lib_numbers_tensor_f64::F64Tensor;

use crate::{
    BoundaryPolicy, ConvolutionAlgorithm, ConvolutionMode, ConvolutionNormalization,
    ConvolutionPlan, ConvolutionReport, LinearOutput, SignalError,
    convolution::{convolve, validate_real_signal},
};

/// Ordering of lags in a correlation result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LagOrder {
    /// Smallest signed lag first.
    Ascending,
    /// Largest signed lag first.
    Descending,
}

/// Scaling applied to cross-correlation sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrelationNormalization {
    /// Preserve raw product sums.
    None,
    /// Divide by the larger input length.
    Biased,
    /// Divide each lag by its exact number of overlapping sample pairs.
    Unbiased,
    /// Divide by the product of the two full-signal Euclidean norms.
    Coefficient,
}

/// Fully explicit real cross-correlation plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorrelationPlan {
    /// Linear retained span or circular period.
    pub mode: ConvolutionMode,
    /// Direct, FFT, or automatic execution.
    pub algorithm: ConvolutionAlgorithm,
    /// Zero or periodic boundary extension.
    pub boundary: BoundaryPolicy,
    /// Correlation scaling convention.
    pub normalization: CorrelationNormalization,
    /// Returned lag ordering.
    pub lag_order: LagOrder,
}

impl CorrelationPlan {
    /// Conventional full, ascending-lag, unnormalized automatic plan.
    pub const fn linear_full() -> Self {
        Self {
            mode: ConvolutionMode::Linear(LinearOutput::Full),
            algorithm: ConvolutionAlgorithm::Auto,
            boundary: BoundaryPolicy::ZeroPad,
            normalization: CorrelationNormalization::None,
            lag_order: LagOrder::Ascending,
        }
    }
}

/// Canonical samples, signed lags, and convolution execution evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrelationResult {
    /// Cross-correlation samples in [`lags`](Self::lags) order.
    pub samples: F64Tensor,
    /// Signed displacement represented by each sample.
    pub lags: Vec<isize>,
    /// Underlying convolution cost, boundary, and retained-span report.
    pub convolution: ConvolutionReport,
    /// Scaling applied to the product sums.
    pub normalization: CorrelationNormalization,
    /// Ordering applied to both samples and lags.
    pub lag_order: LagOrder,
}

/// Computes real cross-correlation as convolution with a policy-correct reversal.
pub fn correlate(
    left: &[f64],
    right: &[f64],
    plan: &CorrelationPlan,
) -> Result<CorrelationResult, SignalError> {
    validate_real_signal(left)?;
    validate_real_signal(right)?;
    let kernel = correlation_kernel(right, plan.mode);
    let convolution_plan = ConvolutionPlan {
        mode: plan.mode,
        algorithm: plan.algorithm,
        boundary: plan.boundary,
        normalization: ConvolutionNormalization::None,
    };
    let result = convolve(left, &kernel, &convolution_plan)?;
    let mut samples = result.samples.as_slice().to_vec();
    let mut lags = correlation_lags(
        plan.mode,
        left.len(),
        right.len(),
        result.report.retained_start,
        result.report.retained_len,
    )?;
    normalize(
        &mut samples,
        &lags,
        left,
        right,
        plan.mode,
        plan.normalization,
    )?;
    if plan.lag_order == LagOrder::Descending {
        samples.reverse();
        lags.reverse();
    }
    let samples =
        F64Tensor::new(vec![samples.len()], samples).expect("correlation shape matches data");
    Ok(CorrelationResult {
        samples,
        lags,
        convolution: result.report,
        normalization: plan.normalization,
        lag_order: plan.lag_order,
    })
}

fn correlation_kernel(right: &[f64], mode: ConvolutionMode) -> Vec<f64> {
    match mode {
        ConvolutionMode::Linear(_) => right.iter().rev().copied().collect(),
        ConvolutionMode::Circular { period } => {
            let folded = fold_periodic(right, period);
            (0..period)
                .map(|index| folded[(period - index) % period])
                .collect()
        }
    }
}

fn correlation_lags(
    mode: ConvolutionMode,
    _left_len: usize,
    right_len: usize,
    retained_start: usize,
    retained_len: usize,
) -> Result<Vec<isize>, SignalError> {
    match mode {
        ConvolutionMode::Linear(_) => {
            let offset =
                isize::try_from(right_len - 1).map_err(|_| SignalError::InvalidLength {
                    len: right_len,
                    reason: "correlation lag does not fit isize",
                })?;
            (retained_start..retained_start + retained_len)
                .map(|index| {
                    isize::try_from(index)
                        .map(|index| index - offset)
                        .map_err(|_| SignalError::InvalidLength {
                            len: index,
                            reason: "correlation lag does not fit isize",
                        })
                })
                .collect()
        }
        ConvolutionMode::Circular { period } => (0..period)
            .map(|lag| {
                isize::try_from(lag).map_err(|_| SignalError::InvalidLength {
                    len: lag,
                    reason: "correlation lag does not fit isize",
                })
            })
            .collect(),
    }
}

fn normalize(
    samples: &mut [f64],
    lags: &[isize],
    left: &[f64],
    right: &[f64],
    mode: ConvolutionMode,
    normalization: CorrelationNormalization,
) -> Result<(), SignalError> {
    match normalization {
        CorrelationNormalization::None => {}
        CorrelationNormalization::Biased => {
            let divisor = match mode {
                ConvolutionMode::Linear(_) => left.len().max(right.len()),
                ConvolutionMode::Circular { period } => period,
            } as f64;
            for value in samples.iter_mut() {
                *value /= divisor;
            }
        }
        CorrelationNormalization::Unbiased => {
            for (value, lag) in samples.iter_mut().zip(lags) {
                let overlap = match mode {
                    ConvolutionMode::Linear(_) => overlap_count(left.len(), right.len(), *lag),
                    ConvolutionMode::Circular { period } => period,
                };
                if overlap == 0 {
                    return Err(SignalError::DegenerateNormalization {
                        normalization: "unbiased-correlation",
                    });
                }
                *value /= overlap as f64;
            }
        }
        CorrelationNormalization::Coefficient => {
            let (left, right) = match mode {
                ConvolutionMode::Linear(_) => (left.to_vec(), right.to_vec()),
                ConvolutionMode::Circular { period } => {
                    (fold_periodic(left, period), fold_periodic(right, period))
                }
            };
            let left_energy = left.iter().map(|value| value * value).sum::<f64>();
            let right_energy = right.iter().map(|value| value * value).sum::<f64>();
            let divisor = (left_energy * right_energy).sqrt();
            if !divisor.is_finite() || divisor <= f64::EPSILON {
                return Err(SignalError::DegenerateNormalization {
                    normalization: "coefficient-correlation",
                });
            }
            for value in samples.iter_mut() {
                *value /= divisor;
            }
        }
    }
    validate_real_signal(samples)
}

fn overlap_count(left_len: usize, right_len: usize, lag: isize) -> usize {
    let left_len = left_len as i128;
    let right_len = right_len as i128;
    let lag = lag as i128;
    let start = 0_i128.max(lag);
    let end = left_len.min(right_len + lag);
    usize::try_from((end - start).max(0)).unwrap_or(0)
}

fn fold_periodic(values: &[f64], period: usize) -> Vec<f64> {
    let mut folded = vec![0.0; period];
    for (index, value) in values.iter().copied().enumerate() {
        folded[index % period] += value;
    }
    folded
}
