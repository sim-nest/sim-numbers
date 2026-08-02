//! Direct and FFT real convolution execution.

use sim_lib_numbers_tensor_f64::F64Tensor;

use crate::{
    BoundaryPolicy, ConvolutionAlgorithm, ConvolutionCostPlan, ConvolutionMode,
    ConvolutionNormalization, ConvolutionPlan, SignalError,
    convolution_plan::{linear_full_len, retained_span},
    fft::{Complex, fft},
};

/// Auditable facts from one convolution execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvolutionReport {
    /// Cost comparison and selected implementation.
    pub cost: ConvolutionCostPlan,
    /// Complete result length before a linear full/same/valid crop.
    pub full_output_len: usize,
    /// First complete-result sample retained by the requested mode.
    pub retained_start: usize,
    /// Number of returned samples.
    pub retained_len: usize,
    /// Boundary extension used by the computation.
    pub boundary: BoundaryPolicy,
    /// Scaling applied after the convolution sum.
    pub normalization: ConvolutionNormalization,
}

/// Canonical real tensor samples and their convolution report.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvolutionResult {
    /// Returned one-dimensional samples.
    pub samples: F64Tensor,
    /// Inspectable selection, span, and policy evidence.
    pub report: ConvolutionReport,
}

/// Convolves two real signals under an explicit plan.
pub fn convolve(
    signal: &[f64],
    kernel: &[f64],
    plan: &ConvolutionPlan,
) -> Result<ConvolutionResult, SignalError> {
    validate_real_signal(signal)?;
    validate_real_signal(kernel)?;
    let cost = plan.inspect(signal.len(), kernel.len())?;
    let raw = match (plan.mode, cost.selected) {
        (ConvolutionMode::Linear(_), ConvolutionAlgorithm::Direct) => {
            direct_linear(signal, kernel)?
        }
        (ConvolutionMode::Linear(_), ConvolutionAlgorithm::Fft) => {
            let full_len = linear_full_len(signal.len(), kernel.len())?;
            fft_product(signal, kernel, cost.fft_len)?[..full_len].to_vec()
        }
        (ConvolutionMode::Circular { period }, ConvolutionAlgorithm::Direct) => {
            direct_circular(signal, kernel, period)?
        }
        (ConvolutionMode::Circular { period }, ConvolutionAlgorithm::Fft) => {
            let signal = fold_periodic(signal, period);
            let kernel = fold_periodic(kernel, period);
            fft_product(&signal, &kernel, period)?
        }
        (_, ConvolutionAlgorithm::Auto) => unreachable!("inspection resolves automatic plans"),
    };
    finish_convolution(signal.len(), kernel, plan, raw, cost)
}

pub(crate) fn validate_real_signal(values: &[f64]) -> Result<(), SignalError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "value",
            });
        }
    }
    Ok(())
}

pub(crate) fn direct_linear(signal: &[f64], kernel: &[f64]) -> Result<Vec<f64>, SignalError> {
    let mut output = vec![0.0; linear_full_len(signal.len(), kernel.len())?];
    for (signal_index, signal_value) in signal.iter().copied().enumerate() {
        for (kernel_index, kernel_value) in kernel.iter().copied().enumerate() {
            output[signal_index + kernel_index] += signal_value * kernel_value;
        }
    }
    validate_real_signal(&output)?;
    Ok(output)
}

pub(crate) fn direct_circular(
    signal: &[f64],
    kernel: &[f64],
    period: usize,
) -> Result<Vec<f64>, SignalError> {
    let signal = fold_periodic(signal, period);
    let kernel = fold_periodic(kernel, period);
    let mut output = vec![0.0; period];
    for (index, slot) in output.iter_mut().enumerate() {
        for (source, signal_value) in signal.iter().copied().enumerate() {
            let kernel_index = (index + period - source) % period;
            *slot += signal_value * kernel[kernel_index];
        }
    }
    validate_real_signal(&output)?;
    Ok(output)
}

pub(crate) fn fft_product(
    signal: &[f64],
    kernel: &[f64],
    fft_len: usize,
) -> Result<Vec<f64>, SignalError> {
    if fft_len == 0 || signal.len() > fft_len || kernel.len() > fft_len {
        return Err(SignalError::InvalidLength {
            len: fft_len,
            reason: "FFT convolution length must contain both inputs",
        });
    }
    let mut left = vec![Complex::ZERO; fft_len];
    let mut right = vec![Complex::ZERO; fft_len];
    for (slot, value) in left.iter_mut().zip(signal) {
        *slot = Complex::new(*value, 0.0);
    }
    for (slot, value) in right.iter_mut().zip(kernel) {
        *slot = Complex::new(*value, 0.0);
    }
    let left = fft(&left, -1.0)?;
    let right = fft(&right, -1.0)?;
    let product = left
        .into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .collect::<Vec<_>>();
    let inverse = fft(&product, 1.0)?;
    let scale = 1.0 / fft_len as f64;
    let output = inverse
        .into_iter()
        .map(|value| value.re * scale)
        .collect::<Vec<_>>();
    validate_real_signal(&output)?;
    Ok(output)
}

pub(crate) fn finish_convolution(
    signal_len: usize,
    kernel: &[f64],
    plan: &ConvolutionPlan,
    mut raw: Vec<f64>,
    cost: ConvolutionCostPlan,
) -> Result<ConvolutionResult, SignalError> {
    let (retained_start, retained_len) = retained_span(plan.mode, signal_len, kernel.len())?;
    let full_output_len = raw.len();
    let retained_end =
        retained_start
            .checked_add(retained_len)
            .ok_or(SignalError::InvalidLength {
                len: retained_len,
                reason: "retained convolution span overflowed",
            })?;
    if retained_end > raw.len() {
        return Err(SignalError::InvalidLength {
            len: raw.len(),
            reason: "convolution engine returned too few samples",
        });
    }
    raw = raw[retained_start..retained_end].to_vec();
    match plan.normalization {
        ConvolutionNormalization::None => {}
        ConvolutionNormalization::KernelSum => {
            let divisor = kernel.iter().sum::<f64>();
            if !divisor.is_finite() || divisor.abs() <= f64::EPSILON {
                return Err(SignalError::DegenerateNormalization {
                    normalization: "kernel-sum",
                });
            }
            for value in &mut raw {
                *value /= divisor;
            }
        }
    }
    validate_real_signal(&raw)?;
    let samples = F64Tensor::new(vec![raw.len()], raw).expect("one-dimensional shape matches data");
    Ok(ConvolutionResult {
        samples,
        report: ConvolutionReport {
            cost,
            full_output_len,
            retained_start,
            retained_len,
            boundary: plan.boundary,
            normalization: plan.normalization,
        },
    })
}

fn fold_periodic(values: &[f64], period: usize) -> Vec<f64> {
    let mut folded = vec![0.0; period];
    for (index, value) in values.iter().copied().enumerate() {
        folded[index % period] += value;
    }
    folded
}
