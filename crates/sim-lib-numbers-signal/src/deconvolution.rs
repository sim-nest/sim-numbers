//! Frequency-domain deconvolution with mandatory singular-bin guards.

use sim_lib_numbers_tensor_f64::F64Tensor;

use crate::{
    SignalError,
    convolution::{direct_circular, direct_linear, validate_real_signal},
    fft::{Complex, fft},
};

/// Geometry used to infer the recovered signal length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeconvolutionMode {
    /// Invert a complete zero-extended linear convolution.
    LinearFull,
    /// Invert a periodic convolution of exactly `period` samples.
    Circular {
        /// Period of the observation, kernel, and recovered signal.
        period: usize,
    },
}

/// Stable spectral inverse applied to every kernel bin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Regularization {
    /// Multiply by the conjugate kernel and divide by `|H|^2 + lambda`.
    Tikhonov {
        /// Positive ridge added to every spectral denominator.
        lambda: f64,
    },
    /// Suppress bins at or below the plan's singular threshold.
    Truncated,
}

/// Explicit guarded-deconvolution policy.
#[derive(Clone, Debug, PartialEq)]
pub struct DeconvolutionPlan {
    /// Linear-full or circular input geometry.
    pub mode: DeconvolutionMode,
    /// Regularized inverse; an unguarded inverse is intentionally absent.
    pub regularization: Regularization,
    /// Kernel-magnitude threshold recorded as singular-bin evidence.
    pub singular_threshold: f64,
}

impl DeconvolutionPlan {
    /// Builds a guarded linear-full Tikhonov plan.
    pub const fn tikhonov(lambda: f64, singular_threshold: f64) -> Self {
        Self {
            mode: DeconvolutionMode::LinearFull,
            regularization: Regularization::Tikhonov { lambda },
            singular_threshold,
        }
    }
}

/// Diagnostics proving how ill-conditioned spectral bins were handled.
#[derive(Clone, Debug, PartialEq)]
pub struct DeconvolutionReport {
    /// Transform length used for the guarded inverse.
    pub fft_len: usize,
    /// Number of recovered samples.
    pub recovered_len: usize,
    /// Regularizer applied to every inverse bin.
    pub regularization: Regularization,
    /// Kernel magnitude at or below which a bin is singular.
    pub singular_threshold: f64,
    /// Exact indices of thresholded or ridge-guarded singular bins.
    pub singular_bins: Vec<usize>,
    /// Smallest observed kernel-bin magnitude.
    pub minimum_kernel_magnitude: f64,
    /// Largest effective inverse gain after regularization.
    pub maximum_inverse_gain: f64,
    /// Euclidean residual between the observation and reconvolved estimate.
    pub residual_l2: f64,
}

/// Canonical recovered samples and finite singular-bin diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct DeconvolutionResult {
    /// Recovered one-dimensional samples.
    pub samples: F64Tensor,
    /// Guard, conditioning, and residual evidence.
    pub report: DeconvolutionReport,
}

/// Deconvolves a real observation without ever dividing by a bare kernel bin.
pub fn deconvolve(
    observation: &[f64],
    kernel: &[f64],
    plan: &DeconvolutionPlan,
) -> Result<DeconvolutionResult, SignalError> {
    validate_real_signal(observation)?;
    validate_real_signal(kernel)?;
    let (recovered_len, fft_len, observation, kernel) = prepare_inputs(observation, kernel, plan)?;
    validate_regularization(plan)?;
    let observation_spectrum = forward_real(&observation, fft_len)?;
    let kernel_spectrum = forward_real(&kernel, fft_len)?;
    let mut singular_bins = Vec::new();
    let mut minimum_kernel_magnitude = f64::INFINITY;
    let mut maximum_inverse_gain = 0.0_f64;
    let mut recovered_spectrum = Vec::with_capacity(fft_len);
    for (index, (observed, kernel)) in observation_spectrum
        .into_iter()
        .zip(kernel_spectrum)
        .enumerate()
    {
        let power = kernel.re * kernel.re + kernel.im * kernel.im;
        let magnitude = power.sqrt();
        minimum_kernel_magnitude = minimum_kernel_magnitude.min(magnitude);
        let singular = magnitude <= plan.singular_threshold;
        if singular {
            singular_bins.push(index);
        }
        let (value, gain) = regularized_bin(observed, kernel, power, singular, plan.regularization);
        maximum_inverse_gain = maximum_inverse_gain.max(gain);
        recovered_spectrum.push(value);
    }
    let inverse = fft(&recovered_spectrum, 1.0)?;
    let scale = 1.0 / fft_len as f64;
    let recovered = inverse
        .into_iter()
        .take(recovered_len)
        .map(|value| value.re * scale)
        .collect::<Vec<_>>();
    validate_real_signal(&recovered)?;
    let residual_l2 = residual_l2(&recovered, &kernel, &observation, plan.mode)?;
    if !minimum_kernel_magnitude.is_finite()
        || !maximum_inverse_gain.is_finite()
        || !residual_l2.is_finite()
    {
        return Err(SignalError::NonFinite {
            index: 0,
            component: "deconvolution diagnostic",
        });
    }
    let samples = F64Tensor::new(vec![recovered.len()], recovered)
        .expect("one-dimensional deconvolution shape matches data");
    Ok(DeconvolutionResult {
        samples,
        report: DeconvolutionReport {
            fft_len,
            recovered_len,
            regularization: plan.regularization,
            singular_threshold: plan.singular_threshold,
            singular_bins,
            minimum_kernel_magnitude,
            maximum_inverse_gain,
            residual_l2,
        },
    })
}

fn prepare_inputs(
    observation: &[f64],
    kernel: &[f64],
    plan: &DeconvolutionPlan,
) -> Result<(usize, usize, Vec<f64>, Vec<f64>), SignalError> {
    if observation.is_empty() || kernel.is_empty() {
        return Err(SignalError::InvalidLength {
            len: observation.len().min(kernel.len()),
            reason: "deconvolution inputs must both be non-empty",
        });
    }
    match plan.mode {
        DeconvolutionMode::LinearFull => {
            if observation.len() < kernel.len() {
                return Err(SignalError::InvalidLength {
                    len: observation.len(),
                    reason: "linear-full observation must be at least as long as the kernel",
                });
            }
            let recovered_len = observation.len() - kernel.len() + 1;
            let fft_len = observation.len().checked_next_power_of_two().ok_or(
                SignalError::InvalidLength {
                    len: observation.len(),
                    reason: "deconvolution FFT length overflowed",
                },
            )?;
            Ok((
                recovered_len,
                fft_len,
                observation.to_vec(),
                kernel.to_vec(),
            ))
        }
        DeconvolutionMode::Circular { period } => {
            if period == 0 || observation.len() != period {
                return Err(SignalError::InvalidLength {
                    len: observation.len(),
                    reason: "circular observation length must equal its nonzero period",
                });
            }
            Ok((
                period,
                period,
                observation.to_vec(),
                fold_periodic(kernel, period),
            ))
        }
    }
}

fn validate_regularization(plan: &DeconvolutionPlan) -> Result<(), SignalError> {
    if !plan.singular_threshold.is_finite() || plan.singular_threshold <= 0.0 {
        return Err(SignalError::InvalidPolicy {
            policy: "singular threshold",
            reason: "deconvolution singular threshold must be finite and positive",
        });
    }
    if let Regularization::Tikhonov { lambda } = plan.regularization
        && (!lambda.is_finite() || lambda <= 0.0)
    {
        return Err(SignalError::InvalidPolicy {
            policy: "Tikhonov lambda",
            reason: "deconvolution lambda must be finite and positive",
        });
    }
    Ok(())
}

fn forward_real(values: &[f64], fft_len: usize) -> Result<Vec<Complex>, SignalError> {
    let mut padded = vec![Complex::ZERO; fft_len];
    for (slot, value) in padded.iter_mut().zip(values) {
        *slot = Complex::new(*value, 0.0);
    }
    fft(&padded, -1.0)
}

fn regularized_bin(
    observed: Complex,
    kernel: Complex,
    power: f64,
    singular: bool,
    regularization: Regularization,
) -> (Complex, f64) {
    let numerator = Complex::new(
        observed.re * kernel.re + observed.im * kernel.im,
        observed.im * kernel.re - observed.re * kernel.im,
    );
    match regularization {
        Regularization::Tikhonov { lambda } => {
            let denominator = power + lambda;
            (
                numerator.scale(1.0 / denominator),
                power.sqrt() / denominator,
            )
        }
        Regularization::Truncated if singular => (Complex::ZERO, 0.0),
        Regularization::Truncated => (numerator.scale(1.0 / power), 1.0 / power.sqrt()),
    }
}

fn residual_l2(
    recovered: &[f64],
    kernel: &[f64],
    observation: &[f64],
    mode: DeconvolutionMode,
) -> Result<f64, SignalError> {
    let predicted = match mode {
        DeconvolutionMode::LinearFull => direct_linear(recovered, kernel)?,
        DeconvolutionMode::Circular { period } => direct_circular(recovered, kernel, period)?,
    };
    Ok(predicted
        .iter()
        .zip(observation)
        .map(|(predicted, observed)| (predicted - observed).powi(2))
        .sum::<f64>()
        .sqrt())
}

fn fold_periodic(values: &[f64], period: usize) -> Vec<f64> {
    let mut folded = vec![0.0; period];
    for (index, value) in values.iter().copied().enumerate() {
        folded[index % period] += value;
    }
    folded
}
