//! Hilbert analytic signals, unwrapped phase, instantaneous frequency, and envelopes.

use std::f64::consts::{PI, TAU};

use crate::{
    Normalization, SignConvention, SignalError,
    fft::{Complex, fft},
    interpolate::{forward_scale, reconstruction_scale},
};

/// Fourier convention and resource limits for analytic-signal construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticSignalPlan {
    /// Coefficient scaling used internally and retained in the report.
    pub normalization: Normalization,
    /// Forward/inverse exponential sign pair.
    pub sign: SignConvention,
    /// Largest admitted signal length.
    pub max_len: usize,
    /// Conservative transform work ceiling.
    pub max_work: u64,
}

impl Default for AnalyticSignalPlan {
    fn default() -> Self {
        Self {
            normalization: Normalization::Inverse,
            sign: SignConvention::NegativeForward,
            max_len: 1_048_576,
            max_work: 1_000_000_000,
        }
    }
}

/// Convention and work evidence for a discrete Hilbert construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticSignalReport {
    /// Number of real input samples and complex output samples.
    pub len: usize,
    /// Internal coefficient normalization.
    pub normalization: Normalization,
    /// Internal Fourier sign pair.
    pub sign: SignConvention,
    /// Whether the even-length Nyquist bin was retained without doubling.
    pub retained_nyquist: bool,
    /// Conservative work charged by the construction.
    pub work_units: u64,
    /// Work ceiling that admitted the construction.
    pub work_limit: u64,
}

/// Complex analytic samples and their explicit Hilbert-transform evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalyticSignal {
    /// `(real, Hilbert imaginary)` samples.
    pub samples: Vec<(f64, f64)>,
    /// Transform and work evidence.
    pub report: AnalyticSignalReport,
}

/// Constructs the discrete analytic signal by zeroing negative-frequency bins.
pub fn analytic_signal(
    samples: &[f64],
    plan: &AnalyticSignalPlan,
) -> Result<AnalyticSignal, SignalError> {
    if samples.is_empty() || samples.len() > plan.max_len {
        return Err(SignalError::InvalidLength {
            len: samples.len(),
            reason: "analytic signal length must be nonzero and within the plan limit",
        });
    }
    for (index, value) in samples.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "value",
            });
        }
    }
    let work_units = analytic_work(samples.len())?;
    if work_units > plan.max_work {
        return Err(SignalError::WorkLimit {
            required: work_units,
            maximum: plan.max_work,
        });
    }
    let mut bins = fft(
        &samples
            .iter()
            .map(|value| Complex::new(*value, 0.0))
            .collect::<Vec<_>>(),
        plan.sign.angle_sign(crate::Direction::Forward),
    )?;
    let coefficient_scale = forward_scale(plan.normalization, samples.len());
    for bin in &mut bins {
        *bin = bin.scale(coefficient_scale);
    }
    let positive_end = samples.len().div_ceil(2);
    for bin in bins.iter_mut().take(positive_end).skip(1) {
        *bin = bin.scale(2.0);
    }
    for bin in bins.iter_mut().skip(samples.len() / 2 + 1) {
        *bin = Complex::ZERO;
    }
    let output_scale = reconstruction_scale(plan.normalization, samples.len());
    let samples = fft(&bins, plan.sign.angle_sign(crate::Direction::Inverse))?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value.scale(output_scale);
            if !value.re.is_finite() {
                return Err(SignalError::NonFinite {
                    index,
                    component: "real",
                });
            }
            if !value.im.is_finite() {
                return Err(SignalError::NonFinite {
                    index,
                    component: "imag",
                });
            }
            Ok(value.into())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let len = samples.len();
    Ok(AnalyticSignal {
        samples,
        report: AnalyticSignalReport {
            len,
            normalization: plan.normalization,
            sign: plan.sign,
            retained_nyquist: len.is_multiple_of(2),
            work_units,
            work_limit: plan.max_work,
        },
    })
}

/// Returns the Hilbert-transform component of a real signal.
pub fn hilbert_transform(
    samples: &[f64],
    plan: &AnalyticSignalPlan,
) -> Result<Vec<f64>, SignalError> {
    Ok(analytic_signal(samples, plan)?
        .samples
        .into_iter()
        .map(|(_, imaginary)| imaginary)
        .collect())
}

/// Unwraps a phase sequence by removing jumps larger than `discontinuity`.
pub fn unwrap_phase(phases: &[f64], discontinuity: f64) -> Result<Vec<f64>, SignalError> {
    if !discontinuity.is_finite() || !(PI..=TAU).contains(&discontinuity) {
        return Err(SignalError::InvalidPolicy {
            policy: "phase unwrap discontinuity",
            reason: "the finite threshold must lie between pi and one turn",
        });
    }
    for (index, phase) in phases.iter().copied().enumerate() {
        if !phase.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "phase",
            });
        }
    }
    let Some(first) = phases.first().copied() else {
        return Ok(Vec::new());
    };
    let mut output = Vec::with_capacity(phases.len());
    output.push(first);
    let mut correction = 0.0;
    for index in 1..phases.len() {
        let delta = phases[index] - phases[index - 1];
        if delta.abs() > discontinuity {
            correction += (delta + PI).rem_euclid(TAU) - PI - delta;
        }
        output.push(phases[index] + correction);
    }
    Ok(output)
}

/// Midpoint grid, unwrapped phase, and interval frequency from analytic samples.
#[derive(Clone, Debug, PartialEq)]
pub struct InstantaneousFrequency {
    /// Midpoint time of each adjacent-sample interval in seconds.
    pub time_seconds: Vec<f64>,
    /// Cycles per second over each adjacent-sample interval.
    pub frequency_hz: Vec<f64>,
    /// Unwrapped phase at each input sample.
    pub unwrapped_phase: Vec<f64>,
    /// Sample rate used to construct the interval grid.
    pub sample_rate_hz: f64,
}

/// Derives unwrapped phase and adjacent-interval instantaneous frequency.
pub fn instantaneous_frequency(
    analytic: &[(f64, f64)],
    sample_rate_hz: f64,
) -> Result<InstantaneousFrequency, SignalError> {
    if analytic.len() < 2 {
        return Err(SignalError::InvalidLength {
            len: analytic.len(),
            reason: "instantaneous frequency requires at least two analytic samples",
        });
    }
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(SignalError::InvalidPolicy {
            policy: "instantaneous-frequency grid",
            reason: "sample rate must be finite and positive",
        });
    }
    let phases = analytic
        .iter()
        .copied()
        .enumerate()
        .map(|(index, (real, imag))| {
            if !real.is_finite() || !imag.is_finite() {
                Err(SignalError::NonFinite {
                    index,
                    component: "analytic sample",
                })
            } else {
                Ok(imag.atan2(real))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let unwrapped_phase = unwrap_phase(&phases, PI)?;
    let frequency_hz = unwrapped_phase
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) * sample_rate_hz / TAU)
        .collect::<Vec<_>>();
    let time_seconds = (0..frequency_hz.len())
        .map(|index| (index as f64 + 0.5) / sample_rate_hz)
        .collect();
    Ok(InstantaneousFrequency {
        time_seconds,
        frequency_hz,
        unwrapped_phase,
        sample_rate_hz,
    })
}

/// Returns the magnitude envelope of complex analytic samples.
pub fn analytic_envelope(analytic: &[(f64, f64)]) -> Result<Vec<f64>, SignalError> {
    analytic
        .iter()
        .copied()
        .enumerate()
        .map(|(index, (real, imag))| {
            let magnitude = real.hypot(imag);
            if magnitude.is_finite() {
                Ok(magnitude)
            } else {
                Err(SignalError::NonFinite {
                    index,
                    component: "analytic envelope",
                })
            }
        })
        .collect()
}

/// Time constants and grid for a rectified attack/release envelope follower.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvelopeFollowerPlan {
    /// Sample rate in hertz.
    pub sample_rate_hz: f64,
    /// Attack time constant in seconds; zero follows rising edges immediately.
    pub attack_seconds: f64,
    /// Release time constant in seconds; zero follows falling edges immediately.
    pub release_seconds: f64,
    /// Initial nonnegative envelope state.
    pub initial_value: f64,
}

/// Applies a deterministic rectified one-pole attack/release envelope follower.
pub fn envelope_follow(
    samples: &[f64],
    plan: &EnvelopeFollowerPlan,
) -> Result<Vec<f64>, SignalError> {
    if !plan.sample_rate_hz.is_finite()
        || plan.sample_rate_hz <= 0.0
        || !plan.attack_seconds.is_finite()
        || plan.attack_seconds < 0.0
        || !plan.release_seconds.is_finite()
        || plan.release_seconds < 0.0
        || !plan.initial_value.is_finite()
        || plan.initial_value < 0.0
    {
        return Err(SignalError::InvalidPolicy {
            policy: "envelope follower",
            reason: "positive sample rate and finite nonnegative time constants/state are required",
        });
    }
    let attack = smoothing_coefficient(plan.attack_seconds, plan.sample_rate_hz);
    let release = smoothing_coefficient(plan.release_seconds, plan.sample_rate_hz);
    let mut state = plan.initial_value;
    let mut output = Vec::with_capacity(samples.len());
    for (index, sample) in samples.iter().copied().enumerate() {
        if !sample.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "value",
            });
        }
        let target = sample.abs();
        let coefficient = if target > state { attack } else { release };
        state = coefficient * state + (1.0 - coefficient) * target;
        output.push(state);
    }
    Ok(output)
}

fn smoothing_coefficient(seconds: f64, sample_rate_hz: f64) -> f64 {
    if seconds == 0.0 {
        0.0
    } else {
        (-1.0 / (seconds * sample_rate_hz)).exp()
    }
}

fn analytic_work(len: usize) -> Result<u64, SignalError> {
    let len = u64::try_from(len).map_err(|_| SignalError::InvalidPolicy {
        policy: "analytic signal work",
        reason: "length does not fit the work counter",
    })?;
    let stages = u64::from(usize::BITS - (len as usize).leading_zeros()).max(1);
    len.checked_mul(stages)
        .and_then(|value| value.checked_mul(4))
        .ok_or(SignalError::InvalidPolicy {
            policy: "analytic signal work",
            reason: "work-unit arithmetic overflowed",
        })
}
