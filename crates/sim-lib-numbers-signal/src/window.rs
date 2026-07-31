//! Reference-defined analysis windows and explicit normalization evidence.

use std::f64::consts::TAU;

use crate::SignalError;

/// Whether a formula includes both endpoints or one period's left endpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowSampling {
    /// Include both endpoints. This is appropriate for finite, non-repeating
    /// records and uses `N - 1` as the cosine denominator.
    #[default]
    Symmetric,
    /// Sample one period without repeating its endpoint, using `N` as the
    /// cosine denominator.
    Periodic,
}

/// A reference analysis-window formula or caller-supplied coefficient vector.
#[derive(Clone, Debug, PartialEq)]
pub enum WindowFunction {
    /// Constant unit coefficients.
    Rectangular,
    /// Hann's raised cosine, `0.5 - 0.5 cos(theta)`.
    Hann,
    /// Hamming's raised cosine, `0.54 - 0.46 cos(theta)`.
    Hamming,
    /// General three-term Blackman window.
    Blackman {
        /// Third-term coefficient. The exact Blackman definition uses `0.16`.
        alpha: f64,
    },
    /// Four-term minimum-sidelobe Blackman-Harris window.
    BlackmanHarris,
    /// Kaiser window based on the order-zero modified Bessel function.
    Kaiser {
        /// Non-negative sidelobe/width trade-off parameter.
        beta: f64,
    },
    /// Coefficients supplied explicitly by the caller.
    Explicit(Vec<f64>),
}

/// Scale applied after the reference coefficients are generated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowNormalization {
    /// Preserve the reference coefficients exactly.
    #[default]
    None,
    /// Scale the coefficients so their arithmetic mean is one.
    CoherentGain,
    /// Scale the coefficients so their sum of squares is one.
    UnitEnergy,
}

/// Complete, reusable policy for generating an analysis window.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowSpec {
    /// Reference formula or explicit coefficient vector.
    pub function: WindowFunction,
    /// Endpoint convention for formula-based windows.
    pub sampling: WindowSampling,
    /// Post-generation scaling policy.
    pub normalization: WindowNormalization,
}

impl WindowSpec {
    /// Creates a symmetric, unnormalized window policy.
    pub fn new(function: WindowFunction) -> Self {
        Self {
            function,
            sampling: WindowSampling::Symmetric,
            normalization: WindowNormalization::None,
        }
    }

    /// Generates `len` coefficients and their reconstructable metrics.
    pub fn generate(&self, len: usize) -> Result<Window, SignalError> {
        if len == 0 {
            return Err(SignalError::InvalidLength {
                len,
                reason: "an analysis window requires at least one coefficient",
            });
        }
        let mut samples = match &self.function {
            WindowFunction::Explicit(samples) => {
                if samples.len() != len {
                    return Err(SignalError::LengthMismatch {
                        expected: len,
                        actual: samples.len(),
                    });
                }
                samples.clone()
            }
            function => formula_window(function, self.sampling, len)?,
        };
        for (index, value) in samples.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(SignalError::NonFinite {
                    index,
                    component: "window",
                });
            }
        }

        let raw_sum = stable_sum(&samples);
        let raw_energy = stable_sum(
            &samples
                .iter()
                .map(|value| value * value)
                .collect::<Vec<_>>(),
        );
        let normalization_scale = match self.normalization {
            WindowNormalization::None => 1.0,
            WindowNormalization::CoherentGain => {
                if !raw_sum.is_finite() || raw_sum.abs() <= f64::EPSILON {
                    return Err(SignalError::DegenerateNormalization {
                        normalization: "window coherent-gain",
                    });
                }
                len as f64 / raw_sum
            }
            WindowNormalization::UnitEnergy => {
                if !raw_energy.is_finite() || raw_energy <= f64::EPSILON {
                    return Err(SignalError::DegenerateNormalization {
                        normalization: "window unit-energy",
                    });
                }
                raw_energy.sqrt().recip()
            }
        };
        for value in &mut samples {
            *value *= normalization_scale;
        }
        let sum = stable_sum(&samples);
        let energy = stable_sum(
            &samples
                .iter()
                .map(|value| value * value)
                .collect::<Vec<_>>(),
        );
        let equivalent_noise_bandwidth_bins = if sum.abs() <= f64::EPSILON {
            None
        } else {
            Some(len as f64 * energy / (sum * sum))
        };
        Ok(Window {
            samples,
            metrics: WindowMetrics {
                len,
                raw_coherent_gain: raw_sum / len as f64,
                raw_energy,
                normalization_scale,
                coherent_gain: sum / len as f64,
                energy,
                equivalent_noise_bandwidth_bins,
            },
        })
    }
}

impl Default for WindowSpec {
    fn default() -> Self {
        Self::new(WindowFunction::Hann)
    }
}

/// Generated coefficients and the exact gain/energy facts used to scale them.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    /// Coefficients after the requested normalization.
    pub samples: Vec<f64>,
    /// Metrics before and after normalization.
    pub metrics: WindowMetrics,
}

/// Gain, energy, and equivalent-bandwidth evidence for a generated window.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowMetrics {
    /// Number of coefficients.
    pub len: usize,
    /// Arithmetic mean of the unnormalized reference coefficients.
    pub raw_coherent_gain: f64,
    /// Sum of squared unnormalized reference coefficients.
    pub raw_energy: f64,
    /// Multiplier applied by [`WindowNormalization`].
    pub normalization_scale: f64,
    /// Arithmetic mean after normalization.
    pub coherent_gain: f64,
    /// Sum of squared coefficients after normalization.
    pub energy: f64,
    /// Scale-invariant equivalent noise bandwidth in FFT bins. A zero-sum
    /// explicit window has no finite equivalent bandwidth.
    pub equivalent_noise_bandwidth_bins: Option<f64>,
}

fn formula_window(
    function: &WindowFunction,
    sampling: WindowSampling,
    len: usize,
) -> Result<Vec<f64>, SignalError> {
    if len == 1 {
        return Ok(vec![1.0]);
    }
    let denominator = match sampling {
        WindowSampling::Symmetric => (len - 1) as f64,
        WindowSampling::Periodic => len as f64,
    };
    match function {
        WindowFunction::Blackman { alpha }
            if !alpha.is_finite() || !(0.0..=1.0).contains(alpha) =>
        {
            return Err(SignalError::InvalidPolicy {
                policy: "Blackman alpha",
                reason: "alpha must be finite and between zero and one",
            });
        }
        WindowFunction::Kaiser { beta } if !beta.is_finite() || *beta < 0.0 => {
            return Err(SignalError::InvalidPolicy {
                policy: "Kaiser beta",
                reason: "beta must be finite and non-negative",
            });
        }
        _ => {}
    }
    Ok((0..len)
        .map(|index| {
            let theta = TAU * index as f64 / denominator;
            match function {
                WindowFunction::Rectangular => 1.0,
                WindowFunction::Hann => 0.5 - 0.5 * theta.cos(),
                WindowFunction::Hamming => 0.54 - 0.46 * theta.cos(),
                WindowFunction::Blackman { alpha } => {
                    let a0 = (1.0 - alpha) / 2.0;
                    a0 - 0.5 * theta.cos() + alpha / 2.0 * (2.0 * theta).cos()
                }
                WindowFunction::BlackmanHarris => {
                    0.35875 - 0.48829 * theta.cos() + 0.14128 * (2.0 * theta).cos()
                        - 0.01168 * (3.0 * theta).cos()
                }
                WindowFunction::Kaiser { beta } => {
                    let position = 2.0 * index as f64 / denominator - 1.0;
                    modified_bessel_i0(beta * (1.0 - position * position).max(0.0).sqrt())
                        / modified_bessel_i0(*beta)
                }
                WindowFunction::Explicit(_) => unreachable!("handled before formula generation"),
            }
        })
        .collect())
}

// The convergent power series is stable across the practical Kaiser range and
// avoids a platform-dependent special-function dependency.
fn modified_bessel_i0(value: f64) -> f64 {
    let quarter_square = value * value / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for order in 1..=100 {
        term *= quarter_square / (order as f64 * order as f64);
        sum += term;
        if term <= sum.abs() * f64::EPSILON {
            break;
        }
    }
    sum
}

fn stable_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = *value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}
