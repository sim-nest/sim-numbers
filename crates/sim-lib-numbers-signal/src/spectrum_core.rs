//! Shared frequency-grid, Fourier-evaluation, and scaling machinery.

use std::f64::consts::TAU;

use crate::{
    EstimatorLimits, FrequencyGridPolicy, SignalError, SpectrumScaling, SpectrumScalingKind,
    SpectrumSide, WindowMetrics,
    fft::{Complex, fft},
};

#[derive(Clone, Debug)]
pub(crate) struct ResolvedGrid {
    pub(crate) frequency: Vec<f64>,
    pub(crate) fft_indices: Option<Vec<usize>>,
    pub(crate) side: SpectrumSide,
}

pub(crate) fn resolve_grid(
    policy: &FrequencyGridPolicy,
    sample_rate_hz: f64,
    fft_len: usize,
    limits: EstimatorLimits,
) -> Result<ResolvedGrid, SignalError> {
    let nyquist = sample_rate_hz / 2.0;
    let resolved = match policy {
        FrequencyGridPolicy::FftBins { side } => match side {
            SpectrumSide::OneSided => {
                let indices = (0..=fft_len / 2).collect::<Vec<_>>();
                let frequency = indices
                    .iter()
                    .map(|index| *index as f64 * sample_rate_hz / fft_len as f64)
                    .collect();
                ResolvedGrid {
                    frequency,
                    fft_indices: Some(indices),
                    side: *side,
                }
            }
            SpectrumSide::TwoSided => {
                let negative_start = fft_len / 2 + 1;
                let mut indices = (negative_start..fft_len).collect::<Vec<_>>();
                indices.extend(0..=fft_len / 2);
                let frequency = indices
                    .iter()
                    .map(|index| {
                        if *index > fft_len / 2 {
                            (*index as f64 - fft_len as f64) * sample_rate_hz / fft_len as f64
                        } else {
                            *index as f64 * sample_rate_hz / fft_len as f64
                        }
                    })
                    .collect();
                ResolvedGrid {
                    frequency,
                    fft_indices: Some(indices),
                    side: *side,
                }
            }
        },
        FrequencyGridPolicy::Linear {
            start_hz,
            end_hz,
            bins,
            side,
        } => {
            validate_frequency_span(*start_hz, *end_hz, *bins, *side, nyquist)?;
            let frequency = if *bins == 1 {
                vec![*start_hz]
            } else {
                let step = (*end_hz - *start_hz) / (*bins - 1) as f64;
                (0..*bins)
                    .map(|index| *start_hz + index as f64 * step)
                    .collect()
            };
            ResolvedGrid {
                frequency,
                fft_indices: None,
                side: *side,
            }
        }
        FrequencyGridPolicy::Explicit {
            frequencies_hz,
            side,
        } => {
            if frequencies_hz.is_empty() {
                return Err(SignalError::InvalidPolicy {
                    policy: "explicit frequency grid",
                    reason: "at least one frequency is required",
                });
            }
            for (index, frequency) in frequencies_hz.iter().copied().enumerate() {
                if !frequency.is_finite() {
                    return Err(SignalError::NonFinite {
                        index,
                        component: "frequency",
                    });
                }
                if (matches!(side, SpectrumSide::OneSided) && !(0.0..=nyquist).contains(&frequency))
                    || (matches!(side, SpectrumSide::TwoSided)
                        && !(-nyquist..=nyquist).contains(&frequency))
                {
                    return Err(SignalError::InvalidPolicy {
                        policy: "explicit frequency grid",
                        reason: "frequency lies outside the selected Nyquist interval",
                    });
                }
                if index > 0 && frequency <= frequencies_hz[index - 1] {
                    return Err(SignalError::InvalidPolicy {
                        policy: "explicit frequency grid",
                        reason: "frequencies must be strictly increasing",
                    });
                }
            }
            ResolvedGrid {
                frequency: frequencies_hz.clone(),
                fft_indices: None,
                side: *side,
            }
        }
    };
    if resolved.frequency.len() > limits.max_frequency_bins {
        return Err(SignalError::InvalidPolicy {
            policy: "frequency-bin limit",
            reason: "resolved grid exceeds the estimator limit",
        });
    }
    Ok(resolved)
}

fn validate_frequency_span(
    start_hz: f64,
    end_hz: f64,
    bins: usize,
    side: SpectrumSide,
    nyquist: f64,
) -> Result<(), SignalError> {
    if !start_hz.is_finite() || !end_hz.is_finite() || end_hz < start_hz || bins == 0 {
        return Err(SignalError::InvalidPolicy {
            policy: "linear frequency grid",
            reason: "finite ordered endpoints and at least one bin are required",
        });
    }
    let valid = match side {
        SpectrumSide::OneSided => start_hz >= 0.0 && end_hz <= nyquist,
        SpectrumSide::TwoSided => start_hz >= -nyquist && end_hz <= nyquist,
    };
    if !valid {
        return Err(SignalError::InvalidPolicy {
            policy: "linear frequency grid",
            reason: "endpoints lie outside the selected Nyquist interval",
        });
    }
    Ok(())
}

pub(crate) fn validate_samples(samples: &[f64]) -> Result<(), SignalError> {
    if samples.is_empty() {
        return Err(SignalError::InvalidLength {
            len: 0,
            reason: "a spectral estimate requires at least one sample",
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
    Ok(())
}

pub(crate) fn evaluate_windowed(
    samples: &[f64],
    window: &[f64],
    sample_rate_hz: f64,
    fft_len: usize,
    grid: &ResolvedGrid,
) -> Result<Vec<Complex>, SignalError> {
    debug_assert_eq!(samples.len(), window.len());
    if let Some(indices) = &grid.fft_indices {
        let mut input = vec![Complex::ZERO; fft_len];
        for (slot, (sample, coefficient)) in input
            .iter_mut()
            .zip(samples.iter().copied().zip(window.iter().copied()))
        {
            *slot = Complex::new(sample * coefficient, 0.0);
        }
        let transformed = fft(&input, -1.0)?;
        return Ok(indices.iter().map(|index| transformed[*index]).collect());
    }
    Ok(grid
        .frequency
        .iter()
        .map(|frequency| {
            samples
                .iter()
                .copied()
                .zip(window.iter().copied())
                .enumerate()
                .fold(Complex::ZERO, |sum, (index, (sample, coefficient))| {
                    let phase = -TAU * frequency * index as f64 / sample_rate_hz;
                    sum + Complex::cis(phase).scale(sample * coefficient)
                })
        })
        .collect())
}

pub(crate) fn spectrum_work(
    segment_len: usize,
    fft_len: usize,
    grid: &ResolvedGrid,
) -> Result<u64, SignalError> {
    let len = u64::try_from(segment_len).map_err(|_| work_overflow())?;
    let bins = u64::try_from(grid.frequency.len()).map_err(|_| work_overflow())?;
    if grid.fft_indices.is_some() {
        let transform = u64::try_from(fft_len).map_err(|_| work_overflow())?;
        transform
            .checked_mul(transform)
            .and_then(|value| value.checked_add(len))
            .ok_or_else(work_overflow)
    } else {
        len.checked_mul(bins).ok_or_else(work_overflow)
    }
}

pub(crate) fn checked_product(left: u64, right: u64) -> Result<u64, SignalError> {
    left.checked_mul(right).ok_or_else(work_overflow)
}

pub(crate) fn scaling(
    kind: SpectrumScalingKind,
    sample_rate_hz: f64,
    metrics: &WindowMetrics,
    side: SpectrumSide,
) -> Result<SpectrumScaling, SignalError> {
    let denominator = match kind {
        SpectrumScalingKind::Power => {
            let sum = metrics.coherent_gain * metrics.len as f64;
            sum * sum
        }
        SpectrumScalingKind::Density => sample_rate_hz * metrics.energy,
        SpectrumScalingKind::LombScargleNormalized => {
            return Err(SignalError::InvalidPolicy {
                policy: "Fourier spectrum scaling",
                reason: "Lomb-Scargle normalization is not a Fourier window scaling",
            });
        }
    };
    if !denominator.is_finite() || denominator <= f64::EPSILON {
        return Err(SignalError::DegenerateNormalization {
            normalization: "spectrum scaling",
        });
    }
    Ok(SpectrumScaling {
        kind,
        sample_rate_hz: Some(sample_rate_hz),
        normalization_denominator: denominator,
        one_sided: matches!(side, SpectrumSide::OneSided),
        interior_bin_multiplier: if matches!(side, SpectrumSide::OneSided) {
            2.0
        } else {
            1.0
        },
    })
}

pub(crate) fn bin_multiplier(frequency: f64, sample_rate_hz: f64, side: SpectrumSide) -> f64 {
    if !matches!(side, SpectrumSide::OneSided) {
        return 1.0;
    }
    let tolerance = sample_rate_hz.abs().max(1.0) * f64::EPSILON * 16.0;
    if frequency.abs() <= tolerance || (frequency - sample_rate_hz / 2.0).abs() <= tolerance {
        1.0
    } else {
        2.0
    }
}

fn work_overflow() -> SignalError {
    SignalError::InvalidPolicy {
        policy: "estimator work",
        reason: "work-unit arithmetic overflowed",
    }
}
