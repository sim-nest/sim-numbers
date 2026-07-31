//! Generalized Lomb-Scargle estimation for uneven sample times.

use std::f64::consts::TAU;

use crate::{
    EstimatorEvidence, EstimatorKind, LombScarglePlan, SignalError, SpectrumEstimate,
    SpectrumScaling, SpectrumScalingKind, SpectrumSide,
    spectrum_core::{checked_product, resolve_grid, validate_samples},
    spectrum_types::{admit_work, validate_common},
};

/// Fits a sinusoid plus floating mean at each requested uneven-sample frequency.
pub fn lomb_scargle(
    times_seconds: &[f64],
    samples: &[f64],
    plan: &LombScarglePlan,
) -> Result<SpectrumEstimate, SignalError> {
    validate_samples(samples)?;
    if times_seconds.len() != samples.len() {
        return Err(SignalError::LengthMismatch {
            expected: samples.len(),
            actual: times_seconds.len(),
        });
    }
    if samples.len() < 3 {
        return Err(SignalError::InvalidLength {
            len: samples.len(),
            reason: "Lomb-Scargle requires at least three observations",
        });
    }
    validate_times(times_seconds)?;
    validate_common(plan.sample_rate_hz, plan.fft_len, 1, plan.limits)?;
    let grid = resolve_grid(&plan.grid, plan.sample_rate_hz, plan.fft_len, plan.limits)?;
    if matches!(grid.side, SpectrumSide::TwoSided) {
        return Err(SignalError::InvalidPolicy {
            policy: "Lomb-Scargle frequency side",
            reason: "real uneven samples require a non-redundant one-sided grid",
        });
    }
    let work_units = checked_product(
        checked_product(samples.len() as u64, grid.frequency.len() as u64)?,
        24,
    )?;
    admit_work(work_units, plan.limits)?;

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let total_sum_squares = samples
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>();
    if total_sum_squares <= f64::EPSILON || !total_sum_squares.is_finite() {
        return Err(SignalError::DegenerateNormalization {
            normalization: "Lomb-Scargle centered variance",
        });
    }
    let origin = times_seconds[0];
    let power = grid
        .frequency
        .iter()
        .map(|frequency| {
            if frequency.abs() <= f64::EPSILON {
                return 0.0;
            }
            let mut normal = [[0.0; 4]; 3];
            for (time, observed) in times_seconds.iter().zip(samples) {
                let phase = TAU * frequency * (time - origin);
                let row = [phase.cos(), phase.sin(), 1.0];
                for left in 0..3 {
                    for right in 0..3 {
                        normal[left][right] += row[left] * row[right];
                    }
                    normal[left][3] += row[left] * observed;
                }
            }
            let Some(coefficients) = solve_3x3(normal) else {
                return 0.0;
            };
            let residual = times_seconds
                .iter()
                .zip(samples)
                .map(|(time, observed)| {
                    let phase = TAU * frequency * (time - origin);
                    let fitted = coefficients[0] * phase.cos()
                        + coefficients[1] * phase.sin()
                        + coefficients[2];
                    let error = observed - fitted;
                    error * error
                })
                .sum::<f64>();
            ((total_sum_squares - residual) / total_sum_squares).clamp(0.0, 1.0)
        })
        .collect::<Vec<_>>();
    let frequency_bins = power.len();
    Ok(SpectrumEstimate {
        frequency: grid.frequency,
        power,
        scaling: SpectrumScaling {
            kind: SpectrumScalingKind::LombScargleNormalized,
            sample_rate_hz: None,
            normalization_denominator: total_sum_squares,
            one_sided: true,
            interior_bin_multiplier: 1.0,
        },
        evidence: EstimatorEvidence {
            estimator: EstimatorKind::LombScargle,
            input_len: samples.len(),
            fft_len: plan.fft_len,
            segment_len: samples.len(),
            segment_count: 1,
            taper_count: 0,
            frequency_bins,
            work_units,
            work_limit: plan.limits.max_work,
            degrees_of_freedom: (samples.len() - 3) as f64,
            frequency_grid: plan.grid.clone(),
            window: None,
            taper_concentrations: Vec::new(),
        },
    })
}

fn validate_times(times_seconds: &[f64]) -> Result<(), SignalError> {
    for (index, time) in times_seconds.iter().copied().enumerate() {
        if !time.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "sample time",
            });
        }
        if index > 0 && time <= times_seconds[index - 1] {
            return Err(SignalError::InvalidPolicy {
                policy: "Lomb-Scargle sample times",
                reason: "sample times must be strictly increasing",
            });
        }
    }
    Ok(())
}

fn solve_3x3(mut matrix: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for pivot in 0..3 {
        let best = (pivot..3).max_by(|left, right| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right][pivot].abs())
        })?;
        matrix.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        if divisor.abs() <= f64::EPSILON * 64.0 {
            return None;
        }
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= divisor;
        }
        let pivot_row = matrix[pivot];
        for (row_index, row) in matrix.iter_mut().enumerate() {
            if row_index == pivot {
                continue;
            }
            let factor = row[pivot];
            for (value, pivot_value) in row.iter_mut().zip(pivot_row).skip(pivot) {
                *value -= factor * pivot_value;
            }
        }
    }
    Some([matrix[0][3], matrix[1][3], matrix[2][3]])
}
