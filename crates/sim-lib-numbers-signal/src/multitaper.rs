//! Discrete-prolate (Slepian) multitaper spectral estimation.

use sim_lib_numbers_tensor_decomp::{EigenPlan, VectorPolicy, symmetric_eigen_f64};
use std::f64::consts::{PI, TAU};

use crate::{
    EstimatorEvidence, EstimatorKind, MultitaperPlan, SignalError, SpectrumEstimate,
    SpectrumScaling, SpectrumScalingKind,
    spectrum_core::{
        bin_multiplier, checked_product, evaluate_windowed, resolve_grid, spectrum_work,
        validate_samples,
    },
    spectrum_types::{admit_work, validate_common},
};

const JACOBI_SWEEPS: u64 = 32;

/// Averages the leading Slepian-taper power-density estimates.
pub fn multitaper(samples: &[f64], plan: &MultitaperPlan) -> Result<SpectrumEstimate, SignalError> {
    validate_samples(samples)?;
    validate_common(
        plan.sample_rate_hz,
        plan.fft_len,
        samples.len(),
        plan.limits,
    )?;
    validate_taper_policy(samples.len(), plan)?;
    let grid = resolve_grid(&plan.grid, plan.sample_rate_hz, plan.fft_len, plan.limits)?;
    let taper_work = taper_work(samples.len(), plan.taper_count)?;
    let transform_work = checked_product(
        spectrum_work(samples.len(), plan.fft_len, &grid)?,
        plan.taper_count as u64,
    )?;
    let work_units = taper_work
        .checked_add(transform_work)
        .ok_or(SignalError::InvalidPolicy {
            policy: "multitaper work",
            reason: "work-unit arithmetic overflowed",
        })?;
    admit_work(work_units, plan.limits)?;

    let (tapers, concentrations) =
        dpss_tapers(samples.len(), plan.time_bandwidth, plan.taper_count)?;
    let denominator = plan.sample_rate_hz;
    let mut power = vec![0.0; grid.frequency.len()];
    for taper in &tapers {
        let transformed =
            evaluate_windowed(samples, taper, plan.sample_rate_hz, plan.fft_len, &grid)?;
        for ((slot, value), frequency) in power.iter_mut().zip(transformed).zip(&grid.frequency) {
            *slot += bin_multiplier(*frequency, plan.sample_rate_hz, grid.side)
                * (value.re * value.re + value.im * value.im)
                / denominator;
        }
    }
    for value in &mut power {
        *value /= plan.taper_count as f64;
    }
    let frequency_bins = power.len();
    Ok(SpectrumEstimate {
        frequency: grid.frequency,
        power,
        scaling: SpectrumScaling {
            kind: SpectrumScalingKind::Density,
            sample_rate_hz: Some(plan.sample_rate_hz),
            normalization_denominator: denominator,
            one_sided: matches!(grid.side, crate::SpectrumSide::OneSided),
            interior_bin_multiplier: if matches!(grid.side, crate::SpectrumSide::OneSided) {
                2.0
            } else {
                1.0
            },
        },
        evidence: EstimatorEvidence {
            estimator: EstimatorKind::SlepianMultitaper,
            input_len: samples.len(),
            fft_len: plan.fft_len,
            segment_len: samples.len(),
            segment_count: 1,
            taper_count: plan.taper_count,
            frequency_bins,
            work_units,
            work_limit: plan.limits.max_work,
            degrees_of_freedom: 2.0 * plan.taper_count as f64,
            frequency_grid: plan.grid.clone(),
            window: None,
            taper_concentrations: concentrations,
        },
    })
}

fn validate_taper_policy(len: usize, plan: &MultitaperPlan) -> Result<(), SignalError> {
    if len < 2
        || !plan.time_bandwidth.is_finite()
        || plan.time_bandwidth <= 0.0
        || plan.time_bandwidth >= len as f64 / 2.0
    {
        return Err(SignalError::InvalidPolicy {
            policy: "Slepian time-bandwidth",
            reason: "N*W must be finite, positive, and smaller than N/2",
        });
    }
    let shannon_count = (2.0 * plan.time_bandwidth).floor() as usize;
    if plan.taper_count == 0 || plan.taper_count > shannon_count {
        return Err(SignalError::InvalidPolicy {
            policy: "Slepian taper count",
            reason: "taper count must be positive and no larger than floor(2*N*W)",
        });
    }
    if plan.taper_count > plan.limits.max_tapers {
        return Err(SignalError::InvalidPolicy {
            policy: "taper limit",
            reason: "Slepian taper count exceeds the estimator limit",
        });
    }
    Ok(())
}

fn taper_work(len: usize, taper_count: usize) -> Result<u64, SignalError> {
    let len = u64::try_from(len).map_err(|_| work_overflow())?;
    let taper_count = u64::try_from(taper_count).map_err(|_| work_overflow())?;
    let cube = checked_product(checked_product(len, len)?, len)?;
    let decomposition = checked_product(JACOBI_SWEEPS, cube)?;
    let concentration = checked_product(checked_product(taper_count, len)?, len)?;
    decomposition
        .checked_add(concentration)
        .ok_or_else(work_overflow)
}

fn jacobi_rotation_limit(len: usize) -> Result<usize, SignalError> {
    let pairs = len
        .checked_mul(len.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(work_overflow)?;
    (JACOBI_SWEEPS as usize)
        .checked_mul(pairs)
        .ok_or_else(work_overflow)
}

pub(crate) fn dpss_tapers(
    len: usize,
    time_bandwidth: f64,
    taper_count: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), SignalError> {
    let half_bandwidth = time_bandwidth / len as f64;
    let mut matrix = vec![0.0; len * len];
    let center = (len - 1) as f64 / 2.0;
    for index in 0..len {
        let position = center - index as f64;
        matrix[index * len + index] = position * position * (TAU * half_bandwidth).cos();
        if index + 1 < len {
            let off_diagonal = (index + 1) as f64 * (len - index - 1) as f64 / 2.0;
            matrix[index * len + index + 1] = off_diagonal;
            matrix[(index + 1) * len + index] = off_diagonal;
        }
    }
    let decomposition = symmetric_eigen_f64(
        &matrix,
        len,
        EigenPlan {
            max_dimension: len,
            // `symmetric_eigen_f64` counts Jacobi rotations, while the
            // multitaper work policy is expressed in complete sweeps.
            max_iterations: jacobi_rotation_limit(len)?,
            tolerance: f64::EPSILON * 16.0,
            symmetry_tolerance: Some(f64::EPSILON * 16.0),
            vectors: VectorPolicy::Compute,
            reconstruction_tolerance: 1e-8,
        },
    )
    .map_err(|_| SignalError::InvalidPolicy {
        policy: "Slepian eigendecomposition",
        reason: "bounded symmetric eigendecomposition failed",
    })?;
    let eigenvalues = decomposition.eigenvalues;
    let eigenvectors = decomposition
        .eigenvectors
        .expect("vector policy requests eigenvectors");
    let mut order = (0..len).collect::<Vec<_>>();
    order.sort_by(|left, right| eigenvalues[*right].total_cmp(&eigenvalues[*left]));
    let mut tapers = Vec::with_capacity(taper_count);
    let mut concentrations = Vec::with_capacity(taper_count);
    for column in order.into_iter().take(taper_count) {
        let mut taper = (0..len)
            .map(|row| eigenvectors[row * len + column])
            .collect::<Vec<_>>();
        let norm = taper.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Err(SignalError::DegenerateNormalization {
                normalization: "Slepian taper",
            });
        }
        for value in &mut taper {
            *value /= norm;
        }
        let pivot = taper
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .map(|(index, _)| index)
            .unwrap_or(0);
        if taper[pivot] < 0.0 {
            for value in &mut taper {
                *value = -*value;
            }
        }
        concentrations.push(spectral_concentration(&taper, half_bandwidth));
        tapers.push(taper);
    }
    Ok((tapers, concentrations))
}

fn spectral_concentration(taper: &[f64], half_bandwidth: f64) -> f64 {
    let mut value = 0.0;
    for (left, left_value) in taper.iter().copied().enumerate() {
        for (right, right_value) in taper.iter().copied().enumerate() {
            let distance = left.abs_diff(right);
            let kernel = if distance == 0 {
                2.0 * half_bandwidth
            } else {
                (TAU * half_bandwidth * distance as f64).sin() / (PI * distance as f64)
            };
            value += left_value * kernel * right_value;
        }
    }
    value.clamp(0.0, 1.0)
}

fn work_overflow() -> SignalError {
    SignalError::InvalidPolicy {
        policy: "multitaper work",
        reason: "work-unit arithmetic overflowed",
    }
}
