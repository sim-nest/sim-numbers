//! Periodogram, Welch, cross-spectrum, and coherence estimators.

use crate::{
    CrossSpectrumEstimate, EstimatorEvidence, EstimatorKind, PeriodogramPlan, SignalError,
    SpectrumEstimate, SpectrumScalingKind, WelchPlan,
    spectrum_core::{
        bin_multiplier, checked_product, evaluate_windowed, resolve_grid, scaling, spectrum_work,
        validate_samples,
    },
    spectrum_types::{admit_work, validate_common},
};

/// Estimates one record's power spectrum under an explicit window and grid.
pub fn periodogram(
    samples: &[f64],
    plan: &PeriodogramPlan,
) -> Result<SpectrumEstimate, SignalError> {
    validate_samples(samples)?;
    validate_common(
        plan.sample_rate_hz,
        plan.fft_len,
        samples.len(),
        plan.limits,
    )?;
    let grid = resolve_grid(&plan.grid, plan.sample_rate_hz, plan.fft_len, plan.limits)?;
    let work_units = spectrum_work(samples.len(), plan.fft_len, &grid)?;
    admit_work(work_units, plan.limits)?;
    let window = plan.window.generate(samples.len())?;
    let scaling = scaling(
        plan.scaling,
        plan.sample_rate_hz,
        &window.metrics,
        grid.side,
    )?;
    let transformed = evaluate_windowed(
        samples,
        &window.samples,
        plan.sample_rate_hz,
        plan.fft_len,
        &grid,
    )?;
    let power = transformed
        .iter()
        .zip(&grid.frequency)
        .map(|(value, frequency)| {
            bin_multiplier(*frequency, plan.sample_rate_hz, grid.side)
                * (value.re * value.re + value.im * value.im)
                / scaling.normalization_denominator
        })
        .collect();
    Ok(SpectrumEstimate {
        frequency: grid.frequency,
        power,
        scaling,
        evidence: EstimatorEvidence {
            estimator: EstimatorKind::Periodogram,
            input_len: samples.len(),
            fft_len: plan.fft_len,
            segment_len: samples.len(),
            segment_count: 1,
            taper_count: 1,
            frequency_bins: transformed.len(),
            work_units,
            work_limit: plan.limits.max_work,
            degrees_of_freedom: 2.0,
            frequency_grid: plan.grid.clone(),
            window: Some(window.metrics),
            taper_concentrations: Vec::new(),
        },
    })
}

/// Averages complete overlapping segments using Welch's method.
pub fn welch(samples: &[f64], plan: &WelchPlan) -> Result<SpectrumEstimate, SignalError> {
    let prepared = prepare_segments(samples, plan, 1)?;
    let mut power = vec![0.0; prepared.grid.frequency.len()];
    for start in prepared.starts.iter().copied() {
        let transformed = evaluate_windowed(
            &samples[start..start + plan.segment_len],
            &prepared.window.samples,
            plan.sample_rate_hz,
            plan.fft_len,
            &prepared.grid,
        )?;
        for ((slot, value), frequency) in power
            .iter_mut()
            .zip(transformed)
            .zip(&prepared.grid.frequency)
        {
            *slot += bin_multiplier(*frequency, plan.sample_rate_hz, prepared.grid.side)
                * (value.re * value.re + value.im * value.im)
                / prepared.scaling.normalization_denominator;
        }
    }
    let divisor = prepared.starts.len() as f64;
    for value in &mut power {
        *value /= divisor;
    }
    let evidence = prepared.evidence(EstimatorKind::Welch, samples.len());
    Ok(SpectrumEstimate {
        frequency: prepared.grid.frequency,
        power,
        scaling: prepared.scaling,
        evidence,
    })
}

/// Estimates averaged complex cross power, auto power, and coherence.
pub fn cross_spectrum(
    x: &[f64],
    y: &[f64],
    plan: &WelchPlan,
) -> Result<CrossSpectrumEstimate, SignalError> {
    if x.len() != y.len() {
        return Err(SignalError::LengthMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }
    validate_samples(y)?;
    let prepared = prepare_segments(x, plan, 2)?;
    let bins = prepared.grid.frequency.len();
    let mut cross_power = vec![(0.0, 0.0); bins];
    let mut x_power = vec![0.0; bins];
    let mut y_power = vec![0.0; bins];
    for start in prepared.starts.iter().copied() {
        let x_transformed = evaluate_windowed(
            &x[start..start + plan.segment_len],
            &prepared.window.samples,
            plan.sample_rate_hz,
            plan.fft_len,
            &prepared.grid,
        )?;
        let y_transformed = evaluate_windowed(
            &y[start..start + plan.segment_len],
            &prepared.window.samples,
            plan.sample_rate_hz,
            plan.fft_len,
            &prepared.grid,
        )?;
        for index in 0..bins {
            let left = x_transformed[index];
            let right = y_transformed[index];
            let scale = bin_multiplier(
                prepared.grid.frequency[index],
                plan.sample_rate_hz,
                prepared.grid.side,
            ) / prepared.scaling.normalization_denominator;
            x_power[index] += scale * (left.re * left.re + left.im * left.im);
            y_power[index] += scale * (right.re * right.re + right.im * right.im);
            cross_power[index].0 += scale * (left.re * right.re + left.im * right.im);
            cross_power[index].1 += scale * (left.im * right.re - left.re * right.im);
        }
    }
    let divisor = prepared.starts.len() as f64;
    for index in 0..bins {
        x_power[index] /= divisor;
        y_power[index] /= divisor;
        cross_power[index].0 /= divisor;
        cross_power[index].1 /= divisor;
    }
    let coherence = (0..bins)
        .map(|index| {
            let denominator = x_power[index] * y_power[index];
            if denominator <= f64::EPSILON {
                0.0
            } else {
                ((cross_power[index].0 * cross_power[index].0
                    + cross_power[index].1 * cross_power[index].1)
                    / denominator)
                    .clamp(0.0, 1.0)
            }
        })
        .collect();
    let evidence = prepared.evidence(EstimatorKind::CrossSpectrum, x.len());
    Ok(CrossSpectrumEstimate {
        frequency: prepared.grid.frequency,
        cross_power,
        x_power,
        y_power,
        coherence,
        scaling: prepared.scaling,
        evidence,
    })
}

struct PreparedSegments {
    starts: Vec<usize>,
    grid: crate::spectrum_core::ResolvedGrid,
    window: crate::Window,
    scaling: crate::SpectrumScaling,
    work_units: u64,
    work_limit: u64,
    fft_len: usize,
    segment_len: usize,
    frequency_grid: crate::FrequencyGridPolicy,
}

impl PreparedSegments {
    fn evidence(&self, estimator: EstimatorKind, input_len: usize) -> EstimatorEvidence {
        EstimatorEvidence {
            estimator,
            input_len,
            fft_len: self.fft_len,
            segment_len: self.segment_len,
            segment_count: self.starts.len(),
            taper_count: 1,
            frequency_bins: self.grid.frequency.len(),
            work_units: self.work_units,
            work_limit: self.work_limit,
            degrees_of_freedom: 2.0 * self.starts.len() as f64,
            frequency_grid: self.frequency_grid.clone(),
            window: Some(self.window.metrics.clone()),
            taper_concentrations: Vec::new(),
        }
    }
}

fn prepare_segments(
    samples: &[f64],
    plan: &WelchPlan,
    transforms_per_segment: u64,
) -> Result<PreparedSegments, SignalError> {
    validate_samples(samples)?;
    validate_common(
        plan.sample_rate_hz,
        plan.fft_len,
        plan.segment_len,
        plan.limits,
    )?;
    if plan.overlap >= plan.segment_len {
        return Err(SignalError::InvalidPolicy {
            policy: "Welch overlap",
            reason: "overlap must be smaller than the segment length",
        });
    }
    if samples.len() < plan.segment_len {
        return Err(SignalError::InvalidLength {
            len: samples.len(),
            reason: "Welch input must contain one complete segment",
        });
    }
    if matches!(plan.scaling, SpectrumScalingKind::LombScargleNormalized) {
        return Err(SignalError::InvalidPolicy {
            policy: "Welch scaling",
            reason: "Lomb-Scargle normalization is not a Welch scaling",
        });
    }
    let hop = plan.segment_len - plan.overlap;
    let segment_count = 1 + (samples.len() - plan.segment_len) / hop;
    if segment_count > plan.limits.max_segments {
        return Err(SignalError::InvalidPolicy {
            policy: "segment limit",
            reason: "Welch segment count exceeds the estimator limit",
        });
    }
    let starts = (0..segment_count)
        .map(|index| index * hop)
        .collect::<Vec<_>>();
    let grid = resolve_grid(&plan.grid, plan.sample_rate_hz, plan.fft_len, plan.limits)?;
    let per_transform = spectrum_work(plan.segment_len, plan.fft_len, &grid)?;
    let work_units = checked_product(
        checked_product(per_transform, segment_count as u64)?,
        transforms_per_segment,
    )?;
    admit_work(work_units, plan.limits)?;
    let window = plan.window.generate(plan.segment_len)?;
    let scaling = scaling(
        plan.scaling,
        plan.sample_rate_hz,
        &window.metrics,
        grid.side,
    )?;
    Ok(PreparedSegments {
        starts,
        grid,
        window,
        scaling,
        work_units,
        work_limit: plan.limits.max_work,
        fft_len: plan.fft_len,
        segment_len: plan.segment_len,
        frequency_grid: plan.grid.clone(),
    })
}
