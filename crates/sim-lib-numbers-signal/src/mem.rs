//! Maximum-entropy spectra from stable autoregressive models.

use crate::{
    ArModel, EstimatorEvidence, EstimatorKind, EstimatorLimits, FrequencyGridPolicy, SignalError,
    SpectrumEstimate, SpectrumScaling, SpectrumScalingKind,
    spectrum_core::{bin_multiplier, resolve_grid},
    spectrum_types::admit_work,
};

/// Frequency-grid and work policy for a maximum-entropy spectrum.
#[derive(Clone, Debug, PartialEq)]
pub struct MemSpectrumPlan {
    /// Sample rate in hertz.
    pub sample_rate_hz: f64,
    /// Reference DFT length used to construct FFT-bin grids.
    pub fft_len: usize,
    /// Frequencies and one/two-sided reporting policy.
    pub grid: FrequencyGridPolicy,
    /// Denominator magnitude-squared floor below which the model is rejected.
    pub denominator_floor: f64,
    /// Resource ceilings.
    pub limits: EstimatorLimits,
}

impl MemSpectrumPlan {
    /// Creates a one-sided FFT-bin maximum-entropy spectrum plan.
    pub fn new(sample_rate_hz: f64, fft_len: usize) -> Self {
        Self {
            sample_rate_hz,
            fft_len,
            grid: FrequencyGridPolicy::default(),
            denominator_floor: 1.0e-18,
            limits: EstimatorLimits::default(),
        }
    }
}

/// Evaluates the maximum-entropy power spectral density of a stable AR model.
pub fn mem_spectrum(
    model: &ArModel,
    plan: &MemSpectrumPlan,
) -> Result<SpectrumEstimate, SignalError> {
    if !plan.sample_rate_hz.is_finite() || plan.sample_rate_hz <= 0.0 || plan.fft_len == 0 {
        return Err(SignalError::InvalidPolicy {
            policy: "MEM spectrum grid",
            reason: "positive finite sample rate and transform length are required",
        });
    }
    if !plan.denominator_floor.is_finite() || plan.denominator_floor <= 0.0 {
        return Err(SignalError::InvalidPolicy {
            policy: "MEM denominator floor",
            reason: "a finite positive floor is required",
        });
    }
    let grid = resolve_grid(&plan.grid, plan.sample_rate_hz, plan.fft_len, plan.limits)?;
    let work_units = u64::try_from(grid.frequency.len())
        .ok()
        .and_then(|bins| {
            u64::try_from(model.coefficients.len() + 1)
                .ok()?
                .checked_mul(bins)
        })
        .ok_or(SignalError::InvalidPolicy {
            policy: "MEM spectrum work",
            reason: "work-unit arithmetic overflowed",
        })?;
    admit_work(work_units, plan.limits)?;
    let mut power = Vec::with_capacity(grid.frequency.len());
    for (index, frequency) in grid.frequency.iter().copied().enumerate() {
        let omega = std::f64::consts::TAU * frequency / plan.sample_rate_hz;
        let mut real = 1.0;
        let mut imag = 0.0;
        for (lag, coefficient) in model.coefficients.iter().copied().enumerate() {
            let phase = -omega * (lag + 1) as f64;
            real += coefficient * phase.cos();
            imag += coefficient * phase.sin();
        }
        let denominator = real * real + imag * imag;
        if !denominator.is_finite() || denominator < plan.denominator_floor {
            return Err(SignalError::SingularModel { order: index });
        }
        power.push(
            bin_multiplier(frequency, plan.sample_rate_hz, grid.side) * model.innovation_variance
                / (plan.sample_rate_hz * denominator),
        );
    }
    let frequency_bins = grid.frequency.len();
    Ok(SpectrumEstimate {
        frequency: grid.frequency,
        power,
        scaling: SpectrumScaling {
            kind: SpectrumScalingKind::Density,
            sample_rate_hz: Some(plan.sample_rate_hz),
            normalization_denominator: plan.sample_rate_hz,
            one_sided: matches!(grid.side, crate::SpectrumSide::OneSided),
            interior_bin_multiplier: if matches!(grid.side, crate::SpectrumSide::OneSided) {
                2.0
            } else {
                1.0
            },
        },
        evidence: EstimatorEvidence {
            estimator: EstimatorKind::MaximumEntropy,
            input_len: model.evidence.input_len,
            fft_len: plan.fft_len,
            segment_len: model.evidence.input_len,
            segment_count: 1,
            taper_count: 0,
            frequency_bins,
            work_units,
            work_limit: plan.limits.max_work,
            degrees_of_freedom: (model.evidence.input_len - model.evidence.effective_order) as f64,
            frequency_grid: plan.grid.clone(),
            window: None,
            taper_concentrations: Vec::new(),
        },
    })
}
