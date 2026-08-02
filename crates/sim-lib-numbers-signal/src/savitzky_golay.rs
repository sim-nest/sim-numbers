//! Savitzky-Golay design and finite-signal application.

use sim_lib_numbers_tensor_linalg::{DenseSolveOptions, DenseSolveReport, solve_dense_f64};

use crate::{
    BoundaryMode, SignalError, convolution::validate_real_signal, linalg_support::dense_error,
    wavelet::extended,
};

/// Canonical centered Savitzky-Golay design policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavitzkyGolaySpec {
    /// Odd number of samples in the local polynomial window.
    pub window_length: usize,
    /// Maximum fitted polynomial degree.
    pub polynomial_order: usize,
    /// Derivative order returned by the filter (`0` means smoothing).
    pub derivative_order: usize,
    /// Positive spacing between consecutive samples.
    pub sample_spacing: f64,
    /// Relative scaled-pivot threshold used by filter design.
    pub singularity_threshold: f64,
}

impl Default for SavitzkyGolaySpec {
    fn default() -> Self {
        Self {
            window_length: 5,
            polynomial_order: 2,
            derivative_order: 0,
            sample_spacing: 1.0,
            singularity_threshold: 1e-12,
        }
    }
}

/// Designed finite impulse response and reconstructable polynomial-fit evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteImpulseResponse {
    /// Coefficients ordered from the most negative to most positive sample offset.
    pub coefficients: Vec<f64>,
    /// Design policy, including derivative and physical sample spacing.
    pub spec: SavitzkyGolaySpec,
    /// Pivot and residual evidence for the moment-system solve.
    pub design_report: DenseSolveReport,
}

/// Designs a centered Savitzky-Golay smoothing or derivative filter.
///
/// Derivative filters always include the canonical `derivative_order! /
/// sample_spacing^derivative_order` scale. This API has no implicit unscaled
/// legacy dialect.
pub fn savitzky_golay(spec: SavitzkyGolaySpec) -> Result<FiniteImpulseResponse, SignalError> {
    validate_spec(spec)?;
    let terms = spec.polynomial_order + 1;
    let radius = spec.window_length / 2;
    let offsets = (0..spec.window_length)
        .map(|index| index as f64 - radius as f64)
        .collect::<Vec<_>>();
    let mut gram = vec![0.0; terms * terms];
    for row in 0..terms {
        for column in 0..terms {
            gram[row * terms + column] = offsets
                .iter()
                .map(|offset| offset.powi((row + column) as i32))
                .sum();
        }
    }
    let mut target = vec![0.0; terms];
    target[spec.derivative_order] =
        factorial(spec.derivative_order) / spec.sample_spacing.powi(spec.derivative_order as i32);
    let solution = solve_dense_f64(
        &gram,
        &target,
        DenseSolveOptions {
            singularity_threshold: spec.singularity_threshold,
        },
    )
    .map_err(|error| dense_error(error, "Savitzky-Golay design"))?;
    let coefficients = offsets
        .iter()
        .map(|offset| {
            solution
                .values
                .iter()
                .enumerate()
                .map(|(power, coefficient)| coefficient * offset.powi(power as i32))
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    validate_real_signal(&coefficients)?;
    Ok(FiniteImpulseResponse {
        coefficients,
        spec,
        design_report: solution.report,
    })
}

/// Applies a designed Savitzky-Golay FIR to every sample under an edge policy.
pub fn apply_savitzky_golay(
    signal: &[f64],
    filter: &FiniteImpulseResponse,
    boundary: BoundaryMode,
) -> Result<Vec<f64>, SignalError> {
    if signal.is_empty() {
        return Err(SignalError::InvalidLength {
            len: 0,
            reason: "Savitzky-Golay application requires at least one sample",
        });
    }
    validate_real_signal(signal)?;
    if filter.coefficients.len() != filter.spec.window_length {
        return Err(SignalError::LengthMismatch {
            expected: filter.spec.window_length,
            actual: filter.coefficients.len(),
        });
    }
    validate_real_signal(&filter.coefficients)?;
    let radius = (filter.coefficients.len() / 2) as isize;
    let output = (0..signal.len())
        .map(|center| {
            filter
                .coefficients
                .iter()
                .enumerate()
                .map(|(tap, coefficient)| {
                    let sample =
                        extended(signal, center as isize + tap as isize - radius, boundary);
                    coefficient * sample
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    validate_real_signal(&output)?;
    Ok(output)
}

fn validate_spec(spec: SavitzkyGolaySpec) -> Result<(), SignalError> {
    if spec.window_length == 0 || spec.window_length.is_multiple_of(2) {
        return Err(SignalError::InvalidPolicy {
            policy: "Savitzky-Golay window length",
            reason: "the centered window length must be positive and odd",
        });
    }
    if spec.polynomial_order >= spec.window_length {
        return Err(SignalError::InvalidPolicy {
            policy: "Savitzky-Golay polynomial order",
            reason: "the polynomial order must be smaller than the window length",
        });
    }
    if spec.derivative_order > spec.polynomial_order {
        return Err(SignalError::InvalidPolicy {
            policy: "Savitzky-Golay derivative order",
            reason: "the derivative order must not exceed the fitted polynomial order",
        });
    }
    if !spec.sample_spacing.is_finite() || spec.sample_spacing <= 0.0 {
        return Err(SignalError::InvalidPolicy {
            policy: "Savitzky-Golay sample spacing",
            reason: "sample spacing must be finite and strictly positive",
        });
    }
    Ok(())
}

fn factorial(value: usize) -> f64 {
    (1..=value).fold(1.0, |product, factor| product * factor as f64)
}
