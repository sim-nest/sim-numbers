//! Multilevel discrete wavelet transforms with explicit edge extension.

use crate::{SignalError, convolution::validate_real_signal};

const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Extension applied when a wavelet lifting step reaches beyond a finite signal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Repeat the signal with its declared period.
    #[default]
    Periodic,
    /// Reflect about the half-sample boundary, retaining the edge sample.
    Symmetric,
    /// Supply zero beyond either end of the signal.
    Zero,
}

/// Reversible analysis/synthesis wavelet family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wavelet {
    /// Orthonormal two-sample Haar transform.
    #[default]
    Haar,
    /// Biorthogonal Le Gall 5/3 lifting transform.
    LeGall53,
}

/// Complete policy for a multilevel discrete wavelet transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveletPlan {
    /// Analysis/synthesis wavelet family.
    pub wavelet: Wavelet,
    /// Number of recursively decomposed approximation levels.
    pub levels: usize,
    /// Extension used by boundary-crossing lifting steps.
    pub boundary: BoundaryMode,
}

impl WaveletPlan {
    /// Creates a plan with an explicit wavelet and decomposition depth.
    pub const fn new(wavelet: Wavelet, levels: usize) -> Self {
        Self {
            wavelet,
            levels,
            boundary: BoundaryMode::Periodic,
        }
    }
}

/// Detail coefficients and reconstruction shape for one analysis level.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveletLevel {
    /// Signal length entering this level, retained for exact odd-length synthesis.
    pub input_len: usize,
    /// High-pass/detail coefficients in increasing sample-pair order.
    pub detail: Vec<f64>,
}

/// Multilevel wavelet coefficients and the policy needed to interpret them.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveletCoefficients {
    /// Low-pass coefficients after the final requested level.
    pub approximation: Vec<f64>,
    /// Detail records from the finest level to the coarsest level.
    pub levels: Vec<WaveletLevel>,
    /// Wavelet used for analysis.
    pub wavelet: Wavelet,
    /// Boundary extension used for analysis.
    pub boundary: BoundaryMode,
    /// Original sample count.
    pub original_len: usize,
}

/// Computes a multilevel discrete wavelet transform.
pub fn dwt(signal: &[f64], plan: &WaveletPlan) -> Result<WaveletCoefficients, SignalError> {
    validate_real_signal(signal)?;
    validate_plan(signal.len(), plan)?;
    let mut approximation = signal.to_vec();
    let mut levels = Vec::with_capacity(plan.levels);
    for _ in 0..plan.levels {
        let input_len = approximation.len();
        let (next, detail) = match plan.wavelet {
            Wavelet::Haar => haar_analysis(&approximation),
            Wavelet::LeGall53 => legall_analysis(&approximation, plan.boundary),
        };
        validate_real_signal(&next)?;
        validate_real_signal(&detail)?;
        approximation = next;
        levels.push(WaveletLevel { input_len, detail });
    }
    Ok(WaveletCoefficients {
        approximation,
        levels,
        wavelet: plan.wavelet,
        boundary: plan.boundary,
        original_len: signal.len(),
    })
}

/// Reconstructs samples from multilevel wavelet coefficients.
pub fn idwt(
    coefficients: &WaveletCoefficients,
    plan: &WaveletPlan,
) -> Result<Vec<f64>, SignalError> {
    if coefficients.wavelet != plan.wavelet || coefficients.boundary != plan.boundary {
        return Err(SignalError::InvalidPolicy {
            policy: "wavelet synthesis",
            reason: "the synthesis plan must match the coefficient policy",
        });
    }
    if coefficients.levels.len() != plan.levels {
        return Err(SignalError::LengthMismatch {
            expected: plan.levels,
            actual: coefficients.levels.len(),
        });
    }
    validate_real_signal(&coefficients.approximation)?;
    let mut approximation = coefficients.approximation.clone();
    for level in coefficients.levels.iter().rev() {
        validate_real_signal(&level.detail)?;
        approximation = match plan.wavelet {
            Wavelet::Haar => haar_synthesis(&approximation, &level.detail, level.input_len)?,
            Wavelet::LeGall53 => legall_synthesis(
                &approximation,
                &level.detail,
                level.input_len,
                plan.boundary,
            )?,
        };
    }
    if approximation.len() != coefficients.original_len {
        return Err(SignalError::LengthMismatch {
            expected: coefficients.original_len,
            actual: approximation.len(),
        });
    }
    validate_real_signal(&approximation)?;
    Ok(approximation)
}

fn validate_plan(len: usize, plan: &WaveletPlan) -> Result<(), SignalError> {
    if len < 2 {
        return Err(SignalError::InvalidLength {
            len,
            reason: "a wavelet transform requires at least two samples",
        });
    }
    if plan.levels == 0 {
        return Err(SignalError::InvalidPolicy {
            policy: "wavelet levels",
            reason: "at least one decomposition level is required",
        });
    }
    let mut available = len;
    for _ in 0..plan.levels {
        if available < 2 {
            return Err(SignalError::InvalidPolicy {
                policy: "wavelet levels",
                reason: "the requested depth exceeds the available approximation samples",
            });
        }
        available = available.div_ceil(2);
    }
    Ok(())
}

fn haar_analysis(input: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut approximation = Vec::with_capacity(input.len().div_ceil(2));
    let mut detail = Vec::with_capacity(input.len() / 2);
    let pairs = input.len() / 2;
    for pair in 0..pairs {
        let even = input[2 * pair];
        let odd = input[2 * pair + 1];
        approximation.push((even + odd) / SQRT_2);
        detail.push((even - odd) / SQRT_2);
    }
    if input.len() % 2 == 1 {
        approximation.push(input[input.len() - 1]);
    }
    (approximation, detail)
}

fn haar_synthesis(
    approximation: &[f64],
    detail: &[f64],
    input_len: usize,
) -> Result<Vec<f64>, SignalError> {
    let expected_approximation = input_len.div_ceil(2);
    let expected_detail = input_len / 2;
    validate_level_lengths(
        approximation,
        detail,
        expected_approximation,
        expected_detail,
    )?;
    let mut output = Vec::with_capacity(input_len);
    for (&approximation, &detail) in approximation.iter().zip(detail) {
        output.push((approximation + detail) / SQRT_2);
        output.push((approximation - detail) / SQRT_2);
    }
    if input_len % 2 == 1 {
        output.push(approximation[expected_approximation - 1]);
    }
    Ok(output)
}

fn legall_analysis(input: &[f64], boundary: BoundaryMode) -> (Vec<f64>, Vec<f64>) {
    let mut approximation = input.iter().step_by(2).copied().collect::<Vec<_>>();
    let mut detail = input.iter().skip(1).step_by(2).copied().collect::<Vec<_>>();
    for index in 0..detail.len() {
        detail[index] -=
            0.5 * (approximation[index] + extended(&approximation, index as isize + 1, boundary));
    }
    if !detail.is_empty() {
        for (index, value) in approximation.iter_mut().enumerate() {
            *value += 0.25
                * (extended(&detail, index as isize - 1, boundary)
                    + extended(&detail, index as isize, boundary));
        }
    }
    (approximation, detail)
}

fn legall_synthesis(
    approximation: &[f64],
    detail: &[f64],
    input_len: usize,
    boundary: BoundaryMode,
) -> Result<Vec<f64>, SignalError> {
    validate_level_lengths(approximation, detail, input_len.div_ceil(2), input_len / 2)?;
    let mut even = approximation.to_vec();
    let mut odd = detail.to_vec();
    if !odd.is_empty() {
        for (index, value) in even.iter_mut().enumerate() {
            *value -= 0.25
                * (extended(&odd, index as isize - 1, boundary)
                    + extended(&odd, index as isize, boundary));
        }
    }
    for (index, value) in odd.iter_mut().enumerate() {
        *value += 0.5 * (even[index] + extended(&even, index as isize + 1, boundary));
    }
    let mut output = Vec::with_capacity(input_len);
    for (index, value) in even.iter().enumerate() {
        output.push(*value);
        if let Some(value) = odd.get(index) {
            output.push(*value);
        }
    }
    Ok(output)
}

fn validate_level_lengths(
    approximation: &[f64],
    detail: &[f64],
    expected_approximation: usize,
    expected_detail: usize,
) -> Result<(), SignalError> {
    if approximation.len() != expected_approximation {
        return Err(SignalError::LengthMismatch {
            expected: expected_approximation,
            actual: approximation.len(),
        });
    }
    if detail.len() != expected_detail {
        return Err(SignalError::LengthMismatch {
            expected: expected_detail,
            actual: detail.len(),
        });
    }
    Ok(())
}

pub(crate) fn extended(values: &[f64], index: isize, boundary: BoundaryMode) -> f64 {
    if index >= 0 && (index as usize) < values.len() {
        return values[index as usize];
    }
    match boundary {
        BoundaryMode::Zero => 0.0,
        BoundaryMode::Periodic => {
            let len = values.len() as isize;
            values[index.rem_euclid(len) as usize]
        }
        BoundaryMode::Symmetric => {
            let len = values.len() as isize;
            let reflected = index.rem_euclid(2 * len);
            let reflected = if reflected < len {
                reflected
            } else {
                2 * len - 1 - reflected
            };
            values[reflected as usize]
        }
    }
}
