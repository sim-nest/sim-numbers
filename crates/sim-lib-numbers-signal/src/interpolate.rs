//! Continuous periodic evaluation and integration of complete DFT coefficient sets.

use std::f64::consts::TAU;

use crate::{Normalization, SignConvention, SignalError, fft::Complex};

/// Whether query coordinates wrap or must remain in one principal period.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Periodicity {
    /// Reduce every finite coordinate modulo the declared period.
    #[default]
    Wrap,
    /// Reject coordinates outside the declared principal period.
    PrincipalPeriod,
}

/// Whether the principal grid contains a duplicate sample at its upper endpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EndpointConvention {
    /// Samples occupy `[origin, origin + period)` with no duplicate endpoint.
    #[default]
    Excluded,
    /// The upper endpoint is admitted and equals the periodic origin.
    Included,
}

/// Signed-frequency assigned to the unique even-length Nyquist bin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NyquistConvention {
    /// Interpret bin `N/2` as positive Nyquist frequency.
    #[default]
    Positive,
    /// Interpret bin `N/2` as negative Nyquist frequency.
    Negative,
}

/// Complete DFT-series convention and deterministic resource ceilings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DftSeriesPlan {
    /// Coordinate corresponding to phase zero.
    pub origin: f64,
    /// Positive coordinate width of one period.
    pub period: f64,
    /// Wrapping or principal-period admission.
    pub periodicity: Periodicity,
    /// Duplicate-upper-endpoint convention for principal coordinates.
    pub endpoint: EndpointConvention,
    /// Scaling used when the supplied bins were produced by a forward transform.
    pub normalization: Normalization,
    /// Exponential sign used by the supplying forward transform.
    pub sign: SignConvention,
    /// Signed-frequency interpretation of an even-length Nyquist bin.
    pub nyquist: NyquistConvention,
    /// Maximum number of coordinates in one interpolation request.
    pub max_points: usize,
    /// Maximum coefficient-point products in one request.
    pub max_work: u64,
}

impl Default for DftSeriesPlan {
    fn default() -> Self {
        Self {
            origin: 0.0,
            period: 1.0,
            periodicity: Periodicity::Wrap,
            endpoint: EndpointConvention::Excluded,
            normalization: Normalization::Inverse,
            sign: SignConvention::NegativeForward,
            nyquist: NyquistConvention::Positive,
            max_points: 16_384,
            max_work: 100_000_000,
        }
    }
}

/// Convention and bounded-work evidence for DFT series evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DftSeriesReport {
    /// Number of complete DFT bins.
    pub bins: usize,
    /// Number of coordinates evaluated.
    pub points: usize,
    /// Coordinate origin.
    pub origin: f64,
    /// Coordinate period.
    pub period: f64,
    /// Wrapping convention.
    pub periodicity: Periodicity,
    /// Endpoint convention.
    pub endpoint: EndpointConvention,
    /// Coefficient normalization.
    pub normalization: Normalization,
    /// Forward exponential sign.
    pub sign: SignConvention,
    /// Even-length Nyquist interpretation.
    pub nyquist: NyquistConvention,
    /// Coefficient-point products charged by the request.
    pub work_units: u64,
    /// Work ceiling that admitted the request.
    pub work_limit: u64,
}

/// Interpolated complex values and their explicit DFT-series evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct DftInterpolation {
    /// Complex series values in query order.
    pub values: Vec<(f64, f64)>,
    /// Query positions expressed in cycles from the declared origin.
    pub phase_cycles: Vec<f64>,
    /// Grid, transform, and work evidence.
    pub report: DftSeriesReport,
}

/// Exact integral of a DFT series between two coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct DftIntegral {
    /// Complex definite integral `(real, imaginary)`.
    pub value: (f64, f64),
    /// Start and end positions expressed in cycles from the declared origin.
    pub phase_cycles: (f64, f64),
    /// Grid, transform, and work evidence.
    pub report: DftSeriesReport,
}

/// Evaluates the periodic trigonometric interpolant represented by complete DFT bins.
pub fn dft_interpolate(
    bins: &[(f64, f64)],
    coordinates: &[f64],
    plan: &DftSeriesPlan,
) -> Result<DftInterpolation, SignalError> {
    validate_series(bins, plan)?;
    let work_units = admit_points(bins.len(), coordinates.len(), plan)?;
    let phase_cycles = coordinates
        .iter()
        .copied()
        .enumerate()
        .map(|(index, coordinate)| coordinate_phase(coordinate, index, plan))
        .collect::<Result<Vec<_>, _>>()?;
    let scale = reconstruction_scale(plan.normalization, bins.len());
    let inverse_sign = plan.sign.angle_sign(crate::Direction::Inverse);
    let values = phase_cycles
        .iter()
        .map(|phase| {
            bins.iter()
                .copied()
                .enumerate()
                .fold(Complex::ZERO, |sum, (index, bin)| {
                    let frequency = signed_frequency(index, bins.len(), plan.nyquist);
                    sum + Complex::from(bin) * Complex::cis(inverse_sign * TAU * frequency * phase)
                })
                .scale(scale)
                .into()
        })
        .collect::<Vec<_>>();
    validate_complex(&values)?;
    Ok(DftInterpolation {
        values,
        phase_cycles,
        report: report(bins.len(), coordinates.len(), work_units, plan),
    })
}

/// Integrates the periodic trigonometric interpolant exactly over a coordinate interval.
pub fn dft_integrate(
    bins: &[(f64, f64)],
    start: f64,
    end: f64,
    plan: &DftSeriesPlan,
) -> Result<DftIntegral, SignalError> {
    validate_series(bins, plan)?;
    let work_units = admit_points(bins.len(), 2, plan)?;
    let start_phase = coordinate_phase(start, 0, plan)?;
    let end_phase = coordinate_phase(end, 1, plan)?;
    let scale = reconstruction_scale(plan.normalization, bins.len());
    let inverse_sign = plan.sign.angle_sign(crate::Direction::Inverse);
    let mut integral = Complex::ZERO;
    for (index, bin) in bins.iter().copied().enumerate() {
        let coefficient = Complex::from(bin).scale(scale);
        let frequency = signed_frequency(index, bins.len(), plan.nyquist);
        if frequency == 0.0 {
            integral += coefficient.scale((end_phase - start_phase) * plan.period);
            continue;
        }
        let angular = inverse_sign * TAU * frequency;
        let delta = Complex::cis(angular * end_phase) - Complex::cis(angular * start_phase);
        integral += coefficient * Complex::new(0.0, -plan.period / angular) * delta;
    }
    let value: (f64, f64) = integral.into();
    validate_complex(&[value])?;
    Ok(DftIntegral {
        value,
        phase_cycles: (start_phase, end_phase),
        report: report(bins.len(), 2, work_units, plan),
    })
}

/// Evaluates one integer DFT bin directly without allocating the complete spectrum.
pub fn dft_bin(
    samples: &[(f64, f64)],
    bin: usize,
    normalization: Normalization,
    sign: SignConvention,
) -> Result<(f64, f64), SignalError> {
    if samples.is_empty() {
        return Err(SignalError::InvalidLength {
            len: 0,
            reason: "single-bin DFT evaluation requires at least one sample",
        });
    }
    if bin >= samples.len() {
        return Err(SignalError::InvalidPolicy {
            policy: "single-bin DFT index",
            reason: "bin must be smaller than the transform length",
        });
    }
    validate_complex(samples)?;
    let angle_sign = sign.angle_sign(crate::Direction::Forward);
    let len = samples.len() as f64;
    let value = samples
        .iter()
        .copied()
        .enumerate()
        .fold(Complex::ZERO, |sum, (index, value)| {
            sum + Complex::from(value)
                * Complex::cis(angle_sign * TAU * bin as f64 * index as f64 / len)
        })
        .scale(forward_scale(normalization, samples.len()));
    Ok(value.into())
}

/// Evaluates one integer DFT bin of a real signal directly.
pub fn dft_bin_real(
    samples: &[f64],
    bin: usize,
    normalization: Normalization,
    sign: SignConvention,
) -> Result<(f64, f64), SignalError> {
    let complex = samples
        .iter()
        .map(|value| (*value, 0.0))
        .collect::<Vec<_>>();
    dft_bin(&complex, bin, normalization, sign)
}

pub(crate) fn forward_scale(normalization: Normalization, len: usize) -> f64 {
    match normalization {
        Normalization::None | Normalization::Inverse => 1.0,
        Normalization::Forward => 1.0 / len as f64,
        Normalization::Orthonormal => 1.0 / (len as f64).sqrt(),
    }
}

pub(crate) fn reconstruction_scale(normalization: Normalization, len: usize) -> f64 {
    1.0 / (len as f64 * forward_scale(normalization, len))
}

fn validate_series(bins: &[(f64, f64)], plan: &DftSeriesPlan) -> Result<(), SignalError> {
    if bins.is_empty() {
        return Err(SignalError::InvalidLength {
            len: 0,
            reason: "DFT interpolation requires at least one complete bin",
        });
    }
    validate_complex(bins)?;
    if !plan.origin.is_finite() || !plan.period.is_finite() || plan.period <= 0.0 {
        return Err(SignalError::InvalidPolicy {
            policy: "DFT interpolation grid",
            reason: "finite origin and positive finite period are required",
        });
    }
    Ok(())
}

fn coordinate_phase(
    coordinate: f64,
    index: usize,
    plan: &DftSeriesPlan,
) -> Result<f64, SignalError> {
    if !coordinate.is_finite() {
        return Err(SignalError::NonFinite {
            index,
            component: "coordinate",
        });
    }
    let phase = (coordinate - plan.origin) / plan.period;
    if plan.periodicity == Periodicity::Wrap {
        return Ok(phase);
    }
    let admitted = match plan.endpoint {
        EndpointConvention::Excluded => (0.0..1.0).contains(&phase),
        EndpointConvention::Included => (0.0..=1.0).contains(&phase),
    };
    if !admitted {
        return Err(SignalError::InvalidPolicy {
            policy: "DFT interpolation coordinate",
            reason: "coordinate lies outside the declared principal-period endpoint policy",
        });
    }
    Ok(phase)
}

fn admit_points(bins: usize, points: usize, plan: &DftSeriesPlan) -> Result<u64, SignalError> {
    if points > plan.max_points {
        return Err(SignalError::InvalidPolicy {
            policy: "DFT interpolation point limit",
            reason: "request contains more coordinates than the plan admits",
        });
    }
    let work = u64::try_from(bins)
        .ok()
        .and_then(|bins| u64::try_from(points).ok()?.checked_mul(bins))
        .ok_or(SignalError::InvalidPolicy {
            policy: "DFT interpolation work",
            reason: "work-unit arithmetic overflowed",
        })?;
    if work > plan.max_work {
        return Err(SignalError::WorkLimit {
            required: work,
            maximum: plan.max_work,
        });
    }
    Ok(work)
}

fn signed_frequency(index: usize, len: usize, nyquist: NyquistConvention) -> f64 {
    if index > len / 2
        || (len.is_multiple_of(2) && index == len / 2 && nyquist == NyquistConvention::Negative)
    {
        index as f64 - len as f64
    } else {
        index as f64
    }
}

fn validate_complex(values: &[(f64, f64)]) -> Result<(), SignalError> {
    for (index, (real, imag)) in values.iter().copied().enumerate() {
        if !real.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "real",
            });
        }
        if !imag.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "imag",
            });
        }
    }
    Ok(())
}

fn report(bins: usize, points: usize, work_units: u64, plan: &DftSeriesPlan) -> DftSeriesReport {
    DftSeriesReport {
        bins,
        points,
        origin: plan.origin,
        period: plan.period,
        periodicity: plan.periodicity,
        endpoint: plan.endpoint,
        normalization: plan.normalization,
        sign: plan.sign,
        nyquist: plan.nyquist,
        work_units,
        work_limit: plan.max_work,
    }
}
