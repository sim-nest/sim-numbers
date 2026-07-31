//! Public plans and reports shared by classical spectral estimators.

use crate::{SignalError, WindowMetrics, WindowSpec};

/// Whether reported non-DC power represents one or both frequency signs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpectrumSide {
    /// Report non-negative frequencies and fold negative-frequency power into
    /// interior bins.
    #[default]
    OneSided,
    /// Report the complete signed spectrum without mirror folding.
    TwoSided,
}

/// Policy that constructs the frequencies evaluated by an estimator.
#[derive(Clone, Debug, PartialEq)]
pub enum FrequencyGridPolicy {
    /// Use the exact DFT bins implied by the plan's transform length.
    FftBins {
        /// One-sided folded bins or the complete signed grid.
        side: SpectrumSide,
    },
    /// Evaluate an inclusive, evenly spaced grid by direct Fourier sums.
    Linear {
        /// First frequency in hertz.
        start_hz: f64,
        /// Last frequency in hertz.
        end_hz: f64,
        /// Number of frequencies, including both endpoints.
        bins: usize,
        /// Whether negative-frequency power is folded into the result.
        side: SpectrumSide,
    },
    /// Evaluate caller-supplied, strictly increasing frequencies.
    Explicit {
        /// Frequencies in hertz.
        frequencies_hz: Vec<f64>,
        /// Whether negative-frequency power is folded into the result.
        side: SpectrumSide,
    },
}

impl Default for FrequencyGridPolicy {
    fn default() -> Self {
        Self::FftBins {
            side: SpectrumSide::OneSided,
        }
    }
}

/// Output units for a Fourier power estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpectrumScalingKind {
    /// Squared signal units, corrected for coherent window gain.
    Power,
    /// Squared signal units per hertz, corrected for window energy.
    #[default]
    Density,
    /// Fraction of centered variance explained by a sinusoidal least-squares
    /// fit. This is the generalized Lomb-Scargle normalization.
    LombScargleNormalized,
}

/// Exact denominator and folding convention used to reconstruct output scale.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumScaling {
    /// Semantic output units.
    pub kind: SpectrumScalingKind,
    /// Sample rate used by Fourier estimators, when applicable.
    pub sample_rate_hz: Option<f64>,
    /// Divisor applied to squared Fourier magnitude or variance reduction.
    pub normalization_denominator: f64,
    /// Whether negative-frequency power is folded into positive frequencies.
    pub one_sided: bool,
    /// Multiplier for non-DC, non-Nyquist bins under one-sided folding.
    pub interior_bin_multiplier: f64,
}

/// Explicit resource ceilings applied before estimator work begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EstimatorLimits {
    /// Maximum transform length.
    pub max_fft_len: usize,
    /// Maximum number of Welch segments.
    pub max_segments: usize,
    /// Maximum number of Slepian tapers.
    pub max_tapers: usize,
    /// Maximum number of requested output frequencies.
    pub max_frequency_bins: usize,
    /// Conservative deterministic work-unit ceiling.
    pub max_work: u64,
}

impl Default for EstimatorLimits {
    fn default() -> Self {
        Self {
            max_fft_len: 16_384,
            max_segments: 4_096,
            max_tapers: 16,
            max_frequency_bins: 16_385,
            max_work: 100_000_000,
        }
    }
}

/// Estimator that produced a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimatorKind {
    /// Single-window periodogram.
    Periodogram,
    /// Averaged, overlapping Welch periodogram.
    Welch,
    /// Averaged auto/cross spectrum and coherence.
    CrossSpectrum,
    /// Slepian discrete-prolate multitaper estimate.
    SlepianMultitaper,
    /// Uneven-sample generalized Lomb-Scargle estimate.
    LombScargle,
}

/// Work, averaging, grid, and window facts retained beside an estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct EstimatorEvidence {
    /// Estimator family.
    pub estimator: EstimatorKind,
    /// Number of input observations admitted by the plan.
    pub input_len: usize,
    /// Transform length or reference grid length.
    pub fft_len: usize,
    /// Samples per segment or taper.
    pub segment_len: usize,
    /// Number of admitted data segments.
    pub segment_count: usize,
    /// Number of Slepian tapers.
    pub taper_count: usize,
    /// Number of output frequencies.
    pub frequency_bins: usize,
    /// Conservative work charged before execution.
    pub work_units: u64,
    /// Work ceiling that admitted this execution.
    pub work_limit: u64,
    /// Nominal independent chi-square or residual degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Resolved frequency-grid policy.
    pub frequency_grid: FrequencyGridPolicy,
    /// Generated window metrics for periodogram-family estimates.
    pub window: Option<WindowMetrics>,
    /// Spectral concentration of each selected Slepian taper.
    pub taper_concentrations: Vec<f64>,
}

/// Power values and all policy/evidence needed to interpret their scale.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumEstimate {
    /// Evaluated frequencies in hertz.
    pub frequency: Vec<f64>,
    /// Power, density, or normalized variance reduction at each frequency.
    pub power: Vec<f64>,
    /// Exact scaling convention.
    pub scaling: SpectrumScaling,
    /// Bounded-work and averaging evidence.
    pub evidence: EstimatorEvidence,
}

/// Complex cross power, auto power, and magnitude-squared coherence.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSpectrumEstimate {
    /// Evaluated frequencies in hertz.
    pub frequency: Vec<f64>,
    /// Complex `X * conjugate(Y)` values as `(real, imaginary)` pairs.
    pub cross_power: Vec<(f64, f64)>,
    /// Auto power for the first signal.
    pub x_power: Vec<f64>,
    /// Auto power for the second signal.
    pub y_power: Vec<f64>,
    /// Magnitude-squared coherence, clamped to `[0, 1]`.
    pub coherence: Vec<f64>,
    /// Exact scaling convention shared by cross and auto power.
    pub scaling: SpectrumScaling,
    /// Bounded-work and averaging evidence.
    pub evidence: EstimatorEvidence,
}

/// Single-record periodogram policy.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodogramPlan {
    /// Sample rate in hertz.
    pub sample_rate_hz: f64,
    /// Transform length, including any zero padding.
    pub fft_len: usize,
    /// Analysis-window policy.
    pub window: WindowSpec,
    /// Frequency-grid policy.
    pub grid: FrequencyGridPolicy,
    /// Power or power-density output.
    pub scaling: SpectrumScalingKind,
    /// Resource ceilings.
    pub limits: EstimatorLimits,
}

impl PeriodogramPlan {
    /// Creates a one-sided density periodogram with a symmetric Hann window.
    pub fn new(sample_rate_hz: f64, fft_len: usize) -> Self {
        Self {
            sample_rate_hz,
            fft_len,
            window: WindowSpec::default(),
            grid: FrequencyGridPolicy::default(),
            scaling: SpectrumScalingKind::Density,
            limits: EstimatorLimits::default(),
        }
    }
}

/// Segment, overlap, and scaling policy shared by Welch and cross spectra.
#[derive(Clone, Debug, PartialEq)]
pub struct WelchPlan {
    /// Sample rate in hertz.
    pub sample_rate_hz: f64,
    /// Samples in each complete segment.
    pub segment_len: usize,
    /// Reused samples between adjacent segments.
    pub overlap: usize,
    /// Transform length, including segment zero padding.
    pub fft_len: usize,
    /// Analysis-window policy.
    pub window: WindowSpec,
    /// Frequency-grid policy.
    pub grid: FrequencyGridPolicy,
    /// Power or power-density output.
    pub scaling: SpectrumScalingKind,
    /// Resource ceilings.
    pub limits: EstimatorLimits,
}

impl WelchPlan {
    /// Creates a 50%-overlapped, one-sided Hann density plan.
    pub fn new(sample_rate_hz: f64, segment_len: usize) -> Self {
        Self {
            sample_rate_hz,
            segment_len,
            overlap: segment_len / 2,
            fft_len: segment_len,
            window: WindowSpec::default(),
            grid: FrequencyGridPolicy::default(),
            scaling: SpectrumScalingKind::Density,
            limits: EstimatorLimits::default(),
        }
    }
}

/// Slepian multitaper policy with explicit bandwidth and taper count.
#[derive(Clone, Debug, PartialEq)]
pub struct MultitaperPlan {
    /// Sample rate in hertz.
    pub sample_rate_hz: f64,
    /// Transform length, including zero padding.
    pub fft_len: usize,
    /// Time-half-bandwidth product `N * W`, strictly between zero and `N / 2`.
    pub time_bandwidth: f64,
    /// Number of leading discrete prolate tapers to average.
    pub taper_count: usize,
    /// Frequency-grid policy.
    pub grid: FrequencyGridPolicy,
    /// Resource ceilings.
    pub limits: EstimatorLimits,
}

impl MultitaperPlan {
    /// Creates a one-sided density plan.
    pub fn new(
        sample_rate_hz: f64,
        fft_len: usize,
        time_bandwidth: f64,
        taper_count: usize,
    ) -> Self {
        Self {
            sample_rate_hz,
            fft_len,
            time_bandwidth,
            taper_count,
            grid: FrequencyGridPolicy::default(),
            limits: EstimatorLimits::default(),
        }
    }
}

/// Uneven-sample generalized Lomb-Scargle policy.
#[derive(Clone, Debug, PartialEq)]
pub struct LombScarglePlan {
    /// Reference sample rate used to validate the frequency grid.
    pub sample_rate_hz: f64,
    /// Reference DFT length used by [`FrequencyGridPolicy::FftBins`].
    pub fft_len: usize,
    /// Frequency-grid policy. A one-sided positive linear or explicit grid is
    /// usually clearest for uneven samples.
    pub grid: FrequencyGridPolicy,
    /// Resource ceilings.
    pub limits: EstimatorLimits,
}

impl LombScarglePlan {
    /// Creates a one-sided FFT-bin grid under the supplied reference rate.
    pub fn new(sample_rate_hz: f64, fft_len: usize) -> Self {
        Self {
            sample_rate_hz,
            fft_len,
            grid: FrequencyGridPolicy::default(),
            limits: EstimatorLimits::default(),
        }
    }
}

pub(crate) fn validate_common(
    sample_rate_hz: f64,
    fft_len: usize,
    segment_len: usize,
    limits: EstimatorLimits,
) -> Result<(), SignalError> {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(SignalError::InvalidPolicy {
            policy: "sample rate",
            reason: "sample rate must be finite and positive",
        });
    }
    if segment_len == 0 || fft_len < segment_len {
        return Err(SignalError::InvalidLength {
            len: fft_len,
            reason: "spectral transform length must contain the non-empty segment",
        });
    }
    if fft_len > limits.max_fft_len {
        return Err(SignalError::InvalidPolicy {
            policy: "FFT length limit",
            reason: "transform length exceeds the estimator limit",
        });
    }
    Ok(())
}

pub(crate) fn admit_work(required: u64, limits: EstimatorLimits) -> Result<(), SignalError> {
    if required > limits.max_work {
        Err(SignalError::WorkLimit {
            required,
            maximum: limits.max_work,
        })
    } else {
        Ok(())
    }
}
