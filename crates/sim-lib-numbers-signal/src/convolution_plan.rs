//! Typed convolution policies and inspectable execution plans.

use crate::SignalError;

/// Retained portion of a linear convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearOutput {
    /// Retain all `signal_len + kernel_len - 1` samples.
    Full,
    /// Retain `signal_len` centered samples.
    Same,
    /// Retain only samples for which the kernel is entirely inside the signal.
    Valid,
}

/// Linear or periodic convolution geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolutionMode {
    /// Zero-extended linear convolution with an explicit retained span.
    Linear(LinearOutput),
    /// Periodic convolution with exactly `period` output samples.
    Circular {
        /// Period of both the folded inputs and the result.
        period: usize,
    },
}

/// Treatment of samples outside the declared signal span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryPolicy {
    /// Samples outside a finite signal are zero.
    ZeroPad,
    /// Samples repeat with the circular period.
    Periodic,
}

/// Scaling applied after convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolutionNormalization {
    /// Preserve the mathematical convolution sum.
    None,
    /// Divide every result by the sum of the kernel coefficients.
    KernelSum,
}

/// Requested or selected convolution implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolutionAlgorithm {
    /// Select the lower estimated operation cost.
    Auto,
    /// Definition-level nested summation.
    Direct,
    /// Frequency-domain multiplication through the crate's FFT engine.
    Fft,
}

/// Inspectable cost comparison used to select a convolution implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvolutionCostPlan {
    /// Algorithm requested by the caller.
    pub requested: ConvolutionAlgorithm,
    /// Algorithm selected after comparing costs.
    pub selected: ConvolutionAlgorithm,
    /// Multiplication/addition work units for direct summation.
    pub direct_cost_units: usize,
    /// Transform and pointwise work units for the FFT path.
    pub fft_cost_units: usize,
    /// Transform length used by the FFT path.
    pub fft_len: usize,
    /// Conservative temporary bytes used by the FFT path.
    pub fft_scratch_bytes: usize,
}

/// Fully explicit plan for real one-dimensional convolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvolutionPlan {
    /// Linear retained span or circular period.
    pub mode: ConvolutionMode,
    /// Requested implementation.
    pub algorithm: ConvolutionAlgorithm,
    /// Extension policy outside the signal span.
    pub boundary: BoundaryPolicy,
    /// Output scaling policy.
    pub normalization: ConvolutionNormalization,
}

impl ConvolutionPlan {
    /// Conventional full, zero-extended, unnormalized automatic plan.
    pub const fn linear_full() -> Self {
        Self {
            mode: ConvolutionMode::Linear(LinearOutput::Full),
            algorithm: ConvolutionAlgorithm::Auto,
            boundary: BoundaryPolicy::ZeroPad,
            normalization: ConvolutionNormalization::None,
        }
    }

    /// Builds a periodic, unnormalized automatic plan.
    pub const fn circular(period: usize) -> Self {
        Self {
            mode: ConvolutionMode::Circular { period },
            algorithm: ConvolutionAlgorithm::Auto,
            boundary: BoundaryPolicy::Periodic,
            normalization: ConvolutionNormalization::None,
        }
    }

    /// Validates lengths and returns the exact implementation cost plan.
    pub fn inspect(
        &self,
        signal_len: usize,
        kernel_len: usize,
    ) -> Result<ConvolutionCostPlan, SignalError> {
        self.validate(signal_len, kernel_len)?;
        let full_len = linear_full_len(signal_len, kernel_len)?;
        let fft_len = match self.mode {
            ConvolutionMode::Linear(_) => {
                full_len
                    .checked_next_power_of_two()
                    .ok_or(SignalError::InvalidLength {
                        len: full_len,
                        reason: "convolution FFT length overflowed",
                    })?
            }
            ConvolutionMode::Circular { period } => period,
        };
        let direct_cost_units = match self.mode {
            ConvolutionMode::Linear(_) => signal_len.checked_mul(kernel_len),
            ConvolutionMode::Circular { period } => period.checked_mul(period),
        }
        .ok_or(SignalError::InvalidLength {
            len: signal_len,
            reason: "direct convolution cost overflowed",
        })?;
        let stages = usize::try_from(usize::BITS - fft_len.leading_zeros()).unwrap_or(usize::MAX);
        let fft_cost_units = fft_len
            .checked_mul(stages)
            .and_then(|cost| cost.checked_mul(3))
            .and_then(|cost| cost.checked_add(fft_len))
            .ok_or(SignalError::InvalidLength {
                len: fft_len,
                reason: "FFT convolution cost overflowed",
            })?;
        let fft_scratch_bytes = fft_len
            .checked_mul(3)
            .and_then(|cells| cells.checked_mul(2 * size_of::<f64>()))
            .ok_or(SignalError::InvalidLength {
                len: fft_len,
                reason: "FFT convolution scratch size overflowed",
            })?;
        let selected = match self.algorithm {
            ConvolutionAlgorithm::Auto if direct_cost_units <= fft_cost_units => {
                ConvolutionAlgorithm::Direct
            }
            ConvolutionAlgorithm::Auto => ConvolutionAlgorithm::Fft,
            selected => selected,
        };
        Ok(ConvolutionCostPlan {
            requested: self.algorithm,
            selected,
            direct_cost_units,
            fft_cost_units,
            fft_len,
            fft_scratch_bytes,
        })
    }

    fn validate(&self, signal_len: usize, kernel_len: usize) -> Result<(), SignalError> {
        if signal_len == 0 || kernel_len == 0 {
            return Err(SignalError::InvalidLength {
                len: signal_len.min(kernel_len),
                reason: "convolution inputs must both be non-empty",
            });
        }
        match (self.mode, self.boundary) {
            (ConvolutionMode::Linear(_), BoundaryPolicy::ZeroPad)
            | (ConvolutionMode::Circular { .. }, BoundaryPolicy::Periodic) => {}
            (ConvolutionMode::Linear(_), BoundaryPolicy::Periodic) => {
                return Err(SignalError::InvalidPolicy {
                    policy: "boundary",
                    reason: "linear convolution requires zero padding",
                });
            }
            (ConvolutionMode::Circular { .. }, BoundaryPolicy::ZeroPad) => {
                return Err(SignalError::InvalidPolicy {
                    policy: "boundary",
                    reason: "circular convolution requires periodic extension",
                });
            }
        }
        if let ConvolutionMode::Circular { period: 0 } = self.mode {
            return Err(SignalError::InvalidLength {
                len: 0,
                reason: "circular convolution period must be nonzero",
            });
        }
        if self.mode == ConvolutionMode::Linear(LinearOutput::Valid) && signal_len < kernel_len {
            return Err(SignalError::InvalidLength {
                len: signal_len,
                reason: "valid convolution requires signal length at least kernel length",
            });
        }
        Ok(())
    }
}

pub(crate) fn linear_full_len(signal_len: usize, kernel_len: usize) -> Result<usize, SignalError> {
    signal_len
        .checked_add(kernel_len)
        .and_then(|len| len.checked_sub(1))
        .ok_or(SignalError::InvalidLength {
            len: signal_len,
            reason: "linear convolution output length overflowed",
        })
}

pub(crate) fn retained_span(
    mode: ConvolutionMode,
    signal_len: usize,
    kernel_len: usize,
) -> Result<(usize, usize), SignalError> {
    Ok(match mode {
        ConvolutionMode::Linear(LinearOutput::Full) => {
            (0, linear_full_len(signal_len, kernel_len)?)
        }
        ConvolutionMode::Linear(LinearOutput::Same) => ((kernel_len - 1) / 2, signal_len),
        ConvolutionMode::Linear(LinearOutput::Valid) => {
            (kernel_len - 1, signal_len - kernel_len + 1)
        }
        ConvolutionMode::Circular { period } => (0, period),
    })
}
