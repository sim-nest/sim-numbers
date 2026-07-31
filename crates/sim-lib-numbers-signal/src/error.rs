//! Errors shared by transform planning and execution.

use std::{error::Error, fmt};

/// Error returned when a transform plan or signal buffer is invalid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignalError {
    /// A transform length was zero or too short for its definition.
    InvalidLength {
        /// Requested logical transform length.
        len: usize,
        /// Definition-level requirement that was not met.
        reason: &'static str,
    },
    /// A policy combination is contradictory or unsupported.
    InvalidPolicy {
        /// Name of the policy that was rejected.
        policy: &'static str,
        /// Reason the policy cannot be used.
        reason: &'static str,
    },
    /// A stride step of zero was requested.
    ZeroStride,
    /// Offset/stride arithmetic overflowed `usize`.
    StrideOverflow,
    /// The selected logical input length disagrees with the plan.
    LengthMismatch {
        /// Length required by the plan.
        expected: usize,
        /// Length available through the selected view.
        actual: usize,
    },
    /// A transform received real data where complex data was required, or the
    /// reverse.
    InputKind {
        /// Input representation required by the plan.
        expected: &'static str,
        /// Input representation supplied by the caller.
        actual: &'static str,
    },
    /// A signal contains a NaN or infinity.
    NonFinite {
        /// Logical signal position containing the invalid component.
        index: usize,
        /// Component name (`real`, `imag`, or `value`).
        component: &'static str,
    },
}

impl fmt::Display for SignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { len, reason } => {
                write!(f, "invalid transform length {len}: {reason}")
            }
            Self::InvalidPolicy { policy, reason } => {
                write!(f, "invalid {policy} policy: {reason}")
            }
            Self::ZeroStride => write!(f, "signal stride must be nonzero"),
            Self::StrideOverflow => write!(f, "signal offset/stride arithmetic overflowed"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "signal length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InputKind { expected, actual } => {
                write!(f, "transform expects {expected} input, got {actual}")
            }
            Self::NonFinite { index, component } => {
                write!(
                    f,
                    "signal {component} component at index {index} is not finite"
                )
            }
        }
    }
}

impl Error for SignalError {}
