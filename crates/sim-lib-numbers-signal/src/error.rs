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
    /// A tensor shape or physical layout is not a valid non-overlapping view.
    InvalidTensorView {
        /// View invariant that was not satisfied.
        reason: &'static str,
    },
    /// A requested transform axis does not exist in the tensor rank.
    AxisOutOfBounds {
        /// Requested axis.
        axis: usize,
        /// Number of tensor dimensions.
        rank: usize,
    },
    /// A transform axis appeared more than once.
    DuplicateAxis {
        /// Repeated axis.
        axis: usize,
    },
    /// A bounded transform plan would exceed the caller's scratch limit.
    ScratchLimit {
        /// Peak scratch bytes required by the selected plan.
        required: usize,
        /// Caller-declared scratch-byte ceiling.
        maximum: usize,
    },
    /// A spectral estimator would exceed its declared deterministic work limit.
    WorkLimit {
        /// Conservative work units required by the selected plan.
        required: u64,
        /// Caller-declared work-unit ceiling.
        maximum: u64,
    },
    /// A Table/Dir block-store operation failed or returned invalid data.
    BlockStore {
        /// Operation being performed.
        operation: &'static str,
        /// Backend or encoding diagnostic.
        message: String,
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
    /// A requested normalization has a zero or non-finite divisor.
    DegenerateNormalization {
        /// Name of the normalization whose divisor was unusable.
        normalization: &'static str,
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
            Self::InvalidTensorView { reason } => write!(f, "invalid tensor view: {reason}"),
            Self::AxisOutOfBounds { axis, rank } => {
                write!(f, "transform axis {axis} is outside tensor rank {rank}")
            }
            Self::DuplicateAxis { axis } => {
                write!(f, "transform axis {axis} was declared more than once")
            }
            Self::ScratchLimit { required, maximum } => {
                write!(
                    f,
                    "transform needs {required} scratch bytes, exceeding limit {maximum}"
                )
            }
            Self::WorkLimit { required, maximum } => {
                write!(
                    f,
                    "spectral estimator needs {required} work units, exceeding limit {maximum}"
                )
            }
            Self::BlockStore { operation, message } => {
                write!(f, "block store {operation} failed: {message}")
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
            Self::DegenerateNormalization { normalization } => {
                write!(f, "{normalization} normalization has a degenerate divisor")
            }
        }
    }
}

impl Error for SignalError {}
