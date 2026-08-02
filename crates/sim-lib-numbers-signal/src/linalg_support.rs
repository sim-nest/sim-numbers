//! Translation from shared dense-solver evidence into signal-domain errors.

use sim_lib_numbers_tensor_linalg::DenseSolveError;

use crate::SignalError;

pub(crate) fn dense_error(error: DenseSolveError, operation: &'static str) -> SignalError {
    match error {
        DenseSolveError::InvalidDimensions { rhs_len, .. } => SignalError::InvalidLength {
            len: rhs_len,
            reason: "the numerical system dimensions are inconsistent",
        },
        DenseSolveError::InvalidThreshold { .. } => SignalError::InvalidPolicy {
            policy: "singularity threshold",
            reason: "the threshold must be finite and strictly positive",
        },
        DenseSolveError::NonFinite { index, component } => {
            SignalError::NonFinite { index, component }
        }
        DenseSolveError::Singular {
            step,
            pivot_magnitude,
            threshold,
        } => SignalError::SingularSystem {
            operation,
            step,
            pivot_magnitude,
            threshold,
        },
        DenseSolveError::NonFiniteResult { index } => SignalError::NonFinite {
            index,
            component: "numerical solution",
        },
    }
}
