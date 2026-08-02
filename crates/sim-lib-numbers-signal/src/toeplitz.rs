//! Stable solution of real Toeplitz systems with explicit diagnostics.

use sim_lib_numbers_tensor_linalg::{DenseSolveOptions, solve_dense_f64};

use crate::{SignalError, convolution::validate_real_signal, linalg_support::dense_error};

/// Numerical policy for a Toeplitz solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToeplitzPlan {
    /// Relative scaled-pivot threshold below which the system is singular.
    pub singularity_threshold: f64,
    /// Relative tolerance allowed between the first row and column diagonals.
    pub diagonal_tolerance: f64,
}

impl Default for ToeplitzPlan {
    fn default() -> Self {
        Self {
            singularity_threshold: 1e-12,
            diagonal_tolerance: 1e-12,
        }
    }
}

/// Conditioning and residual evidence from a successful Toeplitz solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToeplitzDiagnostics {
    /// System dimension.
    pub dimension: usize,
    /// Row interchanges selected by scaled partial pivoting.
    pub pivot_swaps: usize,
    /// Smallest absolute accepted pivot.
    pub min_abs_pivot: f64,
    /// Largest absolute accepted pivot.
    pub max_abs_pivot: f64,
    /// Conservative smallest/largest pivot ratio.
    pub reciprocal_pivot_condition: f64,
    /// Euclidean norm of `T x - b`.
    pub residual_l2: f64,
    /// Relative singularity threshold used for admission.
    pub singularity_threshold: f64,
}

/// Toeplitz solution vector and retained numerical evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ToeplitzSolution {
    /// Solution values in row/column order.
    pub values: Vec<f64>,
    /// Pivot and residual diagnostics.
    pub diagnostics: ToeplitzDiagnostics,
}

/// Solves a real Toeplitz system using a structure-aware input and stable dense pivoting.
pub fn solve_toeplitz(
    first_column: &[f64],
    first_row: &[f64],
    rhs: &[f64],
    plan: ToeplitzPlan,
) -> Result<ToeplitzSolution, SignalError> {
    let dimension = rhs.len();
    if dimension == 0 {
        return Err(SignalError::InvalidLength {
            len: 0,
            reason: "a Toeplitz solve requires a non-empty right-hand side",
        });
    }
    if first_column.len() != dimension {
        return Err(SignalError::LengthMismatch {
            expected: dimension,
            actual: first_column.len(),
        });
    }
    if first_row.len() != dimension {
        return Err(SignalError::LengthMismatch {
            expected: dimension,
            actual: first_row.len(),
        });
    }
    validate_real_signal(first_column)?;
    validate_real_signal(first_row)?;
    validate_real_signal(rhs)?;
    if !plan.diagonal_tolerance.is_finite() || plan.diagonal_tolerance < 0.0 {
        return Err(SignalError::InvalidPolicy {
            policy: "Toeplitz diagonal tolerance",
            reason: "the tolerance must be finite and non-negative",
        });
    }
    let diagonal_scale = first_column[0].abs().max(first_row[0].abs()).max(1.0);
    if (first_column[0] - first_row[0]).abs() > plan.diagonal_tolerance * diagonal_scale {
        return Err(SignalError::InvalidPolicy {
            policy: "Toeplitz diagonal",
            reason: "the first row and column must declare the same diagonal",
        });
    }
    let mut matrix = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            matrix[row * dimension + column] = if row >= column {
                first_column[row - column]
            } else {
                first_row[column - row]
            };
        }
    }
    let solution = solve_dense_f64(
        &matrix,
        rhs,
        DenseSolveOptions {
            singularity_threshold: plan.singularity_threshold,
        },
    )
    .map_err(|error| dense_error(error, "Toeplitz"))?;
    Ok(ToeplitzSolution {
        values: solution.values,
        diagnostics: ToeplitzDiagnostics {
            dimension,
            pivot_swaps: solution.report.pivot_swaps,
            min_abs_pivot: solution.report.min_abs_pivot,
            max_abs_pivot: solution.report.max_abs_pivot,
            reciprocal_pivot_condition: solution.report.reciprocal_pivot_condition,
            residual_l2: solution.report.residual_l2,
            singularity_threshold: plan.singularity_threshold,
        },
    })
}
