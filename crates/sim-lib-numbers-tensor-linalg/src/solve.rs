//! Dependency-light dense f64 solving shared by numerical domain libraries.

use std::{error::Error, fmt};

/// Numerical admission policy for a dense linear solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseSolveOptions {
    /// Smallest admitted pivot relative to the selected row's scale.
    pub singularity_threshold: f64,
}

impl Default for DenseSolveOptions {
    fn default() -> Self {
        Self {
            singularity_threshold: 1e-12,
        }
    }
}

/// Numerical evidence retained by a successful dense solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseSolveReport {
    /// Number of rows and columns in the square system.
    pub dimension: usize,
    /// Row interchanges selected by scaled partial pivoting.
    pub pivot_swaps: usize,
    /// Smallest absolute accepted pivot.
    pub min_abs_pivot: f64,
    /// Largest absolute accepted pivot.
    pub max_abs_pivot: f64,
    /// Conservative pivot-ratio conditioning indicator in `0.0..=1.0`.
    pub reciprocal_pivot_condition: f64,
    /// Euclidean norm of `A x - b`, evaluated against the original inputs.
    pub residual_l2: f64,
}

/// Solution vector and the numerical evidence that admitted it.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseSolution {
    /// Solution in matrix-column order.
    pub values: Vec<f64>,
    /// Pivot and residual evidence.
    pub report: DenseSolveReport,
}

/// Failure returned by [`solve_dense_f64`].
#[derive(Clone, Debug, PartialEq)]
pub enum DenseSolveError {
    /// The row-major matrix is not square with the right-hand side dimension.
    InvalidDimensions {
        /// Number of matrix coefficients supplied.
        matrix_len: usize,
        /// Number of right-hand-side values supplied.
        rhs_len: usize,
    },
    /// The relative singularity threshold is not finite and strictly positive.
    InvalidThreshold {
        /// Rejected threshold.
        threshold: f64,
    },
    /// A matrix or right-hand-side value is NaN or infinite.
    NonFinite {
        /// Flat row-major matrix index, or right-hand-side index.
        index: usize,
        /// Input containing the invalid value.
        component: &'static str,
    },
    /// Scaled partial pivoting found no numerically admissible pivot.
    Singular {
        /// Zero-based elimination step.
        step: usize,
        /// Largest candidate pivot magnitude at the step.
        pivot_magnitude: f64,
        /// Absolute threshold required for the selected row.
        threshold: f64,
    },
    /// Elimination produced a non-finite solution or diagnostic.
    NonFiniteResult {
        /// Solution index being evaluated.
        index: usize,
    },
}

impl fmt::Display for DenseSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions {
                matrix_len,
                rhs_len,
            } => write!(
                f,
                "dense matrix length {matrix_len} is not square for RHS length {rhs_len}"
            ),
            Self::InvalidThreshold { threshold } => write!(
                f,
                "dense solve singularity threshold must be finite and positive, got {threshold}"
            ),
            Self::NonFinite { index, component } => {
                write!(f, "dense solve {component} value {index} is not finite")
            }
            Self::Singular {
                step,
                pivot_magnitude,
                threshold,
            } => write!(
                f,
                "dense system is singular at step {step}: pivot {pivot_magnitude} <= {threshold}"
            ),
            Self::NonFiniteResult { index } => {
                write!(
                    f,
                    "dense solve produced a non-finite result at index {index}"
                )
            }
        }
    }
}

impl Error for DenseSolveError {}

/// Solves a row-major square f64 system with scaled partial pivoting.
pub fn solve_dense_f64(
    matrix: &[f64],
    rhs: &[f64],
    options: DenseSolveOptions,
) -> Result<DenseSolution, DenseSolveError> {
    let dimension = rhs.len();
    if dimension == 0 || dimension.checked_mul(dimension) != Some(matrix.len()) {
        return Err(DenseSolveError::InvalidDimensions {
            matrix_len: matrix.len(),
            rhs_len: rhs.len(),
        });
    }
    if !options.singularity_threshold.is_finite() || options.singularity_threshold <= 0.0 {
        return Err(DenseSolveError::InvalidThreshold {
            threshold: options.singularity_threshold,
        });
    }
    validate_finite(matrix, "matrix")?;
    validate_finite(rhs, "right-hand side")?;

    let original_matrix = matrix.to_vec();
    let original_rhs = rhs.to_vec();
    let mut matrix = matrix.to_vec();
    let mut rhs = rhs.to_vec();
    let mut row_scales = matrix
        .chunks_exact(dimension)
        .map(|row| row.iter().map(|value| value.abs()).fold(0.0, f64::max))
        .collect::<Vec<_>>();
    let mut pivot_swaps = 0;
    let mut min_abs_pivot = f64::INFINITY;
    let mut max_abs_pivot: f64 = 0.0;

    for column in 0..dimension {
        let pivot_row = (column..dimension)
            .max_by(|&left, &right| {
                scaled_pivot(&matrix, &row_scales, dimension, left, column).total_cmp(
                    &scaled_pivot(&matrix, &row_scales, dimension, right, column),
                )
            })
            .expect("the elimination range is non-empty");
        let pivot_magnitude = matrix[pivot_row * dimension + column].abs();
        let threshold = options.singularity_threshold * row_scales[pivot_row];
        if pivot_magnitude <= threshold || row_scales[pivot_row] == 0.0 {
            return Err(DenseSolveError::Singular {
                step: column,
                pivot_magnitude,
                threshold,
            });
        }
        if pivot_row != column {
            for index in 0..dimension {
                matrix.swap(column * dimension + index, pivot_row * dimension + index);
            }
            rhs.swap(column, pivot_row);
            row_scales.swap(column, pivot_row);
            pivot_swaps += 1;
        }
        let pivot = matrix[column * dimension + column];
        min_abs_pivot = min_abs_pivot.min(pivot.abs());
        max_abs_pivot = max_abs_pivot.max(pivot.abs());
        for row in column + 1..dimension {
            let factor = matrix[row * dimension + column] / pivot;
            matrix[row * dimension + column] = 0.0;
            for index in column + 1..dimension {
                matrix[row * dimension + index] -= factor * matrix[column * dimension + index];
            }
            rhs[row] -= factor * rhs[column];
        }
    }

    let mut values = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let known = (row + 1..dimension)
            .map(|column| matrix[row * dimension + column] * values[column])
            .sum::<f64>();
        values[row] = (rhs[row] - known) / matrix[row * dimension + row];
        if !values[row].is_finite() {
            return Err(DenseSolveError::NonFiniteResult { index: row });
        }
    }
    let residual_l2 = original_matrix
        .chunks_exact(dimension)
        .zip(original_rhs)
        .map(|(row, expected)| {
            let residual = row.iter().zip(&values).map(|(a, x)| a * x).sum::<f64>() - expected;
            residual * residual
        })
        .sum::<f64>()
        .sqrt();
    if !residual_l2.is_finite() {
        return Err(DenseSolveError::NonFiniteResult { index: dimension });
    }
    Ok(DenseSolution {
        values,
        report: DenseSolveReport {
            dimension,
            pivot_swaps,
            min_abs_pivot,
            max_abs_pivot,
            reciprocal_pivot_condition: min_abs_pivot / max_abs_pivot,
            residual_l2,
        },
    })
}

fn scaled_pivot(
    matrix: &[f64],
    row_scales: &[f64],
    dimension: usize,
    row: usize,
    column: usize,
) -> f64 {
    let scale = row_scales[row];
    if scale == 0.0 {
        0.0
    } else {
        matrix[row * dimension + column].abs() / scale
    }
}

fn validate_finite(values: &[f64], component: &'static str) -> Result<(), DenseSolveError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DenseSolveError::NonFinite { index, component });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_pivoting_solves_and_reports_the_original_residual() {
        let solution = solve_dense_f64(
            &[0.0, 2.0, 1.0, 3.0],
            &[4.0, 7.0],
            DenseSolveOptions::default(),
        )
        .unwrap();
        assert!((solution.values[0] - 1.0).abs() < 1e-12);
        assert!((solution.values[1] - 2.0).abs() < 1e-12);
        assert_eq!(solution.report.pivot_swaps, 1);
        assert!(solution.report.residual_l2 < 1e-12);
    }

    #[test]
    fn singularity_returns_pivot_evidence() {
        let error = solve_dense_f64(
            &[1.0, 2.0, 2.0, 4.0],
            &[1.0, 2.0],
            DenseSolveOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            DenseSolveError::Singular {
                step: 1,
                pivot_magnitude,
                threshold,
            } if pivot_magnitude <= threshold
        ));
    }
}
