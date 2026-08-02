//! Shared finite row-stochastic transition representation.

use std::{error::Error, fmt};

const STOCHASTIC_TOLERANCE: f64 = 1.0e-10;

/// A finite state vocabulary and row-stochastic transition matrix.
///
/// This is the shared transition representation used by observable Markov
/// models and hidden-state models. State order is caller-owned and retained.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteTransitionMatrix<S> {
    states: Vec<S>,
    probabilities: Vec<Vec<f64>>,
}

impl<S: Eq + Clone> FiniteTransitionMatrix<S> {
    /// Builds a checked matrix from ordered states and probability rows.
    pub fn new(states: Vec<S>, probabilities: Vec<Vec<f64>>) -> Result<Self, TransitionError> {
        if states.is_empty() {
            return Err(TransitionError::EmptyStates);
        }
        for (index, state) in states.iter().enumerate() {
            if states[..index].contains(state) {
                return Err(TransitionError::DuplicateState { index });
            }
        }
        if probabilities.len() != states.len() {
            return Err(TransitionError::RowCount {
                expected: states.len(),
                actual: probabilities.len(),
            });
        }
        for (row, probabilities) in probabilities.iter().enumerate() {
            validate_distribution("transition", row, probabilities, states.len())?;
        }
        Ok(Self {
            states,
            probabilities,
        })
    }

    pub(crate) fn from_normalized(states: Vec<S>, probabilities: Vec<Vec<f64>>) -> Self {
        Self {
            states,
            probabilities,
        }
    }

    /// Returns the ordered finite state vocabulary.
    pub fn states(&self) -> &[S] {
        &self.states
    }

    /// Returns the state count and matrix dimension.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns whether the state vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns all row-stochastic probability rows.
    pub fn rows(&self) -> &[Vec<f64>] {
        &self.probabilities
    }

    /// Returns a probability by state indices.
    pub fn probability_by_index(&self, from: usize, to: usize) -> Option<f64> {
        self.probabilities
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
    }

    /// Returns a probability by state values.
    pub fn probability(&self, from: &S, to: &S) -> Option<f64> {
        let from = self.states.iter().position(|state| state == from)?;
        let to = self.states.iter().position(|state| state == to)?;
        self.probability_by_index(from, to)
    }
}

/// Failure while constructing a finite row-stochastic transition matrix.
#[derive(Clone, Debug, PartialEq)]
pub enum TransitionError {
    /// No states were supplied.
    EmptyStates,
    /// A state duplicated an earlier state.
    DuplicateState {
        /// Position of the duplicate.
        index: usize,
    },
    /// The matrix row count did not match the state count.
    RowCount {
        /// Required row count.
        expected: usize,
        /// Supplied row count.
        actual: usize,
    },
    /// A row length did not match the state count.
    ColumnCount {
        /// Zero-based row index.
        row: usize,
        /// Required column count.
        expected: usize,
        /// Supplied column count.
        actual: usize,
    },
    /// A probability was non-finite or negative.
    InvalidProbability {
        /// Stable matrix or distribution name.
        distribution: &'static str,
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        column: usize,
        /// Rejected probability.
        value: f64,
    },
    /// A probability row did not sum to one.
    ProbabilityMass {
        /// Stable matrix or distribution name.
        distribution: &'static str,
        /// Zero-based row index.
        row: usize,
        /// Observed probability mass.
        sum: f64,
    },
}

impl fmt::Display for TransitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStates => write!(formatter, "finite transitions require at least one state"),
            Self::DuplicateState { index } => {
                write!(formatter, "finite transition state {index} is duplicated")
            }
            Self::RowCount { expected, actual } => write!(
                formatter,
                "finite transitions require {expected} rows, got {actual}"
            ),
            Self::ColumnCount {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "finite transition row {row} requires {expected} columns, got {actual}"
            ),
            Self::InvalidProbability {
                distribution,
                row,
                column,
                value,
            } => write!(
                formatter,
                "{distribution} probability {row}:{column} must be finite and nonnegative, got {value}"
            ),
            Self::ProbabilityMass {
                distribution,
                row,
                sum,
            } => write!(
                formatter,
                "{distribution} probability row {row} must sum to one, got {sum}"
            ),
        }
    }
}

impl Error for TransitionError {}

pub(crate) fn validate_distribution(
    distribution: &'static str,
    row: usize,
    probabilities: &[f64],
    expected: usize,
) -> Result<(), TransitionError> {
    if probabilities.len() != expected {
        return Err(TransitionError::ColumnCount {
            row,
            expected,
            actual: probabilities.len(),
        });
    }
    let mut sum = 0.0;
    for (column, probability) in probabilities.iter().copied().enumerate() {
        if !probability.is_finite() || probability < 0.0 {
            return Err(TransitionError::InvalidProbability {
                distribution,
                row,
                column,
                value: probability,
            });
        }
        sum += probability;
    }
    if (sum - 1.0).abs() > STOCHASTIC_TOLERANCE {
        return Err(TransitionError::ProbabilityMass {
            distribution,
            row,
            sum,
        });
    }
    Ok(())
}
