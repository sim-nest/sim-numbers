//! Finite hidden-state model and emission representations.

use super::transition::{FiniteTransitionMatrix, TransitionError, validate_distribution};
use std::{error::Error, f64::consts::TAU, fmt};

/// Discrete or scalar Gaussian emissions for a finite hidden-state model.
#[derive(Clone, Debug, PartialEq)]
pub enum EmissionModel {
    /// A row-stochastic categorical distribution for every hidden state.
    Discrete {
        /// State-major probability rows over symbols `0..symbol_count`.
        probabilities: Vec<Vec<f64>>,
    },
    /// One univariate Gaussian distribution for every hidden state.
    Gaussian {
        /// Mean for each hidden state.
        means: Vec<f64>,
        /// Strictly positive variance for each hidden state.
        variances: Vec<f64>,
        /// Fitting floor retained with the model as numerical policy.
        variance_floor: f64,
    },
}

impl EmissionModel {
    /// Returns the number of categorical symbols, or `None` for Gaussian emissions.
    pub fn symbol_count(&self) -> Option<usize> {
        match self {
            Self::Discrete { probabilities } => probabilities.first().map(Vec::len),
            Self::Gaussian { .. } => None,
        }
    }

    /// Returns the state-major categorical rows, when this is a discrete model.
    pub fn discrete_probabilities(&self) -> Option<&[Vec<f64>]> {
        match self {
            Self::Discrete { probabilities } => Some(probabilities),
            Self::Gaussian { .. } => None,
        }
    }

    /// Returns Gaussian means, variances, and variance floor when applicable.
    pub fn gaussian_parameters(&self) -> Option<(&[f64], &[f64], f64)> {
        match self {
            Self::Gaussian {
                means,
                variances,
                variance_floor,
            } => Some((means, variances, *variance_floor)),
            Self::Discrete { .. } => None,
        }
    }

    fn validate(&self, states: usize) -> Result<(), HmmError> {
        match self {
            Self::Discrete { probabilities } => {
                if probabilities.len() != states {
                    return Err(HmmError::EmissionStateCount {
                        expected: states,
                        actual: probabilities.len(),
                    });
                }
                let symbols = probabilities.first().map_or(0, Vec::len);
                if symbols == 0 {
                    return Err(HmmError::InvalidModel {
                        field: "emission.symbols",
                        reason: "must be greater than zero",
                    });
                }
                for (state, row) in probabilities.iter().enumerate() {
                    validate_distribution("emission", state, row, symbols)?;
                }
            }
            Self::Gaussian {
                means,
                variances,
                variance_floor,
            } => {
                if means.len() != states || variances.len() != states {
                    return Err(HmmError::EmissionStateCount {
                        expected: states,
                        actual: means.len().min(variances.len()),
                    });
                }
                if !variance_floor.is_finite() || *variance_floor <= 0.0 {
                    return Err(HmmError::InvalidModel {
                        field: "emission.variance_floor",
                        reason: "must be finite and greater than zero",
                    });
                }
                for (state, (&mean, &variance)) in means.iter().zip(variances).enumerate() {
                    if !mean.is_finite() {
                        return Err(HmmError::InvalidGaussian {
                            state,
                            field: "mean",
                            value: mean,
                        });
                    }
                    if !variance.is_finite() || variance < *variance_floor {
                        return Err(HmmError::InvalidGaussian {
                            state,
                            field: "variance",
                            value: variance,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn log_discrete(&self, state: usize, symbol: usize) -> Result<f64, HmmError> {
        let Self::Discrete { probabilities } = self else {
            return Err(HmmError::EmissionKind {
                expected: "discrete",
                actual: "continuous",
            });
        };
        let symbol_count = probabilities.first().map_or(0, Vec::len);
        if symbol >= symbol_count {
            return Err(HmmError::UnknownSymbol {
                symbol,
                symbol_count,
            });
        }
        Ok(log_probability(probabilities[state][symbol]))
    }

    pub(crate) fn log_continuous(&self, state: usize, value: f64) -> Result<f64, HmmError> {
        if !value.is_finite() {
            return Err(HmmError::NonFiniteObservation { value });
        }
        let Self::Gaussian {
            means, variances, ..
        } = self
        else {
            return Err(HmmError::EmissionKind {
                expected: "continuous",
                actual: "discrete",
            });
        };
        let difference = value - means[state];
        Ok(-0.5 * (difference * difference / variances[state] + (TAU * variances[state]).ln()))
    }
}

/// Observation accepted by generic HMM inference.
pub trait HmmObservation: Copy {
    /// Returns this observation's log likelihood in `state`.
    fn emission_log_likelihood(
        self,
        emissions: &EmissionModel,
        state: usize,
    ) -> Result<f64, HmmError>;
}

impl HmmObservation for usize {
    fn emission_log_likelihood(
        self,
        emissions: &EmissionModel,
        state: usize,
    ) -> Result<f64, HmmError> {
        emissions.log_discrete(state, self)
    }
}

impl HmmObservation for f64 {
    fn emission_log_likelihood(
        self,
        emissions: &EmissionModel,
        state: usize,
    ) -> Result<f64, HmmError> {
        emissions.log_continuous(state, self)
    }
}

/// A finite hidden Markov model with inspectable transition and emission rows.
#[derive(Clone, Debug, PartialEq)]
pub struct HiddenMarkovModel<S> {
    initial: Vec<f64>,
    transitions: FiniteTransitionMatrix<S>,
    emissions: EmissionModel,
}

impl<S: Eq + Clone> HiddenMarkovModel<S> {
    /// Builds a model with categorical observations indexed from zero.
    pub fn discrete(
        states: Vec<S>,
        initial: Vec<f64>,
        transitions: Vec<Vec<f64>>,
        emissions: Vec<Vec<f64>>,
    ) -> Result<Self, HmmError> {
        Self::from_transition_matrix(
            initial,
            FiniteTransitionMatrix::new(states, transitions)?,
            EmissionModel::Discrete {
                probabilities: emissions,
            },
        )
    }

    /// Builds a model with scalar Gaussian observations.
    pub fn gaussian(
        states: Vec<S>,
        initial: Vec<f64>,
        transitions: Vec<Vec<f64>>,
        means: Vec<f64>,
        variances: Vec<f64>,
        variance_floor: f64,
    ) -> Result<Self, HmmError> {
        Self::from_transition_matrix(
            initial,
            FiniteTransitionMatrix::new(states, transitions)?,
            EmissionModel::Gaussian {
                means,
                variances,
                variance_floor,
            },
        )
    }

    /// Builds a model from the same finite transition representation exposed
    /// by [`crate::MarkovModel::transition_matrix`].
    pub fn from_transition_matrix(
        initial: Vec<f64>,
        transitions: FiniteTransitionMatrix<S>,
        emissions: EmissionModel,
    ) -> Result<Self, HmmError> {
        validate_distribution("initial", 0, &initial, transitions.len())?;
        emissions.validate(transitions.len())?;
        Ok(Self {
            initial,
            transitions,
            emissions,
        })
    }

    /// Returns the ordered hidden-state vocabulary.
    pub fn states(&self) -> &[S] {
        self.transitions.states()
    }

    /// Returns the normalized initial-state probabilities.
    pub fn initial_probabilities(&self) -> &[f64] {
        &self.initial
    }

    /// Returns the shared finite transition representation.
    pub fn transitions(&self) -> &FiniteTransitionMatrix<S> {
        &self.transitions
    }

    /// Returns the categorical or Gaussian emission representation.
    pub fn emissions(&self) -> &EmissionModel {
        &self.emissions
    }

    pub(crate) fn state_count(&self) -> usize {
        self.transitions.len()
    }

    pub(crate) fn emission_log<O: HmmObservation>(
        &self,
        state: usize,
        observation: O,
    ) -> Result<f64, HmmError> {
        observation.emission_log_likelihood(&self.emissions, state)
    }
}

/// Failure while constructing, fitting, or running hidden-state inference.
#[derive(Clone, Debug, PartialEq)]
pub enum HmmError {
    /// A finite transition matrix was malformed.
    Transition(TransitionError),
    /// No observations or fitting sequences were supplied.
    EmptyInput,
    /// One fitting sequence contained no observations.
    EmptySequence {
        /// Zero-based sequence index.
        index: usize,
    },
    /// The initial or emission state dimension was wrong.
    EmissionStateCount {
        /// Required hidden-state count.
        expected: usize,
        /// Supplied count.
        actual: usize,
    },
    /// A model-level field was invalid.
    InvalidModel {
        /// Invalid field.
        field: &'static str,
        /// Concrete requirement.
        reason: &'static str,
    },
    /// A Gaussian parameter was invalid.
    InvalidGaussian {
        /// Hidden-state index.
        state: usize,
        /// Invalid parameter name.
        field: &'static str,
        /// Rejected value.
        value: f64,
    },
    /// Observation and emission kinds did not match.
    EmissionKind {
        /// Required observation kind.
        expected: &'static str,
        /// Model emission kind.
        actual: &'static str,
    },
    /// A categorical observation was outside the model vocabulary.
    UnknownSymbol {
        /// Rejected symbol index.
        symbol: usize,
        /// Model symbol count.
        symbol_count: usize,
    },
    /// A continuous observation was not finite.
    NonFiniteObservation {
        /// Rejected observation.
        value: f64,
    },
    /// Every state path had zero probability at one observation.
    ImpossibleSequence {
        /// Zero-based observation index.
        position: usize,
    },
    /// Fitting data mixed discrete and continuous sequences.
    MixedSequenceKinds,
    /// A fitting specification or control was invalid.
    InvalidFitControl {
        /// Invalid field.
        field: &'static str,
        /// Concrete requirement.
        reason: &'static str,
    },
    /// Checked fitting work arithmetic overflowed.
    WorkOverflow,
}

impl fmt::Display for HmmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transition(error) => write!(formatter, "{error}"),
            Self::EmptyInput => write!(formatter, "HMM inference requires observations"),
            Self::EmptySequence { index } => {
                write!(formatter, "HMM fitting sequence {index} is empty")
            }
            Self::EmissionStateCount { expected, actual } => write!(
                formatter,
                "HMM emissions require {expected} state rows, got {actual}"
            ),
            Self::InvalidModel { field, reason } => {
                write!(formatter, "invalid HMM model {field}: {reason}")
            }
            Self::InvalidGaussian {
                state,
                field,
                value,
            } => write!(
                formatter,
                "HMM Gaussian state {state} {field} is invalid: {value}"
            ),
            Self::EmissionKind { expected, actual } => write!(
                formatter,
                "HMM expects {expected} observations, model emissions are {actual}"
            ),
            Self::UnknownSymbol {
                symbol,
                symbol_count,
            } => write!(
                formatter,
                "HMM symbol {symbol} is outside vocabulary 0..{symbol_count}"
            ),
            Self::NonFiniteObservation { value } => {
                write!(formatter, "HMM observation is not finite: {value}")
            }
            Self::ImpossibleSequence { position } => write!(
                formatter,
                "HMM observation {position} has zero probability under every state path"
            ),
            Self::MixedSequenceKinds => {
                write!(formatter, "HMM fitting data mixes observation kinds")
            }
            Self::InvalidFitControl { field, reason } => {
                write!(formatter, "invalid HMM fit control {field}: {reason}")
            }
            Self::WorkOverflow => write!(formatter, "HMM fitting work bound overflow"),
        }
    }
}

impl Error for HmmError {}

impl From<TransitionError> for HmmError {
    fn from(value: TransitionError) -> Self {
        Self::Transition(value)
    }
}

pub(crate) fn log_probability(probability: f64) -> f64 {
    if probability == 0.0 {
        f64::NEG_INFINITY
    } else {
        probability.ln()
    }
}
