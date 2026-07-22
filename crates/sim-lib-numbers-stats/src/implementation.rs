//! Implementation of the statistics helpers: descriptive statistics,
//! probability, and fairness metrics over f64 data, with their error type.

use std::{error::Error, fmt};

#[path = "agent_fixtures.rs"]
mod agent_fixtures;
#[path = "claim.rs"]
mod claim;
#[path = "function.rs"]
mod function;
#[path = "runtime.rs"]
mod runtime;

pub use claim::{
    FairnessClaimValue, FairnessEvidence, StatsClaimEvidence, StatsClaimValue, fairness_claim,
    fairness_claim_value, stats_result_claim, stats_result_claim_value,
};
pub use function::{
    StatsNumbersLib, stats_claims_symbol, stats_disparate_impact_claim_symbol,
    stats_entropy_claim_symbol, stats_mean_claim_symbol, stats_variance_claim_symbol,
};

const FOUR_FIFTHS_THRESHOLD: f64 = 0.8;
const PROBABILITY_TOLERANCE: f64 = 1.0e-12;

/// Result alias for the statistics helpers, fixing the error to [`StatsError`].
pub type StatsResult<T> = Result<T, StatsError>;

/// Errors returned by probability, statistics, and fairness helpers.
///
/// Each variant carries the `metric` name of the helper that rejected the
/// input, so the failure can be reported without the caller tracking context.
#[derive(Clone, Debug, PartialEq)]
pub enum StatsError {
    /// A helper requiring at least one value was given an empty slice.
    EmptyInput {
        /// Name of the helper that rejected the input.
        metric: &'static str,
    },
    /// Fewer values were supplied than the helper requires.
    InsufficientInput {
        /// Name of the helper that rejected the input.
        metric: &'static str,
        /// Smallest number of values the helper accepts.
        minimum: usize,
        /// Number of values actually supplied.
        actual: usize,
    },
    /// A value was not finite (`NaN` or infinite).
    NonFinite {
        /// Name of the helper that rejected the input.
        metric: &'static str,
        /// Position of the offending value, if it came from a slice.
        index: Option<usize>,
        /// The offending value.
        value: f64,
    },
    /// A probability fell outside the closed range `0.0..=1.0`.
    ProbabilityOutOfRange {
        /// Name of the helper that rejected the input.
        metric: &'static str,
        /// Position of the offending value, if it came from a slice.
        index: Option<usize>,
        /// The offending value.
        value: f64,
    },
    /// A probability vector did not sum to one within tolerance.
    ProbabilityMass {
        /// Name of the helper that rejected the input.
        metric: &'static str,
        /// The actual sum of the probabilities.
        sum: f64,
    },
    /// A Bayesian update was given zero total evidence.
    ZeroEvidence {
        /// Name of the helper that rejected the input.
        metric: &'static str,
    },
    /// A counts pair was given a zero total.
    ZeroTotal {
        /// Label of the count whose total was zero.
        label: &'static str,
    },
    /// A fairness ratio was given a zero reference rate to divide by.
    ZeroReferenceRate {
        /// Name of the helper that rejected the input.
        metric: &'static str,
    },
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { metric } => write!(f, "{metric} requires at least one value"),
            Self::InsufficientInput {
                metric,
                minimum,
                actual,
            } => write!(
                f,
                "{metric} requires at least {minimum} values, got {actual}"
            ),
            Self::NonFinite {
                metric,
                index,
                value,
            } => match index {
                Some(index) => write!(f, "{metric} value {index} is not finite: {value}"),
                None => write!(f, "{metric} value is not finite: {value}"),
            },
            Self::ProbabilityOutOfRange {
                metric,
                index,
                value,
            } => match index {
                Some(index) => write!(
                    f,
                    "{metric} probability {index} must be between 0 and 1, got {value}"
                ),
                None => write!(
                    f,
                    "{metric} probability must be between 0 and 1, got {value}"
                ),
            },
            Self::ProbabilityMass { metric, sum } => {
                write!(f, "{metric} probabilities must sum to 1, got {sum}")
            }
            Self::ZeroEvidence { metric } => write!(f, "{metric} evidence must be nonzero"),
            Self::ZeroTotal { label } => write!(f, "{label} total must be nonzero"),
            Self::ZeroReferenceRate { metric } => {
                write!(f, "{metric} reference rate must be nonzero")
            }
        }
    }
}

impl Error for StatsError {}

/// Counts for a binary selection or outcome table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryOutcomeCounts {
    /// Number of selected (positive-outcome) cases.
    pub selected: u64,
    /// Total number of cases; the denominator of the selection rate.
    pub total: u64,
}

impl BinaryOutcomeCounts {
    /// Builds a count pair and rejects impossible or empty totals.
    pub fn new(selected: u64, total: u64) -> StatsResult<Self> {
        if total == 0 {
            return Err(StatsError::ZeroTotal {
                label: "outcome counts",
            });
        }
        if selected > total {
            return Err(StatsError::ProbabilityOutOfRange {
                metric: "outcome counts",
                index: None,
                value: selected as f64 / total as f64,
            });
        }
        Ok(Self { selected, total })
    }

    /// Returns `selected / total` as an f64 rate.
    pub fn selection_rate(self) -> f64 {
        self.selected as f64 / self.total as f64
    }
}

/// Disparate-impact summary for two binary-outcome groups.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisparateImpact {
    /// Selection rate of the reference group.
    pub reference_rate: f64,
    /// Selection rate of the comparison group.
    pub comparison_rate: f64,
    /// Comparison rate divided by reference rate.
    pub ratio: f64,
    /// Whether the ratio meets the four-fifths (0.8) threshold.
    pub passes_four_fifths: bool,
}

/// Computes `prior * likelihood / evidence` and validates the posterior.
///
/// Every argument and the result must be a probability in `0.0..=1.0`, and
/// `evidence` must be nonzero.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_stats::bayesian_update;
///
/// let posterior = bayesian_update(0.2, 0.75, 0.3).unwrap();
/// assert!((posterior - 0.5).abs() < 1e-12);
/// ```
pub fn bayesian_update(prior: f64, likelihood: f64, evidence: f64) -> StatsResult<f64> {
    validate_probability("bayesian_update", None, prior)?;
    validate_probability("bayesian_update", None, likelihood)?;
    validate_probability("bayesian_update", None, evidence)?;
    if evidence == 0.0 {
        return Err(StatsError::ZeroEvidence {
            metric: "bayesian_update",
        });
    }
    let posterior = (prior * likelihood) / evidence;
    validate_probability("bayesian_update", None, posterior)?;
    Ok(posterior)
}

/// Computes a binary-test posterior from prior, true-positive, and false-positive rates.
pub fn bayesian_update_binary(
    prior: f64,
    true_positive_rate: f64,
    false_positive_rate: f64,
) -> StatsResult<f64> {
    validate_probability("bayesian_update_binary", None, prior)?;
    validate_probability("bayesian_update_binary", None, true_positive_rate)?;
    validate_probability("bayesian_update_binary", None, false_positive_rate)?;
    let evidence = prior * true_positive_rate + (1.0 - prior) * false_positive_rate;
    bayesian_update(prior, true_positive_rate, evidence)
}

/// Computes Shannon entropy in bits for a probability vector that sums to one.
pub fn entropy(probabilities: &[f64]) -> StatsResult<f64> {
    if probabilities.is_empty() {
        return Err(StatsError::EmptyInput { metric: "entropy" });
    }
    let mut sum = 0.0;
    let mut bits = 0.0;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        validate_probability("entropy", Some(index), probability)?;
        sum += probability;
        if probability > 0.0 {
            bits -= probability * probability.log2();
        }
    }
    if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(StatsError::ProbabilityMass {
            metric: "entropy",
            sum,
        });
    }
    Ok(bits)
}

/// Computes the arithmetic mean of finite values.
///
/// Returns [`StatsError::EmptyInput`] for an empty slice and
/// [`StatsError::NonFinite`] if any value is `NaN` or infinite.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_stats::mean;
///
/// assert_eq!(mean(&[2.0, 4.0, 6.0]).unwrap(), 4.0);
/// ```
pub fn mean(values: &[f64]) -> StatsResult<f64> {
    validate_values("mean", values)?;
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

/// Computes population variance.
///
/// Alias for [`population_variance`].
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_stats::variance;
///
/// let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// assert!((variance(&values).unwrap() - 4.0).abs() < 1e-12);
/// ```
pub fn variance(values: &[f64]) -> StatsResult<f64> {
    population_variance(values)
}

/// Computes population variance with divisor `n`.
pub fn population_variance(values: &[f64]) -> StatsResult<f64> {
    validate_values("population_variance", values)?;
    let mean = mean(values)?;
    Ok(values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64)
}

/// Computes sample variance with divisor `n - 1`.
pub fn sample_variance(values: &[f64]) -> StatsResult<f64> {
    validate_values("sample_variance", values)?;
    if values.len() < 2 {
        return Err(StatsError::InsufficientInput {
            metric: "sample_variance",
            minimum: 2,
            actual: values.len(),
        });
    }
    let mean = mean(values)?;
    Ok(values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64)
}

/// Computes the comparison/reference rate ratio used by the four-fifths rule.
pub fn four_fifths_ratio(reference_rate: f64, comparison_rate: f64) -> StatsResult<f64> {
    validate_probability("four_fifths_ratio", None, reference_rate)?;
    validate_probability("four_fifths_ratio", None, comparison_rate)?;
    if reference_rate == 0.0 {
        return Err(StatsError::ZeroReferenceRate {
            metric: "four_fifths_ratio",
        });
    }
    Ok(comparison_rate / reference_rate)
}

/// Computes disparate impact and the four-fifths pass/fail flag.
///
/// Divides the comparison group's selection rate by the reference group's and
/// flags whether the resulting ratio clears the four-fifths (0.8) threshold.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_stats::{BinaryOutcomeCounts, disparate_impact};
///
/// let reference = BinaryOutcomeCounts::new(80, 100).unwrap();
/// let comparison = BinaryOutcomeCounts::new(60, 100).unwrap();
/// let impact = disparate_impact(reference, comparison).unwrap();
///
/// assert!((impact.ratio - 0.75).abs() < 1e-12);
/// assert!(!impact.passes_four_fifths);
/// ```
pub fn disparate_impact(
    reference: BinaryOutcomeCounts,
    comparison: BinaryOutcomeCounts,
) -> StatsResult<DisparateImpact> {
    let reference_rate = reference.selection_rate();
    let comparison_rate = comparison.selection_rate();
    let ratio = four_fifths_ratio(reference_rate, comparison_rate)?;
    Ok(DisparateImpact {
        reference_rate,
        comparison_rate,
        ratio,
        passes_four_fifths: ratio >= FOUR_FIFTHS_THRESHOLD,
    })
}

fn validate_values(metric: &'static str, values: &[f64]) -> StatsResult<()> {
    if values.is_empty() {
        return Err(StatsError::EmptyInput { metric });
    }
    for (index, value) in values.iter().copied().enumerate() {
        validate_finite(metric, Some(index), value)?;
    }
    Ok(())
}

fn validate_probability(metric: &'static str, index: Option<usize>, value: f64) -> StatsResult<()> {
    validate_finite(metric, index, value)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(StatsError::ProbabilityOutOfRange {
            metric,
            index,
            value,
        });
    }
    Ok(())
}

fn validate_finite(metric: &'static str, index: Option<usize>, value: f64) -> StatsResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(StatsError::NonFinite {
            metric,
            index,
            value,
        })
    }
}
