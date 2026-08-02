//! Deterministically initialized, bounded hidden Markov model fitting.

use super::hmm_baum_welch::baum_welch_step;
use super::hmm_inference::forward_backward;
use super::hmm_model::{HiddenMarkovModel, HmmError};

/// Stable numeric hidden-state identifier produced by [`fit_hmm`].
pub type StateId = usize;

/// One homogeneous observation sequence accepted by HMM fitting.
#[derive(Clone, Debug, PartialEq)]
pub enum Sequence {
    /// Categorical symbols indexed from zero.
    Discrete(Vec<usize>),
    /// Finite scalar observations.
    Continuous(Vec<f64>),
}

impl Sequence {
    fn len(&self) -> usize {
        match self {
            Self::Discrete(values) => values.len(),
            Self::Continuous(values) => values.len(),
        }
    }
}

/// Hidden-state and emission family requested from [`fit_hmm`].
#[derive(Clone, Debug, PartialEq)]
pub enum HmmSpec {
    /// A finite model with categorical emissions.
    Discrete {
        /// Number of hidden states.
        states: usize,
        /// Number of observation symbols.
        symbols: usize,
        /// Pseudo-count added to fitted initial, transition, and emission rows.
        additive_smoothing: f64,
    },
    /// A finite model with scalar Gaussian emissions.
    Gaussian {
        /// Number of hidden states.
        states: usize,
        /// Pseudo-count added to fitted initial and transition rows.
        additive_smoothing: f64,
        /// Hard lower bound for fitted variances.
        variance_floor: f64,
    },
}

impl HmmSpec {
    pub(crate) fn states(&self) -> usize {
        match self {
            Self::Discrete { states, .. } | Self::Gaussian { states, .. } => *states,
        }
    }

    pub(crate) fn smoothing(&self) -> f64 {
        match self {
            Self::Discrete {
                additive_smoothing, ..
            }
            | Self::Gaussian {
                additive_smoothing, ..
            } => *additive_smoothing,
        }
    }

    fn validate(&self) -> Result<(), HmmError> {
        if self.states() == 0 {
            return Err(HmmError::InvalidFitControl {
                field: "spec.states",
                reason: "must be greater than zero",
            });
        }
        if !self.smoothing().is_finite() || self.smoothing() <= 0.0 {
            return Err(HmmError::InvalidFitControl {
                field: "spec.additive_smoothing",
                reason: "must be finite and greater than zero",
            });
        }
        match self {
            Self::Discrete { symbols: 0, .. } => Err(HmmError::InvalidFitControl {
                field: "spec.symbols",
                reason: "must be greater than zero",
            }),
            Self::Gaussian { variance_floor, .. }
                if !variance_floor.is_finite() || *variance_floor <= 0.0 =>
            {
                Err(HmmError::InvalidFitControl {
                    field: "spec.variance_floor",
                    reason: "must be finite and greater than zero",
                })
            }
            _ => Ok(()),
        }
    }
}

/// Deterministic initialization, convergence, and work policy for Baum-Welch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HmmFitControl {
    /// Caller-owned deterministic initialization seed.
    pub seed: u64,
    /// Hard maximum number of accepted Baum-Welch updates.
    pub max_iterations: usize,
    /// Relative log-likelihood convergence tolerance.
    pub tolerance: f64,
    /// Hard maximum charged state-transition work.
    pub max_work: u64,
    /// Probability floor used while normalizing fitted rows.
    pub probability_floor: f64,
}

impl HmmFitControl {
    /// Builds checked fitting control.
    pub fn new(
        seed: u64,
        max_iterations: usize,
        tolerance: f64,
        max_work: u64,
        probability_floor: f64,
    ) -> Result<Self, HmmError> {
        let control = Self {
            seed,
            max_iterations,
            tolerance,
            max_work,
            probability_floor,
        };
        control.validate()?;
        Ok(control)
    }

    fn validate(&self) -> Result<(), HmmError> {
        for (field, valid, reason) in [
            (
                "max_iterations",
                self.max_iterations > 0,
                "must be greater than zero",
            ),
            ("max_work", self.max_work > 0, "must be greater than zero"),
            (
                "tolerance",
                self.tolerance.is_finite() && self.tolerance >= 0.0,
                "must be finite and nonnegative",
            ),
            (
                "probability_floor",
                self.probability_floor.is_finite()
                    && self.probability_floor > 0.0
                    && self.probability_floor < 1.0,
                "must be finite and in the open interval (0, 1)",
            ),
        ] {
            if !valid {
                return Err(HmmError::InvalidFitControl { field, reason });
            }
        }
        Ok(())
    }
}

/// Why bounded Baum-Welch stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HmmTermination {
    /// Relative log-likelihood improvement met the tolerance.
    Converged,
    /// The configured iteration count was exhausted.
    IterationLimit,
    /// The next complete expectation/maximization update exceeded `max_work`.
    WorkLimit,
    /// A candidate update reduced likelihood beyond numerical tolerance and
    /// was rejected, leaving the last non-decreasing model in the report.
    LikelihoodDecrease,
}

/// Convergence, likelihood, repair, seed, and termination evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct HmmFitEvidence {
    /// Initial model log likelihood before any update.
    pub initial_log_likelihood: f64,
    /// Final accepted model log likelihood.
    pub log_likelihood: f64,
    /// Initial value plus every accepted likelihood, in order.
    pub likelihood_history: Vec<f64>,
    /// Number of accepted Baum-Welch updates.
    pub iterations: usize,
    /// Whether convergence tolerance caused termination.
    pub converged: bool,
    /// Count of probability or variance floor repairs.
    pub numerical_repairs: u64,
    /// Caller-supplied initialization seed.
    pub seed: u64,
    /// Charged state-transition work, never greater than the control bound.
    pub work: u64,
    /// Concrete reason fitting stopped.
    pub termination: HmmTermination,
}

/// A fitted hidden-state model together with complete termination evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct HmmFitReport<M> {
    /// Last accepted inspectable model.
    pub model: M,
    /// Fitting evidence and bounds.
    pub evidence: HmmFitEvidence,
}

/// Fits a discrete- or continuous-emission HMM with bounded Baum-Welch.
pub fn fit_hmm(
    data: &[Sequence],
    spec: HmmSpec,
    control: HmmFitControl,
) -> Result<HmmFitReport<HiddenMarkovModel<StateId>>, HmmError> {
    spec.validate()?;
    control.validate()?;
    validate_data(data, &spec)?;
    let unit_work = inference_work(data, spec.states())?;
    if unit_work > control.max_work {
        return Err(HmmError::InvalidFitControl {
            field: "max_work",
            reason: "must admit the initial likelihood sweep",
        });
    }
    let mut model = initialize_model(data, &spec, control.seed)?;
    let initial_log_likelihood = score_data(&model, data)?;
    let mut history = vec![initial_log_likelihood];
    let mut work = unit_work;
    let mut iterations = 0;
    let mut numerical_repairs = 0_u64;

    let termination = loop {
        if iterations == control.max_iterations {
            break HmmTermination::IterationLimit;
        }
        let update_work = unit_work.checked_mul(2).ok_or(HmmError::WorkOverflow)?;
        if work
            .checked_add(update_work)
            .is_none_or(|next| next > control.max_work)
        {
            break HmmTermination::WorkLimit;
        }
        let (candidate, repairs) = baum_welch_step(&model, data, &spec, control.probability_floor)?;
        let likelihood = score_data(&candidate, data)?;
        work += update_work;
        numerical_repairs = numerical_repairs.saturating_add(repairs);
        let previous = *history.last().unwrap_or(&initial_log_likelihood);
        let scale = previous.abs().max(1.0);
        if likelihood + control.tolerance * scale < previous {
            break HmmTermination::LikelihoodDecrease;
        }
        model = candidate;
        history.push(likelihood);
        iterations += 1;
        if (likelihood - previous).abs() <= control.tolerance * scale {
            break HmmTermination::Converged;
        }
    };

    let log_likelihood = *history.last().unwrap_or(&initial_log_likelihood);
    Ok(HmmFitReport {
        model,
        evidence: HmmFitEvidence {
            initial_log_likelihood,
            log_likelihood,
            likelihood_history: history,
            iterations,
            converged: termination == HmmTermination::Converged,
            numerical_repairs,
            seed: control.seed,
            work,
            termination,
        },
    })
}

fn validate_data(data: &[Sequence], spec: &HmmSpec) -> Result<(), HmmError> {
    if data.is_empty() {
        return Err(HmmError::EmptyInput);
    }
    for (index, sequence) in data.iter().enumerate() {
        if sequence.len() == 0 {
            return Err(HmmError::EmptySequence { index });
        }
        match (sequence, spec) {
            (Sequence::Discrete(values), HmmSpec::Discrete { symbols, .. }) => {
                if let Some(&symbol) = values.iter().find(|&&symbol| symbol >= *symbols) {
                    return Err(HmmError::UnknownSymbol {
                        symbol,
                        symbol_count: *symbols,
                    });
                }
            }
            (Sequence::Continuous(values), HmmSpec::Gaussian { .. }) => {
                if let Some(&value) = values.iter().find(|value| !value.is_finite()) {
                    return Err(HmmError::NonFiniteObservation { value });
                }
            }
            _ => return Err(HmmError::MixedSequenceKinds),
        }
    }
    Ok(())
}

fn inference_work(data: &[Sequence], states: usize) -> Result<u64, HmmError> {
    let observations = data.iter().try_fold(0_u64, |sum, sequence| {
        sum.checked_add(sequence.len() as u64)
            .ok_or(HmmError::WorkOverflow)
    })?;
    observations
        .checked_mul(states as u64)
        .and_then(|value| value.checked_mul(states as u64))
        .ok_or(HmmError::WorkOverflow)
}

fn score_data(model: &HiddenMarkovModel<StateId>, data: &[Sequence]) -> Result<f64, HmmError> {
    data.iter().try_fold(0.0, |sum, sequence| {
        let likelihood = match sequence {
            Sequence::Discrete(values) => forward_backward(model, values)?.evidence.log_likelihood,
            Sequence::Continuous(values) => {
                forward_backward(model, values)?.evidence.log_likelihood
            }
        };
        Ok(sum + likelihood)
    })
}

fn initialize_model(
    data: &[Sequence],
    spec: &HmmSpec,
    seed: u64,
) -> Result<HiddenMarkovModel<StateId>, HmmError> {
    let states = spec.states();
    let state_ids = (0..states).collect::<Vec<_>>();
    let mut random = SplitMix64::new(seed);
    let initial = random_distribution(states, &mut random);
    let transitions = (0..states)
        .map(|_| random_distribution(states, &mut random))
        .collect::<Vec<_>>();
    match spec {
        HmmSpec::Discrete { symbols, .. } => {
            let emissions = (0..states)
                .map(|_| random_distribution(*symbols, &mut random))
                .collect();
            HiddenMarkovModel::discrete(state_ids, initial, transitions, emissions)
        }
        HmmSpec::Gaussian { variance_floor, .. } => {
            let values = data
                .iter()
                .flat_map(|sequence| match sequence {
                    Sequence::Continuous(values) => values.as_slice(),
                    Sequence::Discrete(_) => &[],
                })
                .copied()
                .collect::<Vec<_>>();
            let global_mean = values.iter().sum::<f64>() / values.len() as f64;
            let global_variance = values
                .iter()
                .map(|value| (value - global_mean).powi(2))
                .sum::<f64>()
                / values.len() as f64;
            let variance = global_variance.max(*variance_floor);
            let means = (0..states)
                .map(|_| values[random.index(values.len())])
                .collect();
            HiddenMarkovModel::gaussian(
                state_ids,
                initial,
                transitions,
                means,
                vec![variance; states],
                *variance_floor,
            )
        }
    }
}

fn random_distribution(length: usize, random: &mut SplitMix64) -> Vec<f64> {
    let mut values = (0..length)
        .map(|_| 0.5 + random.unit_interval())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f64>();
    for value in &mut values {
        *value /= sum;
    }
    values
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn unit_interval(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next() % length as u64) as usize
    }
}
