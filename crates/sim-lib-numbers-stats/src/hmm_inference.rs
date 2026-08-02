//! Underflow-safe normalized forward/backward and path inference.

use super::hmm_model::{HiddenMarkovModel, HmmError, HmmObservation, log_probability};

/// Numerical and likelihood evidence from normalized sequence inference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InferenceEvidence {
    /// Natural logarithm of the full observation-sequence likelihood.
    pub log_likelihood: f64,
    /// Count of finite renormalizations needed beyond log-domain scaling.
    pub numerical_repairs: u64,
    /// Number of normalized time steps.
    pub normalized_steps: usize,
}

/// Normalized forward, backward, and smoothed posterior state probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardBackward {
    /// Time-major normalized filtering probabilities.
    pub forward: Vec<Vec<f64>>,
    /// Time-major normalized backward likelihood weights.
    pub backward: Vec<Vec<f64>>,
    /// Time-major normalized smoothed state probabilities.
    pub posterior: Vec<Vec<f64>>,
    /// Likelihood and normalization diagnostics.
    pub evidence: InferenceEvidence,
}

/// Maximum-probability hidden-state path and its joint log probability.
#[derive(Clone, Debug, PartialEq)]
pub struct ViterbiPath<S> {
    /// Hidden states in observation order.
    pub states: Vec<S>,
    /// State indices in the model's stable order.
    pub state_indices: Vec<usize>,
    /// Natural logarithm of the path and observations' joint probability.
    pub log_probability: f64,
    /// Number of numerical repairs; log-domain Viterbi normally reports zero.
    pub numerical_repairs: u64,
}

/// Per-position maximum-posterior state path.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorPath<S> {
    /// Hidden states in observation order.
    pub states: Vec<S>,
    /// State indices in the model's stable order.
    pub state_indices: Vec<usize>,
    /// Posterior probability of each selected state.
    pub confidence: Vec<f64>,
    /// Likelihood and normalization diagnostics.
    pub evidence: InferenceEvidence,
}

/// Runs normalized forward/backward inference in the log domain.
pub fn forward_backward<O: HmmObservation, S: Eq + Clone>(
    model: &HiddenMarkovModel<S>,
    observations: &[O],
) -> Result<ForwardBackward, HmmError> {
    if observations.is_empty() {
        return Err(HmmError::EmptyInput);
    }
    let states = model.state_count();
    let mut repairs = 0_u64;
    let mut forward = Vec::with_capacity(observations.len());
    let mut first = (0..states)
        .map(|state| {
            Ok(log_probability(model.initial_probabilities()[state])
                + model.emission_log(state, observations[0])?)
        })
        .collect::<Result<Vec<_>, HmmError>>()?;
    let mut log_likelihood = normalize_logs(&mut first, 0, &mut repairs)?;
    forward.push(first);

    for (position, observation) in observations.iter().copied().enumerate().skip(1) {
        let previous = forward.last().expect("forward row exists");
        let mut row = Vec::with_capacity(states);
        for to in 0..states {
            let terms = (0..states).map(|from| {
                log_probability(previous[from])
                    + log_probability(
                        model
                            .transitions()
                            .probability_by_index(from, to)
                            .unwrap_or(0.0),
                    )
            });
            row.push(log_sum_exp(terms) + model.emission_log(to, observation)?);
        }
        log_likelihood += normalize_logs(&mut row, position, &mut repairs)?;
        forward.push(row);
    }

    let mut backward = vec![vec![0.0; states]; observations.len()];
    backward[observations.len() - 1].fill(1.0 / states as f64);
    for position in (0..observations.len() - 1).rev() {
        let mut row = Vec::with_capacity(states);
        for from in 0..states {
            let mut terms = Vec::with_capacity(states);
            for (to, &backward_probability) in backward[position + 1].iter().enumerate() {
                terms.push(
                    log_probability(
                        model
                            .transitions()
                            .probability_by_index(from, to)
                            .unwrap_or(0.0),
                    ) + model.emission_log(to, observations[position + 1])?
                        + log_probability(backward_probability),
                );
            }
            row.push(log_sum_exp(terms));
        }
        normalize_logs(&mut row, position, &mut repairs)?;
        backward[position] = row;
    }

    let posterior = forward
        .iter()
        .zip(&backward)
        .enumerate()
        .map(|(position, (alpha, beta))| {
            let mut row = alpha
                .iter()
                .zip(beta)
                .map(|(alpha, beta)| alpha * beta)
                .collect::<Vec<_>>();
            normalize_weights(&mut row, position)?;
            Ok(row)
        })
        .collect::<Result<Vec<_>, HmmError>>()?;
    Ok(ForwardBackward {
        forward,
        backward,
        posterior,
        evidence: InferenceEvidence {
            log_likelihood,
            numerical_repairs: repairs,
            normalized_steps: observations.len(),
        },
    })
}

/// Finds the maximum joint-probability hidden-state path in the log domain.
pub fn viterbi<O: HmmObservation, S: Eq + Clone>(
    model: &HiddenMarkovModel<S>,
    observations: &[O],
) -> Result<ViterbiPath<S>, HmmError> {
    if observations.is_empty() {
        return Err(HmmError::EmptyInput);
    }
    let states = model.state_count();
    let mut scores = (0..states)
        .map(|state| {
            Ok(log_probability(model.initial_probabilities()[state])
                + model.emission_log(state, observations[0])?)
        })
        .collect::<Result<Vec<_>, HmmError>>()?;
    require_possible(&scores, 0)?;
    let mut backpointers = Vec::with_capacity(observations.len().saturating_sub(1));
    for (position, observation) in observations.iter().copied().enumerate().skip(1) {
        let mut next = vec![f64::NEG_INFINITY; states];
        let mut pointers = vec![0; states];
        for to in 0..states {
            for (from, &score) in scores.iter().enumerate() {
                let candidate = score
                    + log_probability(
                        model
                            .transitions()
                            .probability_by_index(from, to)
                            .unwrap_or(0.0),
                    );
                if candidate > next[to] {
                    next[to] = candidate;
                    pointers[to] = from;
                }
            }
            next[to] += model.emission_log(to, observation)?;
        }
        require_possible(&next, position)?;
        scores = next;
        backpointers.push(pointers);
    }
    let (mut state, &log_probability) = scores
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .expect("non-empty hidden state set");
    let mut state_indices = vec![state];
    for pointers in backpointers.iter().rev() {
        state = pointers[state];
        state_indices.push(state);
    }
    state_indices.reverse();
    let states = state_indices
        .iter()
        .map(|&index| model.states()[index].clone())
        .collect();
    Ok(ViterbiPath {
        states,
        state_indices,
        log_probability,
        numerical_repairs: 0,
    })
}

/// Decodes the independently most probable hidden state at each position.
pub fn posterior_decode<O: HmmObservation, S: Eq + Clone>(
    model: &HiddenMarkovModel<S>,
    observations: &[O],
) -> Result<PosteriorPath<S>, HmmError> {
    let inference = forward_backward(model, observations)?;
    let mut state_indices = Vec::with_capacity(observations.len());
    let mut confidence = Vec::with_capacity(observations.len());
    for row in &inference.posterior {
        let (state, &probability) = row
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .expect("non-empty hidden state set");
        state_indices.push(state);
        confidence.push(probability);
    }
    let states = state_indices
        .iter()
        .map(|&index| model.states()[index].clone())
        .collect();
    Ok(PosteriorPath {
        states,
        state_indices,
        confidence,
        evidence: inference.evidence,
    })
}

pub(crate) fn log_sum_exp(values: impl IntoIterator<Item = f64>) -> f64 {
    let values = values.into_iter().collect::<Vec<_>>();
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return maximum;
    }
    maximum
        + values
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn normalize_logs(values: &mut [f64], position: usize, repairs: &mut u64) -> Result<f64, HmmError> {
    let log_total = log_sum_exp(values.iter().copied());
    if !log_total.is_finite() {
        return Err(HmmError::ImpossibleSequence { position });
    }
    for value in values.iter_mut() {
        *value = (*value - log_total).exp();
    }
    normalize_probabilities(values, position, repairs)?;
    Ok(log_total)
}

fn normalize_probabilities(
    values: &mut [f64],
    position: usize,
    repairs: &mut u64,
) -> Result<(), HmmError> {
    let sum = values.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(HmmError::ImpossibleSequence { position });
    }
    if (sum - 1.0).abs() > 1.0e-12 {
        *repairs = repairs.saturating_add(1);
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

fn normalize_weights(values: &mut [f64], position: usize) -> Result<(), HmmError> {
    let sum = values.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(HmmError::ImpossibleSequence { position });
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

fn require_possible(scores: &[f64], position: usize) -> Result<(), HmmError> {
    if scores.iter().any(|score| score.is_finite()) {
        Ok(())
    } else {
        Err(HmmError::ImpossibleSequence { position })
    }
}
