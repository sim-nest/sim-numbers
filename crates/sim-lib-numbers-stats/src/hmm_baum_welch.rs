//! Expectation/maximization update for finite hidden Markov models.

use super::hmm_fit::{HmmSpec, Sequence, StateId};
use super::hmm_inference::{ForwardBackward, forward_backward, log_sum_exp};
use super::hmm_model::{HiddenMarkovModel, HmmError, log_probability};

struct ExpectedCounts {
    initial: Vec<f64>,
    transitions: Vec<Vec<f64>>,
    discrete: Option<Vec<Vec<f64>>>,
    gaussian_weight: Option<Vec<f64>>,
    gaussian_sum: Option<Vec<f64>>,
    gaussian_square_sum: Option<Vec<f64>>,
}

impl ExpectedCounts {
    fn new(spec: &HmmSpec) -> Self {
        let states = spec.states();
        Self {
            initial: vec![0.0; states],
            transitions: vec![vec![0.0; states]; states],
            discrete: match spec {
                HmmSpec::Discrete { symbols, .. } => Some(vec![vec![0.0; *symbols]; states]),
                HmmSpec::Gaussian { .. } => None,
            },
            gaussian_weight: matches!(spec, HmmSpec::Gaussian { .. }).then(|| vec![0.0; states]),
            gaussian_sum: matches!(spec, HmmSpec::Gaussian { .. }).then(|| vec![0.0; states]),
            gaussian_square_sum: matches!(spec, HmmSpec::Gaussian { .. })
                .then(|| vec![0.0; states]),
        }
    }
}

pub(crate) fn baum_welch_step(
    model: &HiddenMarkovModel<StateId>,
    data: &[Sequence],
    spec: &HmmSpec,
    probability_floor: f64,
) -> Result<(HiddenMarkovModel<StateId>, u64), HmmError> {
    let mut counts = ExpectedCounts::new(spec);
    for sequence in data {
        match sequence {
            Sequence::Discrete(observations) => {
                let inference = accumulate(model, observations, &mut counts)?;
                let emission = counts
                    .discrete
                    .as_mut()
                    .ok_or(HmmError::MixedSequenceKinds)?;
                for (symbol, posterior) in observations.iter().zip(&inference.posterior) {
                    for (state, probability) in posterior.iter().copied().enumerate() {
                        emission[state][*symbol] += probability;
                    }
                }
            }
            Sequence::Continuous(observations) => {
                let inference = accumulate(model, observations, &mut counts)?;
                let weights = counts
                    .gaussian_weight
                    .as_mut()
                    .ok_or(HmmError::MixedSequenceKinds)?;
                let sums = counts
                    .gaussian_sum
                    .as_mut()
                    .ok_or(HmmError::MixedSequenceKinds)?;
                let squares = counts
                    .gaussian_square_sum
                    .as_mut()
                    .ok_or(HmmError::MixedSequenceKinds)?;
                for (&value, posterior) in observations.iter().zip(&inference.posterior) {
                    for (state, probability) in posterior.iter().copied().enumerate() {
                        weights[state] += probability;
                        sums[state] += probability * value;
                        squares[state] += probability * value * value;
                    }
                }
            }
        }
    }

    maximize(model, spec, counts, probability_floor)
}

fn accumulate<O: super::hmm_model::HmmObservation>(
    model: &HiddenMarkovModel<StateId>,
    observations: &[O],
    counts: &mut ExpectedCounts,
) -> Result<ForwardBackward, HmmError> {
    let inference = forward_backward(model, observations)?;
    for (state, probability) in inference.posterior[0].iter().copied().enumerate() {
        counts.initial[state] += probability;
    }
    let states = model.states().len();
    for position in 0..observations.len().saturating_sub(1) {
        let mut log_weights = Vec::with_capacity(states * states);
        for from in 0..states {
            for to in 0..states {
                log_weights.push(
                    log_probability(inference.forward[position][from])
                        + log_probability(
                            model
                                .transitions()
                                .probability_by_index(from, to)
                                .unwrap_or(0.0),
                        )
                        + model.emission_log(to, observations[position + 1])?
                        + log_probability(inference.backward[position + 1][to]),
                );
            }
        }
        let normalizer = log_sum_exp(log_weights.iter().copied());
        if !normalizer.is_finite() {
            return Err(HmmError::ImpossibleSequence {
                position: position + 1,
            });
        }
        for from in 0..states {
            for to in 0..states {
                counts.transitions[from][to] +=
                    (log_weights[from * states + to] - normalizer).exp();
            }
        }
    }
    Ok(inference)
}

fn maximize(
    model: &HiddenMarkovModel<StateId>,
    spec: &HmmSpec,
    counts: ExpectedCounts,
    probability_floor: f64,
) -> Result<(HiddenMarkovModel<StateId>, u64), HmmError> {
    let mut repairs = 0_u64;
    let states = spec.states();
    let initial = normalize_counts(
        &counts.initial,
        spec.smoothing(),
        probability_floor,
        &mut repairs,
    );
    let transitions = counts
        .transitions
        .iter()
        .map(|row| normalize_counts(row, spec.smoothing(), probability_floor, &mut repairs))
        .collect::<Vec<_>>();
    let state_ids = (0..states).collect();
    let candidate = match spec {
        HmmSpec::Discrete { .. } => {
            let emissions = counts
                .discrete
                .ok_or(HmmError::MixedSequenceKinds)?
                .iter()
                .map(|row| normalize_counts(row, spec.smoothing(), probability_floor, &mut repairs))
                .collect();
            HiddenMarkovModel::discrete(state_ids, initial, transitions, emissions)?
        }
        HmmSpec::Gaussian { variance_floor, .. } => {
            let weights = counts.gaussian_weight.ok_or(HmmError::MixedSequenceKinds)?;
            let sums = counts.gaussian_sum.ok_or(HmmError::MixedSequenceKinds)?;
            let squares = counts
                .gaussian_square_sum
                .ok_or(HmmError::MixedSequenceKinds)?;
            let (old_means, old_variances, _) = model
                .emissions()
                .gaussian_parameters()
                .ok_or(HmmError::MixedSequenceKinds)?;
            let mut means = Vec::with_capacity(states);
            let mut variances = Vec::with_capacity(states);
            for state in 0..states {
                if weights[state] <= probability_floor {
                    means.push(old_means[state]);
                    variances.push(old_variances[state]);
                    repairs = repairs.saturating_add(1);
                    continue;
                }
                let mean = sums[state] / weights[state];
                let raw_variance = squares[state] / weights[state] - mean * mean;
                means.push(mean);
                if raw_variance < *variance_floor {
                    repairs = repairs.saturating_add(1);
                }
                variances.push(raw_variance.max(*variance_floor));
            }
            HiddenMarkovModel::gaussian(
                state_ids,
                initial,
                transitions,
                means,
                variances,
                *variance_floor,
            )?
        }
    };
    Ok((candidate, repairs))
}

fn normalize_counts(counts: &[f64], smoothing: f64, floor: f64, repairs: &mut u64) -> Vec<f64> {
    let mut probabilities = counts
        .iter()
        .map(|count| {
            let smoothed = count + smoothing;
            if smoothed < floor {
                *repairs = repairs.saturating_add(1);
            }
            smoothed.max(floor)
        })
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f64>();
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}
