//! Burg autoregressive estimation, maximum-entropy spectra, and bounded prediction.

use crate::{EstimatorLimits, SignalError, spectrum_types::admit_work};

/// Criterion used to choose an autoregressive order up to the declared maximum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArOrderCriterion {
    /// Use the requested maximum order exactly.
    #[default]
    Fixed,
    /// Minimize Akaike's information criterion.
    Akaike,
    /// Minimize the Bayesian information criterion.
    Bayesian,
    /// Minimize the finite-sample final prediction error.
    FinalPredictionError,
}

/// Handling when Burg recursion reaches a singular or unstable higher order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BurgStability {
    /// Reject the complete fit instead of returning a questionable model.
    #[default]
    Reject,
    /// Stop at the last stable, nonsingular order and report the reduction.
    ReduceOrder,
}

/// Why Burg recursion stopped at the reported effective order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BurgTermination {
    /// The complete requested order was admitted.
    RequestedOrder,
    /// A later stage had no usable residual energy.
    SingularAt(usize),
    /// A later reflection coefficient crossed the stability margin.
    UnstableAt(usize),
}

/// Explicit policy and resource ceiling for a Burg fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BurgPlan {
    /// Largest autoregressive order considered.
    pub max_order: usize,
    /// Declared order-selection criterion.
    pub criterion: ArOrderCriterion,
    /// Reject or retain the preceding stable order on a failed stage.
    pub stability: BurgStability,
    /// Relative energy floor used to detect singular recursion.
    pub singular_tolerance: f64,
    /// Required distance of every reflection coefficient from unit magnitude.
    pub stability_margin: f64,
    /// Conservative deterministic work-unit ceiling.
    pub max_work: u64,
}

impl BurgPlan {
    /// Creates a fixed-order, fail-closed Burg plan.
    pub const fn new(order: usize) -> Self {
        Self {
            max_order: order,
            criterion: ArOrderCriterion::Fixed,
            stability: BurgStability::Reject,
            singular_tolerance: 1.0e-12,
            stability_margin: 1.0e-10,
            max_work: 100_000_000,
        }
    }
}

/// Score and residual evidence for one admitted candidate order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArOrderScore {
    /// Candidate order.
    pub order: usize,
    /// Mean squared one-step residual after this candidate's warm-up span.
    pub residual_variance: f64,
    /// Value of the plan's declared selection criterion.
    pub criterion_score: f64,
}

/// Evidence retained beside an autoregressive model.
#[derive(Clone, Debug, PartialEq)]
pub struct BurgEvidence {
    /// Number of input samples.
    pub input_len: usize,
    /// Maximum order requested by the caller.
    pub requested_order: usize,
    /// Stable order actually selected.
    pub effective_order: usize,
    /// Declared selection criterion.
    pub criterion: ArOrderCriterion,
    /// Reason recursion stopped.
    pub termination: BurgTermination,
    /// Sum of squared one-step residuals for the selected order.
    pub residual_energy: f64,
    /// Minimum distance of an admitted reflection coefficient from unit magnitude.
    pub minimum_reflection_margin: f64,
    /// Candidate scores considered by the selector.
    pub candidate_scores: Vec<ArOrderScore>,
    /// Conservative work charged before fitting.
    pub work_units: u64,
    /// Work ceiling that admitted the fit.
    pub work_limit: u64,
}

/// Stable real autoregressive model in `x[t] + sum(a[k] x[t-k-1]) = e[t]` form.
#[derive(Clone, Debug, PartialEq)]
pub struct ArModel {
    /// Prediction coefficients in increasing lag order.
    pub coefficients: Vec<f64>,
    /// Burg reflection coefficients in increasing order.
    pub reflection_coefficients: Vec<f64>,
    /// Mean removed before fitting and restored during prediction.
    pub mean: f64,
    /// Selected model's one-step innovation variance.
    pub innovation_variance: f64,
    /// Order, residual, stability, and work evidence.
    pub evidence: BurgEvidence,
}

#[derive(Clone)]
struct Candidate {
    coefficients: Vec<f64>,
    reflections: Vec<f64>,
    residual_energy: f64,
    residual_variance: f64,
}

/// Estimates a centered real autoregressive model with Burg's lattice recursion.
pub fn burg(samples: &[f64], plan: &BurgPlan) -> Result<ArModel, SignalError> {
    validate_burg(samples, plan)?;
    let work_units = burg_work(samples.len(), plan.max_order)?;
    admit_work(
        work_units,
        EstimatorLimits {
            max_work: plan.max_work,
            ..EstimatorLimits::default()
        },
    )?;
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let centered = samples.iter().map(|value| value - mean).collect::<Vec<_>>();
    let initial_energy = centered.iter().map(|value| value * value).sum::<f64>();
    if initial_energy <= plan.singular_tolerance * samples.len() as f64 {
        return Err(SignalError::SingularModel { order: 1 });
    }

    let mut forward = centered[1..].to_vec();
    let mut backward = centered[..centered.len() - 1].to_vec();
    let mut coefficients = Vec::new();
    let mut reflections = Vec::new();
    let mut candidates = Vec::with_capacity(plan.max_order);
    let mut termination = BurgTermination::RequestedOrder;

    for order in 1..=plan.max_order {
        let denominator = forward
            .iter()
            .zip(&backward)
            .map(|(front, back)| front * front + back * back)
            .sum::<f64>();
        if !denominator.is_finite()
            || denominator <= plan.singular_tolerance * initial_energy.max(1.0)
        {
            termination = BurgTermination::SingularAt(order);
            if plan.stability == BurgStability::Reject || candidates.is_empty() {
                return Err(SignalError::SingularModel { order });
            }
            break;
        }
        let numerator = -2.0
            * forward
                .iter()
                .zip(&backward)
                .map(|(front, back)| front * back)
                .sum::<f64>();
        let reflection = numerator / denominator;
        if !reflection.is_finite() || reflection.abs() >= 1.0 - plan.stability_margin {
            termination = BurgTermination::UnstableAt(order);
            if plan.stability == BurgStability::Reject || candidates.is_empty() {
                return Err(SignalError::UnstableModel { order });
            }
            break;
        }

        let previous = coefficients.clone();
        coefficients.resize(order, 0.0);
        for index in 0..order - 1 {
            coefficients[index] = previous[index] + reflection * previous[order - 2 - index];
        }
        coefficients[order - 1] = reflection;
        reflections.push(reflection);
        let (residual_energy, residual_variance) = residual(&centered, &coefficients);
        if !residual_variance.is_finite()
            || residual_energy <= plan.singular_tolerance * initial_energy
        {
            termination = BurgTermination::SingularAt(order);
            reflections.pop();
            if plan.stability == BurgStability::Reject || candidates.is_empty() {
                return Err(SignalError::SingularModel { order });
            }
            break;
        }
        candidates.push(Candidate {
            coefficients: coefficients.clone(),
            reflections: reflections.clone(),
            residual_energy,
            residual_variance,
        });

        if order < plan.max_order {
            let next_len = forward.len() - 1;
            let mut next_forward = Vec::with_capacity(next_len);
            let mut next_backward = Vec::with_capacity(next_len);
            for index in 0..next_len {
                next_forward.push(forward[index + 1] + reflection * backward[index + 1]);
                next_backward.push(backward[index] + reflection * forward[index]);
            }
            forward = next_forward;
            backward = next_backward;
        }
    }

    let scores = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| ArOrderScore {
            order: index + 1,
            residual_variance: candidate.residual_variance,
            criterion_score: criterion_score(
                plan.criterion,
                samples.len(),
                index + 1,
                candidate.residual_variance,
            ),
        })
        .collect::<Vec<_>>();
    let selected = if plan.criterion == ArOrderCriterion::Fixed {
        candidates.len() - 1
    } else {
        scores
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.criterion_score.total_cmp(&right.1.criterion_score))
            .map(|(index, _)| index)
            .ok_or(SignalError::SingularModel { order: 1 })?
    };
    let candidate = &candidates[selected];
    let minimum_reflection_margin = candidate
        .reflections
        .iter()
        .map(|value| 1.0 - value.abs())
        .fold(1.0, f64::min);
    Ok(ArModel {
        coefficients: candidate.coefficients.clone(),
        reflection_coefficients: candidate.reflections.clone(),
        mean,
        innovation_variance: candidate.residual_variance,
        evidence: BurgEvidence {
            input_len: samples.len(),
            requested_order: plan.max_order,
            effective_order: selected + 1,
            criterion: plan.criterion,
            termination,
            residual_energy: candidate.residual_energy,
            minimum_reflection_margin,
            candidate_scores: scores,
            work_units,
            work_limit: plan.max_work,
        },
    })
}

fn validate_burg(samples: &[f64], plan: &BurgPlan) -> Result<(), SignalError> {
    validate_real(samples)?;
    if plan.max_order == 0 || plan.max_order >= samples.len() - 1 {
        return Err(SignalError::InvalidLength {
            len: plan.max_order,
            reason: "Burg order must be positive and leave at least two residual samples",
        });
    }
    if !plan.singular_tolerance.is_finite() || plan.singular_tolerance <= 0.0 {
        return Err(SignalError::InvalidPolicy {
            policy: "Burg singular tolerance",
            reason: "a finite positive tolerance is required",
        });
    }
    if !plan.stability_margin.is_finite()
        || plan.stability_margin <= 0.0
        || plan.stability_margin >= 1.0
    {
        return Err(SignalError::InvalidPolicy {
            policy: "Burg stability margin",
            reason: "the margin must lie strictly between zero and one",
        });
    }
    Ok(())
}

fn validate_real(samples: &[f64]) -> Result<(), SignalError> {
    for (index, value) in samples.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "value",
            });
        }
    }
    Ok(())
}

fn residual(samples: &[f64], coefficients: &[f64]) -> (f64, f64) {
    let order = coefficients.len();
    let energy = (order..samples.len())
        .map(|index| {
            let error = samples[index]
                + coefficients
                    .iter()
                    .enumerate()
                    .map(|(lag, coefficient)| coefficient * samples[index - lag - 1])
                    .sum::<f64>();
            error * error
        })
        .sum::<f64>();
    (energy, energy / (samples.len() - order) as f64)
}

fn criterion_score(
    criterion: ArOrderCriterion,
    sample_count: usize,
    order: usize,
    variance: f64,
) -> f64 {
    let n = sample_count as f64;
    let p = order as f64;
    match criterion {
        ArOrderCriterion::Fixed => p,
        ArOrderCriterion::Akaike => n * variance.ln() + 2.0 * p,
        ArOrderCriterion::Bayesian => n * variance.ln() + p * n.ln(),
        ArOrderCriterion::FinalPredictionError => variance * (n + p) / (n - p),
    }
}

fn burg_work(sample_count: usize, order: usize) -> Result<u64, SignalError> {
    let samples = u64::try_from(sample_count).map_err(|_| SignalError::InvalidPolicy {
        policy: "Burg work",
        reason: "sample count does not fit the work counter",
    })?;
    let order = u64::try_from(order).map_err(|_| SignalError::InvalidPolicy {
        policy: "Burg work",
        reason: "order does not fit the work counter",
    })?;
    samples
        .checked_mul(order)
        .and_then(|value| value.checked_mul(8))
        .and_then(|value| value.checked_add(order.checked_mul(order)?))
        .ok_or(SignalError::InvalidPolicy {
            policy: "Burg work",
            reason: "work-unit arithmetic overflowed",
        })
}
