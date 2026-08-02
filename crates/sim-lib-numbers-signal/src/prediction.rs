//! Bounded forward and backward prediction from stable autoregressive models.

use crate::{ArModel, SignalError};

/// Explicit limits for recursive forward or backward prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictionPlan {
    /// Number of samples to predict.
    pub horizon: usize,
    /// Maximum admitted prediction horizon.
    pub max_horizon: usize,
    /// Maximum absolute centered or restored prediction.
    pub max_abs_value: f64,
    /// Conservative multiply-add work ceiling.
    pub max_work: u64,
}

impl PredictionPlan {
    /// Creates a prediction request with finite default ceilings.
    pub const fn new(horizon: usize) -> Self {
        Self {
            horizon,
            max_horizon: 16_384,
            max_abs_value: 1.0e12,
            max_work: 100_000_000,
        }
    }
}

/// Direction of recursive autoregressive prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionDirection {
    /// Extend after the most recent observation.
    Forward,
    /// Extend before the earliest observation by applying the model to reversed history.
    Backward,
}

/// Samples and exact bounds retained from a prediction.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictionResult {
    /// Predicted samples in increasing distance from the observed boundary.
    pub samples: Vec<f64>,
    /// Direction requested by the caller.
    pub direction: PredictionDirection,
    /// Work charged for the recursive evaluation.
    pub work_units: u64,
    /// Absolute-value ceiling applied to every prediction.
    pub max_abs_value: f64,
}

/// Predicts samples after the supplied chronological history.
pub fn predict_forward(
    model: &ArModel,
    history: &[f64],
    plan: &PredictionPlan,
) -> Result<PredictionResult, SignalError> {
    predict(model, history, plan, PredictionDirection::Forward)
}

/// Predicts samples before the supplied chronological history.
pub fn predict_backward(
    model: &ArModel,
    history: &[f64],
    plan: &PredictionPlan,
) -> Result<PredictionResult, SignalError> {
    predict(model, history, plan, PredictionDirection::Backward)
}

fn predict(
    model: &ArModel,
    history: &[f64],
    plan: &PredictionPlan,
    direction: PredictionDirection,
) -> Result<PredictionResult, SignalError> {
    let order = model.coefficients.len();
    if history.len() < order {
        return Err(SignalError::LengthMismatch {
            expected: order,
            actual: history.len(),
        });
    }
    if plan.horizon > plan.max_horizon
        || !plan.max_abs_value.is_finite()
        || plan.max_abs_value <= 0.0
    {
        return Err(SignalError::InvalidPolicy {
            policy: "prediction bounds",
            reason: "horizon and finite positive amplitude bounds must be admitted",
        });
    }
    validate_real(history)?;
    let work_units = u64::try_from(order)
        .ok()
        .and_then(|order| u64::try_from(plan.horizon).ok()?.checked_mul(order))
        .ok_or(SignalError::InvalidPolicy {
            policy: "prediction work",
            reason: "work-unit arithmetic overflowed",
        })?;
    if work_units > plan.max_work {
        return Err(SignalError::WorkLimit {
            required: work_units,
            maximum: plan.max_work,
        });
    }
    let mut state = match direction {
        PredictionDirection::Forward => history.to_vec(),
        PredictionDirection::Backward => history.iter().rev().copied().collect(),
    };
    let mut output = Vec::with_capacity(plan.horizon);
    for index in 0..plan.horizon {
        let centered = model
            .coefficients
            .iter()
            .enumerate()
            .map(|(lag, coefficient)| coefficient * (state[state.len() - 1 - lag] - model.mean))
            .sum::<f64>();
        let value = model.mean - centered;
        if !value.is_finite() || value.abs() > plan.max_abs_value {
            return Err(SignalError::PredictionLimit { index });
        }
        state.push(value);
        output.push(value);
    }
    Ok(PredictionResult {
        samples: output,
        direction,
        work_units,
        max_abs_value: plan.max_abs_value,
    })
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
