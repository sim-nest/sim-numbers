//! Covariance preparation, validation, and model-selection math for GMM fitting.

use std::f64::consts::PI;

use super::{
    clustering::{ClusteringError, compare_vectors},
    gmm::{CovarianceType, GaussianCovariance, GmmModel, ModelSelectionEvidence},
};

pub(super) enum PreparedCovariance {
    Diagonal {
        inverse: Vec<f64>,
        log_determinant: f64,
    },
    Full {
        lower: Vec<Vec<f64>>,
        log_determinant: f64,
    },
}

impl PreparedCovariance {
    pub(super) fn work(&self) -> Result<u64, ClusteringError> {
        match self {
            Self::Diagonal { inverse, .. } => {
                u64::try_from(inverse.len()).map_err(|_| ClusteringError::ArithmeticOverflow {
                    operation: "GMM covariance dimension",
                })
            }
            Self::Full { lower, .. } => {
                let dimensions = u64::try_from(lower.len()).map_err(|_| {
                    ClusteringError::ArithmeticOverflow {
                        operation: "GMM covariance dimension",
                    }
                })?;
                dimensions
                    .checked_mul(dimensions)
                    .and_then(|value| value.checked_add(dimensions))
                    .ok_or(ClusteringError::ArithmeticOverflow {
                        operation: "GMM covariance work",
                    })
            }
        }
    }

    pub(super) fn log_density(&self, point: &[f64], mean: &[f64]) -> f64 {
        let (mahalanobis, log_determinant): (f64, f64) = match self {
            Self::Diagonal {
                inverse,
                log_determinant,
            } => (
                point
                    .iter()
                    .zip(mean)
                    .zip(inverse)
                    .map(|((&point, &mean), inverse)| {
                        let difference = point - mean;
                        difference * difference * inverse
                    })
                    .sum(),
                *log_determinant,
            ),
            Self::Full {
                lower,
                log_determinant,
            } => {
                let mut solved = vec![0.0; point.len()];
                for row in 0..point.len() {
                    let prior = (0..row)
                        .map(|column| lower[row][column] * solved[column])
                        .sum::<f64>();
                    solved[row] = (point[row] - mean[row] - prior) / lower[row][row];
                }
                (
                    solved.iter().map(|value| value * value).sum(),
                    *log_determinant,
                )
            }
        };
        -0.5 * (point.len() as f64 * (2.0 * PI).ln() + log_determinant + mahalanobis)
    }
}

pub(super) fn prepare_model(
    model: &GmmModel,
    dimensions: usize,
) -> Result<Vec<PreparedCovariance>, ClusteringError> {
    model
        .covariances
        .iter()
        .map(|covariance| prepare_covariance(covariance, dimensions))
        .collect()
}

pub(super) fn prepare_covariance(
    covariance: &GaussianCovariance,
    dimensions: usize,
) -> Result<PreparedCovariance, ClusteringError> {
    match covariance {
        GaussianCovariance::Diagonal(variances) => {
            if variances.len() != dimensions
                || variances
                    .iter()
                    .any(|variance| !variance.is_finite() || *variance <= 0.0)
            {
                return Err(ClusteringError::SingularComponent { component: 0 });
            }
            Ok(PreparedCovariance::Diagonal {
                inverse: variances.iter().map(|variance| 1.0 / variance).collect(),
                log_determinant: variances.iter().map(|variance| variance.ln()).sum(),
            })
        }
        GaussianCovariance::Full(matrix) => {
            if matrix.len() != dimensions || matrix.iter().any(|row| row.len() != dimensions) {
                return Err(ClusteringError::SingularComponent { component: 0 });
            }
            let lower = cholesky(matrix)?;
            let log_determinant = 2.0
                * (0..dimensions)
                    .map(|index| lower[index][index].ln())
                    .sum::<f64>();
            Ok(PreparedCovariance::Full {
                lower,
                log_determinant,
            })
        }
    }
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ClusteringError> {
    let dimensions = matrix.len();
    let mut lower = vec![vec![0.0; dimensions]; dimensions];
    for row in 0..dimensions {
        for column in 0..=row {
            let inner = (0..column)
                .map(|index| lower[row][index] * lower[column][index])
                .sum::<f64>();
            if row == column {
                let diagonal = matrix[row][row] - inner;
                if !diagonal.is_finite() || diagonal <= 0.0 {
                    return Err(ClusteringError::SingularComponent { component: 0 });
                }
                lower[row][column] = diagonal.sqrt();
            } else {
                lower[row][column] = (matrix[row][column] - inner) / lower[column][column];
                if !lower[row][column].is_finite() {
                    return Err(ClusteringError::SingularComponent { component: 0 });
                }
            }
        }
    }
    Ok(lower)
}

pub(super) fn validate_model(model: &GmmModel, dimensions: usize) -> Result<(), ClusteringError> {
    let components = model.weights.len();
    if components == 0 || model.means.len() != components || model.covariances.len() != components {
        return Err(ClusteringError::InvalidControl {
            field: "model.components",
            reason: "weights, means, and covariances must be non-empty and aligned",
        });
    }
    if model
        .means
        .iter()
        .any(|mean| mean.len() != dimensions || mean.iter().any(|value| !value.is_finite()))
    {
        return Err(ClusteringError::InvalidControl {
            field: "model.means",
            reason: "must be finite and match the point dimension",
        });
    }
    if model
        .weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
        || (model.weights.iter().sum::<f64>() - 1.0).abs() > 1.0e-9
    {
        return Err(ClusteringError::InvalidControl {
            field: "model.weights",
            reason: "must be positive, finite, and normalized",
        });
    }
    prepare_model(model, dimensions).map(|_| ())
}

pub(super) fn model_selection(
    log_likelihood: f64,
    observations: usize,
    dimensions: usize,
    components: usize,
    covariance: CovarianceType,
) -> Result<ModelSelectionEvidence, ClusteringError> {
    let covariance_parameters = match covariance {
        CovarianceType::Diagonal => dimensions,
        CovarianceType::Full => dimensions
            .checked_mul(dimensions + 1)
            .and_then(|value| value.checked_div(2))
            .ok_or(ClusteringError::ArithmeticOverflow {
                operation: "GMM covariance parameter count",
            })?,
    };
    let per_component = dimensions.checked_add(covariance_parameters).ok_or(
        ClusteringError::ArithmeticOverflow {
            operation: "GMM parameter count",
        },
    )?;
    let parameters = components
        .checked_mul(per_component)
        .and_then(|value| value.checked_add(components - 1))
        .ok_or(ClusteringError::ArithmeticOverflow {
            operation: "GMM parameter count",
        })?;
    let aic = 2.0 * parameters as f64 - 2.0 * log_likelihood;
    let bic = (observations as f64).ln() * parameters as f64 - 2.0 * log_likelihood;
    Ok(ModelSelectionEvidence {
        log_likelihood,
        parameters,
        aic,
        bic,
        observations,
    })
}

pub(super) fn canonicalize_model(model: &mut GmmModel) {
    let mut order = (0..model.means.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| compare_vectors(&model.means[left], &model.means[right]));
    model.weights = order.iter().map(|&index| model.weights[index]).collect();
    model.means = order
        .iter()
        .map(|&index| model.means[index].clone())
        .collect();
    model.covariances = order
        .iter()
        .map(|&index| model.covariances[index].clone())
        .collect();
}

pub(super) fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(f64::NEG_INFINITY);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum
        + values
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
}

pub(super) fn require_finite_covariance(
    values: impl IntoIterator<Item = f64>,
) -> Result<(), ClusteringError> {
    if values.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(ClusteringError::NumericalFailure {
            operation: "GMM covariance update",
        })
    }
}
