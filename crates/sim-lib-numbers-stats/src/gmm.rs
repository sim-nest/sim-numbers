//! Regularized Gaussian-mixture EM with log-domain evidence.

use super::clustering::{
    ClusteringError, SplitMix64, WorkMeter, checked_product, kmeans_plus_plus, validate_components,
    validate_points,
};
use super::gmm_math::{
    PreparedCovariance, canonicalize_model, log_sum_exp, model_selection, prepare_covariance,
    prepare_model, require_finite_covariance, validate_model,
};

/// Covariance representation fitted for every Gaussian component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovarianceType {
    /// One regularized variance per coordinate.
    Diagonal,
    /// One regularized symmetric covariance matrix per component.
    Full,
}

/// Inspectable covariance parameters for one Gaussian component.
#[derive(Clone, Debug, PartialEq)]
pub enum GaussianCovariance {
    /// Coordinate variances in point-coordinate order.
    Diagonal(Vec<f64>),
    /// Symmetric row-major covariance matrix.
    Full(Vec<Vec<f64>>),
}

/// Policy for an EM component with negligible responsibility or singular covariance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SingularComponentPolicy {
    /// Reinitialize from a stable worst-fit observation and the global covariance.
    Reinitialize {
        /// Smallest accepted fraction of total responsibility mass.
        minimum_weight: f64,
    },
    /// Fail closed instead of changing a singular component.
    Fail {
        /// Smallest accepted fraction of total responsibility mass.
        minimum_weight: f64,
    },
}

impl SingularComponentPolicy {
    fn minimum_weight(self) -> f64 {
        match self {
            Self::Reinitialize { minimum_weight } | Self::Fail { minimum_weight } => minimum_weight,
        }
    }

    fn validate(self) -> Result<(), ClusteringError> {
        let weight = self.minimum_weight();
        if !weight.is_finite() || !(0.0..1.0).contains(&weight) {
            return Err(ClusteringError::InvalidControl {
                field: "singular_policy.minimum_weight",
                reason: "must be finite and in the open interval (0, 1)",
            });
        }
        Ok(())
    }
}

impl Default for SingularComponentPolicy {
    fn default() -> Self {
        Self::Reinitialize {
            minimum_weight: 1.0e-8,
        }
    }
}

/// Component count, covariance family, regularization, and singular policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmmSpec {
    /// Number of Gaussian components.
    pub components: usize,
    /// Covariance representation shared by all components.
    pub covariance: CovarianceType,
    /// Positive value added to every fitted covariance diagonal.
    pub regularization: f64,
    /// Explicit behavior for empty or numerically singular components.
    pub singular_policy: SingularComponentPolicy,
}

impl GmmSpec {
    /// Builds a checked regularized mixture specification.
    pub fn new(
        components: usize,
        covariance: CovarianceType,
        regularization: f64,
        singular_policy: SingularComponentPolicy,
    ) -> Result<Self, ClusteringError> {
        let spec = Self {
            components,
            covariance,
            regularization,
            singular_policy,
        };
        spec.validate()?;
        Ok(spec)
    }

    fn validate(self) -> Result<(), ClusteringError> {
        if self.components == 0 {
            return Err(ClusteringError::InvalidControl {
                field: "components",
                reason: "must be greater than zero",
            });
        }
        if !self.regularization.is_finite() || self.regularization <= 0.0 {
            return Err(ClusteringError::InvalidControl {
                field: "regularization",
                reason: "must be finite and greater than zero",
            });
        }
        self.singular_policy.validate()
    }
}

/// Deterministic convergence and work policy for Gaussian-mixture EM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmmControl {
    /// Seed used by k-means++ mean initialization.
    pub seed: u64,
    /// Maximum number of accepted EM updates.
    pub max_iterations: usize,
    /// Relative log-likelihood convergence tolerance.
    pub tolerance: f64,
    /// Hard bound on initialization, responsibility, and parameter-update work.
    pub max_work: u64,
}

impl GmmControl {
    /// Builds checked EM control.
    pub fn new(
        seed: u64,
        max_iterations: usize,
        tolerance: f64,
        max_work: u64,
    ) -> Result<Self, ClusteringError> {
        let control = Self {
            seed,
            max_iterations,
            tolerance,
            max_work,
        };
        control.validate()?;
        Ok(control)
    }

    fn validate(self) -> Result<(), ClusteringError> {
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
        ] {
            if !valid {
                return Err(ClusteringError::InvalidControl { field, reason });
            }
        }
        Ok(())
    }
}

impl Default for GmmControl {
    fn default() -> Self {
        Self {
            seed: 0,
            max_iterations: 100,
            tolerance: 1.0e-8,
            max_work: 1_000_000,
        }
    }
}

/// Why bounded EM stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GmmTermination {
    /// Relative log-likelihood improvement met the tolerance.
    Converged,
    /// The configured iteration count was exhausted.
    IterationLimit,
    /// The work budget could not admit another complete update or score.
    WorkLimit,
    /// A candidate reduced likelihood and was rejected.
    LikelihoodDecrease,
}

/// AIC and BIC evidence for comparing fitted component counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelSelectionEvidence {
    /// Maximized natural-log likelihood.
    pub log_likelihood: f64,
    /// Number of independently fitted free parameters.
    pub parameters: usize,
    /// Akaike information criterion; lower is preferred.
    pub aic: f64,
    /// Bayesian information criterion; lower is preferred.
    pub bic: f64,
    /// Number of observations used by BIC.
    pub observations: usize,
}

/// Inspectable fitted Gaussian-mixture parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct GmmModel {
    /// Normalized component weights.
    pub weights: Vec<f64>,
    /// Component means in lexicographic order.
    pub means: Vec<Vec<f64>>,
    /// Covariance parameters aligned with [`Self::means`].
    pub covariances: Vec<GaussianCovariance>,
}

impl GmmModel {
    /// Returns posterior component probabilities for every supplied point.
    pub fn responsibilities(&self, points: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ClusteringError> {
        validate_points(points)?;
        validate_model(self, points[0].len())?;
        let mut meter = WorkMeter::new(u64::MAX);
        Ok(expectation(points, self, &mut meter)?.responsibilities)
    }

    /// Returns the summed natural-log likelihood of supplied points.
    pub fn log_likelihood(&self, points: &[Vec<f64>]) -> Result<f64, ClusteringError> {
        validate_points(points)?;
        validate_model(self, points[0].len())?;
        let mut meter = WorkMeter::new(u64::MAX);
        Ok(expectation(points, self, &mut meter)?.log_likelihood)
    }

    /// Returns the maximum-posterior component index for every point.
    pub fn predict(&self, points: &[Vec<f64>]) -> Result<Vec<usize>, ClusteringError> {
        self.responsibilities(points).map(|rows| {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(left_index, left), (right_index, right)| {
                            left.total_cmp(right)
                                .then_with(|| right_index.cmp(left_index))
                        })
                        .map(|(index, _)| index)
                        .expect("fitted model has components")
                })
                .collect()
        })
    }
}

/// Convergence, regularization, work, and selection evidence from EM.
#[derive(Clone, Debug, PartialEq)]
pub struct GmmEvidence {
    /// Initial mixture log likelihood.
    pub initial_log_likelihood: f64,
    /// Final accepted mixture log likelihood.
    pub log_likelihood: f64,
    /// Initial value followed by every accepted likelihood.
    pub likelihood_history: Vec<f64>,
    /// Number of accepted EM updates.
    pub iterations: usize,
    /// Whether tolerance caused termination.
    pub converged: bool,
    /// Count of components repaired under the singular policy.
    pub singular_component_repairs: u64,
    /// Caller-supplied initialization seed.
    pub seed: u64,
    /// Charged work, never greater than the configured limit.
    pub work: u64,
    /// Concrete termination reason.
    pub termination: GmmTermination,
    /// AIC/BIC evidence for component-count selection.
    pub model_selection: ModelSelectionEvidence,
}

/// Fitted mixture and complete EM evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct GmmReport {
    /// Last accepted model in canonical component order.
    pub model: GmmModel,
    /// Numerical, convergence, and selection evidence.
    pub evidence: GmmEvidence,
}

/// Fits a regularized diagonal or full-covariance Gaussian mixture.
///
/// Responsibilities and likelihood are computed in the log domain. Candidate
/// updates that decrease likelihood are rejected, and work exhaustion returns
/// the last complete model rather than a partially updated parameter set.
pub fn fit_gmm(
    points: &[Vec<f64>],
    spec: GmmSpec,
    control: GmmControl,
) -> Result<GmmReport, ClusteringError> {
    let dimensions = validate_points(points)?;
    spec.validate()?;
    validate_components(points.len(), spec.components)?;
    control.validate()?;

    let mut meter = WorkMeter::new(control.max_work);
    let global_covariance = global_covariance(points, spec.covariance, spec.regularization)?;
    let mut random = SplitMix64::new(control.seed);
    let means = kmeans_plus_plus(points, spec.components, &mut random, &mut meter)?;
    let mut model = GmmModel {
        weights: vec![1.0 / spec.components as f64; spec.components],
        means,
        covariances: vec![global_covariance.clone(); spec.components],
    };
    let mut state = expectation(points, &model, &mut meter)?;
    let initial_log_likelihood = state.log_likelihood;
    let mut history = vec![state.log_likelihood];
    let mut iterations = 0;
    let mut repairs = 0_u64;
    let mut termination = GmmTermination::IterationLimit;

    while iterations < control.max_iterations {
        let candidate = maximize(points, &state, spec, &global_covariance, &mut meter);
        let (candidate, candidate_repairs) = match candidate {
            Ok(value) => value,
            Err(ClusteringError::WorkLimit { .. }) => {
                termination = GmmTermination::WorkLimit;
                break;
            }
            Err(error) => return Err(error),
        };
        let next_state = match expectation(points, &candidate, &mut meter) {
            Ok(value) => value,
            Err(ClusteringError::WorkLimit { .. }) => {
                termination = GmmTermination::WorkLimit;
                break;
            }
            Err(error) => return Err(error),
        };
        let previous = state.log_likelihood;
        let scale = previous.abs().max(1.0);
        if next_state.log_likelihood + control.tolerance * scale < previous {
            termination = GmmTermination::LikelihoodDecrease;
            break;
        }
        model = candidate;
        state = next_state;
        repairs = repairs.saturating_add(candidate_repairs);
        history.push(state.log_likelihood);
        iterations += 1;
        if (state.log_likelihood - previous).abs() <= control.tolerance * scale {
            termination = GmmTermination::Converged;
            break;
        }
    }

    canonicalize_model(&mut model);
    let model_selection = model_selection(
        state.log_likelihood,
        points.len(),
        dimensions,
        spec.components,
        spec.covariance,
    )?;
    Ok(GmmReport {
        model,
        evidence: GmmEvidence {
            initial_log_likelihood,
            log_likelihood: state.log_likelihood,
            likelihood_history: history,
            iterations,
            converged: termination == GmmTermination::Converged,
            singular_component_repairs: repairs,
            seed: control.seed,
            work: meter.used,
            termination,
            model_selection,
        },
    })
}

struct ExpectationState {
    responsibilities: Vec<Vec<f64>>,
    point_log_likelihoods: Vec<f64>,
    log_likelihood: f64,
}

fn expectation(
    points: &[Vec<f64>],
    model: &GmmModel,
    meter: &mut WorkMeter,
) -> Result<ExpectationState, ClusteringError> {
    let dimensions = points[0].len();
    let prepared = prepare_model(model, dimensions)?;
    let component_cost =
        prepared
            .iter()
            .map(PreparedCovariance::work)
            .try_fold(0_u64, |sum, work| {
                sum.checked_add(work?)
                    .ok_or(ClusteringError::ArithmeticOverflow {
                        operation: "GMM likelihood work",
                    })
            })?;
    let point_count =
        u64::try_from(points.len()).map_err(|_| ClusteringError::ArithmeticOverflow {
            operation: "GMM point count",
        })?;
    meter.charge(component_cost.checked_mul(point_count).ok_or(
        ClusteringError::ArithmeticOverflow {
            operation: "GMM likelihood work",
        },
    )?)?;

    let mut responsibilities = Vec::with_capacity(points.len());
    let mut point_log_likelihoods = Vec::with_capacity(points.len());
    let mut log_likelihood = 0.0;
    for point in points {
        let log_weights = model
            .weights
            .iter()
            .zip(&model.means)
            .zip(&prepared)
            .map(|((&weight, mean), covariance)| weight.ln() + covariance.log_density(point, mean))
            .collect::<Vec<_>>();
        let normalizer = log_sum_exp(&log_weights);
        if !normalizer.is_finite() {
            return Err(ClusteringError::NumericalFailure {
                operation: "GMM log-domain normalization",
            });
        }
        responsibilities.push(
            log_weights
                .iter()
                .map(|weight| (weight - normalizer).exp())
                .collect(),
        );
        point_log_likelihoods.push(normalizer);
        log_likelihood += normalizer;
    }
    if !log_likelihood.is_finite() {
        return Err(ClusteringError::NumericalFailure {
            operation: "GMM log likelihood",
        });
    }
    Ok(ExpectationState {
        responsibilities,
        point_log_likelihoods,
        log_likelihood,
    })
}

fn maximize(
    points: &[Vec<f64>],
    state: &ExpectationState,
    spec: GmmSpec,
    global_covariance: &GaussianCovariance,
    meter: &mut WorkMeter,
) -> Result<(GmmModel, u64), ClusteringError> {
    let dimensions = points[0].len();
    let point_component_work =
        checked_product(points.len(), spec.components, "GMM maximization work")?;
    let dimension_work =
        u64::try_from(dimensions).map_err(|_| ClusteringError::ArithmeticOverflow {
            operation: "GMM dimensions",
        })?;
    meter.charge(point_component_work.checked_mul(dimension_work).ok_or(
        ClusteringError::ArithmeticOverflow {
            operation: "GMM maximization work",
        },
    )?)?;

    let mut masses = vec![0.0; spec.components];
    for row in &state.responsibilities {
        for (mass, responsibility) in masses.iter_mut().zip(row) {
            *mass += responsibility;
        }
    }
    let minimum_mass = spec.singular_policy.minimum_weight() * points.len() as f64;
    let mut means = vec![vec![0.0; dimensions]; spec.components];
    for (point, row) in points.iter().zip(&state.responsibilities) {
        for (component, &responsibility) in row.iter().enumerate() {
            for (sum, &coordinate) in means[component].iter_mut().zip(point) {
                *sum += responsibility * coordinate;
            }
        }
    }
    for (mean, &mass) in means.iter_mut().zip(&masses) {
        if mass > minimum_mass && mass.is_finite() {
            for coordinate in mean {
                *coordinate /= mass;
            }
        }
    }

    let mut covariances = (0..spec.components)
        .map(|component| {
            component_covariance(
                points,
                &state.responsibilities,
                component,
                &means[component],
                masses[component],
                spec.covariance,
                spec.regularization,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut repairs = 0_u64;
    let mut used_points = vec![false; points.len()];
    for component in 0..spec.components {
        let singular_mass = !masses[component].is_finite() || masses[component] <= minimum_mass;
        let singular_covariance = prepare_covariance(&covariances[component], dimensions).is_err();
        if !singular_mass && !singular_covariance {
            continue;
        }
        match spec.singular_policy {
            SingularComponentPolicy::Fail { .. } => {
                return Err(ClusteringError::SingularComponent { component });
            }
            SingularComponentPolicy::Reinitialize { .. } => {
                let point = worst_unused_point(&state.point_log_likelihoods, &used_points);
                used_points[point] = true;
                means[component].clone_from(&points[point]);
                covariances[component] = global_covariance.clone();
                masses[component] = minimum_mass.max(1.0);
                repairs += 1;
            }
        }
    }
    let total_mass = masses.iter().sum::<f64>();
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return Err(ClusteringError::NumericalFailure {
            operation: "GMM component weights",
        });
    }
    let weights = masses.iter().map(|mass| mass / total_mass).collect();
    Ok((
        GmmModel {
            weights,
            means,
            covariances,
        },
        repairs,
    ))
}

fn component_covariance(
    points: &[Vec<f64>],
    responsibilities: &[Vec<f64>],
    component: usize,
    mean: &[f64],
    mass: f64,
    covariance: CovarianceType,
    regularization: f64,
) -> Result<GaussianCovariance, ClusteringError> {
    let dimensions = mean.len();
    if !mass.is_finite() || mass <= 0.0 {
        return Ok(match covariance {
            CovarianceType::Diagonal => GaussianCovariance::Diagonal(vec![0.0; dimensions]),
            CovarianceType::Full => {
                GaussianCovariance::Full(vec![vec![0.0; dimensions]; dimensions])
            }
        });
    }
    match covariance {
        CovarianceType::Diagonal => {
            let mut variances = vec![0.0; dimensions];
            for (point, row) in points.iter().zip(responsibilities) {
                for coordinate in 0..dimensions {
                    let difference = point[coordinate] - mean[coordinate];
                    variances[coordinate] += row[component] * difference * difference;
                }
            }
            for variance in &mut variances {
                *variance = (*variance / mass) + regularization;
            }
            require_finite_covariance(variances.iter().copied())?;
            Ok(GaussianCovariance::Diagonal(variances))
        }
        CovarianceType::Full => {
            let mut matrix = vec![vec![0.0; dimensions]; dimensions];
            for (point, row) in points.iter().zip(responsibilities) {
                for left in 0..dimensions {
                    let left_difference = point[left] - mean[left];
                    for right in 0..=left {
                        let right_difference = point[right] - mean[right];
                        matrix[left][right] += row[component] * left_difference * right_difference;
                    }
                }
            }
            for (left, row) in matrix.iter_mut().enumerate() {
                for value in row.iter_mut().take(left + 1) {
                    *value /= mass;
                }
                row[left] += regularization;
            }
            for left in 0..dimensions {
                let (prior_rows, current_rows) = matrix.split_at_mut(left);
                let row = &current_rows[0];
                for (right, prior_row) in prior_rows.iter_mut().enumerate() {
                    prior_row[left] = row[right];
                }
            }
            require_finite_covariance(matrix.iter().flatten().copied())?;
            Ok(GaussianCovariance::Full(matrix))
        }
    }
}

fn global_covariance(
    points: &[Vec<f64>],
    covariance: CovarianceType,
    regularization: f64,
) -> Result<GaussianCovariance, ClusteringError> {
    let dimensions = points[0].len();
    let mut mean = vec![0.0; dimensions];
    for point in points {
        for (sum, &coordinate) in mean.iter_mut().zip(point) {
            *sum += coordinate;
        }
    }
    for coordinate in &mut mean {
        *coordinate /= points.len() as f64;
    }
    let responsibilities = vec![vec![1.0]; points.len()];
    component_covariance(
        points,
        &responsibilities,
        0,
        &mean,
        points.len() as f64,
        covariance,
        regularization,
    )
}

fn worst_unused_point(log_likelihoods: &[f64], used: &[bool]) -> usize {
    log_likelihoods
        .iter()
        .enumerate()
        .filter(|(index, _)| !used[*index])
        .min_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}
