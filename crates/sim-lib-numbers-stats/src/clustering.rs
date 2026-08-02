//! Deterministic, bounded k-means clustering and shared clustering substrate.

use std::{error::Error, fmt};

/// Errors returned by clustering and mixture-model fitting.
#[derive(Clone, Debug, PartialEq)]
pub enum ClusteringError {
    /// No observations were supplied.
    EmptyInput,
    /// An observation had no coordinates.
    ZeroDimension,
    /// Observation dimensions were inconsistent.
    DimensionMismatch {
        /// Expected coordinate count.
        expected: usize,
        /// Actual coordinate count.
        actual: usize,
        /// Index of the mismatching observation.
        point: usize,
    },
    /// An observation contained NaN or infinity.
    NonFiniteInput {
        /// Observation index.
        point: usize,
        /// Coordinate index.
        coordinate: usize,
        /// Rejected value.
        value: f64,
    },
    /// A requested component count cannot be fitted to the observations.
    InvalidComponentCount {
        /// Requested component count.
        components: usize,
        /// Number of supplied observations.
        points: usize,
    },
    /// A control or model field was invalid.
    InvalidControl {
        /// Name of the invalid field.
        field: &'static str,
        /// Stable reason for rejection.
        reason: &'static str,
    },
    /// The work bound could not admit one complete initial model.
    WorkLimit {
        /// Configured work bound.
        limit: u64,
        /// Work already charged when admission failed.
        used: u64,
    },
    /// Size or work accounting overflowed.
    ArithmeticOverflow {
        /// Operation whose accounting overflowed.
        operation: &'static str,
    },
    /// A component could not be made numerically nonsingular under policy.
    SingularComponent {
        /// Component index in stable model order.
        component: usize,
    },
    /// Finite inputs produced a non-finite intermediate result.
    NumericalFailure {
        /// Operation that failed.
        operation: &'static str,
    },
}

impl fmt::Display for ClusteringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "clustering requires at least one point"),
            Self::ZeroDimension => write!(f, "clustering points must have at least one coordinate"),
            Self::DimensionMismatch {
                expected,
                actual,
                point,
            } => write!(
                f,
                "clustering point {point} has dimension {actual}, expected {expected}"
            ),
            Self::NonFiniteInput {
                point,
                coordinate,
                value,
            } => write!(
                f,
                "clustering point {point} coordinate {coordinate} is not finite: {value}"
            ),
            Self::InvalidComponentCount { components, points } => write!(
                f,
                "clustering component count must be in 1..={points}, got {components}"
            ),
            Self::InvalidControl { field, reason } => {
                write!(f, "invalid clustering control {field}: {reason}")
            }
            Self::WorkLimit { limit, used } => write!(
                f,
                "clustering work limit {limit} cannot admit another complete step after {used} units"
            ),
            Self::ArithmeticOverflow { operation } => {
                write!(f, "clustering accounting overflowed during {operation}")
            }
            Self::SingularComponent { component } => {
                write!(
                    f,
                    "mixture component {component} is singular under the selected policy"
                )
            }
            Self::NumericalFailure { operation } => {
                write!(
                    f,
                    "clustering produced a non-finite result during {operation}"
                )
            }
        }
    }
}

impl Error for ClusteringError {}

/// Deterministic initialization, convergence, restart, and work policy for k-means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KMeansControl {
    /// Root seed from which bounded restart seeds are derived.
    pub seed: u64,
    /// Maximum number of complete Lloyd updates per restart.
    pub max_iterations: usize,
    /// Maximum centroid displacement accepted as convergence.
    pub tolerance: f64,
    /// Hard bound on point-to-centroid distance evaluations across all restarts.
    pub max_work: u64,
    /// Maximum number of independently seeded candidate results.
    pub restarts: usize,
}

impl KMeansControl {
    /// Builds checked k-means control.
    pub fn new(
        seed: u64,
        max_iterations: usize,
        tolerance: f64,
        max_work: u64,
        restarts: usize,
    ) -> Result<Self, ClusteringError> {
        let control = Self {
            seed,
            max_iterations,
            tolerance,
            max_work,
            restarts,
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
            ("restarts", self.restarts > 0, "must be greater than zero"),
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

impl Default for KMeansControl {
    fn default() -> Self {
        Self {
            seed: 0,
            max_iterations: 100,
            tolerance: 1.0e-8,
            max_work: 100_000,
            restarts: 1,
        }
    }
}

/// Why one k-means restart stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KMeansTermination {
    /// Assignments or centroid displacement met the convergence policy.
    Converged,
    /// The restart exhausted its iteration count.
    IterationLimit,
    /// The shared work budget could not admit another complete Lloyd step.
    WorkLimit,
}

/// Why the bounded multi-restart search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KMeansSearchTermination {
    /// Every requested restart completed.
    Completed,
    /// The shared work budget stopped the current or next restart.
    WorkLimit,
}

/// Inspectable k-means centroids and stable cluster assignments.
#[derive(Clone, Debug, PartialEq)]
pub struct KMeansModel {
    /// Lexicographically ordered centroids.
    pub centroids: Vec<Vec<f64>>,
    /// Cluster index for every input point, in input order.
    pub assignments: Vec<usize>,
}

/// Convergence evidence for one bounded restart.
#[derive(Clone, Debug, PartialEq)]
pub struct KMeansRestartEvidence {
    /// Zero-based restart index.
    pub restart: usize,
    /// Derived seed used by k-means++ initialization.
    pub seed: u64,
    /// Final within-cluster sum of squared distances.
    pub inertia: f64,
    /// Number of complete Lloyd updates.
    pub iterations: usize,
    /// Whether convergence, rather than a bound, stopped the restart.
    pub converged: bool,
    /// Number of empty centroids deterministically reseeded from worst-fit points.
    pub empty_cluster_repairs: u64,
    /// Distance-evaluation work charged by this restart.
    pub work: u64,
    /// Concrete termination reason.
    pub termination: KMeansTermination,
}

/// Selected model and complete multi-restart evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct KMeansReport {
    /// Lowest-inertia model, with stable restart-index tie breaking.
    pub model: KMeansModel,
    /// Index into [`Self::restarts`] selected by the model-selection policy.
    pub selected_restart: usize,
    /// Evidence for every candidate that reached a complete assignment.
    pub restarts: Vec<KMeansRestartEvidence>,
    /// Number of requested candidates.
    pub requested_restarts: usize,
    /// Total charged work across initialization and all candidates.
    pub work: u64,
    /// Concrete search-level termination reason.
    pub termination: KMeansSearchTermination,
}

/// Fits deterministic seeded k-means with k-means++ initialization.
///
/// Empty clusters are reseeded from distinct worst-fit points. Candidate models
/// are compared by inertia, then by restart order, so identical inputs and
/// control produce byte-for-byte identical reports.
pub fn fit_kmeans(
    points: &[Vec<f64>],
    clusters: usize,
    control: KMeansControl,
) -> Result<KMeansReport, ClusteringError> {
    validate_points(points)?;
    validate_components(points.len(), clusters)?;
    control.validate()?;

    let mut meter = WorkMeter::new(control.max_work);
    let mut candidates = Vec::with_capacity(control.restarts);
    let mut models = Vec::with_capacity(control.restarts);
    let mut search_termination = KMeansSearchTermination::Completed;

    for restart in 0..control.restarts {
        let seed = derived_seed(control.seed, restart);
        match run_kmeans(points, clusters, control, restart, seed, &mut meter) {
            Ok((model, evidence)) => {
                let stopped = evidence.termination == KMeansTermination::WorkLimit;
                models.push(model);
                candidates.push(evidence);
                if stopped {
                    search_termination = KMeansSearchTermination::WorkLimit;
                    break;
                }
            }
            Err(ClusteringError::WorkLimit { .. }) if !models.is_empty() => {
                search_termination = KMeansSearchTermination::WorkLimit;
                break;
            }
            Err(error) => return Err(error),
        }
    }

    let selected_restart = candidates
        .iter()
        .enumerate()
        .min_by(|(left_index, left), (right_index, right)| {
            left.inertia
                .total_cmp(&right.inertia)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
        .ok_or(ClusteringError::WorkLimit {
            limit: control.max_work,
            used: meter.used,
        })?;

    Ok(KMeansReport {
        model: models.swap_remove(selected_restart),
        selected_restart,
        restarts: candidates,
        requested_restarts: control.restarts,
        work: meter.used,
        termination: search_termination,
    })
}

fn run_kmeans(
    points: &[Vec<f64>],
    clusters: usize,
    control: KMeansControl,
    restart: usize,
    seed: u64,
    meter: &mut WorkMeter,
) -> Result<(KMeansModel, KMeansRestartEvidence), ClusteringError> {
    let start_work = meter.used;
    let mut random = SplitMix64::new(seed);
    let centroids = kmeans_plus_plus(points, clusters, &mut random, meter)?;
    let (assignments, residuals, inertia) = assign_points(points, &centroids, meter)?;
    let mut model = KMeansModel {
        centroids,
        assignments,
    };
    let mut inertia = inertia;
    let mut residuals = residuals;
    let mut iterations = 0;
    let mut repairs = 0_u64;
    let mut termination = KMeansTermination::IterationLimit;

    while iterations < control.max_iterations {
        let (next_centroids, next_repairs) = update_centroids(points, &model, &residuals, clusters);
        let displacement = centroid_displacement(&model.centroids, &next_centroids)?;
        let previous_assignments = model.assignments.clone();
        let assigned = assign_points(points, &next_centroids, meter);
        let (next_assignments, next_residuals, next_inertia) = match assigned {
            Ok(result) => result,
            Err(ClusteringError::WorkLimit { .. }) => {
                termination = KMeansTermination::WorkLimit;
                break;
            }
            Err(error) => return Err(error),
        };
        model.centroids = next_centroids;
        model.assignments = next_assignments;
        residuals = next_residuals;
        inertia = next_inertia;
        repairs = repairs.saturating_add(next_repairs);
        iterations += 1;
        if displacement <= control.tolerance || model.assignments == previous_assignments {
            termination = KMeansTermination::Converged;
            break;
        }
    }

    canonicalize_kmeans(&mut model);
    Ok((
        model,
        KMeansRestartEvidence {
            restart,
            seed,
            inertia,
            iterations,
            converged: termination == KMeansTermination::Converged,
            empty_cluster_repairs: repairs,
            work: meter.used - start_work,
            termination,
        },
    ))
}

fn update_centroids(
    points: &[Vec<f64>],
    model: &KMeansModel,
    residuals: &[f64],
    clusters: usize,
) -> (Vec<Vec<f64>>, u64) {
    let dimensions = points[0].len();
    let mut centroids = vec![vec![0.0; dimensions]; clusters];
    let mut counts = vec![0_usize; clusters];
    for (point, &cluster) in points.iter().zip(&model.assignments) {
        counts[cluster] += 1;
        for (sum, &coordinate) in centroids[cluster].iter_mut().zip(point) {
            *sum += coordinate;
        }
    }
    for (centroid, &count) in centroids.iter_mut().zip(&counts) {
        if count > 0 {
            for coordinate in centroid {
                *coordinate /= count as f64;
            }
        }
    }

    let mut used_points = vec![false; points.len()];
    let mut repairs = 0_u64;
    for cluster in 0..clusters {
        if counts[cluster] != 0 {
            continue;
        }
        let point = residuals
            .iter()
            .enumerate()
            .filter(|(index, _)| !used_points[*index])
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("clusters never exceed points");
        centroids[cluster].clone_from(&points[point]);
        used_points[point] = true;
        repairs += 1;
    }
    (centroids, repairs)
}

fn centroid_displacement(current: &[Vec<f64>], next: &[Vec<f64>]) -> Result<f64, ClusteringError> {
    let maximum = current
        .iter()
        .zip(next)
        .map(|(left, right)| squared_distance(left, right).sqrt())
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    if maximum.is_finite() {
        Ok(maximum)
    } else {
        Err(ClusteringError::NumericalFailure {
            operation: "centroid displacement",
        })
    }
}

fn canonicalize_kmeans(model: &mut KMeansModel) {
    let mut order = (0..model.centroids.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| compare_vectors(&model.centroids[left], &model.centroids[right]));
    let mut remap = vec![0; order.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new;
    }
    model.centroids = order
        .iter()
        .map(|&index| model.centroids[index].clone())
        .collect();
    for assignment in &mut model.assignments {
        *assignment = remap[*assignment];
    }
}

pub(crate) fn validate_points(points: &[Vec<f64>]) -> Result<usize, ClusteringError> {
    let Some(first) = points.first() else {
        return Err(ClusteringError::EmptyInput);
    };
    if first.is_empty() {
        return Err(ClusteringError::ZeroDimension);
    }
    let dimensions = first.len();
    for (point_index, point) in points.iter().enumerate() {
        if point.len() != dimensions {
            return Err(ClusteringError::DimensionMismatch {
                expected: dimensions,
                actual: point.len(),
                point: point_index,
            });
        }
        for (coordinate, &value) in point.iter().enumerate() {
            if !value.is_finite() {
                return Err(ClusteringError::NonFiniteInput {
                    point: point_index,
                    coordinate,
                    value,
                });
            }
        }
    }
    Ok(dimensions)
}

pub(crate) fn validate_components(points: usize, components: usize) -> Result<(), ClusteringError> {
    if components == 0 || components > points {
        return Err(ClusteringError::InvalidComponentCount { components, points });
    }
    Ok(())
}

pub(crate) fn kmeans_plus_plus(
    points: &[Vec<f64>],
    clusters: usize,
    random: &mut SplitMix64,
    meter: &mut WorkMeter,
) -> Result<Vec<Vec<f64>>, ClusteringError> {
    let first = random.index(points.len());
    let mut selected = vec![first];
    let mut centroids = vec![points[first].clone()];
    while centroids.len() < clusters {
        let work = checked_product(points.len(), centroids.len(), "k-means++ distance work")?;
        meter.charge(work)?;
        let distances = points
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|centroid| squared_distance(point, centroid))
                    .min_by(f64::total_cmp)
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        let total = distances.iter().sum::<f64>();
        if !total.is_finite() {
            return Err(ClusteringError::NumericalFailure {
                operation: "k-means++ weighting",
            });
        }
        let next = if total > 0.0 {
            let threshold = random.unit_interval() * total;
            let mut cumulative = 0.0;
            distances
                .iter()
                .enumerate()
                .find_map(|(index, distance)| {
                    cumulative += distance;
                    (cumulative > threshold).then_some(index)
                })
                .unwrap_or(points.len() - 1)
        } else {
            (0..points.len())
                .find(|index| !selected.contains(index))
                .unwrap_or(0)
        };
        selected.push(next);
        centroids.push(points[next].clone());
    }
    Ok(centroids)
}

pub(crate) fn assign_points(
    points: &[Vec<f64>],
    centroids: &[Vec<f64>],
    meter: &mut WorkMeter,
) -> Result<(Vec<usize>, Vec<f64>, f64), ClusteringError> {
    meter.charge(checked_product(
        points.len(),
        centroids.len(),
        "k-means assignment work",
    )?)?;
    let mut assignments = Vec::with_capacity(points.len());
    let mut residuals = Vec::with_capacity(points.len());
    for point in points {
        let (cluster, distance) = centroids
            .iter()
            .enumerate()
            .map(|(index, centroid)| (index, squared_distance(point, centroid)))
            .min_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| left_index.cmp(right_index))
            })
            .expect("component count was validated");
        assignments.push(cluster);
        residuals.push(distance);
    }
    let inertia = residuals.iter().sum::<f64>();
    if inertia.is_finite() {
        Ok((assignments, residuals, inertia))
    } else {
        Err(ClusteringError::NumericalFailure {
            operation: "k-means inertia",
        })
    }
}

pub(crate) fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum()
}

pub(crate) fn compare_vectors(left: &[f64], right: &[f64]) -> std::cmp::Ordering {
    left.iter()
        .zip(right)
        .find_map(|(left, right)| {
            let ordering = left.total_cmp(right);
            (ordering != std::cmp::Ordering::Equal).then_some(ordering)
        })
        .unwrap_or_else(|| left.len().cmp(&right.len()))
}

pub(crate) fn checked_product(
    left: usize,
    right: usize,
    operation: &'static str,
) -> Result<u64, ClusteringError> {
    let left =
        u64::try_from(left).map_err(|_| ClusteringError::ArithmeticOverflow { operation })?;
    let right =
        u64::try_from(right).map_err(|_| ClusteringError::ArithmeticOverflow { operation })?;
    left.checked_mul(right)
        .ok_or(ClusteringError::ArithmeticOverflow { operation })
}

pub(crate) struct WorkMeter {
    pub(crate) limit: u64,
    pub(crate) used: u64,
}

impl WorkMeter {
    pub(crate) fn new(limit: u64) -> Self {
        Self { limit, used: 0 }
    }

    pub(crate) fn charge(&mut self, amount: u64) -> Result<(), ClusteringError> {
        let Some(next) = self.used.checked_add(amount) else {
            return Err(ClusteringError::ArithmeticOverflow {
                operation: "work charge",
            });
        };
        if next > self.limit {
            return Err(ClusteringError::WorkLimit {
                limit: self.limit,
                used: self.used,
            });
        }
        self.used = next;
        Ok(())
    }
}

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub(crate) fn unit_interval(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    pub(crate) fn index(&mut self, length: usize) -> usize {
        (self.next() % length as u64) as usize
    }
}

fn derived_seed(seed: u64, restart: usize) -> u64 {
    let mut random = SplitMix64::new(seed ^ restart as u64);
    random.next()
}
