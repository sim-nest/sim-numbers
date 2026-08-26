//! Optimization plans, evidence, and shared numerical helpers.

use super::*;

/// Source of first derivatives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivativeSource {
    Analytic,
    Automatic,
    FiniteDifference,
}
/// Globalization strategy; bounded paths are genuine projected/active-set methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepPolicy {
    BrentGolden,
    LevenbergMarquardt,
    TrustRegionReflective,
    ProjectedBfgs,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Termination {
    Converged,
    BoundaryConverged,
    Flat,
    WorkLimit,
    NonFinite,
    NoProgress,
    InvalidPlan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    pub argument: f64,
    pub objective: f64,
    pub gradient: f64,
}
impl Default for Tolerances {
    fn default() -> Self {
        Self {
            argument: 1e-9,
            objective: 1e-12,
            gradient: 1e-8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    pub evaluations: usize,
    pub iterations: usize,
    pub memory_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            evaluations: 10_000,
            iterations: 500,
            memory_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
impl Bounds {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, Error> {
        if lower.len() != upper.len()
            || lower
                .iter()
                .zip(&upper)
                .any(|(l, u)| !l.is_finite() || !u.is_finite() || l > u)
        {
            return Err(Error::InvalidPlan(
                "bounds must be finite, ordered, and equal length",
            ));
        }
        Ok(Self { lower, upper })
    }
    pub(crate) fn project(&self, x: &mut [f64]) {
        for ((x, l), u) in x.iter_mut().zip(&self.lower).zip(&self.upper) {
            *x = x.clamp(*l, *u);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectivePlan {
    pub bounds: Bounds,
    pub scale: Vec<f64>,
    pub derivative: DerivativeSource,
    pub policy: StepPolicy,
    pub tolerances: Tolerances,
    pub limits: Limits,
    pub initial_radius: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct LeastSquaresPlan {
    pub bounds: Option<Bounds>,
    pub variable_scale: Vec<f64>,
    pub residual_scale: Vec<f64>,
    pub derivative: DerivativeSource,
    pub policy: StepPolicy,
    pub tolerances: Tolerances,
    pub limits: Limits,
    pub initial_damping: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Work {
    pub evaluations: usize,
    pub iterations: usize,
    pub memory_bytes: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizeResult {
    pub point: Vec<f64>,
    pub value: f64,
    pub gradient_norm: f64,
    pub active: Vec<usize>,
    pub termination: Termination,
    pub work: Work,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarResult {
    pub minimizer: f64,
    pub value: f64,
    pub final_bracket: (f64, f64),
    pub termination: Termination,
    pub work: Work,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Covariance {
    Available(Vec<Vec<f64>>),
    Unavailable(CovarianceUnavailable),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CovarianceUnavailable {
    RankDeficient,
    InsufficientDegreesOfFreedom,
    StatisticalAssumptionsNotDeclared,
}
#[derive(Clone, Debug, PartialEq)]
pub struct LeastSquaresResult {
    pub point: Vec<f64>,
    pub residuals: Vec<f64>,
    pub residual_norm: f64,
    pub rank: usize,
    pub active: Vec<usize>,
    pub covariance: Covariance,
    pub termination: Termination,
    pub work: Work,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidPlan(&'static str),
    Dimension(&'static str),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(s) | Self::Dimension(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for Error {}

pub(crate) fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}
pub(crate) fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
pub(crate) fn validate_scale(scale: &[f64], n: usize) -> Result<(), Error> {
    if scale.len() != n || scale.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        Err(Error::InvalidPlan(
            "scale must contain one finite positive value per variable",
        ))
    } else {
        Ok(())
    }
}
