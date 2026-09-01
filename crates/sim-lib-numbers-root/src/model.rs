//! Root-solver plans, evidence, results, and shared scalar helpers.

use super::*;

/// A certified sign-changing interval, including endpoint orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootBracket {
    /// Lower coordinate.
    pub lower: f64,
    /// Upper coordinate.
    pub upper: f64,
    /// Function value at `lower`.
    pub f_lower: f64,
    /// Function value at `upper`.
    pub f_upper: f64,
}
impl RootBracket {
    /// Validates finite ordered endpoints and opposite signs (or an endpoint root).
    pub fn new(a: f64, fa: f64, b: f64, fb: f64) -> Result<Self, RootTermination> {
        if ![a, fa, b, fb].into_iter().all(f64::is_finite) {
            return Err(RootTermination::NonFiniteEvaluation);
        }
        if a > b {
            return Self::new(b, fb, a, fa);
        }
        if a == b || !(fa == 0.0 || fb == 0.0 || fa.is_sign_positive() != fb.is_sign_positive()) {
            return Err(RootTermination::InvalidBracket);
        }
        Ok(Self {
            lower: a,
            upper: b,
            f_lower: fa,
            f_upper: fb,
        })
    }
    /// Interval width.
    pub fn width(self) -> f64 {
        self.upper - self.lower
    }
}

/// A residual-carrying point estimate; it makes no enclosure claim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootEstimate {
    /// Estimated root coordinate.
    pub value: f64,
    /// Signed function residual.
    pub residual: f64,
}

/// Concrete end state of a root search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootTermination {
    /// Residual tolerance was reached.
    ResidualConverged,
    /// Bracket-width tolerance was reached.
    BracketConverged,
    /// An endpoint was exactly a root.
    EndpointRoot,
    /// The initial interval did not establish a bracket.
    InvalidBracket,
    /// An evaluation returned NaN or infinity.
    NonFiniteEvaluation,
    /// Interior sampling exposed a sign jump without residual decrease.
    DiscontinuousSignChange,
    /// A derivative was too small to support a step.
    FlatDerivative,
    /// Recent iterates repeated.
    Cycling,
    /// Neither residual nor step made useful progress.
    Stagnation,
    /// The Jacobian lost numerical rank.
    RankLoss,
    /// The iteration bound was exhausted.
    IterationLimit,
    /// The evaluation/work bound was exhausted.
    WorkLimit,
    /// A plan was invalid.
    InvalidPlan,
}

/// Evaluation, iteration, and convergence proof retained by every solver.
#[derive(Clone, Debug, PartialEq)]
pub struct RootEvidence {
    /// Concrete termination.
    pub termination: RootTermination,
    /// Function or residual-vector calls.
    pub evaluations: u64,
    /// Derivative or Jacobian calls/builds.
    pub derivative_evaluations: u64,
    /// Completed solver iterations.
    pub iterations: usize,
    /// Smallest observed residual norm.
    pub best_residual: f64,
    /// Last accepted step norm.
    pub last_step: f64,
    /// Stable execution identity for replay.
    pub execution: ExecutionIdentity,
}

/// Scalar result that preserves whether a bracket was actually proved.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarRoot {
    /// Best residual-carrying point.
    pub estimate: RootEstimate,
    /// Certified enclosure, only for bracket-preserving methods.
    pub bracket: Option<RootBracket>,
    /// Search evidence.
    pub evidence: RootEvidence,
}

/// Shared scalar bounds and tolerances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarPlan {
    /// Absolute residual tolerance.
    pub residual_tolerance: f64,
    /// Absolute step/bracket tolerance.
    pub step_tolerance: f64,
    /// Iteration ceiling.
    pub max_iterations: usize,
    /// Total function-call ceiling.
    pub max_evaluations: u64,
    /// Ratio above which a sign-changing midpoint is treated as discontinuous.
    pub discontinuity_ratio: f64,
}
impl Default for ScalarPlan {
    fn default() -> Self {
        Self {
            residual_tolerance: 1e-12,
            step_tolerance: 1e-12,
            max_iterations: 128,
            max_evaluations: 512,
            discontinuity_ratio: 1e6,
        }
    }
}

/// Explicit source of a scalar derivative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DerivativeSource {
    /// Caller-supplied analytic derivative.
    Analytic,
    /// Derivative supplied by `sim-lib-numbers-ad` or another AD adapter.
    Automatic,
    /// Central finite difference with recorded absolute/relative step policy.
    FiniteDifference(FiniteDifferencePlan),
}

/// Reviewable finite-difference step policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiniteDifferencePlan {
    /// Absolute minimum step.
    pub absolute_step: f64,
    /// Step relative to `max(1, |x|)`.
    pub relative_step: f64,
}
impl FiniteDifferencePlan {
    pub(crate) fn step(self, x: f64) -> f64 {
        self.absolute_step
            .max(self.relative_step * x.abs().max(1.0))
    }
}

pub(crate) fn valid_scalar_plan(p: ScalarPlan) -> bool {
    p.residual_tolerance.is_finite()
        && p.residual_tolerance >= 0.0
        && p.step_tolerance.is_finite()
        && p.step_tolerance >= 0.0
        && p.max_iterations > 0
        && p.max_evaluations > 0
        && p.discontinuity_ratio.is_finite()
        && p.discontinuity_ratio > 1.0
}
pub(crate) fn evidence(
    t: RootTermination,
    ev: u64,
    dev: u64,
    it: usize,
    best: f64,
    step: f64,
    id: ExecutionIdentity,
) -> RootEvidence {
    RootEvidence {
        termination: t,
        evaluations: ev,
        derivative_evaluations: dev,
        iterations: it,
        best_residual: best,
        last_step: step,
        execution: id,
    }
}
pub(crate) fn scalar_result(
    x: f64,
    fx: f64,
    bracket: Option<RootBracket>,
    e: RootEvidence,
) -> ScalarRoot {
    ScalarRoot {
        estimate: RootEstimate {
            value: x,
            residual: fx,
        },
        bracket,
        evidence: e,
    }
}
