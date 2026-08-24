#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Bounded scalar and vector root finding with reviewable evidence.
//!
//! A [`RootBracket`] is deliberately not interchangeable with a
//! [`RootEstimate`]. Only the former proves a sign-changing enclosure.

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_method::ExecutionIdentity;
use sim_lib_numbers_tensor_decomp::{
    SingularCutoff, SvdPlan, VectorForm, least_squares, numerical_rank, svd_f64,
};
use std::{error::Error, fmt};

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
    fn step(self, x: f64) -> f64 {
        self.absolute_step
            .max(self.relative_step * x.abs().max(1.0))
    }
}

fn valid_scalar_plan(p: ScalarPlan) -> bool {
    p.residual_tolerance.is_finite()
        && p.residual_tolerance >= 0.0
        && p.step_tolerance.is_finite()
        && p.step_tolerance >= 0.0
        && p.max_iterations > 0
        && p.max_evaluations > 0
        && p.discontinuity_ratio.is_finite()
        && p.discontinuity_ratio > 1.0
}
fn evidence(
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
fn scalar_result(x: f64, fx: f64, bracket: Option<RootBracket>, e: RootEvidence) -> ScalarRoot {
    ScalarRoot {
        estimate: RootEstimate {
            value: x,
            residual: fx,
        },
        bracket,
        evidence: e,
    }
}

/// Establishes a bracket with exactly two evaluations and no iteration.
pub fn establish_bracket<F: FnMut(f64) -> f64>(
    mut f: F,
    a: f64,
    b: f64,
) -> Result<RootBracket, RootTermination> {
    RootBracket::new(a, f(a), b, f(b))
}

/// Bisection with endpoint, orientation, discontinuity, and work evidence.
pub fn bisection<F: FnMut(f64) -> f64>(
    mut f: F,
    bracket: RootBracket,
    p: ScalarPlan,
    id: ExecutionIdentity,
) -> ScalarRoot {
    if !valid_scalar_plan(p) {
        return scalar_result(
            bracket.lower,
            bracket.f_lower,
            Some(bracket),
            evidence(
                RootTermination::InvalidPlan,
                0,
                0,
                0,
                bracket.f_lower.abs(),
                0.0,
                id,
            ),
        );
    }
    if bracket.f_lower == 0.0 {
        return scalar_result(
            bracket.lower,
            0.0,
            Some(bracket),
            evidence(RootTermination::EndpointRoot, 0, 0, 0, 0.0, 0.0, id),
        );
    }
    if bracket.f_upper == 0.0 {
        return scalar_result(
            bracket.upper,
            0.0,
            Some(bracket),
            evidence(RootTermination::EndpointRoot, 0, 0, 0, 0.0, 0.0, id),
        );
    }
    let (mut a, mut b, mut fa, mut fb) = (
        bracket.lower,
        bracket.upper,
        bracket.f_lower,
        bracket.f_upper,
    );
    let mut best = if fa.abs() <= fb.abs() {
        (a, fa)
    } else {
        (b, fb)
    };
    for it in 1..=p.max_iterations {
        if it as u64 > p.max_evaluations {
            return scalar_result(
                best.0,
                best.1,
                RootBracket::new(a, fa, b, fb).ok(),
                evidence(
                    RootTermination::WorkLimit,
                    p.max_evaluations,
                    0,
                    it - 1,
                    best.1.abs(),
                    b - a,
                    id,
                ),
            );
        }
        let m = a + (b - a) / 2.0;
        let fm = f(m);
        if !fm.is_finite() {
            return scalar_result(
                best.0,
                best.1,
                RootBracket::new(a, fa, b, fb).ok(),
                evidence(
                    RootTermination::NonFiniteEvaluation,
                    it as u64,
                    0,
                    it,
                    best.1.abs(),
                    b - a,
                    id,
                ),
            );
        }
        if fm.abs() < best.1.abs() {
            best = (m, fm);
        }
        if fm == 0.0 || fm.abs() <= p.residual_tolerance {
            return scalar_result(
                m,
                fm,
                RootBracket::new(m, fm, m.next_up(), f(m.next_up()))
                    .ok()
                    .or_else(|| RootBracket::new(a, fa, b, fb).ok()),
                evidence(
                    RootTermination::ResidualConverged,
                    it as u64,
                    0,
                    it,
                    fm.abs(),
                    b - a,
                    id,
                ),
            );
        }
        let endpoint_min = fa
            .abs()
            .min(fb.abs())
            .max(p.residual_tolerance.max(f64::MIN_POSITIVE));
        if fm.abs() / endpoint_min > p.discontinuity_ratio
            && (b - a) <= p.step_tolerance.max(f64::EPSILON * m.abs()) * 8.0
        {
            return scalar_result(
                best.0,
                best.1,
                RootBracket::new(a, fa, b, fb).ok(),
                evidence(
                    RootTermination::DiscontinuousSignChange,
                    it as u64,
                    0,
                    it,
                    best.1.abs(),
                    b - a,
                    id,
                ),
            );
        }
        if fa.is_sign_positive() != fm.is_sign_positive() {
            b = m;
            fb = fm;
        } else {
            a = m;
            fa = fm;
        }
        if b - a <= p.step_tolerance {
            let br = RootBracket::new(a, fa, b, fb).ok();
            return scalar_result(
                best.0,
                best.1,
                br,
                evidence(
                    RootTermination::BracketConverged,
                    it as u64,
                    0,
                    it,
                    best.1.abs(),
                    b - a,
                    id,
                ),
            );
        }
    }
    scalar_result(
        best.0,
        best.1,
        RootBracket::new(a, fa, b, fb).ok(),
        evidence(
            RootTermination::IterationLimit,
            p.max_iterations as u64,
            0,
            p.max_iterations,
            best.1.abs(),
            b - a,
            id,
        ),
    )
}

/// Brent-Dekker interpolation safeguarded by a sign-changing bracket.
pub fn brent_dekker<F: FnMut(f64) -> f64>(
    mut f: F,
    br: RootBracket,
    p: ScalarPlan,
    id: ExecutionIdentity,
) -> ScalarRoot {
    if !valid_scalar_plan(p) {
        return scalar_result(
            br.lower,
            br.f_lower,
            Some(br),
            evidence(
                RootTermination::InvalidPlan,
                0,
                0,
                0,
                br.f_lower.abs(),
                0.0,
                id,
            ),
        );
    }
    if br.f_lower == 0.0 || br.f_upper == 0.0 {
        let (x, fx) = if br.f_lower == 0.0 {
            (br.lower, br.f_lower)
        } else {
            (br.upper, br.f_upper)
        };
        return scalar_result(
            x,
            fx,
            Some(br),
            evidence(RootTermination::EndpointRoot, 0, 0, 0, 0.0, 0.0, id),
        );
    }
    let (mut a, mut b, mut fa, mut fb) = (br.lower, br.upper, br.f_lower, br.f_upper);
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    for it in 1..=p.max_iterations {
        if it as u64 > p.max_evaluations {
            return scalar_result(
                b,
                fb,
                RootBracket::new(a, fa, b, fb)
                    .ok()
                    .or_else(|| RootBracket::new(b, fb, c, fc).ok()),
                evidence(
                    RootTermination::WorkLimit,
                    p.max_evaluations,
                    0,
                    it - 1,
                    fb.abs(),
                    d.abs(),
                    id,
                ),
            );
        }
        if fb.is_sign_positive() == fc.is_sign_positive() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = a;
            fc = fa;
        }
        let tol = p.step_tolerance.max(2.0 * f64::EPSILON * b.abs());
        let m = 0.5 * (c - b);
        if fb.abs() <= p.residual_tolerance {
            return scalar_result(
                b,
                fb,
                RootBracket::new(b, fb, c, fc).ok(),
                evidence(
                    RootTermination::ResidualConverged,
                    (it - 1) as u64,
                    0,
                    it - 1,
                    fb.abs(),
                    d.abs(),
                    id,
                ),
            );
        }
        if m.abs() <= tol {
            return scalar_result(
                b,
                fb,
                RootBracket::new(b, fb, c, fc).ok(),
                evidence(
                    RootTermination::BracketConverged,
                    (it - 1) as u64,
                    0,
                    it - 1,
                    fb.abs(),
                    m.abs() * 2.0,
                    id,
                ),
            );
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut q, mut r) = if a == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if q > 0.0 {
                r = -r;
            } else {
                q = -q;
            }
            let old = e;
            e = d;
            if 2.0 * q < (3.0 * m * r - (tol * r).abs()).min((old * r).abs()) {
                d = q / r;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = f(b);
        if !fb.is_finite() {
            return scalar_result(
                a,
                fa,
                RootBracket::new(a, fa, c, fc).ok(),
                evidence(
                    RootTermination::NonFiniteEvaluation,
                    it as u64,
                    0,
                    it,
                    fa.abs(),
                    d.abs(),
                    id,
                ),
            );
        }
    }
    scalar_result(
        b,
        fb,
        RootBracket::new(b, fb, c, fc).ok(),
        evidence(
            RootTermination::IterationLimit,
            p.max_iterations as u64,
            0,
            p.max_iterations,
            fb.abs(),
            d.abs(),
            id,
        ),
    )
}

/// Safeguarded Newton using a supplied analytic or AD derivative.
pub fn safeguarded_newton<F: FnMut(f64) -> f64, D: FnMut(f64) -> f64>(
    mut f: F,
    mut derivative: D,
    source: DerivativeSource,
    initial: f64,
    bracket: Option<RootBracket>,
    p: ScalarPlan,
    id: ExecutionIdentity,
) -> ScalarRoot {
    newton_impl(&mut f, &mut derivative, source, initial, bracket, p, id)
}

/// Safeguarded Newton with an explicit central finite-difference policy.
pub fn finite_difference_newton<F: FnMut(f64) -> f64>(
    mut f: F,
    initial: f64,
    bracket: Option<RootBracket>,
    fd: FiniteDifferencePlan,
    p: ScalarPlan,
    id: ExecutionIdentity,
) -> ScalarRoot {
    if !fd.absolute_step.is_finite()
        || !fd.relative_step.is_finite()
        || fd.absolute_step <= 0.0
        || fd.relative_step < 0.0
    {
        return scalar_result(
            initial,
            f64::NAN,
            bracket,
            evidence(
                RootTermination::InvalidPlan,
                0,
                0,
                0,
                f64::INFINITY,
                0.0,
                id,
            ),
        );
    }
    let cell = std::cell::RefCell::new(&mut f);
    let mut d = |x: f64| {
        let h = fd.step(x);
        let mut ff = cell.borrow_mut();
        (ff(x + h) - ff(x - h)) / (2.0 * h)
    };
    let mut value = |x: f64| (cell.borrow_mut())(x);
    newton_impl(
        &mut value,
        &mut d,
        DerivativeSource::FiniteDifference(fd),
        initial,
        bracket,
        p,
        id,
    )
}

fn newton_impl<F: FnMut(f64) -> f64, D: FnMut(f64) -> f64>(
    f: &mut F,
    d: &mut D,
    source: DerivativeSource,
    mut x: f64,
    mut br: Option<RootBracket>,
    p: ScalarPlan,
    id: ExecutionIdentity,
) -> ScalarRoot {
    if !valid_scalar_plan(p) {
        return scalar_result(
            x,
            f64::NAN,
            br,
            evidence(
                RootTermination::InvalidPlan,
                0,
                0,
                0,
                f64::INFINITY,
                0.0,
                id,
            ),
        );
    }
    let mut fx = f(x);
    let mut ev = 1;
    let mut best = (x, fx);
    let mut history = [f64::NAN; 3];
    for it in 1..=p.max_iterations {
        if !fx.is_finite() {
            return scalar_result(
                x,
                fx,
                br,
                evidence(
                    RootTermination::NonFiniteEvaluation,
                    ev,
                    it as u64 - 1,
                    it - 1,
                    best.1.abs(),
                    0.0,
                    id,
                ),
            );
        }
        if fx.abs() <= p.residual_tolerance {
            return scalar_result(
                x,
                fx,
                br,
                evidence(
                    RootTermination::ResidualConverged,
                    ev,
                    it as u64 - 1,
                    it - 1,
                    fx.abs(),
                    0.0,
                    id,
                ),
            );
        }
        let dx = d(x);
        let extra = if matches!(source, DerivativeSource::FiniteDifference(_)) {
            2
        } else {
            0
        };
        ev += extra;
        if !dx.is_finite() {
            return scalar_result(
                x,
                fx,
                br,
                evidence(
                    RootTermination::NonFiniteEvaluation,
                    ev,
                    it as u64,
                    it,
                    best.1.abs(),
                    0.0,
                    id,
                ),
            );
        }
        if dx.abs() <= f64::EPSILON.sqrt() * fx.abs().max(1.0) {
            return scalar_result(
                x,
                fx,
                br,
                evidence(
                    RootTermination::FlatDerivative,
                    ev,
                    it as u64,
                    it,
                    best.1.abs(),
                    0.0,
                    id,
                ),
            );
        }
        let raw = x - fx / dx;
        let next = match br {
            Some(b) if raw <= b.lower || raw >= b.upper => b.lower + (b.upper - b.lower) / 2.0,
            _ => raw,
        };
        let step = (next - x).abs();
        if history
            .iter()
            .any(|v| (next - *v).abs() <= p.step_tolerance)
        {
            return scalar_result(
                x,
                fx,
                br,
                evidence(
                    RootTermination::Cycling,
                    ev,
                    it as u64,
                    it,
                    best.1.abs(),
                    step,
                    id,
                ),
            );
        }
        history.rotate_right(1);
        history[0] = x;
        let nf = f(next);
        ev += 1;
        if ev > p.max_evaluations {
            return scalar_result(
                x,
                fx,
                br,
                evidence(
                    RootTermination::WorkLimit,
                    p.max_evaluations,
                    it as u64,
                    it - 1,
                    best.1.abs(),
                    step,
                    id,
                ),
            );
        }
        if nf.abs() < best.1.abs() {
            best = (next, nf);
        }
        if let Some(b) = br {
            br = if b.f_lower.is_sign_positive() != nf.is_sign_positive() {
                RootBracket::new(b.lower, b.f_lower, next, nf).ok()
            } else {
                RootBracket::new(next, nf, b.upper, b.f_upper).ok()
            };
        }
        if step <= p.step_tolerance && nf.abs() > p.residual_tolerance {
            return scalar_result(
                next,
                nf,
                br,
                evidence(
                    RootTermination::Stagnation,
                    ev,
                    it as u64,
                    it,
                    best.1.abs(),
                    step,
                    id,
                ),
            );
        }
        x = next;
        fx = nf;
    }
    scalar_result(
        best.0,
        best.1,
        br,
        evidence(
            RootTermination::IterationLimit,
            ev,
            p.max_iterations as u64,
            p.max_iterations,
            best.1.abs(),
            0.0,
            id,
        ),
    )
}

/// Bounded secant iteration without a derivative claim.
pub fn secant<F: FnMut(f64) -> f64>(
    mut f: F,
    mut x0: f64,
    mut x1: f64,
    p: ScalarPlan,
    id: ExecutionIdentity,
) -> ScalarRoot {
    let (mut f0, mut f1) = (f(x0), f(x1));
    let mut ev = 2;
    for it in 1..=p.max_iterations {
        if !f0.is_finite() || !f1.is_finite() {
            return scalar_result(
                x1,
                f1,
                None,
                evidence(
                    RootTermination::NonFiniteEvaluation,
                    ev,
                    0,
                    it - 1,
                    f1.abs(),
                    (x1 - x0).abs(),
                    id,
                ),
            );
        }
        if f1.abs() <= p.residual_tolerance {
            return scalar_result(
                x1,
                f1,
                None,
                evidence(
                    RootTermination::ResidualConverged,
                    ev,
                    0,
                    it - 1,
                    f1.abs(),
                    (x1 - x0).abs(),
                    id,
                ),
            );
        }
        let den = f1 - f0;
        if den.abs() <= f64::EPSILON.sqrt() * f1.abs().max(1.0) {
            return scalar_result(
                x1,
                f1,
                None,
                evidence(
                    RootTermination::FlatDerivative,
                    ev,
                    0,
                    it,
                    f1.abs(),
                    0.0,
                    id,
                ),
            );
        }
        let x2 = x1 - f1 * (x1 - x0) / den;
        let step = (x2 - x1).abs();
        let f2 = f(x2);
        ev += 1;
        if ev > p.max_evaluations {
            return scalar_result(
                x1,
                f1,
                None,
                evidence(
                    RootTermination::WorkLimit,
                    p.max_evaluations,
                    0,
                    it - 1,
                    f1.abs(),
                    step,
                    id,
                ),
            );
        }
        if step <= p.step_tolerance && f2.abs() > p.residual_tolerance {
            return scalar_result(
                x2,
                f2,
                None,
                evidence(RootTermination::Stagnation, ev, 0, it, f2.abs(), step, id),
            );
        }
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f2;
    }
    scalar_result(
        x1,
        f1,
        None,
        evidence(
            RootTermination::IterationLimit,
            ev,
            0,
            p.max_iterations,
            f1.abs(),
            (x1 - x0).abs(),
            id,
        ),
    )
}

/// Source of a vector Jacobian.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JacobianSource {
    /// Caller analytic matrix.
    Analytic,
    /// Matrix supplied by an AD adapter.
    Automatic,
    /// Forward finite differences with recorded policy.
    FiniteDifference(FiniteDifferencePlan),
}
/// Vector Newton/Broyden policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorPlan {
    /// Residual two-norm tolerance.
    pub residual_tolerance: f64,
    /// Step two-norm tolerance.
    pub step_tolerance: f64,
    /// Iteration ceiling.
    pub max_iterations: usize,
    /// Residual evaluation ceiling.
    pub max_evaluations: u64,
    /// Relative SVD rank cutoff.
    pub rank_cutoff: f64,
    /// Minimum accepted line-search multiplier.
    pub minimum_damping: f64,
    /// Iterations between mandatory Broyden resets.
    pub broyden_reset_interval: usize,
}
impl Default for VectorPlan {
    fn default() -> Self {
        Self {
            residual_tolerance: 1e-10,
            step_tolerance: 1e-12,
            max_iterations: 64,
            max_evaluations: 1024,
            rank_cutoff: 1e-12,
            minimum_damping: 2f64.powi(-20),
            broyden_reset_interval: 8,
        }
    }
}
/// Vector result with residual and rank evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorRoot {
    /// Final point.
    pub value: Vec<f64>,
    /// Final residual.
    pub residual: Vec<f64>,
    /// SVD numerical ranks, one per Newton step.
    pub ranks: Vec<usize>,
    /// Accepted damping multipliers.
    pub damping: Vec<f64>,
    /// Iterations at which Broyden rebuilt the Jacobian.
    pub jacobian_resets: Vec<usize>,
    /// Derivative provenance.
    pub jacobian_source: JacobianSource,
    /// Common evidence.
    pub evidence: RootEvidence,
}
fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn finite_jacobian<F: FnMut(&[f64]) -> Vec<f64>>(
    f: &mut F,
    x: &[f64],
    fx: &[f64],
    fd: FiniteDifferencePlan,
    ev: &mut u64,
) -> Vec<f64> {
    let n = x.len();
    let mut j = vec![0.0; n * n];
    for c in 0..n {
        let mut y = x.to_vec();
        let h = fd.step(x[c]);
        y[c] += h;
        let fy = f(&y);
        *ev += 1;
        for r in 0..n {
            j[r * n + c] = (fy[r] - fx[r]) / h;
        }
    }
    j
}
fn vector_solve(
    j: &[f64],
    r: &[f64],
    n: usize,
    cut: f64,
) -> Result<(Vec<f64>, usize), RootTermination> {
    let s = svd_f64(
        j,
        n,
        n,
        SvdPlan {
            vectors: VectorForm::Full,
            ..SvdPlan::default()
        },
    )
    .map_err(|_| RootTermination::RankLoss)?;
    let c = SingularCutoff::new(cut).map_err(|_| RootTermination::InvalidPlan)?;
    let rank = numerical_rank(&s, c).map_err(|_| RootTermination::RankLoss)?;
    if rank < n {
        return Err(RootTermination::RankLoss);
    }
    let rhs = r.iter().map(|v| -v).collect::<Vec<_>>();
    let step = least_squares(&s, &rhs, c).map_err(|_| RootTermination::RankLoss)?;
    Ok((step, rank))
}

/// Damped vector Newton with caller/AD Jacobian callback.
pub fn damped_newton<F, J>(
    mut f: F,
    mut jacobian: J,
    source: JacobianSource,
    initial: Vec<f64>,
    p: VectorPlan,
    id: ExecutionIdentity,
) -> VectorRoot
where
    F: FnMut(&[f64]) -> Vec<f64>,
    J: FnMut(&[f64]) -> Vec<f64>,
{
    vector_newton_impl(&mut f, &mut jacobian, source, initial, p, id)
}
/// Damped vector Newton with an explicit finite-difference Jacobian plan.
pub fn finite_difference_vector_newton<F: FnMut(&[f64]) -> Vec<f64>>(
    mut f: F,
    initial: Vec<f64>,
    fd: FiniteDifferencePlan,
    p: VectorPlan,
    id: ExecutionIdentity,
) -> VectorRoot {
    let mut dummy = |_: &[f64]| Vec::new();
    vector_newton_impl(
        &mut f,
        &mut dummy,
        JacobianSource::FiniteDifference(fd),
        initial,
        p,
        id,
    )
}
fn vector_newton_impl<F, J>(
    f: &mut F,
    jacobian: &mut J,
    source: JacobianSource,
    mut x: Vec<f64>,
    p: VectorPlan,
    id: ExecutionIdentity,
) -> VectorRoot
where
    F: FnMut(&[f64]) -> Vec<f64>,
    J: FnMut(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let mut r = f(&x);
    let mut ev = 1;
    let mut ranks = Vec::new();
    let mut damping = Vec::new();
    for it in 1..=p.max_iterations {
        let rn = norm(&r);
        if !rn.is_finite() || r.len() != n {
            return VectorRoot {
                value: x,
                residual: r,
                ranks,
                damping,
                jacobian_resets: vec![],
                jacobian_source: source,
                evidence: evidence(
                    RootTermination::NonFiniteEvaluation,
                    ev,
                    it as u64 - 1,
                    it - 1,
                    rn,
                    0.0,
                    id,
                ),
            };
        }
        if rn <= p.residual_tolerance {
            return VectorRoot {
                value: x,
                residual: r,
                ranks,
                damping,
                jacobian_resets: vec![],
                jacobian_source: source,
                evidence: evidence(
                    RootTermination::ResidualConverged,
                    ev,
                    it as u64 - 1,
                    it - 1,
                    rn,
                    0.0,
                    id,
                ),
            };
        }
        let j = match source {
            JacobianSource::FiniteDifference(fd) => finite_jacobian(f, &x, &r, fd, &mut ev),
            _ => jacobian(&x),
        };
        let (step, rank) = match vector_solve(&j, &r, n, p.rank_cutoff) {
            Ok(v) => v,
            Err(t) => {
                return VectorRoot {
                    value: x,
                    residual: r,
                    ranks,
                    damping,
                    jacobian_resets: vec![],
                    jacobian_source: source,
                    evidence: evidence(t, ev, it as u64, it - 1, rn, 0.0, id),
                };
            }
        };
        ranks.push(rank);
        let sn = norm(&step);
        let mut alpha = 1.0;
        let (mut candidate, mut cr): (Vec<f64>, Vec<f64>);
        loop {
            candidate = x.iter().zip(&step).map(|(a, s)| a + alpha * s).collect();
            cr = f(&candidate);
            ev += 1;
            if norm(&cr) < rn || alpha <= p.minimum_damping {
                break;
            }
            alpha *= 0.5;
        }
        damping.push(alpha);
        if ev > p.max_evaluations {
            return VectorRoot {
                value: x,
                residual: r,
                ranks,
                damping,
                jacobian_resets: vec![],
                jacobian_source: source,
                evidence: evidence(
                    RootTermination::WorkLimit,
                    p.max_evaluations,
                    it as u64,
                    it - 1,
                    rn,
                    sn * alpha,
                    id,
                ),
            };
        }
        if sn * alpha <= p.step_tolerance && norm(&cr) > p.residual_tolerance {
            return VectorRoot {
                value: candidate,
                residual: cr,
                ranks,
                damping,
                jacobian_resets: vec![],
                jacobian_source: source,
                evidence: evidence(
                    RootTermination::Stagnation,
                    ev,
                    it as u64,
                    it,
                    rn,
                    sn * alpha,
                    id,
                ),
            };
        }
        x = candidate;
        r = cr;
    }
    let rn = norm(&r);
    VectorRoot {
        value: x,
        residual: r,
        ranks,
        damping,
        jacobian_resets: vec![],
        jacobian_source: source,
        evidence: evidence(
            RootTermination::IterationLimit,
            ev,
            p.max_iterations as u64,
            p.max_iterations,
            rn,
            0.0,
            id,
        ),
    }
}

/// Bounded Broyden iteration with visible full-Jacobian reset points.
pub fn broyden<F, J>(
    mut f: F,
    mut jacobian: J,
    source: JacobianSource,
    mut x: Vec<f64>,
    p: VectorPlan,
    id: ExecutionIdentity,
) -> VectorRoot
where
    F: FnMut(&[f64]) -> Vec<f64>,
    J: FnMut(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let mut r = f(&x);
    let mut ev = 1;
    let mut j = match source {
        JacobianSource::FiniteDifference(fd) => finite_jacobian(&mut f, &x, &r, fd, &mut ev),
        _ => jacobian(&x),
    };
    let mut ranks = Vec::new();
    let mut damping = Vec::new();
    let mut resets = vec![0];
    for it in 1..=p.max_iterations {
        let rn = norm(&r);
        if rn <= p.residual_tolerance {
            let derivative_evaluations = resets.len() as u64;
            return VectorRoot {
                value: x,
                residual: r,
                ranks,
                damping,
                jacobian_resets: resets,
                jacobian_source: source,
                evidence: evidence(
                    RootTermination::ResidualConverged,
                    ev,
                    derivative_evaluations,
                    it - 1,
                    rn,
                    0.0,
                    id,
                ),
            };
        }
        if it > 1 && p.broyden_reset_interval > 0 && (it - 1) % p.broyden_reset_interval == 0 {
            j = match source {
                JacobianSource::FiniteDifference(fd) => {
                    finite_jacobian(&mut f, &x, &r, fd, &mut ev)
                }
                _ => jacobian(&x),
            };
            resets.push(it - 1);
        }
        let (step, rank) = match vector_solve(&j, &r, n, p.rank_cutoff) {
            Ok(v) => v,
            Err(t) => {
                let derivative_evaluations = resets.len() as u64;
                return VectorRoot {
                    value: x,
                    residual: r,
                    ranks,
                    damping,
                    jacobian_resets: resets,
                    jacobian_source: source,
                    evidence: evidence(t, ev, derivative_evaluations, it - 1, rn, 0.0, id),
                };
            }
        };
        ranks.push(rank);
        let nx = x.iter().zip(&step).map(|(a, s)| a + s).collect::<Vec<_>>();
        let nr = f(&nx);
        ev += 1;
        let y = nr.iter().zip(&r).map(|(a, b)| a - b).collect::<Vec<_>>();
        let ss = step.iter().map(|v| v * v).sum::<f64>();
        if ss > 0.0 {
            for row in 0..n {
                let js = (0..n).map(|c| j[row * n + c] * step[c]).sum::<f64>();
                for col in 0..n {
                    j[row * n + col] += (y[row] - js) * step[col] / ss;
                }
            }
        }
        damping.push(1.0);
        x = nx;
        r = nr;
        if ev >= p.max_evaluations {
            let derivative_evaluations = resets.len() as u64;
            let residual_norm = norm(&r);
            return VectorRoot {
                value: x,
                residual: r,
                ranks,
                damping,
                jacobian_resets: resets,
                jacobian_source: source,
                evidence: evidence(
                    RootTermination::WorkLimit,
                    ev,
                    derivative_evaluations,
                    it,
                    residual_norm,
                    norm(&step),
                    id,
                ),
            };
        }
    }
    let rn = norm(&r);
    VectorRoot {
        value: x,
        residual: r,
        ranks,
        damping,
        jacobian_resets: resets,
        jacobian_source: source,
        evidence: evidence(
            RootTermination::IterationLimit,
            ev,
            0,
            p.max_iterations,
            rn,
            0.0,
            id,
        ),
    }
}

/// Cookbook recipes embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
/// Loadable root-finding schema surface.
#[derive(Default)]
pub struct RootLib;
impl RootLib {
    /// Constructs the stateless library.
    pub fn new() -> Self {
        Self
    }
}
impl Lib for RootLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "root"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: root_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(root_schema_symbol(),cx.factory().string("bisection brent-dekker safeguarded-newton secant vector-newton broyden bracket estimate residual rank work execution".to_owned())?)
    }
}
/// Runtime inspection symbol.
pub fn root_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/root", "schema")
}

impl fmt::Display for RootTermination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for RootTermination {}

#[cfg(test)]
mod tests;
