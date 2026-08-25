//! Radau IIA solver and its evidence-bearing problem contracts.

use std::{error::Error, fmt, sync::Arc};

use sim_kernel::{
    AbiVersion, Dependency, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult,
    Symbol, Version,
};
use sim_lib_numbers_codec::numeric_plugin_descriptor_symbol;
use sim_lib_numbers_tensor_linalg::{DenseSolveOptions, solve_dense_f64};

const SQRT6: f64 = 2.449_489_742_783_178;
const C: [f64; 3] = [(4.0 - SQRT6) / 10.0, (4.0 + SQRT6) / 10.0, 1.0];
const A: [[f64; 3]; 3] = [
    [
        (88.0 - 7.0 * SQRT6) / 360.0,
        (296.0 - 169.0 * SQRT6) / 1800.0,
        (-2.0 + 3.0 * SQRT6) / 225.0,
    ],
    [
        (296.0 + 169.0 * SQRT6) / 1800.0,
        (88.0 + 7.0 * SQRT6) / 360.0,
        (-2.0 - 3.0 * SQRT6) / 225.0,
    ],
    [(16.0 - SQRT6) / 36.0, (16.0 + SQRT6) / 36.0, 1.0 / 9.0],
];

type StageAttempt = ([Vec<f64>; 3], Vec<f64>, usize);

/// Vector function used by an ODE or mass-matrix problem.
pub type VectorField = Arc<dyn Fn(f64, &[f64], &mut [f64]) -> Result<(), RadauError> + Send + Sync>;
/// Dense matrix producer in row-major order.
pub type MatrixFunction =
    Arc<dyn Fn(f64, &[f64], &mut [f64]) -> Result<(), RadauError> + Send + Sync>;
/// Residual function `F(t,y,ydot)=0`.
pub type ResidualFunction =
    Arc<dyn Fn(f64, &[f64], &[f64], &mut [f64]) -> Result<(), RadauError> + Send + Sync>;
/// Residual stage Jacobian producer `dF/dy + alpha*dF/dydot`.
pub type ResidualJacobian =
    Arc<dyn Fn(f64, &[f64], &[f64], f64, &mut [f64]) -> Result<(), RadauError> + Send + Sync>;
/// Scalar event function.
pub type EventFunction = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// Provenance of a stage Jacobian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JacobianSource {
    /// Caller-provided exact derivative.
    Analytic,
    /// Derivative produced by automatic differentiation.
    AutomaticDifferentiation,
    /// Explicitly admitted finite differences.
    FiniteDifference,
}

/// Caller-selected Jacobian strategy.
#[derive(Clone)]
pub enum JacobianStrategy {
    /// Exact matrix producer.
    Analytic(MatrixFunction),
    /// AD-derived matrix producer.
    AutomaticDifferentiation(MatrixFunction),
    /// Finite differences, admitted explicitly with a positive relative step.
    FiniteDifference {
        /// Relative perturbation.
        relative_step: f64,
    },
}

/// Exact mathematical problem admitted by the backend.
#[derive(Clone)]
pub enum ImplicitProblem {
    /// Ordinary differential equation `y'=f(t,y)`.
    Ode {
        /// Right-hand side.
        rhs: VectorField,
        /// Jacobian policy.
        jacobian: JacobianStrategy,
    },
    /// Mass-matrix equation `M(t,y)y'=f(t,y)`; singular M denotes a DAE.
    MassMatrix {
        /// Right-hand side.
        rhs: VectorField,
        /// Mass matrix.
        mass: MatrixFunction,
        /// Right-hand-side Jacobian policy.
        jacobian: JacobianStrategy,
        /// Differential-variable mask; false entries are algebraic.
        differential: Vec<bool>,
    },
    /// Declared index-1 residual equation `F(t,y,y')=0`.
    Residual {
        /// Residual callable.
        residual: ResidualFunction,
        /// Stage Jacobian, required for an honest residual DAE.
        jacobian: ResidualJacobian,
        /// Differential-variable mask; must contain both kinds.
        differential: Vec<bool>,
    },
}

/// Direction admitted for an event crossing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventDirection {
    /// Negative to positive.
    Rising,
    /// Positive to negative.
    Falling,
    /// Either direction.
    Either,
}

/// Event request located against the collocation polynomial.
#[derive(Clone)]
pub struct EventSpec {
    /// Function whose zero is located.
    pub function: EventFunction,
    /// Crossing direction.
    pub direction: EventDirection,
    /// Stop at the first such event.
    pub terminal: bool,
}

/// Bounded Newton and adaptive-step policy.
#[derive(Clone, Debug)]
pub struct RadauPlan {
    /// Relative error tolerance.
    pub relative_tolerance: f64,
    /// Absolute error tolerance.
    pub absolute_tolerance: f64,
    /// Initial step magnitude.
    pub initial_step: f64,
    /// Maximum step magnitude.
    pub max_step: f64,
    /// Maximum accepted plus rejected steps.
    pub max_steps: usize,
    /// Maximum simplified-Newton iterations per attempt.
    pub max_newton_iterations: usize,
    /// Scaled Newton convergence tolerance.
    pub newton_tolerance: f64,
    /// Recompute after this many accepted factor reuses.
    pub max_factor_reuse: usize,
}
impl Default for RadauPlan {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-7,
            absolute_tolerance: 1e-9,
            initial_step: 1e-3,
            max_step: f64::INFINITY,
            max_steps: 100_000,
            max_newton_iterations: 10,
            newton_tolerance: 1e-9,
            max_factor_reuse: 4,
        }
    }
}

/// Why a Jacobian was recomputed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecomputeReason {
    /// First attempted step.
    Initial,
    /// Step size changed materially.
    StepChanged,
    /// Reuse budget was consumed.
    ReuseLimit,
    /// Newton failed with the reused matrix.
    NewtonFailure,
}

/// One failed Newton attempt.
#[derive(Clone, Debug)]
pub struct FailedNewtonAttempt {
    /// Attempt start time.
    pub time: f64,
    /// Attempted step.
    pub step: f64,
    /// Completed iterations.
    pub iterations: usize,
    /// Last scaled correction.
    pub correction: f64,
}

/// Reviewable solver evidence.
#[derive(Clone, Debug)]
pub struct RadauEvidence {
    /// Accepted steps.
    pub accepted_steps: usize,
    /// Rejected attempts.
    pub rejected_steps: usize,
    /// Residual/right-hand-side evaluations.
    pub function_evaluations: usize,
    /// Jacobian evaluations.
    pub jacobian_evaluations: usize,
    /// Successful factor reuses.
    pub factor_reuses: usize,
    /// Jacobian provenance.
    pub jacobian_source: JacobianSource,
    /// Reasons for recomputation.
    pub recomputations: Vec<RecomputeReason>,
    /// Minimum observed numerical rank.
    pub minimum_rank: usize,
    /// Worst reciprocal pivot conditioning indicator.
    pub minimum_reciprocal_condition: f64,
    /// Failed Newton attempts.
    pub failed_newton_attempts: Vec<FailedNewtonAttempt>,
    /// Located event brackets.
    pub event_brackets: Vec<(usize, f64, f64)>,
}

/// Fifth-degree collocation segment; it exists only for a fully converged stage solve.
#[derive(Clone, Debug)]
pub struct CollocationSegment {
    /// Start time.
    pub start: f64,
    /// End time.
    pub end: f64,
    /// Start state.
    pub y0: Vec<f64>,
    /// Converged stage derivatives.
    pub stages: [Vec<f64>; 3],
}
impl CollocationSegment {
    /// Evaluates the collocation polynomial at a time inside the segment.
    pub fn evaluate(&self, time: f64) -> Vec<f64> {
        let theta = ((time - self.start) / (self.end - self.start)).clamp(0.0, 1.0);
        let mut out = self.y0.clone();
        for j in 0..3 {
            let w = integrated_lagrange(j, theta) * (self.end - self.start);
            for (v, k) in out.iter_mut().zip(&self.stages[j]) {
                *v += w * k;
            }
        }
        out
    }
}

/// Located event.
#[derive(Clone, Debug)]
pub struct LocatedEvent {
    /// Event index.
    pub event: usize,
    /// Located time.
    pub time: f64,
    /// Interpolated state.
    pub state: Vec<f64>,
    /// Terminal policy.
    pub terminal: bool,
}

/// Successful integration result.
#[derive(Clone, Debug)]
pub struct RadauSolution {
    /// Final time.
    pub time: f64,
    /// Final state.
    pub state: Vec<f64>,
    /// Dense collocation segments.
    pub dense: Vec<CollocationSegment>,
    /// Events.
    pub events: Vec<LocatedEvent>,
    /// Solver evidence.
    pub evidence: RadauEvidence,
}

/// Explicit refusal from admission, bounded Newton, or dense solve.
#[derive(Clone, Debug, PartialEq)]
pub enum RadauError {
    /// Invalid problem or plan.
    Invalid(String),
    /// Singular or rank-deficient stage Jacobian.
    SingularJacobian {
        /// Observed rank.
        rank: usize,
        /// Required rank.
        dimension: usize,
    },
    /// Newton exhausted its bound.
    NewtonDidNotConverge,
    /// Step bound exhausted.
    StepLimit,
    /// User callback failed.
    Callback(String),
}
impl fmt::Display for RadauError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for RadauError {}

/// Integrates a stiff ODE or declared index-1 DAE with three-stage Radau IIA.
pub fn solve_radau(
    problem: &ImplicitProblem,
    span: (f64, f64),
    initial: &[f64],
    plan: &RadauPlan,
    events: &[EventSpec],
) -> Result<RadauSolution, RadauError> {
    validate(problem, span, initial, plan)?;
    let n = initial.len();
    let direction = (span.1 - span.0).signum();
    let mut t = span.0;
    let mut y = initial.to_vec();
    let mut h = direction
        * plan
            .initial_step
            .min(plan.max_step)
            .min((span.1 - span.0).abs());
    let source = jacobian_source(problem);
    let mut ev = RadauEvidence {
        accepted_steps: 0,
        rejected_steps: 0,
        function_evaluations: 0,
        jacobian_evaluations: 0,
        factor_reuses: 0,
        jacobian_source: source,
        recomputations: vec![],
        minimum_rank: usize::MAX,
        minimum_reciprocal_condition: 1.0,
        failed_newton_attempts: vec![],
        event_brackets: vec![],
    };
    let mut dense = vec![];
    let mut located = vec![];
    let mut cached: Option<(f64, Vec<f64>, usize)> = None;
    while direction * (span.1 - t) > 0.0 {
        if ev.accepted_steps + ev.rejected_steps >= plan.max_steps {
            return Err(RadauError::StepLimit);
        }
        if direction * (t + h - span.1) > 0.0 {
            h = span.1 - t;
        }
        let attempt = attempt_step(problem, t, &y, h, plan, &mut ev, &mut cached);
        let (stages, y_full, newton_iterations) = match attempt {
            Ok(v) => v,
            Err(RadauError::NewtonDidNotConverge) => {
                ev.rejected_steps += 1;
                h *= 0.5;
                cached = None;
                continue;
            }
            Err(e) => return Err(e),
        };
        let (half_stages, y_half, _) =
            attempt_step(problem, t, &y, h * 0.5, plan, &mut ev, &mut cached)?;
        let (_, y_two, _) = attempt_step(
            problem,
            t + h * 0.5,
            &y_half,
            h * 0.5,
            plan,
            &mut ev,
            &mut cached,
        )?;
        let error = scaled_error(&y_full, &y_two, &y, plan);
        if error > 1.0 {
            ev.rejected_steps += 1;
            h *= (0.9 * error.powf(-0.2)).clamp(0.2, 0.8);
            cached = None;
            continue;
        }
        let segment = CollocationSegment {
            start: t,
            end: t + h,
            y0: y.clone(),
            stages,
        };
        locate_events(events, &segment, &mut located, &mut ev);
        y = y_two;
        t += h;
        dense.push(segment);
        ev.accepted_steps += 1;
        if located
            .last()
            .is_some_and(|e| e.terminal && (e.time - t).abs() <= h.abs() + f64::EPSILON)
        {
            let e = located.last().expect("present");
            t = e.time;
            y = e.state.clone();
            break;
        }
        let factor = (if error == 0.0 {
            5.0
        } else {
            0.9 * error.powf(-0.2)
        })
        .clamp(0.2, 5.0);
        h = direction
            * (h.abs() * factor)
                .min(plan.max_step)
                .min((span.1 - t).abs());
        let _ = half_stages;
        let _ = newton_iterations;
    }
    if ev.minimum_rank == usize::MAX {
        ev.minimum_rank = n;
    }
    Ok(RadauSolution {
        time: t,
        state: y,
        dense,
        events: located,
        evidence: ev,
    })
}

fn attempt_step(
    problem: &ImplicitProblem,
    t: f64,
    y: &[f64],
    h: f64,
    plan: &RadauPlan,
    ev: &mut RadauEvidence,
    cached: &mut Option<(f64, Vec<f64>, usize)>,
) -> Result<StageAttempt, RadauError> {
    let n = y.len();
    let mut k = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    initial_derivative(problem, t, y, &mut k[0], ev)?;
    k[1] = k[0].clone();
    k[2] = k[0].clone();
    let reuse = cached.as_ref().is_some_and(|(old, _, uses)| {
        ((h / old) - 1.0).abs() < 0.2 && *uses < plan.max_factor_reuse
    });
    let mut matrix = if reuse {
        ev.factor_reuses += 1;
        cached.as_ref().unwrap().1.clone()
    } else {
        ev.recomputations.push(if cached.is_none() {
            RecomputeReason::Initial
        } else {
            RecomputeReason::StepChanged
        });
        build_block_jacobian(problem, t, y, h, &k, ev)?
    };
    for iteration in 0..plan.max_newton_iterations {
        let residual = stage_residual(problem, t, y, h, &k, ev)?;
        let norm = residual.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if norm <= plan.newton_tolerance {
            let mut end = y.to_vec();
            for i in 0..n {
                for j in 0..3 {
                    end[i] += h * A[2][j] * k[j][i];
                }
            }
            *cached = Some((h, matrix, cached.as_ref().map_or(1, |x| x.2 + 1)));
            return Ok((k, end, iteration));
        }
        let rhs = residual.iter().map(|v| -v).collect::<Vec<_>>();
        let opts = DenseSolveOptions {
            singularity_threshold: 1e-13,
        };
        let solved =
            solve_dense_f64(&matrix, &rhs, opts).map_err(|_| RadauError::SingularJacobian {
                rank: estimate_rank(&matrix, 3 * n),
                dimension: 3 * n,
            })?;
        ev.minimum_rank = ev.minimum_rank.min(solved.report.dimension);
        ev.minimum_reciprocal_condition = ev
            .minimum_reciprocal_condition
            .min(solved.report.reciprocal_pivot_condition);
        let correction = solved.values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (j, stage) in k.iter_mut().enumerate() {
            for (i, value) in stage.iter_mut().enumerate() {
                *value += solved.values[j * n + i];
            }
        }
        if !correction.is_finite() {
            break;
        }
        if iteration + 1 == plan.max_newton_iterations {
            ev.failed_newton_attempts.push(FailedNewtonAttempt {
                time: t,
                step: h,
                iterations: iteration + 1,
                correction,
            });
        }
        if iteration == 1 && correction > 1e6 {
            matrix = build_block_jacobian(problem, t, y, h, &k, ev)?;
            ev.recomputations.push(RecomputeReason::NewtonFailure);
        }
    }
    Err(RadauError::NewtonDidNotConverge)
}

fn build_block_jacobian(
    problem: &ImplicitProblem,
    t: f64,
    y: &[f64],
    h: f64,
    k: &[Vec<f64>; 3],
    ev: &mut RadauEvidence,
) -> Result<Vec<f64>, RadauError> {
    let n = y.len();
    let mut block = vec![0.0; 9 * n * n];
    match problem {
        ImplicitProblem::Ode { rhs, jacobian }
        | ImplicitProblem::MassMatrix { rhs, jacobian, .. } => {
            let mut j = vec![0.0; n * n];
            eval_jacobian(jacobian, rhs, t, y, &mut j, ev)?;
            let mut mass = vec![0.0; n * n];
            if let ImplicitProblem::MassMatrix { mass: m, .. } = problem {
                m(t, y, &mut mass)?;
            } else {
                for i in 0..n {
                    mass[i * n + i] = 1.0;
                }
            }
            for s in 0..3 {
                for q in 0..3 {
                    for r in 0..n {
                        for c in 0..n {
                            block[(s * n + r) * (3 * n) + q * n + c] =
                                (if s == q { mass[r * n + c] } else { 0.0 })
                                    - h * A[s][q] * j[r * n + c];
                        }
                    }
                }
            }
        }
        ImplicitProblem::Residual { jacobian, .. } => {
            for s in 0..3 {
                let ys = stage_state(y, h, k, s);
                let mut dy = vec![0.0; n * n];
                let mut dy_plus_dydot = vec![0.0; n * n];
                jacobian(t + C[s] * h, &ys, &k[s], 0.0, &mut dy)?;
                jacobian(t + C[s] * h, &ys, &k[s], 1.0, &mut dy_plus_dydot)?;
                ev.jacobian_evaluations += 2;
                for q in 0..3 {
                    for r in 0..n {
                        for c in 0..n {
                            let dydot = dy_plus_dydot[r * n + c] - dy[r * n + c];
                            block[(s * n + r) * (3 * n) + q * n + c] =
                                h * A[s][q] * dy[r * n + c] + if s == q { dydot } else { 0.0 };
                        }
                    }
                }
            }
        }
    }
    ev.jacobian_evaluations += 1;
    Ok(block)
}
fn stage_residual(
    problem: &ImplicitProblem,
    t: f64,
    y: &[f64],
    h: f64,
    k: &[Vec<f64>; 3],
    ev: &mut RadauEvidence,
) -> Result<Vec<f64>, RadauError> {
    let n = y.len();
    let mut out = vec![0.0; 3 * n];
    for s in 0..3 {
        let ys = stage_state(y, h, k, s);
        match problem {
            ImplicitProblem::Ode { rhs, .. } => {
                let mut f = vec![0.0; n];
                rhs(t + C[s] * h, &ys, &mut f)?;
                for i in 0..n {
                    out[s * n + i] = k[s][i] - f[i];
                }
            }
            ImplicitProblem::MassMatrix { rhs, mass, .. } => {
                let mut f = vec![0.0; n];
                let mut m = vec![0.0; n * n];
                rhs(t + C[s] * h, &ys, &mut f)?;
                mass(t + C[s] * h, &ys, &mut m)?;
                for i in 0..n {
                    out[s * n + i] = (0..n).map(|j| m[i * n + j] * k[s][j]).sum::<f64>() - f[i];
                }
            }
            ImplicitProblem::Residual { residual, .. } => {
                residual(t + C[s] * h, &ys, &k[s], &mut out[s * n..(s + 1) * n])?
            }
        }
        ev.function_evaluations += 1;
    }
    Ok(out)
}
fn stage_state(y: &[f64], h: f64, k: &[Vec<f64>; 3], s: usize) -> Vec<f64> {
    (0..y.len())
        .map(|i| y[i] + h * (0..3).map(|j| A[s][j] * k[j][i]).sum::<f64>())
        .collect()
}
fn initial_derivative(
    problem: &ImplicitProblem,
    t: f64,
    y: &[f64],
    out: &mut [f64],
    ev: &mut RadauEvidence,
) -> Result<(), RadauError> {
    match problem {
        ImplicitProblem::Ode { rhs, .. } => rhs(t, y, out)?,
        ImplicitProblem::MassMatrix { rhs, mass, .. } => {
            let n = y.len();
            let mut b = vec![0.0; n];
            let mut m = vec![0.0; n * n];
            rhs(t, y, &mut b)?;
            mass(t, y, &mut m)?;
            let opts = DenseSolveOptions {
                singularity_threshold: 1e-13,
            };
            match solve_dense_f64(&m, &b, opts) {
                Ok(s) => out.copy_from_slice(&s.values),
                Err(_) => out.fill(0.0),
            }
        }
        ImplicitProblem::Residual { .. } => out.fill(0.0),
    }
    ev.function_evaluations += 1;
    Ok(())
}
fn eval_jacobian(
    strategy: &JacobianStrategy,
    rhs: &VectorField,
    t: f64,
    y: &[f64],
    out: &mut [f64],
    ev: &mut RadauEvidence,
) -> Result<(), RadauError> {
    match strategy {
        JacobianStrategy::Analytic(f) | JacobianStrategy::AutomaticDifferentiation(f) => {
            f(t, y, out)?
        }
        JacobianStrategy::FiniteDifference { relative_step } => {
            let n = y.len();
            let mut base = vec![0.0; n];
            rhs(t, y, &mut base)?;
            for c in 0..n {
                let d = relative_step * y[c].abs().max(1.0);
                let mut yp = y.to_vec();
                yp[c] += d;
                let mut fp = vec![0.0; n];
                rhs(t, &yp, &mut fp)?;
                for r in 0..n {
                    out[r * n + c] = (fp[r] - base[r]) / d;
                }
                ev.function_evaluations += 1;
            }
        }
    }
    ev.jacobian_evaluations += 1;
    Ok(())
}
fn jacobian_source(p: &ImplicitProblem) -> JacobianSource {
    match p {
        ImplicitProblem::Ode { jacobian, .. } | ImplicitProblem::MassMatrix { jacobian, .. } => {
            match jacobian {
                JacobianStrategy::Analytic(_) => JacobianSource::Analytic,
                JacobianStrategy::AutomaticDifferentiation(_) => {
                    JacobianSource::AutomaticDifferentiation
                }
                JacobianStrategy::FiniteDifference { .. } => JacobianSource::FiniteDifference,
            }
        }
        ImplicitProblem::Residual { .. } => JacobianSource::Analytic,
    }
}
fn scaled_error(a: &[f64], b: &[f64], y: &[f64], p: &RadauPlan) -> f64 {
    a.iter()
        .zip(b)
        .zip(y)
        .map(|((x, z), old)| {
            (x - z).abs() / (p.absolute_tolerance + p.relative_tolerance * x.abs().max(old.abs()))
        })
        .fold(0.0_f64, f64::max)
        / 31.0
}
fn integrated_lagrange(j: usize, x: f64) -> f64 {
    let mut poly = vec![1.0];
    let mut denom = 1.0;
    for (m, node) in C.iter().enumerate() {
        if m != j {
            let mut next = vec![0.0; poly.len() + 1];
            for (i, v) in poly.iter().enumerate() {
                next[i] -= v * node;
                next[i + 1] += v;
            }
            poly = next;
            denom *= C[j] - node;
        }
    }
    poly.iter()
        .enumerate()
        .map(|(p, v)| v * x.powi(p as i32 + 1) / (p as f64 + 1.0) / denom)
        .sum()
}
fn locate_events(
    specs: &[EventSpec],
    seg: &CollocationSegment,
    out: &mut Vec<LocatedEvent>,
    ev: &mut RadauEvidence,
) {
    for (i, s) in specs.iter().enumerate() {
        let mut lo = seg.start;
        let mut hi = seg.end;
        let mut flo = (s.function)(lo, &seg.y0);
        let fhi = (s.function)(hi, &seg.evaluate(hi));
        let crosses = match s.direction {
            EventDirection::Rising => flo <= 0.0 && fhi >= 0.0,
            EventDirection::Falling => flo >= 0.0 && fhi <= 0.0,
            EventDirection::Either => flo * fhi <= 0.0,
        };
        if !crosses {
            continue;
        }
        ev.event_brackets.push((i, lo, hi));
        for _ in 0..60 {
            let mid = (lo + hi) * 0.5;
            let fm = (s.function)(mid, &seg.evaluate(mid));
            if flo * fm <= 0.0 {
                hi = mid
            } else {
                lo = mid;
                flo = fm
            }
            if (hi - lo).abs() <= 1e-12 * (1.0 + mid.abs()) {
                break;
            }
        }
        let time = (lo + hi) * 0.5;
        out.push(LocatedEvent {
            event: i,
            time,
            state: seg.evaluate(time),
            terminal: s.terminal,
        });
    }
}
fn estimate_rank(a: &[f64], n: usize) -> usize {
    (0..n).filter(|i| a[i * n + i].abs() > 1e-13).count()
}
fn validate(
    p: &ImplicitProblem,
    span: (f64, f64),
    y: &[f64],
    plan: &RadauPlan,
) -> Result<(), RadauError> {
    if y.is_empty()
        || !span.0.is_finite()
        || !span.1.is_finite()
        || plan.initial_step <= 0.0
        || plan.max_step <= 0.0
        || plan.max_steps == 0
        || plan.max_newton_iterations == 0
        || plan.relative_tolerance <= 0.0
        || plan.absolute_tolerance <= 0.0
    {
        return Err(RadauError::Invalid("invalid Radau problem or plan".into()));
    }
    match p {
        ImplicitProblem::MassMatrix { differential, .. } if differential.len() != y.len() => Err(
            RadauError::Invalid("mass-matrix differential mask must match state".into()),
        ),
        ImplicitProblem::Residual { differential, .. }
            if differential.len() != y.len()
                || differential.iter().all(|x| *x)
                || differential.iter().all(|x| !*x) =>
        {
            Err(RadauError::Invalid(
                "residual DAE must declare matching differential and algebraic variables".into(),
            ))
        }
        ImplicitProblem::Ode {
            jacobian: JacobianStrategy::FiniteDifference { relative_step },
            ..
        }
        | ImplicitProblem::MassMatrix {
            jacobian: JacobianStrategy::FiniteDifference { relative_step },
            ..
        } if *relative_step <= 0.0 => Err(RadauError::Invalid(
            "finite-difference Jacobian step must be positive".into(),
        )),
        _ => Ok(()),
    }
}

/// Loadable runtime library advertising the `radau-iia` DAE backend.
pub struct ImplicitNumbersLib;
impl ImplicitNumbersLib {
    /// Creates the stateless library.
    pub fn new() -> Self {
        Self
    }
}
impl Default for ImplicitNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}
impl Lib for ImplicitNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "implicit"),
            version: Version(env!("CARGO_PKG_VERSION").into()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: vec![],
            exports: vec![Export::Value {
                symbol: numeric_plugin_descriptor_symbol("numbers/implicit", "radau-iia"),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(
            numeric_plugin_descriptor_symbol("numbers/implicit", "radau-iia"),
            cx.factory().table(vec![
                (
                    Symbol::new("kind"),
                    cx.factory().string("numeric-plugin".into())?,
                ),
                (
                    Symbol::new("method"),
                    cx.factory().symbol(Symbol::new("radau-iia"))?,
                ),
                (
                    Symbol::new("plugin-kind"),
                    cx.factory().string("dae".into())?,
                ),
                (Symbol::new("adaptive"), cx.factory().bool(true)?),
                (Symbol::new("dense-path"), cx.factory().bool(true)?),
                (Symbol::new("events"), cx.factory().bool(true)?),
                (Symbol::new("jacobian"), cx.factory().bool(true)?),
                (Symbol::new("mass-matrix"), cx.factory().bool(true)?),
                (Symbol::new("dae-residual"), cx.factory().bool(true)?),
                (
                    Symbol::new("provider"),
                    cx.factory()
                        .symbol(Symbol::qualified("numbers", "implicit"))?,
                ),
            ])?,
        )?;
        Ok(())
    }
}
