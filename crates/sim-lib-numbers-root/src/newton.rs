//! Safeguarded analytic and finite-difference Newton methods.

use super::*;

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

pub(crate) fn newton_impl<F: FnMut(f64) -> f64, D: FnMut(f64) -> f64>(
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
