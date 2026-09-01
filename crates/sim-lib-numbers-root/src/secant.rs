//! Bounded secant root solving.

use super::*;

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
