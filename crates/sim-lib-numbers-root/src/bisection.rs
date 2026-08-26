//! Bracket establishment and bisection.

use super::*;

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
