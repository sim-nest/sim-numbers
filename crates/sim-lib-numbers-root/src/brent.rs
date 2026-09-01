//! Brent-Dekker bracketed root solving.

use super::*;

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
