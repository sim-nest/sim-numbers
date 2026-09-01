//! Collocation interpolation, event location, rank estimates, and admission checks.

use super::*;

pub(super) fn integrated_lagrange(j: usize, x: f64) -> f64 {
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
pub(super) fn locate_events(
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
pub(super) fn estimate_rank(a: &[f64], n: usize) -> usize {
    (0..n).filter(|i| a[i * n + i].abs() > 1e-13).count()
}
pub(super) fn validate(
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
