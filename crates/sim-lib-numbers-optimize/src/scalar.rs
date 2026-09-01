//! Bounded scalar minimization and finite-difference gradients.

use super::*;

/// Bounded Brent minimization. This proves local bracket convergence only.
pub fn minimize_scalar<F>(
    mut f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    limits: Limits,
) -> Result<ScalarResult, Error>
where
    F: FnMut(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() || a >= b || !tol.is_finite() || tol <= 0.0 {
        return Err(Error::InvalidPlan(
            "scalar interval and tolerance must be finite and ordered",
        ));
    }
    let golden = 0.3819660112501051;
    let mut x = a + golden * (b - a);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x);
    let (mut fw, mut fv) = (fx, fx);
    let (mut d, mut e) = (0.0_f64, 0.0_f64);
    let mut evals = 1;
    if !fx.is_finite() {
        return Ok(ScalarResult {
            minimizer: x,
            value: fx,
            final_bracket: (a, b),
            termination: Termination::NonFinite,
            work: Work {
                evaluations: evals,
                iterations: 0,
                memory_bytes: 0,
            },
        });
    }
    for iter in 0..limits.iterations {
        let m = 0.5 * (a + b);
        let t = tol * x.abs() + f64::EPSILON.sqrt();
        if (x - m).abs() <= 2.0 * t - 0.5 * (b - a) {
            return Ok(ScalarResult {
                minimizer: x,
                value: fx,
                final_bracket: (a, b),
                termination: Termination::Converged,
                work: Work {
                    evaluations: evals,
                    iterations: iter,
                    memory_bytes: 0,
                },
            });
        }
        let old = e;
        e = d;
        if old.abs() > t {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p
            } else {
                q = -q
            };
            if p.abs() >= 0.5 * q * old.abs() || p <= q * (a - x) || p >= q * (b - x) {
                e = if x < m { b - x } else { a - x };
                d = golden * e
            } else {
                d = p / q;
            }
        } else {
            e = if x < m { b - x } else { a - x };
            d = golden * e
        }
        let u = x + if d.abs() >= t { d } else { t.copysign(d) };
        if evals >= limits.evaluations {
            return Ok(ScalarResult {
                minimizer: x,
                value: fx,
                final_bracket: (a, b),
                termination: Termination::WorkLimit,
                work: Work {
                    evaluations: evals,
                    iterations: iter,
                    memory_bytes: 0,
                },
            });
        }
        let fu = f(u);
        evals += 1;
        if !fu.is_finite() {
            return Ok(ScalarResult {
                minimizer: x,
                value: fx,
                final_bracket: (a, b),
                termination: Termination::NonFinite,
                work: Work {
                    evaluations: evals,
                    iterations: iter,
                    memory_bytes: 0,
                },
            });
        }
        if fu <= fx {
            if u < x {
                b = x
            } else {
                a = x
            };
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu
        } else {
            if u < x {
                a = u
            } else {
                b = u
            };
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu
            }
        }
    }
    Ok(ScalarResult {
        minimizer: x,
        value: fx,
        final_bracket: (a, b),
        termination: Termination::WorkLimit,
        work: Work {
            evaluations: evals,
            iterations: limits.iterations,
            memory_bytes: 0,
        },
    })
}

pub(crate) fn numerical_gradient<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    x: &[f64],
    fx: f64,
    scale: &[f64],
    evals: &mut usize,
    limit: usize,
) -> Option<Vec<f64>> {
    let mut g = vec![0.0; x.len()];
    for i in 0..x.len() {
        if *evals >= limit {
            return None;
        }
        let mut y = x.to_vec();
        let h = f64::EPSILON.sqrt() * (x[i].abs() + scale[i]);
        y[i] += h;
        let fy = f(&y);
        *evals += 1;
        if !fy.is_finite() {
            return None;
        }
        g[i] = (fy - fx) / h;
    }
    Some(g)
}
