//! Nonlinear and linear least-squares solvers.

use super::*;

/// Damped LM for unconstrained fits and trust-region reflective active-set steps for boxes.
pub fn least_squares<R, J>(
    residual: R,
    jacobian: J,
    mut x: Vec<f64>,
    plan: &LeastSquaresPlan,
) -> Result<LeastSquaresResult, Error>
where
    R: Fn(&[f64]) -> Vec<f64>,
    J: Fn(&[f64]) -> Vec<Vec<f64>>,
{
    let n = x.len();
    validate_scale(&plan.variable_scale, n)?;
    if plan.initial_damping <= 0.0 || !plan.initial_damping.is_finite() {
        return Err(Error::InvalidPlan(
            "initial damping must be finite and positive",
        ));
    }
    if let Some(bounds) = &plan.bounds {
        if bounds.lower.len() != n {
            return Err(Error::Dimension("bounds and point differ"));
        }
        bounds.project(&mut x);
        if plan.policy != StepPolicy::TrustRegionReflective {
            return Err(Error::InvalidPlan(
                "bounded least squares requires trust-region reflective policy",
            ));
        }
    } else if plan.policy != StepPolicy::LevenbergMarquardt {
        return Err(Error::InvalidPlan(
            "unconstrained least squares requires LM",
        ));
    }
    let mut lambda = plan.initial_damping.max(1e-12);
    let mut evals = 0;
    for iter in 0..plan.limits.iterations {
        if evals >= plan.limits.evaluations {
            return Ok(ls_result(
                x,
                &residual,
                &jacobian,
                plan.bounds.as_ref(),
                Termination::WorkLimit,
                evals,
                iter,
                false,
            ));
        }
        let r = residual(&x);
        let j = jacobian(&x);
        evals += 2;
        if plan.residual_scale.len() != r.len()
            || plan
                .residual_scale
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || j.len() != r.len()
            || j.iter().any(|v| v.len() != n)
        {
            return Err(Error::Dimension(
                "residual scaling or Jacobian dimensions differ",
            ));
        }
        let memory = (j.len() * n + n * n) * 8;
        if memory > plan.limits.memory_bytes {
            return Ok(ls_result(
                x,
                &residual,
                &jacobian,
                plan.bounds.as_ref(),
                Termination::WorkLimit,
                evals,
                iter,
                false,
            ));
        }
        if !finite(&r) || j.iter().any(|v| !finite(v)) {
            return Ok(ls_result(
                x,
                &residual,
                &jacobian,
                plan.bounds.as_ref(),
                Termination::NonFinite,
                evals,
                iter,
                false,
            ));
        }
        let mut step = if let Some(bounds) = &plan.bounds {
            reflective_step(
                &j,
                &r,
                &x,
                bounds,
                lambda,
                (1.0 / lambda).sqrt(),
                plan.tolerances.gradient,
            )
        } else {
            let (a, b) = normal(&j, &r, lambda);
            solve(a, b, plan.tolerances.gradient).0
        };
        if norm(&step) <= plan.tolerances.argument {
            return Ok(ls_result(
                x,
                &residual,
                &jacobian,
                plan.bounds.as_ref(),
                Termination::Converged,
                evals,
                iter,
                false,
            ));
        }
        if let Some(bounds) = &plan.bounds {
            // Fraction-to-the-boundary keeps the trial strictly feasible. The
            // diagonal reflective metric above, rather than projection, defines
            // the step direction and trust region.
            for i in 0..n {
                let room = if step[i] > 0.0 {
                    bounds.upper[i] - x[i]
                } else {
                    x[i] - bounds.lower[i]
                };
                if step[i].abs() > room {
                    step[i] = (0.995 * room).copysign(step[i])
                }
            }
        }
        let y = x.iter().zip(&step).map(|(a, b)| a + b).collect::<Vec<_>>();
        let nr = residual(&y);
        evals += 1;
        if !finite(&nr) {
            lambda *= 10.0;
            continue;
        }
        if norm(&nr) < norm(&r) {
            x = y;
            lambda *= 0.3;
            if (norm(&r) - norm(&nr)).abs() <= plan.tolerances.objective {
                return Ok(ls_result(
                    x,
                    &residual,
                    &jacobian,
                    plan.bounds.as_ref(),
                    Termination::Converged,
                    evals,
                    iter + 1,
                    false,
                ));
            }
        } else {
            lambda *= 10.0
        }
    }
    Ok(ls_result(
        x,
        &residual,
        &jacobian,
        plan.bounds.as_ref(),
        Termination::WorkLimit,
        evals,
        plan.limits.iterations,
        false,
    ))
}

/// Active-set bounded linear least squares over the shared, pivot-free Jacobi SVD.
///
/// Each iteration solves the reduced problem for the currently free variables,
/// fixes the first bound encountered, and releases a bound only when its KKT
/// multiplier has the wrong sign. This is a bounded-variable least-squares
/// path, not an unconstrained solve followed by clipping.
pub fn linear_least_squares(
    a: &[Vec<f64>],
    b: &[f64],
    bounds: Bounds,
    tol: f64,
    limits: Limits,
    statistical_assumptions: bool,
) -> Result<LeastSquaresResult, Error> {
    let n = a.first().map_or(0, Vec::len);
    if a.len() != b.len() || bounds.lower.len() != n || a.iter().any(|r| r.len() != n) {
        return Err(Error::Dimension("linear system dimensions differ"));
    }
    if !tol.is_finite() || tol <= 0.0 || !finite(b) || a.iter().any(|r| !finite(r)) {
        return Err(Error::InvalidPlan(
            "linear least-squares data and tolerance must be finite",
        ));
    }
    let memory = a.len().saturating_mul(n).saturating_mul(24);
    if memory > limits.memory_bytes {
        return Ok(linear_result(
            a,
            b,
            vec![0.0; n],
            0,
            Vec::new(),
            0,
            statistical_assumptions,
            Termination::WorkLimit,
        ));
    }
    let mut x = bounds
        .lower
        .iter()
        .zip(&bounds.upper)
        .map(|(l, u)| 0.0_f64.clamp(*l, *u))
        .collect::<Vec<_>>();
    let mut active = vec![None; n]; // Some(false) lower, Some(true) upper.
    let mut iterations = 0;
    for k in 0..limits.iterations {
        iterations = k + 1;
        let free = (0..n).filter(|&i| active[i].is_none()).collect::<Vec<_>>();
        let adjusted = a
            .iter()
            .zip(b)
            .map(|(row, rhs)| {
                rhs - (0..n)
                    .filter(|&i| active[i].is_some())
                    .map(|i| row[i] * x[i])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let reduced = a
            .iter()
            .flat_map(|row| free.iter().map(|&i| row[i]))
            .collect::<Vec<_>>();
        let (candidate, rank) = if free.is_empty() {
            (Vec::new(), 0)
        } else {
            let svd = svd_f64(
                &reduced,
                a.len(),
                free.len(),
                SvdPlan {
                    max_dimension: a.len().max(free.len()),
                    max_work: u64::try_from(limits.evaluations)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(1_000),
                    max_iterations: limits.iterations.max(1),
                    tolerance: tol,
                    vectors: VectorForm::Thin,
                    reconstruction_tolerance: tol.sqrt().max(1e-10),
                    return_partial: false,
                },
            )
            .map_err(|_| {
                Error::InvalidPlan("SVD could not certify the reduced least-squares system")
            })?;
            let rank = numerical_rank(&svd, SingularCutoff(tol))
                .map_err(|_| Error::InvalidPlan("invalid SVD cutoff"))?;
            let z = svd_least_squares(&svd, &adjusted, SingularCutoff(tol))
                .map_err(|_| Error::InvalidPlan("SVD least-squares solve failed"))?;
            (z, rank)
        };
        let mut target = x.clone();
        for (&i, &z) in free.iter().zip(&candidate) {
            target[i] = z;
        }
        let mut alpha = 1.0_f64;
        let mut hit = None;
        for &i in &free {
            let step = target[i] - x[i];
            let (bound, upper) = if step > 0.0 {
                (bounds.upper[i], true)
            } else {
                (bounds.lower[i], false)
            };
            if step != 0.0 {
                let q = (bound - x[i]) / step;
                if q >= 0.0 && q < alpha {
                    alpha = q;
                    hit = Some((i, upper));
                }
            }
        }
        for &i in &free {
            x[i] += alpha * (target[i] - x[i]);
        }
        if let Some((i, upper)) = hit {
            x[i] = if upper {
                bounds.upper[i]
            } else {
                bounds.lower[i]
            };
            active[i] = Some(upper);
            continue;
        }

        let residual = linear_residual(a, b, &x);
        let gradient = (0..n)
            .map(|j| {
                a.iter()
                    .zip(&residual)
                    .map(|(row, r)| row[j] * r)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let release = (0..n)
            .filter(|&i| match active[i] {
                Some(false) => gradient[i] < -tol,
                Some(true) => gradient[i] > tol,
                None => false,
            })
            .max_by(|&i, &j| gradient[i].abs().total_cmp(&gradient[j].abs()));
        if let Some(i) = release {
            active[i] = None;
        } else {
            let indices = (0..n).filter(|&i| active[i].is_some()).collect::<Vec<_>>();
            let termination = if indices.is_empty() {
                Termination::Converged
            } else {
                Termination::BoundaryConverged
            };
            return Ok(linear_result(
                a,
                b,
                x,
                rank + indices.len(),
                indices,
                iterations,
                statistical_assumptions,
                termination,
            ));
        }
        if iterations >= limits.evaluations {
            let indices = (0..n).filter(|&i| active[i].is_some()).collect::<Vec<_>>();
            return Ok(linear_result(
                a,
                b,
                x,
                rank,
                indices,
                iterations,
                statistical_assumptions,
                Termination::WorkLimit,
            ));
        }
    }
    let active = (0..n)
        .filter(|&i| (x[i] - bounds.lower[i]).abs() <= tol || (x[i] - bounds.upper[i]).abs() <= tol)
        .collect();
    Ok(linear_result(
        a,
        b,
        x,
        n,
        active,
        iterations,
        statistical_assumptions,
        Termination::WorkLimit,
    ))
}
pub(crate) fn linear_residual(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b)
        .map(|(row, y)| row.iter().zip(x).map(|(v, z)| v * z).sum::<f64>() - y)
        .collect()
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_result(
    a: &[Vec<f64>],
    b: &[f64],
    x: Vec<f64>,
    rank: usize,
    active: Vec<usize>,
    iterations: usize,
    stats: bool,
    termination: Termination,
) -> LeastSquaresResult {
    let r = linear_residual(a, b, &x);
    let n = x.len();
    let covariance = if rank < n {
        Covariance::Unavailable(CovarianceUnavailable::RankDeficient)
    } else if !stats {
        Covariance::Unavailable(CovarianceUnavailable::StatisticalAssumptionsNotDeclared)
    } else if a.len() <= n {
        Covariance::Unavailable(CovarianceUnavailable::InsufficientDegreesOfFreedom)
    } else {
        let mut gram = vec![vec![0.0; n]; n];
        for row in a {
            for i in 0..n {
                for j in 0..n {
                    gram[i][j] += row[i] * row[j];
                }
            }
        }
        match inverse(gram, 1e-12) {
            Some(mut inv) => {
                let variance = r.iter().map(|v| v * v).sum::<f64>() / (a.len() - n) as f64;
                for row in &mut inv {
                    for v in row {
                        *v *= variance;
                    }
                }
                Covariance::Available(inv)
            }
            None => Covariance::Unavailable(CovarianceUnavailable::RankDeficient),
        }
    };
    LeastSquaresResult {
        point: x,
        residual_norm: norm(&r),
        residuals: r,
        rank,
        active,
        covariance,
        termination,
        work: Work {
            evaluations: iterations,
            iterations,
            memory_bytes: (a.len() * n + n * n) * 8,
        },
    }
}

pub(crate) fn inverse(mut a: Vec<Vec<f64>>, tol: f64) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut inv = vec![vec![0.0; n]; n];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0
    }
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[p][k].abs() <= tol {
            return None;
        }
        a.swap(k, p);
        inv.swap(k, p);
        let d = a[k][k];
        for j in 0..n {
            a[k][j] /= d;
            inv[k][j] /= d
        }
        for i in 0..n {
            if i != k {
                let q = a[i][k];
                for j in 0..n {
                    a[i][j] -= q * a[k][j];
                    inv[i][j] -= q * inv[k][j]
                }
            }
        }
    }
    Some(inv)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn ls_result<R: Fn(&[f64]) -> Vec<f64>, J: Fn(&[f64]) -> Vec<Vec<f64>>>(
    x: Vec<f64>,
    r: &R,
    j: &J,
    bounds: Option<&Bounds>,
    t: Termination,
    e: usize,
    i: usize,
    stats: bool,
) -> LeastSquaresResult {
    let rv = r(&x);
    let jj = j(&x);
    let (_, rank) = solve(normal(&jj, &rv, 0.0).0, vec![0.0; x.len()], 1e-10);
    let n = x.len();
    let active = bounds.map_or_else(Vec::new, |bounds| {
        (0..n)
            .filter(|&i| {
                (x[i] - bounds.lower[i]).abs() <= 10.0 * f64::EPSILON.sqrt()
                    || (x[i] - bounds.upper[i]).abs() <= 10.0 * f64::EPSILON.sqrt()
            })
            .collect()
    });
    LeastSquaresResult {
        point: x,
        residual_norm: norm(&rv),
        residuals: rv,
        rank,
        active,
        covariance: if rank < n {
            Covariance::Unavailable(CovarianceUnavailable::RankDeficient)
        } else if stats {
            Covariance::Available(vec![vec![0.0; n]; n])
        } else {
            Covariance::Unavailable(CovarianceUnavailable::StatisticalAssumptionsNotDeclared)
        },
        termination: t,
        work: Work {
            evaluations: e,
            iterations: i,
            memory_bytes: jj.len() * n * 8,
        },
    }
}
