//! Projected multivariate objective minimization.

use super::*;

/// Projected BFGS/trust-region objective minimization with acceptance ratio.
pub fn minimize<F, G>(
    mut f: F,
    gradient: Option<G>,
    mut x: Vec<f64>,
    plan: &ObjectivePlan,
) -> Result<OptimizeResult, Error>
where
    F: FnMut(&[f64]) -> f64,
    G: Fn(&[f64], &mut [f64]),
{
    let n = x.len();
    if plan.bounds.lower.len() != n {
        return Err(Error::Dimension("bounds and point differ"));
    }
    validate_scale(&plan.scale, n)?;
    if plan.policy != StepPolicy::ProjectedBfgs || plan.initial_radius <= 0.0 {
        return Err(Error::InvalidPlan(
            "multivariate objective requires projected BFGS and positive radius",
        ));
    }
    plan.bounds.project(&mut x);
    let memory = n * n * 8 + n * 40;
    if memory > plan.limits.memory_bytes {
        return Ok(opt_result(
            x,
            f64::NAN,
            vec![],
            Termination::WorkLimit,
            0,
            0,
            memory,
        ));
    }
    let mut h = vec![vec![0.0; n]; n];
    for (i, row) in h.iter_mut().enumerate() {
        row[i] = 1.0
    }
    let mut fx = f(&x);
    let mut evals = 1;
    let mut radius = plan.initial_radius;
    if !fx.is_finite() {
        return Ok(opt_result(
            x,
            fx,
            vec![],
            Termination::NonFinite,
            evals,
            0,
            memory,
        ));
    }
    for iter in 0..plan.limits.iterations {
        let mut g = vec![0.0; n];
        if let Some(ref grad) = gradient {
            grad(&x, &mut g)
        } else if let Some(v) = numerical_gradient(
            &mut f,
            &x,
            fx,
            &plan.scale,
            &mut evals,
            plan.limits.evaluations,
        ) {
            g = v
        } else {
            return Ok(opt_result(
                x,
                fx,
                g,
                Termination::NonFinite,
                evals,
                iter,
                memory,
            ));
        };
        if !finite(&g) {
            return Ok(opt_result(
                x,
                fx,
                g,
                Termination::NonFinite,
                evals,
                iter,
                memory,
            ));
        }
        let mut pg = g.clone();
        for i in 0..n {
            if (x[i] <= plan.bounds.lower[i] && g[i] > 0.0)
                || (x[i] >= plan.bounds.upper[i] && g[i] < 0.0)
            {
                pg[i] = 0.0
            }
        }
        if norm(&pg) <= plan.tolerances.gradient {
            let boundary = pg != g;
            return Ok(opt_result(
                x,
                fx,
                g,
                if boundary {
                    Termination::BoundaryConverged
                } else {
                    Termination::Converged
                },
                evals,
                iter,
                memory,
            ));
        }
        let mut p = vec![0.0; n];
        for i in 0..n {
            p[i] = -h[i].iter().zip(&pg).map(|(a, b)| a * b).sum::<f64>() / plan.scale[i]
        }
        let pn = norm(&p);
        if pn > radius {
            for z in &mut p {
                *z *= radius / pn
            }
        }
        let mut y = x.iter().zip(&p).map(|(a, b)| a + b).collect::<Vec<_>>();
        plan.bounds.project(&mut y);
        let step = y.iter().zip(&x).map(|(a, b)| a - b).collect::<Vec<_>>();
        if norm(&step) <= plan.tolerances.argument {
            return Ok(opt_result(
                x,
                fx,
                g,
                Termination::NoProgress,
                evals,
                iter,
                memory,
            ));
        }
        if evals >= plan.limits.evaluations {
            return Ok(opt_result(
                x,
                fx,
                g,
                Termination::WorkLimit,
                evals,
                iter,
                memory,
            ));
        }
        let fy = f(&y);
        evals += 1;
        if !fy.is_finite() {
            radius *= 0.25;
            continue;
        }
        let predicted = (-pg.iter().zip(&step).map(|(a, b)| a * b).sum::<f64>()).max(f64::EPSILON);
        let ratio = (fx - fy) / predicted;
        if ratio > 0.1 {
            let old = x;
            x = y;
            let old_fx = fx;
            fx = fy;
            let mut ng = vec![0.0; n];
            if let Some(ref grad) = gradient {
                grad(&x, &mut ng)
            } else if let Some(v) = numerical_gradient(
                &mut f,
                &x,
                fx,
                &plan.scale,
                &mut evals,
                plan.limits.evaluations,
            ) {
                ng = v
            } else {
                return Ok(opt_result(
                    x,
                    fx,
                    g,
                    Termination::WorkLimit,
                    evals,
                    iter,
                    memory,
                ));
            };
            let s = x.iter().zip(&old).map(|(a, b)| a - b).collect::<Vec<_>>();
            let q = ng.iter().zip(&g).map(|(a, b)| a - b).collect::<Vec<_>>();
            let sq = s.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>();
            if sq > 1e-14 {
                let rho = 1.0 / sq;
                let hq = h
                    .iter()
                    .map(|r| r.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>())
                    .collect::<Vec<_>>();
                let qhq = q.iter().zip(&hq).map(|(a, b)| a * b).sum::<f64>();
                for i in 0..n {
                    for j in 0..n {
                        h[i][j] += (1.0 + qhq * rho) * rho * s[i] * s[j]
                            - rho * (s[i] * hq[j] + hq[i] * s[j]);
                    }
                }
            }
            if (old_fx - fx).abs() <= plan.tolerances.objective {
                return Ok(opt_result(
                    x,
                    fx,
                    ng,
                    Termination::Converged,
                    evals,
                    iter + 1,
                    memory,
                ));
            }
            if ratio > 0.75 {
                radius *= 2.0
            }
        } else {
            radius *= 0.25
        }
    }
    Ok(opt_result(
        x,
        fx,
        vec![],
        Termination::WorkLimit,
        evals,
        plan.limits.iterations,
        memory,
    ))
}
pub(crate) fn opt_result(
    x: Vec<f64>,
    value: f64,
    g: Vec<f64>,
    termination: Termination,
    evaluations: usize,
    iterations: usize,
    memory_bytes: usize,
) -> OptimizeResult {
    OptimizeResult {
        active: vec![],
        gradient_norm: norm(&g),
        point: x,
        value,
        termination,
        work: Work {
            evaluations,
            iterations,
            memory_bytes,
        },
    }
}

pub(crate) fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, tol: f64) -> (Vec<f64>, usize) {
    let n = b.len();
    let mut rank = 0;
    for k in 0..n {
        let mut p = k;
        for i in k + 1..n {
            if a[i][k].abs() > a[p][k].abs() {
                p = i
            }
        }
        if a[p][k].abs() <= tol {
            continue;
        }
        a.swap(k, p);
        b.swap(k, p);
        rank += 1;
        let pivot_row = a[k].clone();
        for i in k + 1..n {
            let q = a[i][k] / a[k][k];
            for (value, pivot) in a[i][k..].iter_mut().zip(&pivot_row[k..]) {
                *value -= q * pivot
            }
            b[i] -= q * b[k]
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if a[i][i].abs() > tol {
            x[i] = (b[i]
                - a[i]
                    .iter()
                    .enumerate()
                    .skip(i + 1)
                    .map(|(j, v)| v * x[j])
                    .sum::<f64>())
                / a[i][i]
        }
    }
    (x, rank)
}
pub(crate) fn normal(j: &[Vec<f64>], r: &[f64], damping: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = j.first().map_or(0, Vec::len);
    let mut a = vec![vec![0.0; n]; n];
    let mut b = vec![0.0; n];
    for (row, ri) in j.iter().zip(r) {
        for p in 0..n {
            b[p] -= row[p] * ri;
            for q in 0..n {
                a[p][q] += row[p] * row[q]
            }
        }
    }
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += damping
    }
    (a, b)
}

pub(crate) fn reflective_step(
    j: &[Vec<f64>],
    r: &[f64],
    x: &[f64],
    bounds: &Bounds,
    damping: f64,
    radius: f64,
    tolerance: f64,
) -> Vec<f64> {
    let (mut a, b) = normal(j, r, 0.0);
    let distance = b
        .iter()
        .enumerate()
        .map(|(i, descent)| {
            if *descent >= 0.0 {
                bounds.upper[i] - x[i]
            } else {
                x[i] - bounds.lower[i]
            }
        })
        .map(|v| v.max(tolerance).sqrt())
        .collect::<Vec<_>>();
    for i in 0..x.len() {
        for k in 0..x.len() {
            a[i][k] *= distance[i] * distance[k];
        }
        a[i][i] += damping;
    }
    let scaled_rhs = b
        .iter()
        .zip(&distance)
        .map(|(v, d)| v * d)
        .collect::<Vec<_>>();
    let (mut scaled, _) = solve(a, scaled_rhs, tolerance);
    let scaled_norm = norm(&scaled);
    if scaled_norm > radius {
        for value in &mut scaled {
            *value *= radius / scaled_norm;
        }
    }
    scaled
        .iter()
        .zip(distance)
        .map(|(value, d)| value * d)
        .collect()
}
