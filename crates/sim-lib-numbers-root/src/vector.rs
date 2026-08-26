//! Dense vector Newton and Broyden methods.

use super::*;

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
pub(crate) fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
pub(crate) fn finite_jacobian<F: FnMut(&[f64]) -> Vec<f64>>(
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
pub(crate) fn vector_solve(
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
pub(crate) fn vector_newton_impl<F, J>(
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
