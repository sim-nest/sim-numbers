//! Bounded optimization with explicit plans and honest convergence evidence.

use core::fmt;
use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_tensor_decomp::{
    SingularCutoff, SvdPlan, VectorForm, least_squares as svd_least_squares, numerical_rank,
    svd_f64,
};

/// Cookbook recipes embedded for runtime discovery.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

/// Loadable schema surface for optimization policy and evidence discovery.
#[derive(Default)]
pub struct OptimizeLib;
impl OptimizeLib {
    pub fn new() -> Self {
        Self
    }
}
impl Lib for OptimizeLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "optimize"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: optimize_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(optimize_schema_symbol(),cx.factory().string("bounded-brent projected-bfgs levenberg-marquardt trust-region-reflective active-set-linear-least-squares rank covariance termination work assignment-adapter".to_owned())?)
    }
}
/// Runtime inspection symbol.
pub fn optimize_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/optimize", "schema")
}

/// Source of first derivatives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivativeSource {
    Analytic,
    Automatic,
    FiniteDifference,
}
/// Globalization strategy; bounded paths are genuine projected/active-set methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepPolicy {
    BrentGolden,
    LevenbergMarquardt,
    TrustRegionReflective,
    ProjectedBfgs,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Termination {
    Converged,
    BoundaryConverged,
    Flat,
    WorkLimit,
    NonFinite,
    NoProgress,
    InvalidPlan,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    pub argument: f64,
    pub objective: f64,
    pub gradient: f64,
}
impl Default for Tolerances {
    fn default() -> Self {
        Self {
            argument: 1e-9,
            objective: 1e-12,
            gradient: 1e-8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    pub evaluations: usize,
    pub iterations: usize,
    pub memory_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            evaluations: 10_000,
            iterations: 500,
            memory_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
impl Bounds {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, Error> {
        if lower.len() != upper.len()
            || lower
                .iter()
                .zip(&upper)
                .any(|(l, u)| !l.is_finite() || !u.is_finite() || l > u)
        {
            return Err(Error::InvalidPlan(
                "bounds must be finite, ordered, and equal length",
            ));
        }
        Ok(Self { lower, upper })
    }
    fn project(&self, x: &mut [f64]) {
        for ((x, l), u) in x.iter_mut().zip(&self.lower).zip(&self.upper) {
            *x = x.clamp(*l, *u);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectivePlan {
    pub bounds: Bounds,
    pub scale: Vec<f64>,
    pub derivative: DerivativeSource,
    pub policy: StepPolicy,
    pub tolerances: Tolerances,
    pub limits: Limits,
    pub initial_radius: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct LeastSquaresPlan {
    pub bounds: Option<Bounds>,
    pub variable_scale: Vec<f64>,
    pub residual_scale: Vec<f64>,
    pub derivative: DerivativeSource,
    pub policy: StepPolicy,
    pub tolerances: Tolerances,
    pub limits: Limits,
    pub initial_damping: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Work {
    pub evaluations: usize,
    pub iterations: usize,
    pub memory_bytes: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizeResult {
    pub point: Vec<f64>,
    pub value: f64,
    pub gradient_norm: f64,
    pub active: Vec<usize>,
    pub termination: Termination,
    pub work: Work,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarResult {
    pub minimizer: f64,
    pub value: f64,
    pub final_bracket: (f64, f64),
    pub termination: Termination,
    pub work: Work,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Covariance {
    Available(Vec<Vec<f64>>),
    Unavailable(CovarianceUnavailable),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CovarianceUnavailable {
    RankDeficient,
    InsufficientDegreesOfFreedom,
    StatisticalAssumptionsNotDeclared,
}
#[derive(Clone, Debug, PartialEq)]
pub struct LeastSquaresResult {
    pub point: Vec<f64>,
    pub residuals: Vec<f64>,
    pub residual_norm: f64,
    pub rank: usize,
    pub active: Vec<usize>,
    pub covariance: Covariance,
    pub termination: Termination,
    pub work: Work,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidPlan(&'static str),
    Dimension(&'static str),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(s) | Self::Dimension(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for Error {}

fn finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn validate_scale(scale: &[f64], n: usize) -> Result<(), Error> {
    if scale.len() != n || scale.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        Err(Error::InvalidPlan(
            "scale must contain one finite positive value per variable",
        ))
    } else {
        Ok(())
    }
}

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

fn numerical_gradient<F: FnMut(&[f64]) -> f64>(
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
fn opt_result(
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

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, tol: f64) -> (Vec<f64>, usize) {
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
fn normal(j: &[Vec<f64>], r: &[f64], damping: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
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

fn reflective_step(
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
fn linear_residual(a: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b)
        .map(|(row, y)| row.iter().zip(x).map(|(v, z)| v * z).sum::<f64>() - y)
        .collect()
}
#[allow(clippy::too_many_arguments)]
fn linear_result(
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

fn inverse(mut a: Vec<Vec<f64>>, tol: f64) -> Option<Vec<Vec<f64>>> {
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
fn ls_result<R: Fn(&[f64]) -> Vec<f64>, J: Fn(&[f64]) -> Vec<Vec<f64>>>(
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

#[cfg(feature = "assignment")]
pub mod assignment {
    use sim_lib_discrete_graph::{
        Assignment, AssignmentPolicy, CostMatrix, GraphError, min_cost_assignment,
    };
    pub fn assign(
        costs: Vec<Vec<f64>>,
        policy: AssignmentPolicy<f64>,
    ) -> Result<Assignment<f64>, GraphError> {
        let matrix = CostMatrix::try_from(costs)?;
        min_cost_assignment(&matrix, policy)
    }
}

#[cfg(test)]
mod tests;
