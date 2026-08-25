//! Sampled quadrature with explicit policy and reconstructable evidence.
use std::{error::Error, fmt};
/// Floating-point reduction policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SumMode {
    /// Sequential addition.
    Naive,
    /// Balanced addition.
    Pairwise,
    /// Neumaier compensated addition.
    Neumaier,
}
/// Sampled integration rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampledRule {
    /// Trapezoid rule.
    Trapezoid,
    /// Non-uniform three-point Simpson rule.
    Simpson,
}
/// Unmatched Simpson interval policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinalInterval {
    /// Reject it.
    Error,
    /// Apply trapezoid and record it.
    Trapezoid,
}
/// Complete sampled quadrature policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampledPlan {
    /// Rule.
    pub rule: SampledRule,
    /// Tail policy.
    pub final_interval: FinalInterval,
    /// Reduction.
    pub sum: SumMode,
}
impl Default for SampledPlan {
    fn default() -> Self {
        Self {
            rule: SampledRule::Trapezoid,
            final_interval: FinalInterval::Error,
            sum: SumMode::Neumaier,
        }
    }
}
/// Reconstructable execution evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledEvidence {
    /// Plan.
    pub plan: SampledPlan,
    /// Increasing `1`, decreasing `-1`.
    pub orientation: i8,
    /// Input points.
    pub points: usize,
    /// Simpson groups.
    pub simpson_groups: usize,
    /// Tail fallback occurred.
    pub fallback_used: bool,
}
/// Sampled scalar/vector result.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledIntegral {
    /// Per-component values.
    pub value: Vec<f64>,
    /// Evidence.
    pub evidence: SampledEvidence,
}
/// Invalid sampled input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SampledError {
    /// Fewer than two points.
    TooShort,
    /// Length mismatch.
    Misaligned,
    /// Inconsistent component widths.
    Ragged,
    /// Non-finite data.
    NonFinite,
    /// Duplicate or non-monotone coordinates.
    NotStrictlyMonotone,
    /// Undeclared Simpson tail.
    UnmatchedFinalInterval,
    /// Adjacent event coordinates differ.
    EventBoundaryMismatch,
}
impl fmt::Display for SampledError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for SampledError {}
fn valid(x: &[f64], y: &[Vec<f64>]) -> Result<(i8, usize), SampledError> {
    if x.len() < 2 {
        return Err(SampledError::TooShort);
    }
    if x.len() != y.len() {
        return Err(SampledError::Misaligned);
    }
    let n = y[0].len();
    if n == 0 || y.iter().any(|r| r.len() != n) {
        return Err(SampledError::Ragged);
    }
    if x.iter().chain(y.iter().flatten()).any(|v| !v.is_finite()) {
        return Err(SampledError::NonFinite);
    }
    let s = if x[1] > x[0] {
        1
    } else if x[1] < x[0] {
        -1
    } else {
        return Err(SampledError::NotStrictlyMonotone);
    };
    if x.windows(2)
        .any(|w| (s > 0 && w[1] <= w[0]) || (s < 0 && w[1] >= w[0]))
    {
        return Err(SampledError::NotStrictlyMonotone);
    }
    Ok((s, n))
}
fn sum(v: &[f64], m: SumMode) -> f64 {
    match m {
        SumMode::Naive => v.iter().sum(),
        SumMode::Pairwise => {
            if v.len() < 2 {
                v.first().copied().unwrap_or(0.)
            } else {
                let n = v.len() / 2;
                sum(&v[..n], m) + sum(&v[n..], m)
            }
        }
        SumMode::Neumaier => {
            let (mut s, mut c) = (0., 0.);
            for &x in v {
                let t = s + x;
                c += if s.abs() >= x.abs() {
                    (s - t) + x
                } else {
                    (x - t) + s
                };
                s = t
            }
            s + c
        }
    }
}
/// Integrates strictly monotone aligned scalar or vector samples.
pub fn integrate_sampled(
    x: &[f64],
    y: &[Vec<f64>],
    p: SampledPlan,
) -> Result<SampledIntegral, SampledError> {
    let (o, n) = valid(x, y)?;
    let mut t = vec![vec![]; n];
    let (mut i, mut g, mut fb) = (0, 0, false);
    match p.rule {
        SampledRule::Trapezoid => {
            while i + 1 < x.len() {
                let h = x[i + 1] - x[i];
                for c in 0..n {
                    t[c].push(h * (y[i][c] + y[i + 1][c]) * 0.5)
                }
                i += 1
            }
        }
        SampledRule::Simpson => {
            while i + 2 < x.len() {
                let (a, b) = (x[i + 1] - x[i], x[i + 2] - x[i + 1]);
                for c in 0..n {
                    t[c].push(
                        (a + b) / 6.
                            * ((2. - b / a) * y[i][c]
                                + (a + b).powi(2) / (a * b) * y[i + 1][c]
                                + (2. - a / b) * y[i + 2][c]),
                    )
                }
                i += 2;
                g += 1
            }
            if i + 1 < x.len() {
                if p.final_interval == FinalInterval::Error {
                    return Err(SampledError::UnmatchedFinalInterval);
                }
                let h = x[i + 1] - x[i];
                for c in 0..n {
                    t[c].push(h * (y[i][c] + y[i + 1][c]) * 0.5)
                }
                fb = true
            }
        }
    }
    Ok(SampledIntegral {
        value: t.iter().map(|v| sum(v, p.sum)).collect(),
        evidence: SampledEvidence {
            plan: p,
            orientation: o,
            points: x.len(),
            simpson_groups: g,
            fallback_used: fb,
        },
    })
}
/// Cumulative results aligned to every input point.
pub fn cumulative_sampled(
    x: &[f64],
    y: &[Vec<f64>],
    p: SampledPlan,
) -> Result<Vec<Vec<f64>>, SampledError> {
    let (_, n) = valid(x, y)?;
    let mut out = vec![vec![0.; n]; x.len()];
    for i in 1..x.len() {
        let mut q = p;
        if q.rule == SampledRule::Simpson && i % 2 == 1 {
            q.final_interval = FinalInterval::Trapezoid
        }
        out[i] = integrate_sampled(&x[..=i], &y[..=i], q)?.value
    }
    Ok(out)
}
/// Named continuous segment with explicit one-sided endpoint samples.
#[derive(Clone, Debug, PartialEq)]
pub struct EventSegment {
    /// Identity.
    pub id: String,
    /// Coordinates.
    pub x: Vec<f64>,
    /// Samples including explicit endpoint side values.
    pub y: Vec<Vec<f64>>,
}
/// Integrates segments without interpolating across events or impulses.
pub fn integrate_event_segments(
    s: &[EventSegment],
    p: SampledPlan,
) -> Result<Vec<(String, SampledIntegral)>, SampledError> {
    if s.windows(2).any(|w| w[0].x.last() != w[1].x.first()) {
        return Err(SampledError::EventBoundaryMismatch);
    }
    s.iter()
        .map(|v| integrate_sampled(&v.x, &v.y, p).map(|r| (v.id.clone(), r)))
        .collect()
}

/// Vector callable result with its single shared accepted refinement mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorIntegral {
    /// Component values.
    pub value: Vec<f64>,
    /// Accepted intervals shared by every component.
    pub mesh: Vec<(f64, f64)>,
    /// Reduction policy used both within each rule evaluation and across the
    /// accepted mesh.
    pub sum: SumMode,
}
/// Adaptive vector integration using an embedded Gauss/Kronrod pair and one
/// max-component error decision for the shared refinement mesh.
pub fn adaptive_vector_gauss_kronrod<F>(
    mut f: F,
    a: f64,
    b: f64,
    tol: f64,
    depth: usize,
    m: SumMode,
) -> Result<VectorIntegral, SampledError>
where
    F: FnMut(f64) -> Vec<f64>,
{
    const X: [f64; 8] = [
        0.9914553711208126,
        0.9491079123427585,
        0.8648644233597691,
        0.7415311855993945,
        0.5860872354676911,
        0.4058451513773972,
        0.20778495500789847,
        0.,
    ];
    const K: [f64; 8] = [
        0.022935322010529224,
        0.06309209262997856,
        0.10479001032225018,
        0.14065325971552592,
        0.1690047266392679,
        0.19035057806478542,
        0.20443294007529889,
        0.20948214108472782,
    ];
    const G: [f64; 4] = [
        0.1294849661688697,
        0.27970539148927664,
        0.3818300505051189,
        0.4179591836734694,
    ];
    fn go<F: FnMut(f64) -> Vec<f64>>(
        f: &mut F,
        a: f64,
        b: f64,
        tol: f64,
        d: usize,
        m: SumMode,
        parts: &mut Vec<Vec<f64>>,
        mesh: &mut Vec<(f64, f64)>,
    ) -> Result<(), SampledError> {
        let mid = (a + b) * 0.5;
        let h = (b - a) * 0.5;
        let z = f(mid);
        if z.is_empty() || z.iter().any(|v| !v.is_finite()) {
            return Err(SampledError::NonFinite);
        }
        let n = z.len();
        let (mut kt, mut gt) = (vec![vec![]; n], vec![vec![]; n]);
        for (i, &x) in X.iter().enumerate() {
            let v = if x == 0. {
                z.clone()
            } else {
                let (p, q) = (f(mid + h * x), f(mid - h * x));
                if p.len() != n || q.len() != n {
                    return Err(SampledError::Ragged);
                }
                p.iter().zip(q).map(|(a, b)| a + b).collect()
            };
            if v.iter().any(|v| !v.is_finite()) {
                return Err(SampledError::NonFinite);
            }
            for c in 0..n {
                kt[c].push(K[i] * v[c]);
                if let Some(j) = match i {
                    1 => Some(0),
                    3 => Some(1),
                    5 => Some(2),
                    7 => Some(3),
                    _ => None,
                } {
                    gt[c].push(G[j] * v[c])
                }
            }
        }
        let kv: Vec<_> = kt.iter().map(|v| h * sum(v, m)).collect();
        let gv: Vec<_> = gt.iter().map(|v| h * sum(v, m)).collect();
        let e = kv
            .iter()
            .zip(gv)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        if d == 0 || e <= tol {
            parts.push(kv);
            mesh.push((a, b));
            Ok(())
        } else {
            go(f, a, mid, tol * 0.5, d - 1, m, parts, mesh)?;
            go(f, mid, b, tol * 0.5, d - 1, m, parts, mesh)
        }
    }
    if !a.is_finite() || !b.is_finite() || a == b || !tol.is_finite() || tol <= 0. {
        return Err(SampledError::NonFinite);
    }
    let (mut p, mut mesh) = (vec![], vec![]);
    go(&mut f, a, b, tol, depth, m, &mut p, &mut mesh)?;
    let n = p[0].len();
    Ok(VectorIntegral {
        value: (0..n)
            .map(|c| sum(&p.iter().map(|v| v[c]).collect::<Vec<_>>(), m))
            .collect(),
        mesh,
        sum: m,
    })
}
