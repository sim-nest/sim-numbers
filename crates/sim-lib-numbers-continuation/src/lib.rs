#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Bounded pseudo-arclength continuation with explicit numerical evidence.

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_method::ExecutionIdentity;
use sim_lib_numbers_root::{
    JacobianSource, RootTermination, VectorPlan, VectorRoot, damped_newton,
};
use sim_lib_numbers_tensor_decomp::{
    SingularCutoff, SvdPlan, VectorForm, null_space, numerical_rank, svd_f64,
};

/// Cookbook recipes embedded for runtime discovery.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

/// Loadable runtime discovery surface.
#[derive(Default)]
pub struct ContinuationLib;
impl ContinuationLib {
    /// Creates the runtime library.
    pub fn new() -> Self {
        Self
    }
}
impl Lib for ContinuationLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "continuation"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: continuation_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(continuation_schema_symbol(), cx.factory().string("pseudo-arclength residual-manifold seeds derivative orientation step-plan bordered-newton folds rank-loss domain-exit closed-loop evidence".to_owned())?)
    }
}
/// Runtime inspection symbol.
pub fn continuation_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/continuation", "schema")
}

/// Declared source of the residual Jacobian.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivativeSource {
    /// Caller-supplied analytic Jacobian.
    Analytic,
    /// Jacobian supplied by an automatic-differentiation adapter.
    Automatic,
    /// Caller-supplied finite-difference Jacobian with external policy.
    FiniteDifference,
}
/// Desired traversal orientation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrientationPolicy {
    /// Preserve the direction from the first seed to the second.
    SeedDirection,
    /// Prefer increasing parameter when orientation is ambiguous.
    IncreasingParameter,
    /// Prefer decreasing parameter when orientation is ambiguous.
    DecreasingParameter,
}
/// Closed parameter interval. The final coordinate is the continuation parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterDomain {
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
}
/// Adaptive pseudo-arclength step policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepPlan {
    /// Initial step.
    pub initial: f64,
    /// Smallest retry step.
    pub minimum: f64,
    /// Largest accepted step.
    pub maximum: f64,
    /// Growth multiplier.
    pub growth: f64,
    /// Shrink multiplier.
    pub shrink: f64,
    /// Corrector iterations considered easy.
    pub easy_iterations: usize,
    /// Curvature threshold that forces shrinkage.
    pub curvature_threshold: f64,
}
impl Default for StepPlan {
    fn default() -> Self {
        Self {
            initial: 0.1,
            minimum: 1e-4,
            maximum: 0.5,
            growth: 1.4,
            shrink: 0.5,
            easy_iterations: 3,
            curvature_threshold: 0.35,
        }
    }
}
/// Hard method limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MethodLimits {
    /// Maximum accepted continuation steps.
    pub steps: usize,
    /// Maximum rejected predictions.
    pub rejections: usize,
    /// Maximum corrector iterations per attempt.
    pub corrector_iterations: usize,
    /// Maximum corrector evaluations per attempt.
    pub corrector_evaluations: u64,
}
impl Default for MethodLimits {
    fn default() -> Self {
        Self {
            steps: 100,
            rejections: 24,
            corrector_iterations: 16,
            corrector_evaluations: 256,
        }
    }
}
/// A fully declared continuation problem.
pub struct ContinuationProblem<R, J> {
    /// Residual mapping R^(m+1) -> R^m.
    pub residual: R,
    /// Row-major m by (m+1) residual Jacobian.
    pub jacobian: J,
    /// Two admitted points, in traversal order.
    pub seeds: [Vec<f64>; 2],
    /// Parameter domain for the last coordinate.
    pub domain: ParameterDomain,
    /// Derivative provenance.
    pub derivative: DerivativeSource,
    /// Orientation policy.
    pub orientation: OrientationPolicy,
    /// Adaptive step policy.
    pub step: StepPlan,
    /// Corrector tolerances.
    pub corrector: VectorPlan,
    /// Hard bounds.
    pub limits: MethodLimits,
    /// Replay identity.
    pub identity: ExecutionIdentity,
}
/// Fold state at one accepted point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FoldClassification {
    /// No parameter turning detected.
    Regular,
    /// Parameter tangent changed sign.
    Fold,
    /// Parameter tangent is locally near zero.
    NearFold,
}
/// A distinct trace event.
#[derive(Clone, Debug, PartialEq)]
pub enum ContinuationEvent {
    /// A prediction was rejected, retaining its corrector report.
    RejectedPrediction {
        /// Predicted point.
        predicted: Vec<f64>,
        /// Attempted step.
        step: f64,
        /// Corrector evidence.
        corrector: Box<VectorRoot>,
    },
    /// Step length changed.
    StepChanged {
        /// Previous length.
        from: f64,
        /// New length.
        to: f64,
        /// Decision reason.
        reason: StepDecision,
    },
    /// A fold was crossed.
    Fold,
    /// The tangent had to be flipped to maintain traversal.
    BranchOrientationChange,
    /// The bordered corrector lost rank.
    RankLoss,
    /// The unconstrained prediction left the parameter domain.
    DomainExit {
        /// Predicted parameter (not clamped).
        parameter: f64,
    },
    /// The new point approached the first seed.
    ClosedLoopApproach,
    /// The supplied seeds are inconsistent with the manifold.
    SeedInconsistency,
}
/// Why an adaptive step changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepDecision {
    /// Corrector was inexpensive and curvature low.
    EasyCorrector,
    /// Corrector effort was high.
    CorrectorEffort,
    /// Tangent curvature was high.
    Curvature,
    /// A prediction was rejected.
    Rejected,
}
/// One immutable accepted continuation point.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuationPoint {
    coordinates: Vec<f64>,
    residual: Vec<f64>,
    tangent: Vec<f64>,
    normal: Vec<f64>,
    corrector: Option<VectorRoot>,
    fold: FoldClassification,
}
impl ContinuationPoint {
    /// Coordinates, with parameter last.
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }
    /// Residual at the accepted point.
    pub fn residual(&self) -> &[f64] {
        &self.residual
    }
    /// Unit oriented tangent.
    pub fn tangent(&self) -> &[f64] {
        &self.tangent
    }
    /// Unit residual normal (gradient for scalar residuals).
    pub fn normal(&self) -> &[f64] {
        &self.normal
    }
    /// Bordered Newton report, absent only for seeds.
    pub fn corrector(&self) -> Option<&VectorRoot> {
        self.corrector.as_ref()
    }
    /// Fold classification.
    pub fn fold(&self) -> FoldClassification {
        self.fold
    }
}
/// Bounded trace termination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceTermination {
    /// Accepted-step limit reached.
    StepLimit,
    /// Rejection limit reached.
    RejectionLimit,
    /// Prediction left the parameter domain.
    DomainExit,
    /// Corrector rank loss.
    RankLoss,
    /// Seed admission failed.
    SeedInconsistency,
    /// Trace returned near its beginning.
    ClosedLoop,
}
/// Ordered trace and complete decision log.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuationTrace {
    points: Vec<ContinuationPoint>,
    events: Vec<ContinuationEvent>,
    termination: TraceTermination,
}
impl ContinuationTrace {
    /// Accepted points in traversal order.
    pub fn points(&self) -> &[ContinuationPoint] {
        &self.points
    }
    /// Ordered event log.
    pub fn events(&self) -> &[ContinuationEvent] {
        &self.events
    }
    /// Terminal reason.
    pub fn termination(&self) -> TraceTermination {
        self.termination
    }
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn normalized(mut x: Vec<f64>) -> Option<Vec<f64>> {
    let n = norm(&x);
    if !n.is_finite() || n <= f64::EPSILON {
        None
    } else {
        x.iter_mut().for_each(|v| *v /= n);
        Some(x)
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn tangent<J: FnMut(&[f64]) -> Vec<f64>>(
    j: &mut J,
    x: &[f64],
    cut: f64,
) -> Option<(Vec<f64>, Vec<f64>, usize)> {
    let rows = x.len() - 1;
    let a = j(x);
    if a.len() != rows * x.len() {
        return None;
    }
    let svd = svd_f64(
        &a,
        rows,
        x.len(),
        SvdPlan {
            vectors: VectorForm::Full,
            ..SvdPlan::default()
        },
    )
    .ok()?;
    let cutoff = SingularCutoff::new(cut).ok()?;
    let rank = numerical_rank(&svd, cutoff).ok()?;
    let ns = null_space(&svd, cutoff).ok()?;
    if ns.len() != x.len() {
        return None;
    }
    let t = normalized(ns)?;
    let normal = if rows == 1 {
        normalized(a).unwrap_or_else(|| vec![0.0; x.len()])
    } else {
        vec![0.0; x.len()]
    };
    Some((t, normal, rank))
}

/// Trace the declared residual manifold with pseudo-arclength prediction and bordered Newton correction.
pub fn trace<R, J>(mut p: ContinuationProblem<R, J>) -> ContinuationTrace
where
    R: FnMut(&[f64]) -> Vec<f64>,
    J: FnMut(&[f64]) -> Vec<f64>,
{
    let mut events = Vec::new();
    let n = p.seeds[0].len();
    let valid_plan = n >= 2
        && p.seeds[1].len() == n
        && p.domain.lower <= p.domain.upper
        && p.step.minimum > 0.0
        && p.step.minimum <= p.step.initial
        && p.step.initial <= p.step.maximum
        && p.step.shrink > 0.0
        && p.step.shrink < 1.0
        && p.step.growth > 1.0;
    let r0 = (p.residual)(&p.seeds[0]);
    let r1 = (p.residual)(&p.seeds[1]);
    if !valid_plan
        || r0.len() != n - 1
        || r1.len() != n - 1
        || norm(&r0) > p.corrector.residual_tolerance
        || norm(&r1) > p.corrector.residual_tolerance
        || p.seeds
            .iter()
            .any(|s| s[n - 1] < p.domain.lower || s[n - 1] > p.domain.upper)
    {
        events.push(ContinuationEvent::SeedInconsistency);
        return ContinuationTrace {
            points: vec![],
            events,
            termination: TraceTermination::SeedInconsistency,
        };
    }
    let mut sec = normalized(
        p.seeds[1]
            .iter()
            .zip(&p.seeds[0])
            .map(|(b, a)| b - a)
            .collect(),
    )
    .unwrap();
    let prefer = match p.orientation {
        OrientationPolicy::SeedDirection => 0.0,
        OrientationPolicy::IncreasingParameter => 1.0,
        OrientationPolicy::DecreasingParameter => -1.0,
    };
    if prefer != 0.0 && sec[n - 1] * prefer < 0.0 {
        sec.iter_mut().for_each(|v| *v = -*v);
    }
    let seed_point = |x: Vec<f64>, r: Vec<f64>, t: Vec<f64>, normal: Vec<f64>| ContinuationPoint {
        coordinates: x,
        residual: r,
        tangent: t,
        normal,
        corrector: None,
        fold: FoldClassification::Regular,
    };
    let (_, normal0, _) = tangent(&mut p.jacobian, &p.seeds[0], p.corrector.rank_cutoff)
        .unwrap_or((sec.clone(), vec![0.0; n], n - 1));
    let (mut tan, normal1, _) = tangent(&mut p.jacobian, &p.seeds[1], p.corrector.rank_cutoff)
        .unwrap_or((sec.clone(), vec![0.0; n], n - 1));
    if dot(&tan, &sec) < 0.0 {
        tan.iter_mut().for_each(|v| *v = -*v);
    }
    let mut points = vec![
        seed_point(p.seeds[0].clone(), r0, sec.clone(), normal0),
        seed_point(p.seeds[1].clone(), r1, tan.clone(), normal1),
    ];
    let mut h = p.step.initial;
    let mut rejected = 0;
    loop {
        if points.len() - 2 >= p.limits.steps {
            return ContinuationTrace {
                points,
                events,
                termination: TraceTermination::StepLimit,
            };
        }
        let base = points.last().unwrap().coordinates.clone();
        let predicted: Vec<_> = base.iter().zip(&tan).map(|(x, t)| x + h * t).collect();
        if predicted[n - 1] < p.domain.lower || predicted[n - 1] > p.domain.upper {
            events.push(ContinuationEvent::DomainExit {
                parameter: predicted[n - 1],
            });
            return ContinuationTrace {
                points,
                events,
                termination: TraceTermination::DomainExit,
            };
        }
        let anchor = predicted.clone();
        let border = tan.clone();
        let residual = &mut p.residual;
        let jac = &mut p.jacobian;
        let source = match p.derivative {
            DerivativeSource::Analytic => JacobianSource::Analytic,
            DerivativeSource::Automatic => JacobianSource::Automatic,
            DerivativeSource::FiniteDifference => JacobianSource::Analytic,
        };
        let mut cp = p.corrector;
        cp.max_iterations = cp.max_iterations.min(p.limits.corrector_iterations);
        cp.max_evaluations = cp.max_evaluations.min(p.limits.corrector_evaluations);
        let corrector = damped_newton(
            |x| {
                let mut r = residual(x);
                r.push(dot(
                    &border,
                    &x.iter()
                        .zip(&anchor)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                ));
                r
            },
            |x| {
                let mut a = jac(x);
                a.extend_from_slice(&border);
                a
            },
            source,
            predicted.clone(),
            cp,
            p.identity.clone(),
        );
        if corrector.evidence.termination != RootTermination::ResidualConverged {
            let rank_loss = corrector.evidence.termination == RootTermination::RankLoss;
            events.push(ContinuationEvent::RejectedPrediction {
                predicted,
                step: h,
                corrector: Box::new(corrector),
            });
            if rank_loss {
                events.push(ContinuationEvent::RankLoss);
                return ContinuationTrace {
                    points,
                    events,
                    termination: TraceTermination::RankLoss,
                };
            }
            rejected += 1;
            if rejected > p.limits.rejections || h * p.step.shrink < p.step.minimum {
                return ContinuationTrace {
                    points,
                    events,
                    termination: TraceTermination::RejectionLimit,
                };
            }
            let old = h;
            h = (h * p.step.shrink).max(p.step.minimum);
            events.push(ContinuationEvent::StepChanged {
                from: old,
                to: h,
                reason: StepDecision::Rejected,
            });
            continue;
        }
        let x = corrector.value.clone();
        let residual_at = (p.residual)(&x);
        let Some((mut next, normal, rank)) = tangent(&mut p.jacobian, &x, p.corrector.rank_cutoff)
        else {
            events.push(ContinuationEvent::RankLoss);
            return ContinuationTrace {
                points,
                events,
                termination: TraceTermination::RankLoss,
            };
        };
        if rank < n - 1 {
            events.push(ContinuationEvent::RankLoss);
            return ContinuationTrace {
                points,
                events,
                termination: TraceTermination::RankLoss,
            };
        }
        if dot(&next, &tan) < 0.0 {
            next.iter_mut().for_each(|v| *v = -*v);
            events.push(ContinuationEvent::BranchOrientationChange);
        }
        let curvature = norm(
            &next
                .iter()
                .zip(&tan)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );
        let fold = if tan[n - 1] * next[n - 1] < 0.0 {
            events.push(ContinuationEvent::Fold);
            FoldClassification::Fold
        } else if next[n - 1].abs() < p.step.curvature_threshold {
            FoldClassification::NearFold
        } else {
            FoldClassification::Regular
        };
        if points.len() > 4
            && norm(
                &x.iter()
                    .zip(&points[0].coordinates)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>(),
            ) <= h
        {
            events.push(ContinuationEvent::ClosedLoopApproach);
            return ContinuationTrace {
                points,
                events,
                termination: TraceTermination::ClosedLoop,
            };
        }
        let iterations = corrector.evidence.iterations;
        points.push(ContinuationPoint {
            coordinates: x,
            residual: residual_at,
            tangent: next.clone(),
            normal,
            corrector: Some(corrector),
            fold,
        });
        tan = next;
        let (new_h, reason) = if curvature > p.step.curvature_threshold {
            (
                (h * p.step.shrink).max(p.step.minimum),
                Some(StepDecision::Curvature),
            )
        } else if iterations <= p.step.easy_iterations {
            (
                (h * p.step.growth).min(p.step.maximum),
                Some(StepDecision::EasyCorrector),
            )
        } else {
            (
                (h * p.step.shrink).max(p.step.minimum),
                Some(StepDecision::CorrectorEffort),
            )
        };
        if let Some(reason) = reason
            && new_h != h
        {
            events.push(ContinuationEvent::StepChanged {
                from: h,
                to: new_h,
                reason,
            });
            h = new_h;
        }
    }
}

#[cfg(test)]
mod tests;
