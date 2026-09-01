use super::*;

fn identity() -> ExecutionIdentity {
    ExecutionIdentity::new("test", "sim-lib-numbers-continuation", "trace-1").unwrap()
}
fn scalar_problem<R, J>(residual: R, jacobian: J, seeds: [[f64; 2]; 2]) -> ContinuationProblem<R, J>
where
    R: FnMut(&[f64]) -> Vec<f64>,
    J: FnMut(&[f64]) -> Vec<f64>,
{
    ContinuationProblem {
        residual,
        jacobian,
        seeds: seeds.map(Vec::from),
        domain: ParameterDomain {
            lower: -2.0,
            upper: 2.0,
        },
        derivative: DerivativeSource::Analytic,
        orientation: OrientationPolicy::SeedDirection,
        step: StepPlan {
            initial: 0.08,
            maximum: 0.12,
            ..StepPlan::default()
        },
        corrector: VectorPlan::default(),
        limits: MethodLimits {
            steps: 12,
            ..MethodLimits::default()
        },
        identity: identity(),
    }
}

#[test]
fn circle_points_are_ordered_immutable_and_carry_corrector_evidence() {
    let p = scalar_problem(
        |z| vec![z[0] * z[0] + z[1] * z[1] - 1.0],
        |z| vec![2.0 * z[0], 2.0 * z[1]],
        [[1.0, 0.0], [0.9950041653, 0.0998334166]],
    );
    let out = trace(p);
    assert!(out.points().len() > 4);
    for point in &out.points()[2..] {
        assert!(norm(point.residual()) < 1e-9);
        assert!((norm(point.tangent()) - 1.0).abs() < 1e-10);
        assert!(point.corrector().is_some());
    }
    assert!(
        out.events()
            .iter()
            .any(|e| matches!(e, ContinuationEvent::StepChanged { .. }))
    );
}

#[test]
fn fold_and_reversed_seeds_preserve_requested_orientation() {
    let forward = trace(scalar_problem(
        |z| vec![z[0] * z[0] - z[1]],
        |z| vec![2.0 * z[0], -1.0],
        [[-0.2, 0.04], [-0.1, 0.01]],
    ));
    assert!(
        forward
            .events()
            .iter()
            .any(|e| matches!(e, ContinuationEvent::Fold))
            || forward
                .points()
                .iter()
                .any(|p| p.fold() != FoldClassification::Regular)
    );
    let reverse = trace(scalar_problem(
        |z| vec![z[0] * z[0] - z[1]],
        |z| vec![2.0 * z[0], -1.0],
        [[0.1, 0.01], [-0.1, 0.01]],
    ));
    assert!(reverse.points()[1].tangent()[0] < 0.0);
}

#[test]
fn rank_loss_and_seed_inconsistency_are_distinct() {
    let singular = trace(scalar_problem(
        |_| vec![0.0],
        |_| vec![0.0, 0.0],
        [[0.0, 0.0], [0.1, 0.0]],
    ));
    assert_eq!(singular.termination(), TraceTermination::RankLoss);
    assert!(singular.events().contains(&ContinuationEvent::RankLoss));
    let bad = trace(scalar_problem(
        |z| vec![z[0] * z[0] - z[1]],
        |z| vec![2.0 * z[0], -1.0],
        [[0.0, 0.2], [0.1, 0.01]],
    ));
    assert_eq!(bad.termination(), TraceTermination::SeedInconsistency);
}

#[test]
fn domain_exit_is_unclamped_and_work_is_bounded() {
    let mut p = scalar_problem(
        |z| vec![z[0] - z[1]],
        |_| vec![1.0, -1.0],
        [[0.0, 0.0], [0.05, 0.05]],
    );
    p.domain = ParameterDomain {
        lower: 0.0,
        upper: 0.11,
    };
    p.step.initial = 0.1;
    p.step.maximum = 0.1;
    let out = trace(p);
    assert_eq!(out.termination(), TraceTermination::DomainExit);
    let parameter = out
        .events()
        .iter()
        .find_map(|e| {
            if let ContinuationEvent::DomainExit { parameter } = e {
                Some(*parameter)
            } else {
                None
            }
        })
        .unwrap();
    assert!(parameter > 0.11);
    let mut bounded = scalar_problem(
        |z| vec![z[0] - z[1]],
        |_| vec![1.0, -1.0],
        [[0.0, 0.0], [0.05, 0.05]],
    );
    bounded.limits.steps = 2;
    let bounded = trace(bounded);
    assert_eq!(bounded.termination(), TraceTermination::StepLimit);
    assert_eq!(bounded.points().len(), 4);
}

#[test]
fn recorded_points_resume_the_same_branch_and_cusp_is_traced() {
    let make = |seeds| {
        scalar_problem(
            |z| vec![z[0].powi(3) - z[1]],
            |z| vec![3.0 * z[0] * z[0], -1.0],
            seeds,
        )
    };
    let first = trace(make([[-0.3, -0.027], [-0.2, -0.008]]));
    let a = first.points()[2].coordinates();
    let b = first.points()[3].coordinates();
    let resumed = trace(make([[a[0], a[1]], [b[0], b[1]]]));
    assert!(dot(first.points()[3].tangent(), resumed.points()[1].tangent()) > 0.99);
}
// conformance: continuation tests prove folds, bounded correction, restart, and refusal evidence.
