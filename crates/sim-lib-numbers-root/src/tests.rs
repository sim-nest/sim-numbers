use super::*;
fn id() -> ExecutionIdentity {
    ExecutionIdentity::new("test", "sim-lib-numbers-root", "replay-1").unwrap()
}
#[test]
fn bracket_type_rejects_unproved_intervals_and_orients() {
    assert_eq!(RootBracket::new(2.0, 2.0, 0.0, -2.0).unwrap().lower, 0.0);
    assert_eq!(
        RootBracket::new(0.0, 1.0, 2.0, 2.0),
        Err(RootTermination::InvalidBracket)
    );
}
#[test]
fn bisection_and_brent_cover_endpoint_multiple_and_replay() {
    let br = establish_bracket(|x| x * x - 2.0, 0.0, 2.0).unwrap();
    let a = brent_dekker(|x| x * x - 2.0, br, ScalarPlan::default(), id());
    let b = brent_dekker(|x| x * x - 2.0, br, ScalarPlan::default(), id());
    assert_eq!(a, b);
    assert!(a.bracket.is_some());
    assert!(a.estimate.residual.abs() < 1e-10);
    let end = bisection(
        |x| x,
        RootBracket::new(0.0, 0.0, 2.0, 2.0).unwrap(),
        ScalarPlan::default(),
        id(),
    );
    assert_eq!(end.evidence.termination, RootTermination::EndpointRoot);
    let multiple = safeguarded_newton(
        |x| (x - 1.0).powi(2),
        |x| 2.0 * (x - 1.0),
        DerivativeSource::Analytic,
        2.0,
        None,
        ScalarPlan::default(),
        id(),
    );
    assert!(multiple.estimate.residual.abs() < 1e-10);
}
#[test]
fn scalar_failures_are_distinct() {
    let flat = safeguarded_newton(
        |_| 1.0,
        |_| 0.0,
        DerivativeSource::Analytic,
        0.0,
        None,
        ScalarPlan::default(),
        id(),
    );
    assert_eq!(flat.evidence.termination, RootTermination::FlatDerivative);
    let nf = secant(
        |x| if x > 0.5 { f64::NAN } else { x - 1.0 },
        0.0,
        0.25,
        ScalarPlan::default(),
        id(),
    );
    assert_eq!(
        nf.evidence.termination,
        RootTermination::NonFiniteEvaluation
    );
    let exhausted = secant(
        |x| x * x - 2.0,
        0.0,
        2.0,
        ScalarPlan {
            max_evaluations: 2,
            ..ScalarPlan::default()
        },
        id(),
    );
    assert_eq!(exhausted.evidence.termination, RootTermination::WorkLimit);

    let discontinuity = bisection(
        |x| 1.0 / (x - 0.1),
        establish_bracket(|x| 1.0 / (x - 0.1), 0.0, 1.0).unwrap(),
        ScalarPlan {
            step_tolerance: 1e-6,
            discontinuity_ratio: 1.01,
            ..ScalarPlan::default()
        },
        id(),
    );
    assert_eq!(
        discontinuity.evidence.termination,
        RootTermination::DiscontinuousSignChange
    );
}
#[test]
fn vector_newton_handles_scaling_and_reports_rank() {
    let out = damped_newton(
        |x| vec![1e6 * (x[0] - 1.0), 1e-6 * (x[1] - 2.0)],
        |_| vec![1e6, 0.0, 0.0, 1e-6],
        JacobianSource::Analytic,
        vec![0.0, 0.0],
        VectorPlan {
            rank_cutoff: 1e-14,
            ..VectorPlan::default()
        },
        id(),
    );
    assert_eq!(out.evidence.termination, RootTermination::ResidualConverged);
    assert_eq!(out.ranks, [2]);
    let singular = damped_newton(
        |x| vec![x[0] + x[1] - 1.0, 2.0 * x[0] + 2.0 * x[1] - 2.0],
        |_| vec![1.0, 1.0, 2.0, 2.0],
        JacobianSource::Analytic,
        vec![0.0, 0.0],
        VectorPlan::default(),
        id(),
    );
    assert_eq!(singular.evidence.termination, RootTermination::RankLoss);
}
#[test]
fn finite_difference_and_broyden_provenance_survive() {
    let fd = FiniteDifferencePlan {
        absolute_step: 1e-8,
        relative_step: 1e-7,
    };
    let out = finite_difference_vector_newton(
        |x| vec![x[0] * x[0] - 2.0],
        vec![1.0],
        fd,
        VectorPlan::default(),
        id(),
    );
    assert_eq!(out.jacobian_source, JacobianSource::FiniteDifference(fd));
    assert!(out.evidence.derivative_evaluations > 0);
    let b = broyden(
        |x| vec![x[0] * x[0] - 2.0],
        |x| vec![2.0 * x[0]],
        JacobianSource::Analytic,
        vec![1.0],
        VectorPlan {
            broyden_reset_interval: 2,
            ..VectorPlan::default()
        },
        id(),
    );
    assert!(b.jacobian_resets.len() > 1);
}
