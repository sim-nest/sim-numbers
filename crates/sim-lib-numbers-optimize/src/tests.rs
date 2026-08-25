use super::*;

fn limits() -> Limits {
    Limits {
        evaluations: 2_000,
        iterations: 200,
        memory_bytes: 1_000_000,
    }
}

#[test]
fn brent_finds_known_minimum_and_returns_bracket() {
    let out = minimize_scalar(|x| (x - 2.0).powi(2), -4.0, 7.0, 1e-10, limits()).unwrap();
    assert!((out.minimizer - 2.0).abs() < 1e-6);
    assert!(out.final_bracket.0 <= out.minimizer && out.minimizer <= out.final_bracket.1);
    assert_eq!(out.termination, Termination::Converged);
}

#[test]
fn scalar_reports_non_finite_and_work_limit_honestly() {
    assert_eq!(
        minimize_scalar(|_| f64::NAN, 0.0, 1.0, 1e-8, limits())
            .unwrap()
            .termination,
        Termination::NonFinite
    );
    let tiny = Limits {
        evaluations: 1,
        iterations: 1,
        memory_bytes: 0,
    };
    assert_eq!(
        minimize_scalar(|x| (x - 0.3).abs(), 0.0, 1.0, 1e-15, tiny)
            .unwrap()
            .termination,
        Termination::WorkLimit
    );
}

fn objective_plan(bounds: Bounds) -> ObjectivePlan {
    ObjectivePlan {
        bounds,
        scale: vec![1.0, 1.0],
        derivative: DerivativeSource::Analytic,
        policy: StepPolicy::ProjectedBfgs,
        tolerances: Tolerances::default(),
        limits: limits(),
        initial_radius: 1.0,
    }
}

#[test]
fn projected_objective_finds_active_bound_not_false_stationary_point() {
    let bounds = Bounds::new(vec![0.0, -4.0], vec![1.0, 4.0]).unwrap();
    let out = minimize(
        |x| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2),
        Some(|x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 3.0);
            g[1] = 2.0 * (x[1] + 1.0)
        }),
        vec![0.2, 3.0],
        &objective_plan(bounds),
    )
    .unwrap();
    assert!(
        (out.point[0] - 1.0).abs() < 1e-7 && (out.point[1] + 1.0).abs() < 1e-5,
        "{out:?}"
    );
    assert!(matches!(
        out.termination,
        Termination::Converged | Termination::BoundaryConverged
    ));
}

#[test]
fn finite_difference_fallback_and_bad_scaling_are_explicit() {
    let mut p = objective_plan(Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).unwrap());
    p.derivative = DerivativeSource::FiniteDifference;
    let out = minimize::<_, fn(&[f64], &mut [f64])>(
        |x| x[0] * x[0] + x[1] * x[1],
        None,
        vec![2.0, -3.0],
        &p,
    )
    .unwrap();
    assert!(out.value < 1e-8, "{out:?}");
    p.scale[0] = 0.0;
    assert!(matches!(
        minimize::<_, fn(&[f64], &mut [f64])>(|_| 0.0, None, vec![0.0, 0.0], &p),
        Err(Error::InvalidPlan(_))
    ));
}

fn ls_plan(bounds: Option<Bounds>, policy: StepPolicy) -> LeastSquaresPlan {
    LeastSquaresPlan {
        bounds,
        variable_scale: vec![1.0, 1.0],
        residual_scale: vec![1.0, 1.0],
        derivative: DerivativeSource::Analytic,
        policy,
        tolerances: Tolerances::default(),
        limits: limits(),
        initial_damping: 1e-3,
    }
}

#[test]
fn lm_fits_nonlinear_residuals() {
    let out = least_squares(
        |x| vec![x[0] * x[0] - 4.0, x[1] - 3.0],
        |x| vec![vec![2.0 * x[0], 0.0], vec![0.0, 1.0]],
        vec![1.0, 0.0],
        &ls_plan(None, StepPolicy::LevenbergMarquardt),
    )
    .unwrap();
    assert!(
        (out.point[0] - 2.0).abs() < 1e-5 && (out.point[1] - 3.0).abs() < 1e-5,
        "{out:?}"
    );
}

#[test]
fn bounded_least_squares_uses_reflective_path() {
    let bounds = Bounds::new(vec![0.0, 0.0], vec![1.0, 2.0]).unwrap();
    let out = least_squares(
        |x| vec![x[0] - 5.0, x[1] - 1.0],
        |_| vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![0.2, 0.2],
        &ls_plan(Some(bounds), StepPolicy::TrustRegionReflective),
    )
    .unwrap();
    assert!(
        (out.point[0] - 1.0).abs() < 1e-7 && (out.point[1] - 1.0).abs() < 1e-5,
        "{out:?}"
    );
    assert_eq!(out.active, vec![0]);
}

#[test]
fn linear_rank_and_covariance_unavailability_are_evidence() {
    let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    let b = vec![1.0, 2.0];
    let out = linear_least_squares(
        &a,
        &b,
        Bounds::new(vec![-10.0, -10.0], vec![10.0, 10.0]).unwrap(),
        1e-10,
        limits(),
        true,
    )
    .unwrap();
    assert_eq!(out.rank, 1);
    assert_eq!(
        out.covariance,
        Covariance::Unavailable(CovarianceUnavailable::RankDeficient)
    );
}

#[test]
fn bounded_linear_solver_uses_active_set_and_certified_svd_rank() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let b = vec![4.0, 1.0, 5.0];
    let out = linear_least_squares(
        &a,
        &b,
        Bounds::new(vec![0.0, 0.0], vec![2.0, 10.0]).unwrap(),
        1e-10,
        limits(),
        true,
    )
    .unwrap();
    assert!((out.point[0] - 2.0).abs() < 1e-9, "{out:?}");
    assert_eq!(out.active, vec![0]);
    assert_eq!(out.rank, 2);
    assert!(matches!(out.covariance, Covariance::Available(_)));
}

#[test]
fn flat_nonsmooth_and_memory_limits_terminate() {
    let flat = minimize_scalar(|_| 1.0, -1.0, 1.0, 1e-8, limits()).unwrap();
    assert!(matches!(
        flat.termination,
        Termination::Converged | Termination::WorkLimit
    ));
    let nonsmooth = minimize_scalar(|x| x.abs(), -1.0, 2.0, 1e-9, limits()).unwrap();
    assert!(nonsmooth.minimizer.abs() < 1e-5);
    let mut p = objective_plan(Bounds::new(vec![-1.0, -1.0], vec![1.0, 1.0]).unwrap());
    p.limits.memory_bytes = 1;
    assert_eq!(
        minimize(
            |x| x[0],
            Some(|_: &[f64], g: &mut [f64]| g.fill(1.0)),
            vec![0.0, 0.0],
            &p
        )
        .unwrap()
        .termination,
        Termination::WorkLimit
    );
}

#[cfg(feature = "assignment")]
#[test]
fn assignment_adapter_preserves_certificate_and_receipt() {
    use sim_lib_discrete_graph::{AssignmentPolicy, CostMatrix, verify_assignment};
    let costs = vec![vec![1.0, 9.0], vec![8.0, 2.0]];
    let policy = AssignmentPolicy::new(vec![20.0, 20.0], vec![20.0, 20.0]);
    let out = crate::assignment::assign(costs.clone(), policy.clone()).unwrap();
    verify_assignment(&CostMatrix::try_from(costs).unwrap(), &policy, &out).unwrap();
    assert!(!format!("{:?}", out.certificate).is_empty());
    assert!(out.receipt.work_used > 0);
}
