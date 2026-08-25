use super::*;
use std::sync::Arc;

fn analytic_decay(lambda: f64) -> ImplicitProblem {
    ImplicitProblem::Ode {
        rhs: Arc::new(move |_, y, out| {
            out[0] = lambda * y[0];
            Ok(())
        }),
        jacobian: JacobianStrategy::Analytic(Arc::new(move |_, _, out| {
            out[0] = lambda;
            Ok(())
        })),
    }
}
fn plan(step: f64) -> RadauPlan {
    RadauPlan {
        initial_step: step,
        max_step: step,
        relative_tolerance: 1e-8,
        absolute_tolerance: 1e-10,
        ..RadauPlan::default()
    }
}

#[test]
fn observed_fifth_order_and_stiff_decay() {
    let exact = (-1.0_f64).exp();
    let coarse = solve_radau(&analytic_decay(-1.0), (0.0, 1.0), &[1.0], &plan(0.2), &[]).unwrap();
    let fine = solve_radau(&analytic_decay(-1.0), (0.0, 1.0), &[1.0], &plan(0.1), &[]).unwrap();
    let ratio = (coarse.state[0] - exact).abs() / (fine.state[0] - exact).abs();
    assert!(ratio > 20.0, "observed refinement ratio {ratio}");
    let stiff = solve_radau(
        &analytic_decay(-1000.0),
        (0.0, 1.0),
        &[1.0],
        &plan(0.1),
        &[],
    )
    .unwrap();
    assert!(
        stiff.state[0].abs() < 1e-8,
        "L-stable decay {}",
        stiff.state[0]
    );
}

#[test]
fn index_one_constraint_is_retained_and_ode_is_not_a_dae() {
    // y0'=-y0, y1-y0^2=0. The stage Jacobian is dF/dy+alpha*dF/dydot.
    let problem = ImplicitProblem::Residual {
        residual: Arc::new(|_, y, yd, out| {
            out[0] = yd[0] + y[0];
            out[1] = y[1] - y[0] * y[0];
            Ok(())
        }),
        jacobian: Arc::new(|_, y, _, alpha, out| {
            out.copy_from_slice(&[1.0 + alpha, 0.0, -2.0 * y[0], 1.0]);
            Ok(())
        }),
        differential: vec![true, false],
    };
    let solved = solve_radau(&problem, (0.0, 0.5), &[1.0, 1.0], &plan(0.02), &[]).unwrap();
    assert!((solved.state[1] - solved.state[0] * solved.state[0]).abs() < 2e-6);
    let invalid = ImplicitProblem::Residual {
        residual: Arc::new(|_, _, _, _| Ok(())),
        jacobian: Arc::new(|_, _, _, _, _| Ok(())),
        differential: vec![true, true],
    };
    assert!(matches!(
        solve_radau(&invalid, (0.0, 1.0), &[1.0, 1.0], &plan(0.1), &[]),
        Err(RadauError::Invalid(_))
    ));
}

#[test]
fn finite_difference_provenance_event_accuracy_and_work_separation() {
    let problem = ImplicitProblem::Ode {
        rhs: Arc::new(|_, y, out| {
            out[0] = -40.0 * y[0];
            Ok(())
        }),
        jacobian: JacobianStrategy::FiniteDifference {
            relative_step: 1e-7,
        },
    };
    let event = EventSpec {
        function: Arc::new(|_, y| y[0] - 0.5),
        direction: EventDirection::Falling,
        terminal: true,
    };
    let solved = solve_radau(&problem, (0.0, 1.0), &[1.0], &plan(0.01), &[event]).unwrap();
    assert!((solved.events[0].time - 2.0_f64.ln() / 40.0).abs() < 2e-5);
    assert_eq!(
        solved.evidence.jacobian_source,
        JacobianSource::FiniteDifference
    );
    // Forward Euler positivity requires h<1/40: materially more accepted work
    // than the large stable Radau steps, independent of wall-clock timing.
    let mut work_plan = plan(0.1);
    work_plan.relative_tolerance = 1e-4;
    work_plan.absolute_tolerance = 1e-7;
    let implicit =
        solve_radau(&analytic_decay(-40.0), (0.0, 1.0), &[1.0], &work_plan, &[]).unwrap();
    assert!(40 > implicit.evidence.accepted_steps * 2);
}

#[test]
fn singular_stage_jacobian_refuses_and_dense_requires_convergence() {
    let singular = ImplicitProblem::MassMatrix {
        rhs: Arc::new(|_, _, out| {
            out.fill(1.0);
            Ok(())
        }),
        mass: Arc::new(|_, _, out| {
            out.fill(0.0);
            Ok(())
        }),
        jacobian: JacobianStrategy::Analytic(Arc::new(|_, _, out| {
            out.fill(0.0);
            Ok(())
        })),
        differential: vec![true],
    };
    assert!(matches!(
        solve_radau(&singular, (0.0, 1.0), &[1.0], &plan(0.1), &[]),
        Err(RadauError::SingularJacobian { .. })
    ));
    let divergent = ImplicitProblem::Ode {
        rhs: Arc::new(|_, y, out| {
            out[0] = y[0].exp();
            Ok(())
        }),
        jacobian: JacobianStrategy::Analytic(Arc::new(|_, y, out| {
            out[0] = y[0].exp();
            Ok(())
        })),
    };
    let mut bounded = plan(1.0);
    bounded.max_newton_iterations = 1;
    bounded.max_steps = 1;
    assert!(solve_radau(&divergent, (0.0, 1.0), &[10.0], &bounded, &[]).is_err());
}

#[test]
fn ad_provenance_and_robertson_scale_separation() {
    let p = ImplicitProblem::Ode {
        rhs: Arc::new(|_, y, o| {
            o[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
            o[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
            o[2] = 3e7 * y[1] * y[1];
            Ok(())
        }),
        jacobian: JacobianStrategy::AutomaticDifferentiation(Arc::new(|_, y, j| {
            j.copy_from_slice(&[
                -0.04,
                1e4 * y[2],
                1e4 * y[1],
                0.04,
                -1e4 * y[2] - 6e7 * y[1],
                -1e4 * y[1],
                0.0,
                6e7 * y[1],
                0.0,
            ]);
            Ok(())
        })),
    };
    let mut pplan = plan(1e-4);
    pplan.max_step = 1e-2;
    let s = solve_radau(&p, (0.0, 0.1), &[1.0, 0.0, 0.0], &pplan, &[]).unwrap();
    assert!((s.state.iter().sum::<f64>() - 1.0).abs() < 1e-7);
    assert!(s.state[1] < 1e-4 && s.state[2] > 1e-5);
    assert_eq!(
        s.evidence.jacobian_source,
        JacobianSource::AutomaticDifferentiation
    );
}
