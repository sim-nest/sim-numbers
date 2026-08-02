use super::*;

// conformance: Savitzky-Golay polynomial laws and Toeplitz diagnostics.

#[test]
fn savitzky_golay_preserves_fitted_polynomials() {
    let filter = savitzky_golay(SavitzkyGolaySpec {
        window_length: 7,
        polynomial_order: 3,
        ..SavitzkyGolaySpec::default()
    })
    .unwrap();
    let samples = (-10..=10)
        .map(|x| {
            let x = f64::from(x);
            2.0 - 0.5 * x + 0.25 * x * x - 0.125 * x * x * x
        })
        .collect::<Vec<_>>();
    let smoothed = apply_savitzky_golay(&samples, &filter, BoundaryMode::Symmetric).unwrap();
    for index in 3..samples.len() - 3 {
        assert!((smoothed[index] - samples[index]).abs() < 1e-9);
    }
}

#[test]
fn derivatives_include_factorial_and_physical_spacing() {
    let spacing = 0.25;
    let filter = savitzky_golay(SavitzkyGolaySpec {
        window_length: 7,
        polynomial_order: 3,
        derivative_order: 2,
        sample_spacing: spacing,
        ..SavitzkyGolaySpec::default()
    })
    .unwrap();
    let samples = (-10..=10)
        .map(|index| {
            let x = f64::from(index) * spacing;
            3.0 * x * x + 2.0 * x + 1.0
        })
        .collect::<Vec<_>>();
    let derivative = apply_savitzky_golay(&samples, &filter, BoundaryMode::Zero).unwrap();
    for value in &derivative[3..derivative.len() - 3] {
        assert!((*value - 6.0).abs() < 1e-9, "{value}");
    }
}

#[test]
fn toeplitz_solver_reports_pivots_and_residual() {
    let solution = solve_toeplitz(
        &[4.0, 1.0, 0.5],
        &[4.0, 2.0, -1.0],
        &[5.5, 6.0, 3.5],
        ToeplitzPlan::default(),
    )
    .unwrap();
    assert!((solution.values[0] - 1.0).abs() < 1e-12);
    assert!((solution.values[1] - 1.0).abs() < 1e-12);
    assert!((solution.values[2] - 0.5).abs() < 1e-12);
    assert!(solution.diagnostics.reciprocal_pivot_condition > 0.0);
    assert!(solution.diagnostics.residual_l2 < 1e-12);
}

#[test]
fn toeplitz_singularity_returns_threshold_diagnostics() {
    let error = solve_toeplitz(
        &[1.0, 1.0],
        &[1.0, 1.0],
        &[2.0, 2.0],
        ToeplitzPlan::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SignalError::SingularSystem {
            operation: "Toeplitz",
            step: 1,
            pivot_magnitude,
            threshold,
        } if pivot_magnitude <= threshold
    ));
}

#[test]
fn smoothing_policies_fail_closed() {
    assert!(
        savitzky_golay(SavitzkyGolaySpec {
            window_length: 4,
            ..SavitzkyGolaySpec::default()
        })
        .is_err()
    );
    assert!(
        savitzky_golay(SavitzkyGolaySpec {
            derivative_order: 3,
            polynomial_order: 2,
            ..SavitzkyGolaySpec::default()
        })
        .is_err()
    );
    assert!(
        solve_toeplitz(
            &[1.0, 0.0],
            &[2.0, 0.0],
            &[1.0, 1.0],
            ToeplitzPlan::default(),
        )
        .is_err()
    );
}
