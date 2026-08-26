use super::*;
fn plan(method: InterpolationMethod) -> GridInterpolationPlan {
    GridInterpolationPlan {
        axis_order: vec![1, 0],
        interpolation: InterpolationPlan {
            method,
            ..InterpolationPlan::default()
        },
        max_work: 100,
        max_memory_values: 100,
    }
}
#[test]
fn exact_nodes_and_bilinear_plane() {
    let grid = RegularGridInterpolator::new(
        vec![vec![0.0, 1.0], vec![0.0, 2.0]],
        vec![2, 2],
        vec![0.0, 4.0, 1.0, 5.0],
        plan(InterpolationMethod::Linear),
    )
    .unwrap();
    assert_eq!(grid.evaluate(&[1.0, 2.0]).unwrap().value, 5.0);
    assert_eq!(grid.evaluate(&[0.25, 0.5]).unwrap().value, 1.25);
}
#[test]
fn axis_reversal_is_storage_stable() {
    let grid = RegularGridInterpolator::new(
        vec![vec![1.0, 0.0], vec![0.0, 2.0]],
        vec![2, 2],
        vec![1.0, 5.0, 0.0, 4.0],
        plan(InterpolationMethod::Linear),
    )
    .unwrap();
    assert_eq!(grid.evaluate(&[0.25, 0.5]).unwrap().value, 1.25);
}
#[test]
fn extrapolation_and_limits_fail_closed() {
    let axes = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
    let shape = vec![2, 2];
    let mut low = plan(InterpolationMethod::Linear);
    low.max_work = 7;
    assert!(matches!(
        RegularGridInterpolator::new(axes.clone(), shape.clone(), vec![0.0; 4], low),
        Err(SignalError::WorkLimit { .. })
    ));
    let grid =
        RegularGridInterpolator::new(axes, shape, vec![0.0; 4], plan(InterpolationMethod::Linear))
            .unwrap();
    assert!(matches!(
        grid.evaluate(&[2.0, 0.0]),
        Err(SignalError::OutOfDomain { .. })
    ));
}
#[test]
fn degenerate_duplicate_and_false_monotonicity_claim_are_explicit() {
    assert!(
        RegularGridInterpolator::new(
            vec![vec![0.0], vec![0.0, 1.0]],
            vec![1, 2],
            vec![0.0; 2],
            plan(InterpolationMethod::Linear)
        )
        .is_err()
    );
    let grid = RegularGridInterpolator::new(
        vec![vec![0.0, 1.0], vec![0.0, 1.0]],
        vec![2, 2],
        vec![0.0, 0.0, 0.0, 1.0],
        plan(InterpolationMethod::Monotone),
    )
    .unwrap();
    assert!(
        !grid
            .evaluate(&[0.5, 0.5])
            .unwrap()
            .report
            .multidimensional_monotonicity_claimed
    );
}
// conformance: regular-grid tests prove ranked interpolation geometry and bounded work.
