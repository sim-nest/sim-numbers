use super::*;

// conformance: sampled interpolation shape, duplicate, and extrapolation policy.

fn plan(method: InterpolationMethod) -> InterpolationPlan {
    InterpolationPlan {
        method,
        ..InterpolationPlan::default()
    }
}

#[test]
fn linear_and_cubic_interpolants_preserve_affine_data() {
    let x = [0.0, 1.0, 2.5, 4.0];
    let y = x.map(|x| 2.0 * x - 1.0);
    let at = [0.25, 1.5, 3.75];
    for method in [InterpolationMethod::Linear, InterpolationMethod::Cubic] {
        let result = interpolate_samples(&x, &y, &at, plan(method)).unwrap();
        for (&coordinate, &value) in at.iter().zip(&result.values) {
            assert!((value - (2.0 * coordinate - 1.0)).abs() < 1e-10);
        }
    }
}

#[test]
fn monotone_cubic_does_not_overshoot_sample_intervals() {
    let interpolator = SampleInterpolator::new(
        &[0.0, 1.0, 2.0, 4.0],
        &[0.0, 1.0, 1.5, 3.0],
        plan(InterpolationMethod::Monotone),
    )
    .unwrap();
    let at = (0..=80)
        .map(|index| index as f64 / 20.0)
        .collect::<Vec<_>>();
    let values = interpolator.evaluate(&at).unwrap().values;
    assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(values.iter().all(|value| (0.0..=3.0).contains(value)));
}

#[test]
fn duplicate_policies_are_explicit_and_reported() {
    let x = [0.0, 1.0, 1.0, 2.0];
    let y = [0.0, 1.0, 3.0, 4.0];
    assert!(matches!(
        SampleInterpolator::new(&x, &y, InterpolationPlan::default()),
        Err(SignalError::DuplicateCoordinate { index: 2, .. })
    ));
    let interpolator = SampleInterpolator::new(
        &x,
        &y,
        InterpolationPlan {
            method: InterpolationMethod::Linear,
            duplicates: DuplicateXPolicy::Average,
            extrapolation: ExtrapolationPolicy::Reject,
        },
    )
    .unwrap();
    let result = interpolator.evaluate(&[1.0]).unwrap();
    assert_eq!(result.values, vec![2.0]);
    assert_eq!(result.report.duplicates_resolved, 1);
    assert_eq!(result.report.unique_points, 3);
}

#[test]
fn extrapolation_rejects_clamps_or_continues_endpoint_secants() {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 2.0, 3.0];
    assert!(matches!(
        interpolate_samples(&x, &y, &[-1.0], plan(InterpolationMethod::Linear)),
        Err(SignalError::OutOfDomain { index: 0, .. })
    ));
    let clamped = interpolate_samples(
        &x,
        &y,
        &[-1.0, 3.0],
        InterpolationPlan {
            method: InterpolationMethod::Monotone,
            duplicates: DuplicateXPolicy::Reject,
            extrapolation: ExtrapolationPolicy::Clamp,
        },
    )
    .unwrap();
    assert_eq!(clamped.values, vec![0.0, 3.0]);
    assert_eq!(clamped.report.extrapolated_points, 2);
    let extended = interpolate_samples(
        &x,
        &y,
        &[-1.0, 3.0],
        InterpolationPlan {
            method: InterpolationMethod::Cubic,
            duplicates: DuplicateXPolicy::Reject,
            extrapolation: ExtrapolationPolicy::Linear,
        },
    )
    .unwrap();
    assert_eq!(extended.values, vec![-2.0, 4.0]);
}

#[test]
fn interpolation_rejects_unsorted_and_non_finite_inputs() {
    assert!(
        SampleInterpolator::new(
            &[0.0, 2.0, 1.0],
            &[0.0, 1.0, 2.0],
            InterpolationPlan::default(),
        )
        .is_err()
    );
    assert!(
        interpolate_samples(
            &[0.0, 1.0],
            &[0.0, f64::NAN],
            &[0.5],
            InterpolationPlan::default(),
        )
        .is_err()
    );
}
