use super::{
    BootstrapControl, StatsError, bootstrap_mean_difference_interval, median_absolute_deviation,
};

#[test]
fn median_absolute_deviation_obeys_translation_and_scale_laws() {
    let values = [1.0, 2.0, 4.0, 8.0, 16.0];
    let translated = values.map(|value| value + 100.0);
    let scaled = values.map(|value| value * 3.0);
    let mad = median_absolute_deviation(&values).unwrap();

    assert_eq!(mad, 3.0);
    assert_eq!(median_absolute_deviation(&translated).unwrap(), mad);
    assert_eq!(median_absolute_deviation(&scaled).unwrap(), mad * 3.0);
}

#[test]
fn bootstrap_is_seed_reproducible_and_translation_invariant() {
    let baseline = [10.0, 11.0, 9.0, 12.0, 8.0];
    let candidate = [8.0, 9.0, 7.0, 10.0, 6.0];
    let control = BootstrapControl::new(0x5eed, 1_000, 0.95, 10_000).unwrap();
    let first = bootstrap_mean_difference_interval(&baseline, &candidate, control).unwrap();
    let second = bootstrap_mean_difference_interval(&baseline, &candidate, control).unwrap();
    let shifted_baseline = baseline.map(|value| value + 1_000.0);
    let shifted_candidate = candidate.map(|value| value + 1_000.0);
    let shifted =
        bootstrap_mean_difference_interval(&shifted_baseline, &shifted_candidate, control).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.point_effect, -2.0);
    assert!(first.lower <= first.point_effect && first.point_effect <= first.upper);
    assert!((first.point_effect - shifted.point_effect).abs() < 1.0e-12);
    assert!((first.lower - shifted.lower).abs() < 1.0e-12);
    assert!((first.upper - shifted.upper).abs() < 1.0e-12);
}

#[test]
fn robust_statistics_reject_non_finite_input_and_unbounded_work() {
    assert!(matches!(
        median_absolute_deviation(&[1.0, f64::NAN]),
        Err(StatsError::NonFinite { index: Some(1), .. })
    ));
    let control = BootstrapControl::new(7, 100, 0.9, 399).unwrap();
    assert!(matches!(
        bootstrap_mean_difference_interval(&[1.0, 2.0], &[3.0, 4.0], control),
        Err(StatsError::WorkLimitExceeded {
            required: 400,
            limit: 399
        })
    ));
    assert!(matches!(
        bootstrap_mean_difference_interval(&[1.0, f64::INFINITY], &[2.0], control),
        Err(StatsError::NonFinite { index: Some(1), .. })
    ));
}
