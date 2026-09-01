use super::{
    BootstrapControl, ClusterSample, IsotonicPoint, RegisteredLook, RegisteredLookSequence,
    StatsError, ThresholdReadout, bootstrap_mean_difference_interval, clustered_bootstrap_interval,
    exact_binary_interval, fit_isotonic, paired_bootstrap_interval,
};

#[test]
fn exact_binary_reference_and_exhaustive_tiny_cases() {
    let interval = exact_binary_interval(5, 10, 0.95).unwrap();
    assert!((interval.lower - 0.187_086_028_447_398_52).abs() < 1.0e-12);
    assert!((interval.upper - 0.812_913_971_552_601_5).abs() < 1.0e-12);
    for trials in 1..=12 {
        let mut previous_lower = 0.0;
        let mut previous_upper = 0.0;
        for successes in 0..=trials {
            let value = exact_binary_interval(successes, trials, 0.9).unwrap();
            assert!(0.0 <= value.lower && value.lower <= value.upper && value.upper <= 1.0);
            assert!(value.lower >= previous_lower);
            assert!(value.upper >= previous_upper);
            previous_lower = value.lower;
            previous_upper = value.upper;
        }
    }
}

#[test]
fn paired_and_clustered_bootstraps_replay_and_preserve_identity() {
    let control = BootstrapControl::new(91, 400, 0.9, 10_000).unwrap();
    let pairs = [(1.0, 2.0), (4.0, 3.0), (2.0, 4.0)];
    assert_eq!(
        paired_bootstrap_interval(&pairs, control),
        paired_bootstrap_interval(&pairs, control)
    );

    let clusters = vec![
        ClusterSample {
            id: 20,
            pairs: vec![(4.0, 7.0), (2.0, 3.0)],
        },
        ClusterSample {
            id: 10,
            pairs: vec![(1.0, 1.5), (3.0, 4.0)],
        },
        ClusterSample {
            id: 30,
            pairs: vec![(8.0, 7.0)],
        },
    ];
    let mut shuffled = clusters.clone();
    shuffled.reverse();
    for cluster in &mut shuffled {
        cluster.pairs.reverse();
    }
    assert_eq!(
        clustered_bootstrap_interval(&clusters, 3, control),
        clustered_bootstrap_interval(&shuffled, 3, control)
    );
    assert!(matches!(
        clustered_bootstrap_interval(&clusters[..2], 3, control),
        Err(StatsError::InsufficientInput { .. })
    ));
}

#[test]
fn registered_looks_seal_the_optional_stopping_budget() {
    let sequence = RegisteredLookSequence::new(
        vec![
            RegisteredLook {
                samples: 4,
                alpha: 0.01,
            },
            RegisteredLook {
                samples: 8,
                alpha: 0.02,
            },
        ],
        0.03,
    )
    .unwrap();
    assert_eq!(
        sequence.interval(&[0.0; 3]).unwrap_err(),
        StatsError::InvalidControl {
            field: "observations",
            reason: "sample count is not a registered look"
        }
    );
    let first = sequence.interval(&[0.0, 1.0, 1.0, 1.0]).unwrap();
    let second = sequence
        .interval(&[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .unwrap();
    assert_eq!(first.alpha_spent + second.alpha_spent, first.total_budget);
    assert!(first.lower <= first.mean && first.mean <= first.upper);
    assert!(second.lower <= second.mean && second.mean <= second.upper);
}

#[test]
fn weighted_pav_is_monotone_permutation_invariant_and_censored() {
    let points = [
        IsotonicPoint {
            level: 2.0,
            value: 0.2,
            weight: 1.0,
        },
        IsotonicPoint {
            level: 1.0,
            value: 0.8,
            weight: 3.0,
        },
        IsotonicPoint {
            level: 3.0,
            value: 0.9,
            weight: 1.0,
        },
    ];
    let fit = fit_isotonic(&points).unwrap();
    let mut reversed = points;
    reversed.reverse();
    assert_eq!(fit, fit_isotonic(&reversed).unwrap());
    assert!(fit.fitted.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(
        fit.fitted[0..2]
            .iter()
            .all(|value| (*value - 0.65).abs() < 1.0e-15)
    );
    assert_eq!(fit.fitted[2], 0.9);
    assert_eq!(
        fit.threshold(0.1).unwrap(),
        ThresholdReadout::BelowTestedRange
    );
    assert_eq!(
        fit.threshold(0.8).unwrap(),
        ThresholdReadout::Observed { level: 3.0 }
    );
    assert_eq!(
        fit.threshold(1.0).unwrap(),
        ThresholdReadout::AboveTestedRange
    );
    assert!(fit.normalized_area.is_some());
    assert_eq!(fit_isotonic(&points[..1]).unwrap().normalized_area, None);
}

#[test]
fn malformed_inputs_and_work_are_rejected_without_changing_independent_bootstrap() {
    assert!(exact_binary_interval(2, 1, 0.95).is_err());
    assert!(exact_binary_interval(1, 2, f64::NAN).is_err());
    assert!(
        paired_bootstrap_interval(
            &[(0.0, f64::INFINITY)],
            BootstrapControl::new(1, 2, 0.9, 10).unwrap()
        )
        .is_err()
    );
    assert!(
        fit_isotonic(&[IsotonicPoint {
            level: 0.0,
            value: 1.0,
            weight: 0.0
        }])
        .is_err()
    );
    let too_small = BootstrapControl::new(1, 10, 0.9, 9).unwrap();
    assert!(matches!(
        paired_bootstrap_interval(&[(0.0, 1.0)], too_small),
        Err(StatsError::WorkLimitExceeded { .. })
    ));

    let independent = BootstrapControl::new(0x5eed, 1_000, 0.95, 10_000).unwrap();
    let fixture =
        bootstrap_mean_difference_interval(&[10.0, 11.0, 12.0], &[8.0, 9.0, 10.0], independent)
            .unwrap();
    assert_eq!(fixture.seed, 0x5eed);
    assert_eq!(fixture.resamples, 1_000);
    assert_eq!(fixture.point_effect, -2.0);
}
// conformance: decision tests prove registered looks and bounded statistical evidence.
