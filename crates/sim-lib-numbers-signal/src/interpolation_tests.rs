// conformance: generated analytic periodic functions under explicit DFT-series conventions.

use std::f64::consts::TAU;

use super::*;

fn periodic_value(time: f64) -> f64 {
    1.0 + 2.0 * (TAU * time).cos() - 0.5 * (2.0 * TAU * time).sin()
}

fn unitary_bins(len: usize) -> Vec<(f64, f64)> {
    let samples = (0..len)
        .map(|index| (periodic_value(index as f64 / len as f64), 0.0))
        .collect::<Vec<_>>();
    let mut plan = TransformPlan::new(TransformKind::Dft, len);
    plan.normalization = Normalization::Orthonormal;
    let SignalBuffer::Complex(bins) = transform(&plan, SignalView::Complex(&samples)).unwrap()
    else {
        panic!("complex DFT expected")
    };
    bins.as_slice().to_vec()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn dft_interpolation_matches_generated_periodic_function_between_samples() {
    let bins = unitary_bins(16);
    let plan = DftSeriesPlan {
        normalization: Normalization::Orthonormal,
        ..DftSeriesPlan::default()
    };
    let coordinates = [0.1375, 0.375, 1.1375];
    let result = dft_interpolate(&bins, &coordinates, &plan).unwrap();
    for (actual, coordinate) in result.values.iter().zip(coordinates) {
        assert_close(actual.0, periodic_value(coordinate), 1.0e-11);
        assert_close(actual.1, 0.0, 1.0e-11);
    }
    assert_eq!(result.report.work_units, 48);
    assert_eq!(result.report.endpoint, EndpointConvention::Excluded);
    assert_eq!(result.report.periodicity, Periodicity::Wrap);
}

#[test]
fn dft_integration_and_endpoint_policy_are_explicit() {
    let bins = unitary_bins(16);
    let mut plan = DftSeriesPlan {
        normalization: Normalization::Orthonormal,
        periodicity: Periodicity::PrincipalPeriod,
        endpoint: EndpointConvention::Included,
        ..DftSeriesPlan::default()
    };
    let integral = dft_integrate(&bins, 0.0, 1.0, &plan).unwrap();
    assert_close(integral.value.0, 1.0, 1.0e-11);
    assert_close(integral.value.1, 0.0, 1.0e-11);

    plan.endpoint = EndpointConvention::Excluded;
    assert!(matches!(
        dft_interpolate(&bins, &[1.0], &plan),
        Err(SignalError::InvalidPolicy {
            policy: "DFT interpolation coordinate",
            ..
        })
    ));
}

#[test]
fn single_bin_matches_complete_reference_dft() {
    let samples = (0..17)
        .map(|index| {
            let value = periodic_value(index as f64 / 17.0);
            (value, 0.25 * value)
        })
        .collect::<Vec<_>>();
    let full = reference_dft(
        &samples,
        Direction::Forward,
        SignConvention::NegativeForward,
    )
    .unwrap();
    for bin in [0, 1, 2, 9, 16] {
        let single = dft_bin(
            &samples,
            bin,
            Normalization::Inverse,
            SignConvention::NegativeForward,
        )
        .unwrap();
        assert_close(single.0, full[bin].0, 1.0e-11);
        assert_close(single.1, full[bin].1, 1.0e-11);
    }
}
