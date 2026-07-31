use super::*;

// conformance: wavelet inverse and polynomial-annihilation laws.

fn assert_close(left: &[f64], right: &[f64], tolerance: f64) {
    assert_eq!(left.len(), right.len());
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "sample {index}: {left} != {right}"
        );
    }
}

#[test]
fn every_wavelet_and_boundary_round_trips_odd_multilevel_signal() {
    let signal = [0.5, -1.0, 2.5, 4.0, -3.25, 1.5, 0.125, 7.0, -2.0];
    for wavelet in [Wavelet::Haar, Wavelet::LeGall53] {
        for boundary in [
            BoundaryMode::Periodic,
            BoundaryMode::Symmetric,
            BoundaryMode::Zero,
        ] {
            let plan = WaveletPlan {
                wavelet,
                levels: 3,
                boundary,
            };
            let coefficients = dwt(&signal, &plan).unwrap();
            assert_eq!(coefficients.levels.len(), 3);
            assert_eq!(coefficients.levels[0].input_len, signal.len());
            assert_close(&idwt(&coefficients, &plan).unwrap(), &signal, 1e-12);
        }
    }
}

#[test]
fn legall_detail_annihilates_a_linear_polynomial() {
    let signal = (0..9)
        .map(|index| 2.0 * index as f64 - 3.0)
        .collect::<Vec<_>>();
    let plan = WaveletPlan {
        wavelet: Wavelet::LeGall53,
        levels: 1,
        boundary: BoundaryMode::Symmetric,
    };
    let coefficients = dwt(&signal, &plan).unwrap();
    assert!(
        coefficients.levels[0]
            .detail
            .iter()
            .all(|value| value.abs() <= 1e-12)
    );
}

#[test]
fn wavelet_plans_and_coefficients_fail_closed() {
    let plan = WaveletPlan::new(Wavelet::Haar, 0);
    assert!(matches!(
        dwt(&[1.0, 2.0], &plan),
        Err(SignalError::InvalidPolicy {
            policy: "wavelet levels",
            ..
        })
    ));

    let plan = WaveletPlan::new(Wavelet::Haar, 2);
    assert!(dwt(&[1.0, 2.0], &plan).is_err());
    assert!(dwt(&[1.0, f64::NAN], &WaveletPlan::new(Wavelet::Haar, 1)).is_err());

    let mut coefficients = dwt(&[1.0, 2.0, 3.0, 4.0], &WaveletPlan::new(Wavelet::Haar, 1)).unwrap();
    coefficients.levels[0].detail.pop();
    assert!(idwt(&coefficients, &WaveletPlan::new(Wavelet::Haar, 1)).is_err());
}
