// conformance: generated analytic tones, phase, instantaneous frequency, and envelopes.

use std::f64::consts::{PI, TAU};

use super::*;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn hilbert_construction_matches_a_generated_periodic_tone_for_every_scaling() {
    let len = 64;
    let bin = 5;
    let samples = (0..len)
        .map(|index| (TAU * bin as f64 * index as f64 / len as f64).cos())
        .collect::<Vec<_>>();
    for normalization in [
        Normalization::None,
        Normalization::Forward,
        Normalization::Inverse,
        Normalization::Orthonormal,
    ] {
        let plan = AnalyticSignalPlan {
            normalization,
            ..AnalyticSignalPlan::default()
        };
        let analytic = analytic_signal(&samples, &plan).unwrap();
        for (index, (real, imag)) in analytic.samples.iter().copied().enumerate() {
            let phase = TAU * bin as f64 * index as f64 / len as f64;
            assert_close(real, phase.cos(), 1.0e-11);
            assert_close(imag, phase.sin(), 1.0e-11);
        }
        let envelope = analytic_envelope(&analytic.samples).unwrap();
        assert!(envelope.iter().all(|value| (*value - 1.0).abs() < 1.0e-11));
    }
}

#[test]
fn unwrapped_phase_yields_interval_centered_instantaneous_frequency() {
    let sample_rate_hz = 64.0;
    let frequency_hz = 5.0;
    let analytic = (0..64)
        .map(|index| {
            let phase = TAU * frequency_hz * index as f64 / sample_rate_hz;
            (phase.cos(), phase.sin())
        })
        .collect::<Vec<_>>();
    let result = instantaneous_frequency(&analytic, sample_rate_hz).unwrap();
    assert_eq!(result.frequency_hz.len(), 63);
    assert_close(result.time_seconds[0], 0.5 / sample_rate_hz, 1.0e-14);
    for frequency in result.frequency_hz {
        assert_close(frequency, frequency_hz, 1.0e-12);
    }

    let wrapped = [0.75 * PI, -0.75 * PI, -0.5 * PI];
    let unwrapped = unwrap_phase(&wrapped, PI).unwrap();
    assert_close(unwrapped[1], 1.25 * PI, 1.0e-14);
    assert_close(unwrapped[2], 1.5 * PI, 1.0e-14);
}

#[test]
fn attack_release_envelope_is_finite_and_directional() {
    let samples = [0.0, 1.0, 1.0, 0.0, 0.0];
    let plan = EnvelopeFollowerPlan {
        sample_rate_hz: 10.0,
        attack_seconds: 0.1,
        release_seconds: 1.0,
        initial_value: 0.0,
    };
    let envelope = envelope_follow(&samples, &plan).unwrap();
    assert_eq!(envelope[0], 0.0);
    assert!(envelope[1] > 0.5);
    assert!(envelope[2] > envelope[1]);
    assert!(envelope[3] < envelope[2]);
    assert!(envelope[4] < envelope[3]);
    assert!(envelope[3] > 0.5);
}
