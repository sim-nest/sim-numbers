// conformance: reference windows and bounded classical spectrum estimators.

use std::f64::consts::TAU;

use super::*;
use crate::multitaper::dpss_tapers;

const TOLERANCE: f64 = 1e-10;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

fn rectangular() -> WindowSpec {
    WindowSpec::new(WindowFunction::Rectangular)
}

fn tone(len: usize, sample_rate_hz: f64, frequency_hz: f64, amplitude: f64) -> Vec<f64> {
    (0..len)
        .map(|index| amplitude * (TAU * frequency_hz * index as f64 / sample_rate_hz).sin())
        .collect()
}

#[test]
fn reference_windows_match_known_coefficients_and_invariants() {
    let hann = WindowSpec::new(WindowFunction::Hann).generate(5).unwrap();
    for (actual, expected) in hann.samples.iter().zip([0.0, 0.5, 1.0, 0.5, 0.0]) {
        assert_close(*actual, expected, TOLERANCE);
    }
    let hamming = WindowSpec::new(WindowFunction::Hamming)
        .generate(5)
        .unwrap();
    for (actual, expected) in hamming.samples.iter().zip([0.08, 0.54, 1.0, 0.54, 0.08]) {
        assert_close(*actual, expected, TOLERANCE);
    }
    let blackman = WindowSpec::new(WindowFunction::Blackman { alpha: 0.16 })
        .generate(5)
        .unwrap();
    assert_close(blackman.samples[0], 0.0, TOLERANCE);
    assert_close(blackman.samples[2], 1.0, TOLERANCE);
    let blackman_harris = WindowSpec::new(WindowFunction::BlackmanHarris)
        .generate(5)
        .unwrap();
    assert_close(blackman_harris.samples[0], 0.00006, TOLERANCE);
    assert_close(blackman_harris.samples[2], 1.0, TOLERANCE);
    let kaiser = WindowSpec::new(WindowFunction::Kaiser { beta: 8.6 })
        .generate(9)
        .unwrap();
    assert_close(kaiser.samples[0], kaiser.samples[8], TOLERANCE);
    assert_close(kaiser.samples[4], 1.0, TOLERANCE);
    assert!(
        kaiser
            .samples
            .windows(2)
            .take(4)
            .all(|pair| pair[0] < pair[1])
    );
    let rectangular = rectangular().generate(7).unwrap();
    assert!(rectangular.samples.iter().all(|value| *value == 1.0));

    let explicit = WindowSpec::new(WindowFunction::Explicit(vec![1.0, 2.0, 1.0]))
        .generate(3)
        .unwrap();
    assert_eq!(explicit.samples, vec![1.0, 2.0, 1.0]);
    assert_close(explicit.metrics.coherent_gain, 4.0 / 3.0, TOLERANCE);
    assert_close(explicit.metrics.energy, 6.0, TOLERANCE);
}

#[test]
fn window_normalizations_report_raw_and_applied_scale() {
    let mut coherent = WindowSpec::new(WindowFunction::Hann);
    coherent.sampling = WindowSampling::Periodic;
    coherent.normalization = WindowNormalization::CoherentGain;
    let coherent = coherent.generate(64).unwrap();
    assert_close(coherent.metrics.raw_coherent_gain, 0.5, TOLERANCE);
    assert_close(coherent.metrics.normalization_scale, 2.0, TOLERANCE);
    assert_close(coherent.metrics.coherent_gain, 1.0, TOLERANCE);
    assert_close(
        coherent.metrics.equivalent_noise_bandwidth_bins.unwrap(),
        1.5,
        TOLERANCE,
    );

    let mut energy = WindowSpec::new(WindowFunction::Hann);
    energy.sampling = WindowSampling::Periodic;
    energy.normalization = WindowNormalization::UnitEnergy;
    let energy = energy.generate(64).unwrap();
    assert_close(energy.metrics.raw_energy, 24.0, TOLERANCE);
    assert_close(energy.metrics.energy, 1.0, TOLERANCE);
}

#[test]
fn periodogram_recovers_tone_power_and_parseval_noise_energy() {
    let sample_rate_hz = 1024.0;
    let samples = tone(256, sample_rate_hz, 128.0, 2.0);
    let mut plan = PeriodogramPlan::new(sample_rate_hz, samples.len());
    plan.window = rectangular();
    plan.scaling = SpectrumScalingKind::Power;
    let estimate = periodogram(&samples, &plan).unwrap();
    let peak = estimate
        .power
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_close(estimate.frequency[peak], 128.0, TOLERANCE);
    assert_close(estimate.power[peak], 2.0, TOLERANCE);
    assert_eq!(estimate.evidence.degrees_of_freedom, 2.0);
    assert_close(
        estimate.scaling.normalization_denominator,
        (samples.len() * samples.len()) as f64,
        TOLERANCE,
    );

    let noise = (0..256)
        .scan(0x1234_5678_u64, |state, _| {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            Some(((*state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0)
        })
        .collect::<Vec<_>>();
    plan.scaling = SpectrumScalingKind::Density;
    let density = periodogram(&noise, &plan).unwrap();
    let integrated = density.power.iter().sum::<f64>() * sample_rate_hz / plan.fft_len as f64;
    let mean_square = noise.iter().map(|value| value * value).sum::<f64>() / noise.len() as f64;
    assert_close(integrated, mean_square, TOLERANCE);
}

#[test]
fn welch_and_cross_spectrum_report_segments_dof_and_coherence() {
    let sample_rate_hz = 1024.0;
    let samples = tone(768, sample_rate_hz, 128.0, 1.0);
    let mut plan = WelchPlan::new(sample_rate_hz, 256);
    plan.window = rectangular();
    let estimate = welch(&samples, &plan).unwrap();
    assert_eq!(estimate.evidence.segment_count, 5);
    assert_eq!(estimate.evidence.degrees_of_freedom, 10.0);
    let peak = estimate
        .power
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_close(estimate.frequency[peak], 128.0, TOLERANCE);

    let shifted = tone(768, sample_rate_hz, 128.0, 0.25);
    let cross = cross_spectrum(&samples, &shifted, &plan).unwrap();
    assert_close(cross.coherence[peak], 1.0, TOLERANCE);
    assert_eq!(cross.evidence.estimator, EstimatorKind::CrossSpectrum);
    assert_eq!(cross.evidence.degrees_of_freedom, 10.0);
}

#[test]
fn linear_frequency_grid_uses_bounded_direct_fourier_evaluation() {
    let samples = tone(64, 64.0, 7.25, 1.0);
    let mut plan = PeriodogramPlan::new(64.0, 64);
    plan.window = rectangular();
    plan.scaling = SpectrumScalingKind::Power;
    plan.grid = FrequencyGridPolicy::Linear {
        start_hz: 6.0,
        end_hz: 8.0,
        bins: 41,
        side: SpectrumSide::OneSided,
    };
    let estimate = periodogram(&samples, &plan).unwrap();
    assert_eq!(estimate.evidence.work_units, 64 * 41);
    let peak = estimate
        .power
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert!((estimate.frequency[peak] - 7.25).abs() <= 0.05);
}

#[test]
fn slepian_tapers_are_orthonormal_concentrated_and_find_a_tone() {
    let (tapers, concentrations) = dpss_tapers(32, 2.5, 4).unwrap();
    for left in 0..tapers.len() {
        for right in 0..tapers.len() {
            let product = tapers[left]
                .iter()
                .zip(&tapers[right])
                .map(|(a, b)| a * b)
                .sum::<f64>();
            assert_close(product, if left == right { 1.0 } else { 0.0 }, 1e-8);
        }
    }
    assert!(concentrations.windows(2).all(|pair| pair[0] >= pair[1]));
    assert!(concentrations.iter().all(|value| *value > 0.9));

    let samples = tone(32, 32.0, 5.0, 1.0);
    let plan = MultitaperPlan::new(32.0, 32, 2.5, 4);
    let estimate = multitaper(&samples, &plan).unwrap();
    let peak = estimate
        .power
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert!((estimate.frequency[peak] - 5.0).abs() <= 1.0);
    assert_eq!(estimate.evidence.degrees_of_freedom, 8.0);
    assert_eq!(estimate.evidence.taper_concentrations, concentrations);
}

#[test]
fn uneven_sample_lomb_scargle_recovers_known_tone_and_scale() {
    let mut time = 0.0;
    let times = (0..80)
        .map(|index| {
            time += 0.01 + (index % 7) as f64 * 0.0003;
            time
        })
        .collect::<Vec<_>>();
    let samples = times
        .iter()
        .map(|time| 1.25 + 0.8 * (TAU * 7.3 * time).sin())
        .collect::<Vec<_>>();
    let mut plan = LombScarglePlan::new(64.0, 256);
    plan.grid = FrequencyGridPolicy::Linear {
        start_hz: 1.0,
        end_hz: 15.0,
        bins: 281,
        side: SpectrumSide::OneSided,
    };
    let estimate = lomb_scargle(&times, &samples, &plan).unwrap();
    let peak = estimate
        .power
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_close(estimate.frequency[peak], 7.3, 0.05);
    assert!(estimate.power[peak] > 0.999);
    assert_eq!(
        estimate.scaling.kind,
        SpectrumScalingKind::LombScargleNormalized
    );
    assert!(estimate.scaling.normalization_denominator > 0.0);
    assert_eq!(estimate.evidence.degrees_of_freedom, 77.0);
}

#[test]
fn segment_taper_grid_and_work_limits_fail_before_execution() {
    let samples = tone(512, 512.0, 32.0, 1.0);
    let mut welch_plan = WelchPlan::new(512.0, 64);
    welch_plan.limits.max_segments = 2;
    assert!(matches!(
        welch(&samples, &welch_plan),
        Err(SignalError::InvalidPolicy {
            policy: "segment limit",
            ..
        })
    ));

    let mut periodogram_plan = PeriodogramPlan::new(512.0, 512);
    periodogram_plan.limits.max_frequency_bins = 4;
    assert!(matches!(
        periodogram(&samples, &periodogram_plan),
        Err(SignalError::InvalidPolicy {
            policy: "frequency-bin limit",
            ..
        })
    ));
    periodogram_plan.limits.max_frequency_bins = 513;
    periodogram_plan.limits.max_work = 1;
    assert!(matches!(
        periodogram(&samples, &periodogram_plan),
        Err(SignalError::WorkLimit { .. })
    ));

    let mut multitaper_plan = MultitaperPlan::new(512.0, 512, 3.0, 5);
    multitaper_plan.limits.max_tapers = 2;
    assert!(matches!(
        multitaper(&samples, &multitaper_plan),
        Err(SignalError::InvalidPolicy {
            policy: "taper limit",
            ..
        })
    ));
}
