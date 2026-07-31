// conformance: generated AR processes, stable Burg order evidence, MEM, and bounded prediction.

use std::f64::consts::TAU;

use super::*;

fn generated_ar2(len: usize) -> Vec<f64> {
    let radius = 0.9;
    let frequency = 0.125;
    let first = 2.0 * radius * (TAU * frequency).cos();
    let second = -(radius * radius);
    let mut state = 0x5eed_f00d_dead_beef_u64;
    let mut samples = vec![0.0; len];
    for index in 2..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let uniform = (state >> 11) as f64 / (1_u64 << 53) as f64;
        let innovation = (uniform - 0.5) * 0.2;
        samples[index] = first * samples[index - 1] + second * samples[index - 2] + innovation;
    }
    samples
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn burg_recovers_generated_ar_process_and_bic_order() {
    let samples = generated_ar2(4096);
    let fixed = burg(&samples, &BurgPlan::new(2)).unwrap();
    assert_eq!(fixed.evidence.effective_order, 2);
    assert_eq!(fixed.evidence.termination, BurgTermination::RequestedOrder);
    assert_close(fixed.coefficients[0], -1.272792206, 0.04);
    assert_close(fixed.coefficients[1], 0.81, 0.04);
    assert!(fixed.evidence.minimum_reflection_margin > 0.05);
    assert!(fixed.evidence.residual_energy > 0.0);

    let mut selected = BurgPlan::new(8);
    selected.criterion = ArOrderCriterion::Bayesian;
    let selected = burg(&samples, &selected).unwrap();
    assert_eq!(selected.evidence.effective_order, 2);
    assert_eq!(selected.evidence.candidate_scores.len(), 8);
    assert_eq!(selected.evidence.criterion, ArOrderCriterion::Bayesian);
}

#[test]
fn maximum_entropy_spectrum_finds_the_generated_resonance() {
    let samples = generated_ar2(4096);
    let model = burg(&samples, &BurgPlan::new(2)).unwrap();
    let spectrum = mem_spectrum(&model, &MemSpectrumPlan::new(1.0, 512)).unwrap();
    let peak = spectrum
        .power
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_close(spectrum.frequency[peak], 0.125, 2.0 / 512.0);
    assert_eq!(spectrum.evidence.estimator, EstimatorKind::MaximumEntropy);
    assert_eq!(spectrum.evidence.degrees_of_freedom, 4094.0);
    assert!(
        spectrum
            .power
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
}

#[test]
fn forward_and_backward_predictions_are_bounded_before_recursion() {
    let samples = generated_ar2(512);
    let model = burg(&samples, &BurgPlan::new(2)).unwrap();
    let mut plan = PredictionPlan::new(64);
    plan.max_abs_value = 10.0;
    let forward = predict_forward(&model, &samples, &plan).unwrap();
    let backward = predict_backward(&model, &samples, &plan).unwrap();
    assert_eq!(forward.samples.len(), 64);
    assert_eq!(backward.samples.len(), 64);
    assert!(forward.samples.iter().all(|value| value.abs() <= 10.0));
    assert!(backward.samples.iter().all(|value| value.abs() <= 10.0));
    assert_eq!(forward.work_units, 128);

    plan.max_abs_value = 1.0e-12;
    assert!(matches!(
        predict_forward(&model, &samples, &plan),
        Err(SignalError::PredictionLimit { .. })
    ));
}

#[test]
fn singular_and_unstable_models_fail_closed() {
    assert_eq!(
        burg(&[3.0; 16], &BurgPlan::new(2)),
        Err(SignalError::SingularModel { order: 1 })
    );
    assert_eq!(
        burg(&[1.0, -1.0, 1.0, -1.0], &BurgPlan::new(1)),
        Err(SignalError::UnstableModel { order: 1 })
    );
}
