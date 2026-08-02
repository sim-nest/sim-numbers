use std::f64::consts::TAU;

use sim_lib_numbers_signal::{ArOrderCriterion, BurgPlan, MemSpectrumPlan, burg, mem_spectrum};

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
        samples[index] =
            first * samples[index - 1] + second * samples[index - 2] + (uniform - 0.5) * 0.2;
    }
    samples
}

fn main() {
    let samples = generated_ar2(4096);
    let mut plan = BurgPlan::new(8);
    plan.criterion = ArOrderCriterion::Bayesian;
    let model = burg(&samples, &plan).unwrap();
    let spectrum = mem_spectrum(&model, &MemSpectrumPlan::new(1.0, 512)).unwrap();
    let peak = spectrum
        .power
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    println!(
        "effective-order={} residual-variance={:.6} reflection-margin={:.6}",
        model.evidence.effective_order,
        model.innovation_variance,
        model.evidence.minimum_reflection_margin
    );
    println!(
        "mem-peak={:.6} work={}/{}",
        spectrum.frequency[peak], spectrum.evidence.work_units, spectrum.evidence.work_limit
    );
}
