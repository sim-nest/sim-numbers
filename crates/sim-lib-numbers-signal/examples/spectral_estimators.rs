use std::f64::consts::TAU;

use sim_lib_numbers_signal::{
    PeriodogramPlan, SpectrumScalingKind, WindowFunction, WindowSampling, WindowSpec, periodogram,
};

fn main() {
    let sample_rate_hz = 64.0;
    let samples = (0..64)
        .map(|index| (TAU * 8.0 * index as f64 / sample_rate_hz).sin())
        .collect::<Vec<_>>();
    let mut window = WindowSpec::new(WindowFunction::Hann);
    window.sampling = WindowSampling::Periodic;
    let metrics = window.generate(samples.len()).unwrap().metrics;

    let mut plan = PeriodogramPlan::new(sample_rate_hz, samples.len());
    plan.window = window;
    plan.scaling = SpectrumScalingKind::Power;
    let estimate = periodogram(&samples, &plan).unwrap();
    let peak = estimate
        .power
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;

    println!(
        "window coherent-gain={:.3} enbw-bins={:.3}",
        metrics.coherent_gain,
        metrics.equivalent_noise_bandwidth_bins.unwrap()
    );
    println!(
        "peak-frequency={:.1} peak-power={:.3}",
        estimate.frequency[peak], estimate.power[peak]
    );
    println!(
        "dof={:.0} work={}/{}",
        estimate.evidence.degrees_of_freedom,
        estimate.evidence.work_units,
        estimate.evidence.work_limit
    );
}
