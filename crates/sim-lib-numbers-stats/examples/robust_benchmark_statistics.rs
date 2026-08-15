use sim_lib_numbers_stats::{
    BootstrapControl, bootstrap_mean_difference_interval, median_absolute_deviation,
};

fn main() {
    let baseline = [101.0, 99.0, 100.0, 102.0, 98.0];
    let candidate = [91.0, 89.0, 90.0, 92.0, 88.0];
    let control = BootstrapControl::new(19, 2_000, 0.95, 20_000).unwrap();
    let interval = bootstrap_mean_difference_interval(&baseline, &candidate, control).unwrap();

    println!(
        "baseline MAD: {:.1}",
        median_absolute_deviation(&baseline).unwrap()
    );
    println!(
        "candidate-minus-baseline mean: {:.1}",
        interval.point_effect
    );
    println!(
        "95% interval: [{:.1}, {:.1}]",
        interval.lower, interval.upper
    );
    println!("seed: {}, resamples: {}", interval.seed, interval.resamples);
}
