use sim_lib_numbers_stats::{
    HiddenMarkovModel, HmmFitControl, HmmSpec, QuantilePolicy, QuantileSketch, Sequence, fit_hmm,
    forward_backward, viterbi,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let policy = QuantilePolicy::new(0.02, 8, 256)?;
    let mut left = QuantileSketch::new(policy.clone())?;
    let mut right = QuantileSketch::new(policy)?;
    for value in 0..100 {
        if value % 2 == 0 {
            left.insert(value as f64)?;
        } else {
            right.insert(value as f64)?;
        }
    }
    left.merge(&right)?;
    let median = left.estimate(0.5)?;
    println!(
        "quantile value={} rank=[{:.3},{:.3}] retained={} exact={}",
        median.value, median.rank_lower, median.rank_upper, median.retained_entries, median.exact
    );

    let model = HiddenMarkovModel::discrete(
        vec!["quiet", "active"],
        vec![0.6, 0.4],
        vec![vec![0.8, 0.2], vec![0.3, 0.7]],
        vec![vec![0.9, 0.1], vec![0.2, 0.8]],
    )?;
    let observations = [0, 0, 1, 1];
    let inference = forward_backward(&model, &observations)?;
    let path = viterbi(&model, &observations)?;
    println!(
        "inference log-likelihood={:.6} path={:?} repairs={}",
        inference.evidence.log_likelihood, path.states, inference.evidence.numerical_repairs
    );

    let data = [
        Sequence::Discrete(vec![0, 0, 1, 1, 1, 0]),
        Sequence::Discrete(vec![0, 1, 1, 0, 0, 0]),
    ];
    let report = fit_hmm(
        &data,
        HmmSpec::Discrete {
            states: 2,
            symbols: 2,
            additive_smoothing: 1.0e-6,
        },
        HmmFitControl::new(23, 6, 1.0e-7, 10_000, 1.0e-12)?,
    )?;
    println!(
        "fit likelihood={:.6} iterations={} converged={} repairs={} work={} termination={:?} seed={}",
        report.evidence.log_likelihood,
        report.evidence.iterations,
        report.evidence.converged,
        report.evidence.numerical_repairs,
        report.evidence.work,
        report.evidence.termination,
        report.evidence.seed
    );
    Ok(())
}
