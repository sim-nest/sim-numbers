use super::{hmm_fit::*, hmm_inference::*, hmm_model::*, markov::*};

// conformance: finite discrete/Gaussian HMM inference and bounded fitting.

fn discrete_model() -> HiddenMarkovModel<&'static str> {
    HiddenMarkovModel::discrete(
        vec!["fair", "loaded"],
        vec![0.6, 0.4],
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],
        vec![vec![0.5, 0.5], vec![0.1, 0.9]],
    )
    .unwrap()
}

fn path_probability(
    model: &HiddenMarkovModel<&str>,
    path: &[usize],
    observations: &[usize],
) -> f64 {
    let emissions = model.emissions().discrete_probabilities().unwrap();
    let mut probability =
        model.initial_probabilities()[path[0]] * emissions[path[0]][observations[0]];
    for position in 1..path.len() {
        probability *= model
            .transitions()
            .probability_by_index(path[position - 1], path[position])
            .unwrap();
        probability *= emissions[path[position]][observations[position]];
    }
    probability
}

fn enumerate_paths(length: usize) -> Vec<Vec<usize>> {
    (0..(1_usize << length))
        .map(|bits| (0..length).map(|position| (bits >> position) & 1).collect())
        .collect()
}

#[test]
fn normalized_inference_agrees_with_enumerated_tiny_fixture() {
    let model = discrete_model();
    let observations = [0, 1, 1];
    let paths = enumerate_paths(observations.len());
    let weighted = paths
        .iter()
        .map(|path| (path, path_probability(&model, path, &observations)))
        .collect::<Vec<_>>();
    let exact_likelihood = weighted
        .iter()
        .map(|(_, probability)| probability)
        .sum::<f64>();
    let exact_best = weighted
        .iter()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap();

    let inference = forward_backward(&model, &observations).unwrap();
    assert!((inference.evidence.log_likelihood.exp() - exact_likelihood).abs() < 1.0e-14);
    for row in inference
        .forward
        .iter()
        .chain(&inference.backward)
        .chain(&inference.posterior)
    {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }
    for position in 0..observations.len() {
        for state in 0..2 {
            let exact = weighted
                .iter()
                .filter(|(path, _)| path[position] == state)
                .map(|(_, probability)| probability)
                .sum::<f64>()
                / exact_likelihood;
            assert!((inference.posterior[position][state] - exact).abs() < 1.0e-12);
        }
    }

    let path = viterbi(&model, &observations).unwrap();
    assert_eq!(&path.state_indices, exact_best.0);
    assert!((path.log_probability.exp() - exact_best.1).abs() < 1.0e-14);
    let posterior = posterior_decode(&model, &observations).unwrap();
    assert_eq!(posterior.state_indices.len(), observations.len());
    assert_eq!(posterior.evidence, inference.evidence);
}

#[test]
fn long_and_continuous_sequences_remain_finite() {
    let discrete = discrete_model();
    let long = vec![1; 10_000];
    let inference = forward_backward(&discrete, &long).unwrap();
    assert!(inference.evidence.log_likelihood.is_finite());

    let gaussian = HiddenMarkovModel::gaussian(
        vec!["cold", "hot"],
        vec![0.5, 0.5],
        vec![vec![0.95, 0.05], vec![0.08, 0.92]],
        vec![-2.0, 3.0],
        vec![0.5, 0.75],
        1.0e-6,
    )
    .unwrap();
    let observations = [-2.2, -1.8, 2.7, 3.1];
    assert!(
        forward_backward(&gaussian, &observations)
            .unwrap()
            .evidence
            .log_likelihood
            .is_finite()
    );
    assert_eq!(viterbi(&gaussian, &observations).unwrap().states.len(), 4);
}

#[test]
fn hmm_accepts_the_observable_markov_transition_representation() {
    let provenance =
        CorpusProvenance::from_bytes("weather", "fixture", "CC0-1.0", b"weather").unwrap();
    let policy = MarkovPolicy::new(1.0, 0, provenance).unwrap();
    let markov = fit_markov(
        &[vec!["sun", "rain", "sun"], vec!["rain", "sun", "sun"]],
        policy,
    )
    .unwrap()
    .model;
    let hmm = HiddenMarkovModel::from_transition_matrix(
        vec![0.5, 0.5],
        markov.transition_matrix(),
        EmissionModel::Discrete {
            probabilities: vec![vec![0.8, 0.2], vec![0.3, 0.7]],
        },
    )
    .unwrap();
    assert!(forward_backward(&hmm, &[0, 1, 0]).is_ok());
}

#[test]
fn discrete_baum_welch_is_seeded_monotone_and_bounded() {
    let data = vec![
        Sequence::Discrete(vec![0, 0, 1, 1, 1, 0]),
        Sequence::Discrete(vec![0, 1, 1, 1, 0, 0]),
        Sequence::Discrete(vec![1, 1, 1, 0, 0, 0]),
    ];
    let spec = HmmSpec::Discrete {
        states: 2,
        symbols: 2,
        additive_smoothing: 1.0e-6,
    };
    let control = HmmFitControl::new(17, 12, 1.0e-8, 20_000, 1.0e-12).unwrap();
    let first = fit_hmm(&data, spec.clone(), control).unwrap();
    let second = fit_hmm(&data, spec, control).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.evidence.seed, 17);
    assert!(first.evidence.iterations <= control.max_iterations);
    assert!(first.evidence.work <= control.max_work);
    assert!(first.evidence.log_likelihood.is_finite());
    assert!(
        first
            .evidence
            .likelihood_history
            .windows(2)
            .all(|pair| pair[1] + 1.0e-8 >= pair[0])
    );
    assert!(matches!(
        first.evidence.termination,
        HmmTermination::Converged
            | HmmTermination::IterationLimit
            | HmmTermination::WorkLimit
            | HmmTermination::LikelihoodDecrease
    ));
}

#[test]
fn continuous_fitting_retains_variance_and_termination_evidence() {
    let data = vec![
        Sequence::Continuous(vec![-2.2, -2.0, -1.8, 2.8, 3.0, 3.2]),
        Sequence::Continuous(vec![3.1, 2.9, -1.9, -2.1]),
    ];
    let spec = HmmSpec::Gaussian {
        states: 2,
        additive_smoothing: 1.0e-6,
        variance_floor: 1.0e-4,
    };
    let control = HmmFitControl::new(99, 8, 1.0e-7, 10_000, 1.0e-12).unwrap();
    let report = fit_hmm(&data, spec, control).unwrap();
    let (_, variances, floor) = report.model.emissions().gaussian_parameters().unwrap();
    assert!(variances.iter().all(|variance| *variance >= floor));
    assert!(report.evidence.log_likelihood.is_finite());
    assert!(report.evidence.work <= control.max_work);
}

#[test]
fn work_and_input_failures_are_explicit() {
    let data = vec![Sequence::Discrete(vec![0, 1, 0])];
    let spec = HmmSpec::Discrete {
        states: 2,
        symbols: 2,
        additive_smoothing: 0.1,
    };
    let too_small = HmmFitControl::new(1, 4, 1.0e-6, 1, 1.0e-12).unwrap();
    assert!(matches!(
        fit_hmm(&data, spec.clone(), too_small),
        Err(HmmError::InvalidFitControl {
            field: "max_work",
            ..
        })
    ));

    let one_sweep = HmmFitControl::new(1, 4, 1.0e-6, 12, 1.0e-12).unwrap();
    let report = fit_hmm(&data, spec, one_sweep).unwrap();
    assert_eq!(report.evidence.iterations, 0);
    assert_eq!(report.evidence.termination, HmmTermination::WorkLimit);
    assert_eq!(report.evidence.work, 12);

    assert!(matches!(
        forward_backward(&discrete_model(), &[2]),
        Err(HmmError::UnknownSymbol { .. })
    ));
}
