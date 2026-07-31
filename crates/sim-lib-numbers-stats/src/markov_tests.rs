use super::markov::*;

// conformance: generic finite transition estimation, evidence, and serialization.

const FIXTURE: &[u8] = include_bytes!("../fixtures/generated-weather-transitions.tsv");

fn policy(held_out_sequences: usize) -> MarkovPolicy {
    let provenance = CorpusProvenance::from_bytes(
        "generated-weather-transitions-v1",
        "deterministic synthetic finite-state fixture",
        "CC0-1.0",
        FIXTURE,
    )
    .unwrap();
    assert_eq!(provenance.content_hash, "fnv1a64:cfa3c26f7c8d57a0");
    MarkovPolicy::new(1.0, held_out_sequences, provenance).unwrap()
}

#[test]
fn finite_non_music_model_is_smoothed_and_scores_holdout() {
    let sequences = vec![
        vec!["sun", "rain", "sun"],
        vec!["sun", "sun", "rain"],
        vec!["rain", "sun", "rain"],
    ];
    let report = fit_markov(&sequences, policy(1)).unwrap();

    assert_eq!(report.training_sequences, 2);
    assert_eq!(report.held_out_sequences, 1);
    assert_eq!(report.model.transition_count(&"sun", &"rain").unwrap(), 2);
    assert_eq!(report.model.transition_count(&"rain", &"rain").unwrap(), 0);
    assert_eq!(report.held_out_score.unwrap().transitions, 2);
    assert!(report.held_out_score.unwrap().perplexity.is_finite());
}

#[test]
fn stable_serialization_retains_policy_provenance_and_counts() {
    let report = fit_markov(
        &[vec!["sun", "rain", "sun"], vec!["rain", "sun", "rain"]],
        policy(1),
    )
    .unwrap();
    let first = report
        .model
        .to_stable_text(|state| (*state).to_owned())
        .unwrap();
    let second = report
        .model
        .to_stable_text(|state| (*state).to_owned())
        .unwrap();

    assert_eq!(first, second);
    assert!(first.starts_with("SIM-MARKOV-1\n"));
    assert!(first.contains("corpus-license=4343302d312e30"));
    assert!(first.contains("transition=0:1:"));
}

#[test]
fn invalid_holdout_and_unknown_states_fail_closed() {
    let sequences = vec![vec!["sun", "rain"]];
    assert!(matches!(
        fit_markov(&sequences, policy(1)),
        Err(MarkovError::InvalidHoldout { .. })
    ));
    assert!(matches!(
        fit_markov(&[vec!["sun", "rain"], vec!["sun", "snow"]], policy(1)),
        Err(MarkovError::UnknownState {
            sequence: 0,
            position: 1
        })
    ));

    let model = fit_markov(&sequences, policy(0)).unwrap().model;
    assert!(matches!(
        model.score(&[vec!["sun", "snow"]]),
        Err(MarkovError::UnknownState {
            sequence: 0,
            position: 1
        })
    ));
}
