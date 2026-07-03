use crate::{
    BinaryOutcomeCounts, FairnessClaimValue, StatsClaimValue, StatsError, StatsNumbersLib,
    bayesian_update, bayesian_update_binary, disparate_impact, entropy, four_fifths_ratio, mean,
    population_variance, sample_variance, stats_claims_symbol, stats_disparate_impact_claim_symbol,
    stats_entropy_claim_symbol, stats_mean_claim_symbol, stats_variance_claim_symbol, variance,
};
use sim_kernel::{
    Args, Cx, Datum, DatumStore, DefaultFactory, EagerPolicy, Error, Expr, Ref, Symbol, Value,
};
use std::sync::Arc;

fn test_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&StatsNumbersLib::new()).unwrap();
    cx
}

fn u64_value(cx: &mut Cx, value: u64) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "u64"), value.to_string())
        .unwrap()
}

fn f64_value(cx: &mut Cx, value: f64) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
        .unwrap()
}

fn f64_list(cx: &mut Cx, values: &[f64]) -> Value {
    let values = values
        .iter()
        .copied()
        .map(|value| f64_value(cx, value))
        .collect::<Vec<_>>();
    list(cx, values)
}

fn symbol_value(cx: &mut Cx, symbol: Symbol) -> Value {
    cx.factory().symbol(symbol).unwrap()
}

fn list(cx: &mut Cx, values: Vec<Value>) -> Value {
    cx.factory().list(values).unwrap()
}

fn evidence_field(cx: &mut Cx, evidence: &Value, name: &str) -> Value {
    evidence
        .object()
        .as_table_impl()
        .unwrap()
        .get(cx, Symbol::new(name))
        .unwrap()
}

fn value_to_f64(cx: &mut Cx, value: &Value) -> f64 {
    value
        .object()
        .as_number_value()
        .unwrap()
        .number_literal(cx)
        .unwrap()
        .unwrap()
        .canonical
        .parse()
        .unwrap()
}

fn claim_evidence(cx: &mut Cx, claim_value: &Value) -> Value {
    let table = claim_value.object().as_table(cx).unwrap();
    table
        .object()
        .as_table_impl()
        .unwrap()
        .get(cx, Symbol::new("evidence"))
        .unwrap()
}

fn claim_evidence_value(cx: &mut Cx, claim_value: &Value) -> f64 {
    let evidence = claim_evidence(cx, claim_value);
    let value = evidence_field(cx, &evidence, "value");
    value_to_f64(cx, &value)
}

fn close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1.0e-12,
        "expected {left} to be close to {right}"
    );
}

#[test]
fn bayesian_update_normalizes_prior_by_evidence() {
    close(bayesian_update(0.2, 0.75, 0.3).unwrap(), 0.5);
    close(
        bayesian_update_binary(0.01, 0.9, 0.08).unwrap(),
        0.1020408163265306,
    );
}

#[test]
fn entropy_returns_bits_for_probability_vector() {
    close(entropy(&[0.5, 0.25, 0.25]).unwrap(), 1.5);
    close(entropy(&[1.0, 0.0]).unwrap(), 0.0);
}

#[test]
fn mean_and_variance_use_deterministic_samples() {
    let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    close(mean(&values).unwrap(), 5.0);
    close(variance(&values).unwrap(), 4.0);
    close(population_variance(&values).unwrap(), 4.0);
    close(sample_variance(&values).unwrap(), 32.0 / 7.0);
}

#[test]
fn disparate_impact_reports_four_fifths_result() {
    let reference = BinaryOutcomeCounts::new(80, 100).unwrap();
    let comparison = BinaryOutcomeCounts::new(60, 100).unwrap();

    let impact = disparate_impact(reference, comparison).unwrap();

    close(impact.reference_rate, 0.8);
    close(impact.comparison_rate, 0.6);
    close(impact.ratio, 0.75);
    assert!(!impact.passes_four_fifths);
    close(four_fifths_ratio(0.8, 0.64).unwrap(), 0.8);
}

#[test]
fn numeric6_fairness_claim_carries_evidence() {
    let mut cx = test_cx();
    let args = vec![
        u64_value(&mut cx, 80),
        u64_value(&mut cx, 100),
        u64_value(&mut cx, 48),
        u64_value(&mut cx, 80),
    ];
    let claim_value = cx
        .call_function(&stats_disparate_impact_claim_symbol(), Args::new(args))
        .unwrap();

    let fairness_claim = claim_value
        .object()
        .downcast_ref::<FairnessClaimValue>()
        .unwrap();
    let claim = fairness_claim.claim();
    assert_eq!(
        claim.subject,
        Ref::Symbol(Symbol::qualified("stats", "disparate-impact"))
    );
    assert_eq!(claim.predicate, Symbol::new("fairness-result"));
    let Ref::Content(object_id) = &claim.object else {
        panic!("expected evidence object to be content-addressed");
    };
    assert!(matches!(
        cx.datum_store().get(object_id).unwrap(),
        Some(Datum::Node { tag, .. })
            if *tag == Symbol::qualified("stats", "disparate-impact-evidence")
    ));

    let table = claim_value.object().as_table(&mut cx).unwrap();
    let evidence = table
        .object()
        .as_table_impl()
        .unwrap()
        .get(&mut cx, Symbol::new("evidence"))
        .unwrap();
    let ratio = evidence_field(&mut cx, &evidence, "ratio");
    close(value_to_f64(&mut cx, &ratio), 0.75);
    let threshold = evidence_field(&mut cx, &evidence, "threshold");
    close(value_to_f64(&mut cx, &threshold), 0.8);
    let value = evidence_field(&mut cx, &evidence, "value");
    close(value_to_f64(&mut cx, &value), 0.75);
    assert_eq!(
        evidence_field(&mut cx, &evidence, "passes-four-fifths")
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::Bool(false)
    );
}

#[test]
fn descriptive_stats_claims_carry_inputs_and_value() {
    let mut cx = test_cx();
    let mean_samples = f64_list(&mut cx, &[1.0, 2.0, 3.0, 4.0, 5.0]);

    let mean_claim = cx
        .call_function(&stats_mean_claim_symbol(), Args::new(vec![mean_samples]))
        .unwrap();
    let mean = mean_claim
        .object()
        .downcast_ref::<StatsClaimValue>()
        .unwrap();
    assert_eq!(
        mean.claim().subject,
        Ref::Symbol(Symbol::qualified("stats", "mean"))
    );
    assert_eq!(mean.claim().predicate, Symbol::new("stats-result"));
    assert_eq!(mean.evidence().inputs(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    close(claim_evidence_value(&mut cx, &mean_claim), 3.0);

    let variance_samples = f64_list(&mut cx, &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    let variance_claim = cx
        .call_function(
            &stats_variance_claim_symbol(),
            Args::new(vec![variance_samples]),
        )
        .unwrap();
    close(claim_evidence_value(&mut cx, &variance_claim), 4.0);

    let probabilities = f64_list(&mut cx, &[0.5, 0.25, 0.25]);
    let entropy_claim = cx
        .call_function(
            &stats_entropy_claim_symbol(),
            Args::new(vec![probabilities]),
        )
        .unwrap();
    close(claim_evidence_value(&mut cx, &entropy_claim), 1.5);
}

#[test]
fn stats_claims_batches_descriptive_and_fairness_claims() {
    let mut cx = test_cx();
    let mean_metric = symbol_value(&mut cx, Symbol::qualified("stats", "mean"));
    let mean_samples = f64_list(&mut cx, &[1.0, 2.0, 3.0]);
    let mean_pair = list(&mut cx, vec![mean_metric, mean_samples]);
    let entropy_metric = symbol_value(&mut cx, Symbol::qualified("stats", "entropy"));
    let probabilities = f64_list(&mut cx, &[0.5, 0.25, 0.25]);
    let entropy_pair = list(&mut cx, vec![entropy_metric, probabilities]);
    let fairness_metric = symbol_value(&mut cx, Symbol::qualified("stats", "disparate-impact"));
    let reference_selected = u64_value(&mut cx, 80);
    let reference_total = u64_value(&mut cx, 100);
    let comparison_selected = u64_value(&mut cx, 48);
    let comparison_total = u64_value(&mut cx, 80);
    let fairness_pair = list(
        &mut cx,
        vec![
            fairness_metric,
            reference_selected,
            reference_total,
            comparison_selected,
            comparison_total,
        ],
    );
    let pairs = list(&mut cx, vec![mean_pair, entropy_pair, fairness_pair]);

    let claims = cx
        .call_function(&stats_claims_symbol(), Args::new(vec![pairs]))
        .unwrap();
    let claims = claims
        .object()
        .as_list()
        .unwrap()
        .to_vec(&mut cx, None)
        .unwrap();
    assert_eq!(claims.len(), 3);

    let mean = claims[0]
        .object()
        .downcast_ref::<StatsClaimValue>()
        .unwrap();
    assert_eq!(mean.claim().predicate, Symbol::new("stats-result"));
    close(claim_evidence_value(&mut cx, &claims[0]), 2.0);

    let entropy = claims[1]
        .object()
        .downcast_ref::<StatsClaimValue>()
        .unwrap();
    assert_eq!(entropy.claim().predicate, Symbol::new("stats-result"));
    close(claim_evidence_value(&mut cx, &claims[1]), 1.5);

    let fairness = claims[2]
        .object()
        .downcast_ref::<FairnessClaimValue>()
        .unwrap();
    assert_eq!(fairness.claim().predicate, Symbol::new("fairness-result"));
    close(claim_evidence_value(&mut cx, &claims[2]), 0.75);
}

#[test]
fn numeric6_fairness_claim_rejects_zero_totals() {
    let mut cx = test_cx();
    let args = vec![
        u64_value(&mut cx, 1),
        u64_value(&mut cx, 0),
        u64_value(&mut cx, 1),
        u64_value(&mut cx, 1),
    ];
    let err = cx
        .call_function(&stats_disparate_impact_claim_symbol(), Args::new(args))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DomainError { message, .. } if message.contains("total must be nonzero")
    ));
}

#[test]
fn invalid_inputs_fail_closed() {
    assert!(matches!(
        mean(&[]),
        Err(StatsError::EmptyInput { metric: "mean" })
    ));
    assert!(matches!(
        sample_variance(&[1.0]),
        Err(StatsError::InsufficientInput {
            metric: "sample_variance",
            minimum: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        entropy(&[0.7, 0.2]),
        Err(StatsError::ProbabilityMass {
            metric: "entropy",
            ..
        })
    ));
    assert!(matches!(
        bayesian_update(0.2, 0.75, 0.0),
        Err(StatsError::ZeroEvidence {
            metric: "bayesian_update"
        })
    ));
    assert!(matches!(
        bayesian_update(0.9, 0.9, 0.1),
        Err(StatsError::ProbabilityOutOfRange {
            metric: "bayesian_update",
            ..
        })
    ));
    assert!(matches!(
        BinaryOutcomeCounts::new(2, 0),
        Err(StatsError::ZeroTotal {
            label: "outcome counts"
        })
    ));
    assert!(matches!(
        four_fifths_ratio(0.0, 0.2),
        Err(StatsError::ZeroReferenceRate {
            metric: "four_fifths_ratio"
        })
    ));
}
