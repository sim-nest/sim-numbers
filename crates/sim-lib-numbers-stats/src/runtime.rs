//! Runtime dispatch for statistics operations.

use sim_kernel::{Args, Cx, Error, Expr, NumberLiteral, Result, Symbol, Value, force_list_to_vec};

use super::claim::{
    FairnessEvidence, StatsClaimEvidence, fairness_claim_value, stats_error_to_kernel,
    stats_result_claim_value,
};
use crate::{entropy, mean, variance};

/// Calls `stats/mean-claim` with one evaluated list of sample values.
pub fn call_stats_mean_claim(cx: &mut Cx, args: Args) -> Result<Value> {
    let samples = one_f64_list_arg(cx, args, "stats/mean-claim", "samples")?;
    let result = mean(&samples).map_err(stats_error_to_kernel)?;
    stats_result_claim_value(
        cx,
        StatsClaimEvidence::new(Symbol::qualified("stats", "mean"), samples, result),
    )
}

/// Calls `stats/variance-claim` with one evaluated list of sample values.
pub fn call_stats_variance_claim(cx: &mut Cx, args: Args) -> Result<Value> {
    let samples = one_f64_list_arg(cx, args, "stats/variance-claim", "samples")?;
    let result = variance(&samples).map_err(stats_error_to_kernel)?;
    stats_result_claim_value(
        cx,
        StatsClaimEvidence::new(Symbol::qualified("stats", "variance"), samples, result),
    )
}

/// Calls `stats/entropy-claim` with one evaluated probability-vector list.
pub fn call_stats_entropy_claim(cx: &mut Cx, args: Args) -> Result<Value> {
    let probabilities = one_f64_list_arg(cx, args, "stats/entropy-claim", "probabilities")?;
    let result = entropy(&probabilities).map_err(stats_error_to_kernel)?;
    stats_result_claim_value(
        cx,
        StatsClaimEvidence::new(Symbol::qualified("stats", "entropy"), probabilities, result),
    )
}

/// Calls `stats/disparate-impact-claim` with four already-evaluated count values.
pub fn call_stats_disparate_impact_claim(cx: &mut Cx, args: Args) -> Result<Value> {
    let values = args.into_vec();
    let [
        reference_selected,
        reference_total,
        comparison_selected,
        comparison_total,
    ] = values.as_slice()
    else {
        return Err(Error::Eval(
            "stats/disparate-impact-claim expects reference-selected, reference-total, comparison-selected, comparison-total"
                .to_owned(),
        ));
    };
    let evidence = FairnessEvidence::new(
        value_to_u64(cx, reference_selected, "reference-selected")?,
        value_to_u64(cx, reference_total, "reference-total")?,
        value_to_u64(cx, comparison_selected, "comparison-selected")?,
        value_to_u64(cx, comparison_total, "comparison-total")?,
    )
    .map_err(stats_error_to_kernel)?;
    fairness_claim_value(cx, evidence)
}

/// Calls `stats/claims` with one evaluated list of `(metric . args)` lists.
pub fn call_stats_claims(cx: &mut Cx, args: Args) -> Result<Value> {
    let values = args.into_vec();
    let [pairs] = values.as_slice() else {
        return Err(Error::Eval(
            "stats/claims expects one list of metric argument lists".to_owned(),
        ));
    };
    let pairs = value_to_list(cx, pairs, "stats/claims pairs")?;
    let claims = pairs
        .into_iter()
        .map(|pair| dispatch_claim_pair(cx, &pair))
        .collect::<Result<Vec<_>>>()?;
    cx.factory().list(claims)
}

fn dispatch_claim_pair(cx: &mut Cx, pair: &Value) -> Result<Value> {
    let values = value_to_list(cx, pair, "stats/claims pair")?;
    let Some((metric_value, args)) = values.split_first() else {
        return Err(Error::Eval(
            "stats/claims pair expects a metric symbol followed by arguments".to_owned(),
        ));
    };
    let metric = value_to_symbol(cx, metric_value, "stats/claims metric")?;
    dispatch_claim_by_metric(cx, &metric, Args::new(args.to_vec()))
}

fn dispatch_claim_by_metric(cx: &mut Cx, metric: &Symbol, args: Args) -> Result<Value> {
    if let Some(namespace) = &metric.namespace
        && namespace.as_ref() != "stats"
    {
        return Err(Error::Eval(format!(
            "stats/claims metric must be in the stats namespace, got {}",
            metric.as_qualified_str()
        )));
    }
    let name = metric
        .name
        .strip_suffix("-claim")
        .unwrap_or(metric.name.as_ref());
    match name {
        "mean" => call_stats_mean_claim(cx, args),
        "variance" => call_stats_variance_claim(cx, args),
        "entropy" => call_stats_entropy_claim(cx, args),
        "disparate-impact" => call_stats_disparate_impact_claim(cx, args),
        _ => Err(Error::Eval(format!(
            "stats/claims unsupported metric {}",
            metric.as_qualified_str()
        ))),
    }
}

fn one_f64_list_arg(cx: &mut Cx, args: Args, operation: &str, name: &str) -> Result<Vec<f64>> {
    let values = args.into_vec();
    let [samples] = values.as_slice() else {
        return Err(Error::Eval(format!("{operation} expects one {name} list")));
    };
    value_to_f64_list(cx, samples, name)
}

fn value_to_f64_list(cx: &mut Cx, value: &Value, name: &str) -> Result<Vec<f64>> {
    let values = value_to_list(cx, value, name)?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| value_to_f64(cx, value, &format!("{name}[{index}]")))
        .collect()
}

fn value_to_list(cx: &mut Cx, value: &Value, name: &str) -> Result<Vec<Value>> {
    let list = value.object().as_list().ok_or(Error::TypeMismatch {
        expected: "list",
        found: "non-list",
    })?;
    force_list_to_vec(cx, list, name)
}

fn value_to_symbol(cx: &mut Cx, value: &Value, name: &str) -> Result<Symbol> {
    let Expr::Symbol(symbol) = value.object().as_expr(cx)? else {
        return Err(Error::TypeMismatch {
            expected: "symbol",
            found: "non-symbol",
        });
    };
    if symbol.name.is_empty() {
        return Err(Error::Eval(format!("{name} must not be empty")));
    }
    Ok(symbol)
}

fn value_to_f64(cx: &mut Cx, value: &Value, name: &str) -> Result<f64> {
    let literal = value
        .object()
        .as_number_value()
        .ok_or(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        })?
        .number_literal(cx)?
        .ok_or_else(|| Error::Eval(format!("{name} must have a canonical numeric literal")))?;
    literal_to_f64(literal, name)
}

fn value_to_u64(cx: &mut Cx, value: &Value, name: &str) -> Result<u64> {
    let literal = value
        .object()
        .as_number_value()
        .ok_or(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        })?
        .number_literal(cx)?
        .ok_or_else(|| Error::Eval(format!("{name} must have a canonical numeric literal")))?;
    literal_to_u64(literal, name)
}

fn literal_to_u64(literal: NumberLiteral, name: &str) -> Result<u64> {
    literal
        .canonical
        .parse::<u64>()
        .map_err(|_| Error::Eval(format!("{name} must be a non-negative integer count")))
}

fn literal_to_f64(literal: NumberLiteral, name: &str) -> Result<f64> {
    literal
        .canonical
        .parse::<f64>()
        .map_err(|_| Error::Eval(format!("{name} must be a finite number")))
}
