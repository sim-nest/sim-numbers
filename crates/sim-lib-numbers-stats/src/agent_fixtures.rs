//! Deterministic 30-agent descriptor fixtures exposed as evaluated stats calls.

use sim_kernel::{Args, Cx, Error, Expr, NumberLiteral, Result, Symbol, Value};
use sim_lib_numbers_core::domains;

/// Returns the symbols installed for 30-agent descriptor fixtures.
pub(crate) fn fixture_symbols() -> [Symbol; 4] {
    [
        data_analysis_report_symbol(),
        healthcare_intelligence_trace_symbol(),
        financial_advisory_trace_symbol(),
        education_intelligence_trace_symbol(),
    ]
}

/// Calls the descriptor fixture named by `symbol`, when this module owns it.
pub(crate) fn call_fixture(cx: &mut Cx, symbol: &Symbol, args: Args) -> Result<Option<Value>> {
    if *symbol == data_analysis_report_symbol() {
        return fixture_value(cx, args, symbol, data_analysis_report());
    }
    if *symbol == healthcare_intelligence_trace_symbol() {
        return fixture_value(cx, args, symbol, healthcare_intelligence_trace());
    }
    if *symbol == financial_advisory_trace_symbol() {
        return fixture_value(cx, args, symbol, financial_advisory_trace());
    }
    if *symbol == education_intelligence_trace_symbol() {
        return fixture_value(cx, args, symbol, education_intelligence_trace());
    }
    Ok(None)
}

fn data_analysis_report_symbol() -> Symbol {
    Symbol::qualified("stats", "data-analysis-report")
}

fn healthcare_intelligence_trace_symbol() -> Symbol {
    Symbol::qualified("stats", "healthcare-intelligence-trace")
}

fn financial_advisory_trace_symbol() -> Symbol {
    Symbol::qualified("stats", "financial-advisory-trace")
}

fn education_intelligence_trace_symbol() -> Symbol {
    Symbol::qualified("stats", "education-intelligence-trace")
}

fn fixture_value(cx: &mut Cx, args: Args, symbol: &Symbol, expr: Expr) -> Result<Option<Value>> {
    if !args.values().is_empty() {
        return Err(Error::Eval(format!("{} expects no arguments", symbol)));
    }
    cx.factory().expr(expr).map(Some)
}

fn data_analysis_report() -> Expr {
    let number = domains::f64();
    list(vec![
        sym("data-analysis-report"),
        atoms(&number, &["id", "a30-010-data-analysis"]),
        list(vec![
            sym("table"),
            sym("synthetic-demand"),
            atoms(&number, &["row", "s1", "x", "1", "y", "5"]),
            atoms(&number, &["row", "s2", "x", "2", "y", "8"]),
            atoms(&number, &["row", "s3", "x", "3", "y", "11"]),
            atoms(&number, &["row", "s4", "x", "4", "y", "24"]),
        ]),
        list(vec![
            sym("ols"),
            atoms(&number, &["method", "f64-normal-equation"]),
            atoms(&number, &["training-rows", "s1", "s2", "s3"]),
            atoms(&number, &["intercept", "2"]),
            atoms(&number, &["slope", "3"]),
            atoms(&number, &["r2", "100"]),
        ]),
        list(vec![
            sym("anomaly-detection"),
            atoms(&number, &["method", "residual-threshold"]),
            atoms(&number, &["threshold", "5"]),
            list(vec![
                sym("anomaly"),
                atoms(&number, &["row", "s4"]),
                atoms(&number, &["expected", "14"]),
                atoms(&number, &["actual", "24"]),
                atoms(&number, &["residual", "10"]),
            ]),
        ]),
        list(vec![
            sym("confidence-scored-insights"),
            atoms(&number, &["insight", "linear-trend", "confidence", "94"]),
            atoms(&number, &["insight", "investigate-s4", "confidence", "99"]),
        ]),
        atoms(
            &number,
            &["answer", "coefficients-intercept-2-slope-3-anomaly-s4"],
        ),
    ])
}

fn healthcare_intelligence_trace() -> Expr {
    let number = domains::f64();
    list(vec![
        sym("healthcare-intelligence-trace"),
        atoms(&number, &["id", "a30-024-healthcare-intelligence"]),
        list(vec![
            sym("metadata"),
            atoms(&number, &["fixture", "synthetic-symptom-panel"]),
            atoms(&number, &["synthetic-data", "yes"]),
            atoms(&number, &["non-medical-advice", "yes"]),
            atoms(&number, &["review-required", "yes"]),
        ]),
        atoms(
            &number,
            &["prior", "condition", "dehydration", "probability", "20"],
        ),
        list(vec![
            sym("bayesian-update"),
            atoms(
                &number,
                &[
                    "evidence",
                    "dry-mouth",
                    "likelihood-positive",
                    "70",
                    "likelihood-negative",
                    "20",
                    "posterior",
                    "47",
                ],
            ),
            atoms(
                &number,
                &[
                    "evidence",
                    "dizziness",
                    "likelihood-positive",
                    "60",
                    "likelihood-negative",
                    "30",
                    "posterior",
                    "64",
                ],
            ),
        ]),
        list(vec![
            sym("safety-threshold"),
            atoms(&number, &["minimum-confidence-for-self-serve", "85"]),
            atoms(&number, &["posterior", "64"]),
            atoms(&number, &["result", "below-threshold"]),
        ]),
        atoms(&number, &["decision", "escalate-for-clinician-review"]),
        list(vec![
            sym("fail-closed"),
            atoms(&number, &["missing-reviewer-capability", "block"]),
            atoms(&number, &["reason", "medical-review-required"]),
        ]),
        list(vec![
            sym("ledger"),
            atoms(
                &number,
                &["trace", "prior-evidence-posterior-threshold-escalate"],
            ),
            atoms(&number, &["audit", "immutable-synthetic-run"]),
        ]),
        list(vec![
            sym("effect-ledger"),
            atoms(&number, &["effect", "load-synthetic-symptoms", "local"]),
            atoms(&number, &["effect", "bayesian-update", "posterior-64"]),
            atoms(&number, &["effect", "compare-safety-threshold", "below"]),
            atoms(&number, &["effect", "require-review", "escalation"]),
        ]),
    ])
}

fn financial_advisory_trace() -> Expr {
    let number = domains::i64();
    list(vec![
        sym("financial-advisory-trace"),
        atoms(&number, &["id", "a30-026-financial-advisory"]),
        list(vec![
            sym("metadata"),
            atoms(&number, &["fixture", "synthetic-price-series"]),
            atoms(&number, &["synthetic-data", "yes"]),
            atoms(&number, &["not-financial-advice", "yes"]),
            atoms(&number, &["review-required", "yes"]),
        ]),
        list(vec![
            sym("client-profile"),
            atoms(&number, &["risk-tolerance", "moderate"]),
            atoms(&number, &["horizon-years", "12"]),
            atoms(&number, &["liquidity-buffer-months", "9"]),
        ]),
        list(vec![
            sym("money-domain"),
            atoms(&number, &["cash-flow", "fixed-cents"]),
            atoms(&number, &["return-domain", "rational-basis-points"]),
            atoms(&number, &["tensor", "shape-days-8"]),
        ]),
        atoms(
            &number,
            &[
                "price-series",
                "symbol",
                "sim-basket",
                "currency",
                "usd",
                "close-cents",
                "10000",
                "10120",
                "10050",
                "10240",
                "9900",
                "9820",
                "9950",
                "10080",
            ],
        ),
        list(vec![
            sym("risk-metrics"),
            atoms(&number, &["volatility-bps", "1420"]),
            atoms(&number, &["var-95-bps", "minus-210"]),
            atoms(&number, &["max-drawdown-bps", "minus-410"]),
        ]),
        list(vec![
            sym("composite-risk"),
            atoms(&number, &["volatility-score", "42"]),
            atoms(&number, &["drawdown-score", "36"]),
            atoms(&number, &["var-score", "31"]),
            atoms(&number, &["weighted-score", "37"]),
        ]),
        list(vec![
            sym("suitability-gate"),
            atoms(&number, &["capability", "suitability-review"]),
            atoms(&number, &["max-score", "45"]),
            atoms(&number, &["observed-score", "37"]),
            atoms(&number, &["result", "pass"]),
        ]),
        list(vec![
            sym("concentration-gate"),
            atoms(&number, &["max-position-percent", "25"]),
            atoms(&number, &["largest-position-percent", "18"]),
            atoms(&number, &["result", "pass"]),
        ]),
        atoms(&number, &["decision", "eligible-for-advisor-review"]),
        list(vec![
            sym("compliance"),
            atoms(&number, &["disclosure", "not-financial-advice"]),
            atoms(&number, &["advice-status", "educational-synthetic"]),
            atoms(&number, &["human-approval", "required-before-trade"]),
        ]),
        list(vec![
            sym("effect-ledger"),
            atoms(&number, &["effect", "load-synthetic-prices", "local"]),
            atoms(&number, &["effect", "compute-volatility", "bps-1420"]),
            atoms(&number, &["effect", "compute-var95", "minus-210-bps"]),
            atoms(
                &number,
                &["effect", "compute-max-drawdown", "minus-410-bps"],
            ),
            atoms(&number, &["effect", "enforce-suitability", "pass"]),
        ]),
    ])
}

fn education_intelligence_trace() -> Expr {
    let number = domains::f64();
    list(vec![
        sym("education-intelligence-trace"),
        atoms(&number, &["id", "a30-028-education-intelligence"]),
        list(vec![
            sym("metadata"),
            atoms(&number, &["fixture", "synthetic-learner-panel"]),
            atoms(&number, &["synthetic-data", "yes"]),
            atoms(&number, &["deterministic", "yes"]),
        ]),
        list(vec![
            sym("learner-model"),
            atoms(&number, &["learner", "sim-learner-7"]),
            atoms(&number, &["goal", "loop-reasoning"]),
            atoms(
                &number,
                &[
                    "observations",
                    "correct",
                    "incorrect",
                    "incorrect",
                    "review-due",
                ],
            ),
        ]),
        list(vec![
            sym("knowledge-dag"),
            atoms(&number, &["node", "variables", "mastery-92"]),
            atoms(&number, &["node", "branches", "mastery-78"]),
            atoms(&number, &["node", "loops", "mastery-44"]),
            atoms(&number, &["node", "functions", "mastery-35"]),
            atoms(&number, &["edge", "variables", "branches"]),
            atoms(&number, &["edge", "branches", "loops"]),
            atoms(&number, &["edge", "loops", "functions"]),
        ]),
        list(vec![
            sym("placement"),
            atoms(&number, &["method", "irt-2pl"]),
            atoms(&number, &["ability-bps", "5200"]),
            atoms(
                &number,
                &[
                    "item",
                    "loops-diagnostic",
                    "discrimination-bps",
                    "135",
                    "difficulty-bps",
                    "4800",
                    "fisher-info",
                    "82",
                ],
            ),
            atoms(&number, &["placement-node", "loops"]),
        ]),
        list(vec![
            sym("bkt-update"),
            atoms(&number, &["skill", "loops"]),
            atoms(&number, &["prior", "58"]),
            atoms(&number, &["learn", "12"]),
            atoms(&number, &["guess", "20"]),
            atoms(&number, &["slip", "10"]),
            atoms(&number, &["evidence", "incorrect"]),
            atoms(&number, &["posterior-before-transition", "44"]),
            atoms(&number, &["posterior-after-transition", "51"]),
        ]),
        list(vec![
            sym("sm2-spacing"),
            atoms(&number, &["card", "branch-review"]),
            atoms(&number, &["quality", "3"]),
            atoms(&number, &["ease-bps", "230"]),
            atoms(&number, &["interval-days", "6"]),
            atoms(&number, &["due-priority", "high"]),
        ]),
        list(vec![
            sym("next-objective"),
            atoms(&number, &["node", "loops"]),
            atoms(
                &number,
                &["reason", "mastery-below-threshold-and-review-due"],
            ),
            atoms(&number, &["threshold", "85"]),
        ]),
        list(vec![
            sym("effect-ledger"),
            atoms(&number, &["effect", "load-synthetic-learner", "local"]),
            atoms(&number, &["effect", "run-irt-placement", "node-loops"]),
            atoms(&number, &["effect", "update-bkt", "posterior-51"]),
            atoms(&number, &["effect", "schedule-sm2", "branch-review"]),
            atoms(&number, &["effect", "select-next-objective", "loops"]),
        ]),
    ])
}

fn atoms(domain: &Symbol, items: &[&str]) -> Expr {
    list(items.iter().map(|item| atom(domain, item)).collect())
}

fn atom(domain: &Symbol, value: &str) -> Expr {
    if value.as_bytes().iter().all(u8::is_ascii_digit) {
        return Expr::Number(NumberLiteral {
            domain: domain.clone(),
            canonical: value.to_owned(),
        });
    }
    sym(value)
}

fn sym(name: &str) -> Expr {
    Expr::Symbol(Symbol::new(name))
}

fn list(items: Vec<Expr>) -> Expr {
    Expr::List(items)
}
