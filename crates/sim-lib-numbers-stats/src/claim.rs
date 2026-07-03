//! Claim value support for statistics and fairness metrics.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    Claim, Cx, Datum, Error, Expr, Factory, NumberLiteral, Object, ObjectCompat, Ref, Result,
    Symbol, Value,
};
use sim_lib_numbers_core::domains;

use crate::{BinaryOutcomeCounts, StatsResult, disparate_impact};

const FOUR_FIFTHS_THRESHOLD: f64 = 0.8;

/// Evidence carried by the disparate-impact fairness Claim.
#[derive(Clone, Debug, PartialEq)]
pub struct FairnessEvidence {
    /// Number of selected cases in the reference group.
    pub reference_selected: u64,
    /// Total cases in the reference group.
    pub reference_total: u64,
    /// Number of selected cases in the comparison group.
    pub comparison_selected: u64,
    /// Total cases in the comparison group.
    pub comparison_total: u64,
    /// Selection rate for the reference group.
    pub reference_rate: f64,
    /// Selection rate for the comparison group.
    pub comparison_rate: f64,
    /// Comparison rate divided by reference rate.
    pub ratio: f64,
    /// Four-fifths rule threshold.
    pub threshold: f64,
    /// Whether `ratio >= threshold`.
    pub passes_four_fifths: bool,
}

impl FairnessEvidence {
    /// Computes evidence for a disparate-impact Claim from validated counts.
    pub fn new(
        reference_selected: u64,
        reference_total: u64,
        comparison_selected: u64,
        comparison_total: u64,
    ) -> StatsResult<Self> {
        let reference = BinaryOutcomeCounts::new(reference_selected, reference_total)?;
        let comparison = BinaryOutcomeCounts::new(comparison_selected, comparison_total)?;
        let impact = disparate_impact(reference, comparison)?;
        Ok(Self {
            reference_selected,
            reference_total,
            comparison_selected,
            comparison_total,
            reference_rate: impact.reference_rate,
            comparison_rate: impact.comparison_rate,
            ratio: impact.ratio,
            threshold: FOUR_FIFTHS_THRESHOLD,
            passes_four_fifths: impact.passes_four_fifths,
        })
    }

    /// Returns the evidence as a browsable runtime table.
    pub fn table_value(&self, factory: &dyn Factory) -> Result<Value> {
        factory.table(vec![
            (
                Symbol::new("reference-selected"),
                u64_value(factory, self.reference_selected)?,
            ),
            (
                Symbol::new("reference-total"),
                u64_value(factory, self.reference_total)?,
            ),
            (
                Symbol::new("comparison-selected"),
                u64_value(factory, self.comparison_selected)?,
            ),
            (
                Symbol::new("comparison-total"),
                u64_value(factory, self.comparison_total)?,
            ),
            (
                Symbol::new("reference-rate"),
                f64_value(factory, self.reference_rate)?,
            ),
            (
                Symbol::new("comparison-rate"),
                f64_value(factory, self.comparison_rate)?,
            ),
            (Symbol::new("ratio"), f64_value(factory, self.ratio)?),
            (Symbol::new("value"), f64_value(factory, self.ratio)?),
            (
                Symbol::new("threshold"),
                f64_value(factory, self.threshold)?,
            ),
            (
                Symbol::new("passes-four-fifths"),
                factory.bool(self.passes_four_fifths)?,
            ),
        ])
    }

    fn datum(&self) -> Datum {
        Datum::Node {
            tag: Symbol::qualified("stats", "disparate-impact-evidence"),
            fields: vec![
                (
                    Symbol::new("reference-selected"),
                    u64_datum(self.reference_selected),
                ),
                (
                    Symbol::new("reference-total"),
                    u64_datum(self.reference_total),
                ),
                (
                    Symbol::new("comparison-selected"),
                    u64_datum(self.comparison_selected),
                ),
                (
                    Symbol::new("comparison-total"),
                    u64_datum(self.comparison_total),
                ),
                (
                    Symbol::new("reference-rate"),
                    f64_datum(self.reference_rate),
                ),
                (
                    Symbol::new("comparison-rate"),
                    f64_datum(self.comparison_rate),
                ),
                (Symbol::new("ratio"), f64_datum(self.ratio)),
                (Symbol::new("value"), f64_datum(self.ratio)),
                (Symbol::new("threshold"), f64_datum(self.threshold)),
                (
                    Symbol::new("passes-four-fifths"),
                    Datum::Bool(self.passes_four_fifths),
                ),
            ],
        }
    }
}

/// Evidence carried by a descriptive statistics Claim.
#[derive(Clone, Debug, PartialEq)]
pub struct StatsClaimEvidence {
    metric: Symbol,
    inputs: Vec<f64>,
    value: f64,
}

impl StatsClaimEvidence {
    /// Creates descriptive metric evidence from validated inputs and a result.
    pub fn new(metric: Symbol, inputs: Vec<f64>, value: f64) -> Self {
        Self {
            metric,
            inputs,
            value,
        }
    }

    /// Returns the metric named by this evidence.
    pub fn metric(&self) -> &Symbol {
        &self.metric
    }

    /// Returns the input samples or probabilities used to compute the metric.
    pub fn inputs(&self) -> &[f64] {
        &self.inputs
    }

    /// Returns the computed metric value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Returns the evidence as a browsable runtime table.
    pub fn table_value(&self, factory: &dyn Factory) -> Result<Value> {
        factory.table(vec![
            (Symbol::new("metric"), factory.symbol(self.metric.clone())?),
            (
                Symbol::new("inputs"),
                f64_list_value(factory, &self.inputs)?,
            ),
            (
                Symbol::new("count"),
                u64_value(factory, self.inputs.len() as u64)?,
            ),
            (Symbol::new("value"), f64_value(factory, self.value)?),
        ])
    }

    fn datum(&self) -> Datum {
        Datum::Node {
            tag: Symbol::qualified("stats", "result-evidence"),
            fields: vec![
                (Symbol::new("metric"), Datum::Symbol(self.metric.clone())),
                (
                    Symbol::new("inputs"),
                    Datum::Vector(self.inputs.iter().copied().map(f64_datum).collect()),
                ),
                (Symbol::new("count"), u64_datum(self.inputs.len() as u64)),
                (Symbol::new("value"), f64_datum(self.value)),
            ],
        }
    }
}

/// Builds the public descriptive statistics Claim and interns its evidence object.
pub fn stats_result_claim(cx: &mut Cx, evidence: &StatsClaimEvidence) -> Result<Claim> {
    Claim::content_object(
        cx.datum_store_mut(),
        Ref::Symbol(evidence.metric().clone()),
        Symbol::new("stats-result"),
        evidence.datum(),
    )
}

/// A first-class runtime object wrapping a descriptive statistics Claim.
#[derive(Clone, Debug, PartialEq)]
pub struct StatsClaimValue {
    claim: Claim,
    evidence: StatsClaimEvidence,
}

impl StatsClaimValue {
    /// Creates a new runtime Claim wrapper.
    pub fn new(claim: Claim, evidence: StatsClaimEvidence) -> Self {
        Self { claim, evidence }
    }

    /// Returns the underlying kernel Claim record.
    pub fn claim(&self) -> &Claim {
        &self.claim
    }

    /// Returns the evidence used to compute the Claim object.
    pub fn evidence(&self) -> &StatsClaimEvidence {
        &self.evidence
    }

    /// Builds a runtime table exposing the Claim and evidence fields.
    pub fn table_value(&self, cx: &mut Cx) -> Result<Value> {
        let object = match &self.claim.object {
            Ref::Content(id) => cx.factory().string(format!("{id:?}"))?,
            other => cx.factory().string(format!("{other:?}"))?,
        };
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("claim".to_owned())?,
            ),
            (
                Symbol::new("subject"),
                cx.factory().symbol(self.evidence.metric().clone())?,
            ),
            (
                Symbol::new("predicate"),
                cx.factory().symbol(self.claim.predicate.clone())?,
            ),
            (Symbol::new("object"), object),
            (
                Symbol::new("evidence"),
                self.evidence.table_value(cx.factory())?,
            ),
        ])
    }
}

impl Object for StatsClaimValue {
    fn snapshot(&self, _cx: &mut Cx) -> Result<Option<Datum>> {
        Ok(Some(self.claim.canonical_datum()))
    }

    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!(
            "#<claim {} stats-result>",
            self.evidence.metric().as_qualified_str()
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ObjectCompat for StatsClaimValue {
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::from(self.claim.canonical_datum()))
    }

    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        self.table_value(cx)
    }
}

/// Wraps a descriptive statistics Claim as a runtime value.
pub fn stats_result_claim_value(cx: &mut Cx, evidence: StatsClaimEvidence) -> Result<Value> {
    let claim = stats_result_claim(cx, &evidence)?;
    cx.factory()
        .opaque(Arc::new(StatsClaimValue::new(claim, evidence)))
}

/// Builds the public disparate-impact fairness Claim and interns its evidence object.
pub fn fairness_claim(cx: &mut Cx, evidence: &FairnessEvidence) -> Result<Claim> {
    Claim::content_object(
        cx.datum_store_mut(),
        Ref::Symbol(Symbol::qualified("stats", "disparate-impact")),
        Symbol::new("fairness-result"),
        evidence.datum(),
    )
}

/// A first-class runtime object wrapping the fairness Claim and its evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct FairnessClaimValue {
    claim: Claim,
    evidence: FairnessEvidence,
}

impl FairnessClaimValue {
    /// Creates a new runtime Claim wrapper.
    pub fn new(claim: Claim, evidence: FairnessEvidence) -> Self {
        Self { claim, evidence }
    }

    /// Returns the underlying kernel Claim record.
    pub fn claim(&self) -> &Claim {
        &self.claim
    }

    /// Returns the evidence used to compute the Claim object.
    pub fn evidence(&self) -> &FairnessEvidence {
        &self.evidence
    }

    /// Builds a runtime table exposing the Claim and evidence fields.
    pub fn table_value(&self, cx: &mut Cx) -> Result<Value> {
        let object = match &self.claim.object {
            Ref::Content(id) => cx.factory().string(format!("{id:?}"))?,
            other => cx.factory().string(format!("{other:?}"))?,
        };
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("claim".to_owned())?,
            ),
            (
                Symbol::new("subject"),
                cx.factory()
                    .symbol(Symbol::qualified("stats", "disparate-impact"))?,
            ),
            (
                Symbol::new("predicate"),
                cx.factory().symbol(self.claim.predicate.clone())?,
            ),
            (Symbol::new("object"), object),
            (
                Symbol::new("evidence"),
                self.evidence.table_value(cx.factory())?,
            ),
        ])
    }
}

impl Object for FairnessClaimValue {
    fn snapshot(&self, _cx: &mut Cx) -> Result<Option<Datum>> {
        Ok(Some(self.claim.canonical_datum()))
    }

    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("#<claim stats/disparate-impact fairness-result>".to_owned())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ObjectCompat for FairnessClaimValue {
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::from(self.claim.canonical_datum()))
    }

    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        self.table_value(cx)
    }
}

/// Wraps a fairness Claim as a runtime value.
pub fn fairness_claim_value(cx: &mut Cx, evidence: FairnessEvidence) -> Result<Value> {
    let claim = fairness_claim(cx, &evidence)?;
    cx.factory()
        .opaque(Arc::new(FairnessClaimValue::new(claim, evidence)))
}

fn u64_value(factory: &dyn Factory, value: u64) -> Result<Value> {
    factory.number_literal(domains::u64(), value.to_string())
}

fn f64_value(factory: &dyn Factory, value: f64) -> Result<Value> {
    factory.number_literal(domains::f64(), value.to_string())
}

fn f64_list_value(factory: &dyn Factory, values: &[f64]) -> Result<Value> {
    let values = values
        .iter()
        .copied()
        .map(|value| f64_value(factory, value))
        .collect::<Result<Vec<_>>>()?;
    factory.list(values)
}

fn u64_datum(value: u64) -> Datum {
    Datum::Number(NumberLiteral {
        domain: domains::u64(),
        canonical: value.to_string(),
    })
}

fn f64_datum(value: f64) -> Datum {
    Datum::Number(NumberLiteral {
        domain: domains::f64(),
        canonical: value.to_string(),
    })
}

pub(crate) fn stats_error_to_kernel(error: crate::StatsError) -> Error {
    Error::DomainError {
        domain: Symbol::qualified("numbers", "stats"),
        category: Symbol::new("invalid-input"),
        message: error.to_string(),
    }
}
