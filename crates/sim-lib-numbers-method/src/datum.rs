use std::{error::Error, fmt};

use sim_kernel::{Datum, Symbol};

use crate::{
    CriterionId, ErrorMeasure, ExecutionIdentity, MethodError, MethodEvidence, MethodId,
    MethodPlan, PrecisionId, RefusalId, Termination, ToleranceSet, WorkLimit, WorkReceipt,
};

/// Failure to decode a canonical method datum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DatumError {
    /// Datum shape, tag, field, or scalar encoding was invalid.
    Invalid(&'static str),
    /// Decoded values violated record invariants.
    Method(MethodError),
}
impl fmt::Display for DatumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(v) => write!(f, "invalid method datum: {v}"),
            Self::Method(v) => v.fmt(f),
        }
    }
}
impl Error for DatumError {}
impl From<MethodError> for DatumError {
    fn from(value: MethodError) -> Self {
        Self::Method(value)
    }
}

/// Exact canonical [`Datum`] projection for a common method record.
pub trait CanonicalDatum: Sized {
    /// Projects this record.
    fn to_datum(&self) -> Datum;
    /// Validates and reconstructs this record.
    fn from_datum(datum: &Datum) -> Result<Self, DatumError>;
}

fn node(tag: &str, fields: Vec<(&str, Datum)>) -> Datum {
    Datum::Node {
        tag: Symbol::new(tag),
        fields: fields
            .into_iter()
            .map(|(k, v)| (Symbol::new(k), v))
            .collect(),
    }
}
fn fields<'a>(datum: &'a Datum, tag: &str) -> Result<&'a [(Symbol, Datum)], DatumError> {
    match datum {
        Datum::Node {
            tag: actual,
            fields,
        } if actual.name.as_ref() == tag => Ok(fields),
        _ => Err(DatumError::Invalid("node tag")),
    }
}
fn field<'a>(fields: &'a [(Symbol, Datum)], name: &str) -> Result<&'a Datum, DatumError> {
    fields
        .iter()
        .find(|(key, _)| key.name.as_ref() == name)
        .map(|(_, value)| value)
        .ok_or(DatumError::Invalid("missing field"))
}
fn text(d: &Datum) -> Result<&str, DatumError> {
    if let Datum::String(v) = d {
        Ok(v)
    } else {
        Err(DatumError::Invalid("string"))
    }
}
fn u64_d(v: u64) -> Datum {
    Datum::String(v.to_string())
}
fn u64_v(d: &Datum) -> Result<u64, DatumError> {
    text(d)?.parse().map_err(|_| DatumError::Invalid("u64"))
}
fn bool_v(d: &Datum) -> Result<bool, DatumError> {
    if let Datum::Bool(v) = d {
        Ok(*v)
    } else {
        Err(DatumError::Invalid("bool"))
    }
}
fn finite_d(v: f64) -> Datum {
    Datum::String(format!("binary64:{:016x}", v.to_bits()))
}
fn finite_v(d: &Datum) -> Result<f64, DatumError> {
    let bits = text(d)?
        .strip_prefix("binary64:")
        .ok_or(DatumError::Invalid("binary64 prefix"))?;
    let value = f64::from_bits(
        u64::from_str_radix(bits, 16).map_err(|_| DatumError::Invalid("binary64 bits"))?,
    );
    if value.is_finite() {
        Ok(value)
    } else {
        Err(DatumError::Invalid("non-finite binary64"))
    }
}
fn criterion_name(v: CriterionId) -> &'static str {
    match v {
        CriterionId::ResidualNorm => "residual-norm",
        CriterionId::BracketWidth => "bracket-width",
        CriterionId::StepNorm => "step-norm",
        CriterionId::EmbeddedLocalError => "embedded-local-error",
        CriterionId::ObjectiveValue => "objective-value",
        CriterionId::SpectralResolution => "spectral-resolution",
    }
}
fn criterion_v(v: &str) -> Result<CriterionId, DatumError> {
    match v {
        "residual-norm" => Ok(CriterionId::ResidualNorm),
        "bracket-width" => Ok(CriterionId::BracketWidth),
        "step-norm" => Ok(CriterionId::StepNorm),
        "embedded-local-error" => Ok(CriterionId::EmbeddedLocalError),
        "objective-value" => Ok(CriterionId::ObjectiveValue),
        "spectral-resolution" => Ok(CriterionId::SpectralResolution),
        _ => Err(DatumError::Invalid("criterion")),
    }
}

impl CanonicalDatum for WorkLimit {
    fn to_datum(&self) -> Datum {
        node(
            "numbers-method/work-limit-v1",
            vec![("units", u64_d(self.get()))],
        )
    }
    fn from_datum(d: &Datum) -> Result<Self, DatumError> {
        WorkLimit::new(u64_v(field(
            fields(d, "numbers-method/work-limit-v1")?,
            "units",
        )?)?)
        .map_err(Into::into)
    }
}
impl CanonicalDatum for WorkReceipt {
    fn to_datum(&self) -> Datum {
        node(
            "numbers-method/work-receipt-v1",
            vec![
                ("charged", u64_d(self.charged())),
                ("limit", self.limit().to_datum()),
                ("saturated", Datum::Bool(self.saturated())),
            ],
        )
    }
    fn from_datum(d: &Datum) -> Result<Self, DatumError> {
        let f = fields(d, "numbers-method/work-receipt-v1")?;
        WorkReceipt::new(
            u64_v(field(f, "charged")?)?,
            WorkLimit::from_datum(field(f, "limit")?)?,
            bool_v(field(f, "saturated")?)?,
        )
        .map_err(Into::into)
    }
}
impl CanonicalDatum for ErrorMeasure {
    fn to_datum(&self) -> Datum {
        node(
            "numbers-method/error-measure-v1",
            vec![
                (
                    "criterion",
                    Datum::String(criterion_name(self.criterion()).into()),
                ),
                ("value", finite_d(self.value())),
            ],
        )
    }
    fn from_datum(d: &Datum) -> Result<Self, DatumError> {
        let f = fields(d, "numbers-method/error-measure-v1")?;
        ErrorMeasure::new(
            criterion_v(text(field(f, "criterion")?)?)?,
            finite_v(field(f, "value")?)?,
        )
        .map_err(Into::into)
    }
}
impl CanonicalDatum for ToleranceSet {
    fn to_datum(&self) -> Datum {
        node(
            "numbers-method/tolerance-set-v1",
            vec![(
                "measures",
                Datum::Vector(
                    self.measures()
                        .iter()
                        .map(CanonicalDatum::to_datum)
                        .collect(),
                ),
            )],
        )
    }
    fn from_datum(d: &Datum) -> Result<Self, DatumError> {
        let f = fields(d, "numbers-method/tolerance-set-v1")?;
        let Datum::Vector(v) = field(f, "measures")? else {
            return Err(DatumError::Invalid("measure vector"));
        };
        ToleranceSet::new(
            v.iter()
                .map(ErrorMeasure::from_datum)
                .collect::<Result<Vec<_>, _>>()?,
        )
        .map_err(Into::into)
    }
}
impl CanonicalDatum for MethodPlan {
    fn to_datum(&self) -> Datum {
        node(
            "numbers-method/plan-v1",
            vec![
                ("method", Datum::String(self.method().as_str().into())),
                ("work-limit", self.work_limit().to_datum()),
                ("tolerances", self.tolerances().to_datum()),
            ],
        )
    }
    fn from_datum(d: &Datum) -> Result<Self, DatumError> {
        let f = fields(d, "numbers-method/plan-v1")?;
        Ok(MethodPlan::new(
            MethodId::new(text(field(f, "method")?)?)?,
            WorkLimit::from_datum(field(f, "work-limit")?)?,
            ToleranceSet::from_datum(field(f, "tolerances")?)?,
        ))
    }
}
impl CanonicalDatum for ExecutionIdentity {
    fn to_datum(&self) -> Datum {
        node(
            "numbers-method/execution-v1",
            vec![
                ("engine", Datum::String(self.engine().into())),
                (
                    "implementation",
                    Datum::String(self.implementation().into()),
                ),
                ("invocation", Datum::String(self.invocation().into())),
            ],
        )
    }
    fn from_datum(d: &Datum) -> Result<Self, DatumError> {
        let f = fields(d, "numbers-method/execution-v1")?;
        ExecutionIdentity::new(
            text(field(f, "engine")?)?,
            text(field(f, "implementation")?)?,
            text(field(f, "invocation")?)?,
        )
        .map_err(Into::into)
    }
}
impl CanonicalDatum for Termination {
    fn to_datum(&self) -> Datum {
        match self {
            Self::Converged { criterion } => node(
                "numbers-method/termination-v1",
                vec![
                    ("kind", Datum::String("converged".into())),
                    ("detail", Datum::String(criterion_name(*criterion).into())),
                ],
            ),
            Self::WorkLimit => term("work-limit"),
            Self::IterationLimit => term("iteration-limit"),
            Self::Interrupted => term("interrupted"),
            Self::Refused { reason } => node(
                "numbers-method/termination-v1",
                vec![
                    ("kind", Datum::String("refused".into())),
                    (
                        "detail",
                        Datum::String(
                            match reason {
                                RefusalId::InvalidInput => "invalid-input",
                                RefusalId::Unsupported => "unsupported",
                                RefusalId::ResourceAdmission => "resource-admission",
                            }
                            .into(),
                        ),
                    ),
                ],
            ),
        }
    }
    fn from_datum(d: &Datum) -> Result<Self, DatumError> {
        let f = fields(d, "numbers-method/termination-v1")?;
        match text(field(f, "kind")?)? {
            "converged" => Ok(Self::Converged {
                criterion: criterion_v(text(field(f, "detail")?)?)?,
            }),
            "work-limit" => Ok(Self::WorkLimit),
            "iteration-limit" => Ok(Self::IterationLimit),
            "interrupted" => Ok(Self::Interrupted),
            "refused" => Ok(Self::Refused {
                reason: match text(field(f, "detail")?)? {
                    "invalid-input" => RefusalId::InvalidInput,
                    "unsupported" => RefusalId::Unsupported,
                    "resource-admission" => RefusalId::ResourceAdmission,
                    _ => return Err(DatumError::Invalid("refusal")),
                },
            }),
            _ => Err(DatumError::Invalid("termination")),
        }
    }
}
fn term(kind: &str) -> Datum {
    node(
        "numbers-method/termination-v1",
        vec![("kind", Datum::String(kind.into())), ("detail", Datum::Nil)],
    )
}
impl CanonicalDatum for MethodEvidence {
    fn to_datum(&self) -> Datum {
        node(
            "numbers-method/evidence-v1",
            vec![
                ("method", Datum::String(self.method().as_str().into())),
                ("termination", self.termination().to_datum()),
                ("work", self.work().to_datum()),
                ("requested", self.requested().to_datum()),
                (
                    "achieved",
                    Datum::Vector(
                        self.achieved()
                            .iter()
                            .map(CanonicalDatum::to_datum)
                            .collect(),
                    ),
                ),
                ("precision", Datum::String("binary64".into())),
                ("execution", self.execution().to_datum()),
            ],
        )
    }
    fn from_datum(d: &Datum) -> Result<Self, DatumError> {
        let f = fields(d, "numbers-method/evidence-v1")?;
        let Datum::Vector(a) = field(f, "achieved")? else {
            return Err(DatumError::Invalid("achieved vector"));
        };
        if text(field(f, "precision")?)? != "binary64" {
            return Err(DatumError::Invalid("precision"));
        }
        MethodEvidence::new(
            MethodId::new(text(field(f, "method")?)?)?,
            Termination::from_datum(field(f, "termination")?)?,
            WorkReceipt::from_datum(field(f, "work")?)?,
            ToleranceSet::from_datum(field(f, "requested")?)?,
            a.iter()
                .map(ErrorMeasure::from_datum)
                .collect::<Result<Vec<_>, _>>()?,
            PrecisionId::Binary64,
            ExecutionIdentity::from_datum(field(f, "execution")?)?,
        )
        .map_err(Into::into)
    }
}
