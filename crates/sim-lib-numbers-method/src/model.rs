use std::{error::Error, fmt};

/// Validation failure for a common method record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MethodError {
    /// A work bound was zero or could not be represented.
    InvalidWorkLimit,
    /// Charged work exceeded the admitted limit.
    WorkExceeded,
    /// A tolerance or achieved error was negative or non-finite.
    InvalidMeasure,
    /// No tolerances were requested.
    EmptyToleranceSet,
    /// A method identity was not in the closed registry.
    UnknownMethod(String),
    /// Execution identity was empty or contained control characters.
    InvalidExecutionIdentity,
    /// Evidence contradicted its termination claim or requested criteria.
    ContradictoryEvidence,
}

impl fmt::Display for MethodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::InvalidWorkLimit => "work limit must be positive and representable",
                Self::WorkExceeded => "charged work exceeds its admitted limit",
                Self::InvalidMeasure => "measure must be finite and non-negative",
                Self::EmptyToleranceSet => "at least one tolerance is required",
                Self::UnknownMethod(_) => "unknown numerical method identity",
                Self::InvalidExecutionIdentity => "invalid execution identity",
                Self::ContradictoryEvidence => "termination and numerical evidence contradict",
            }
        )
    }
}
impl Error for MethodError {}

/// Stable identity of a registered numerical method.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MethodId(String);
impl MethodId {
    /// Scaled-partial-pivoting dense solve.
    pub const DENSE_SCALED_PIVOT: &'static str = "numbers/dense-scaled-pivot-v1";
    /// Lloyd k-means iteration.
    pub const KMEANS_LLOYD: &'static str = "numbers/kmeans-lloyd-v1";
    /// Periodogram-family signal estimator.
    pub const SIGNAL_ESTIMATOR: &'static str = "numbers/signal-estimator-v1";
    /// Admits an identity from the closed method registry.
    pub fn new(value: impl Into<String>) -> Result<Self, MethodError> {
        let value = value.into();
        match value.as_str() {
            Self::DENSE_SCALED_PIVOT | Self::KMEANS_LLOYD | Self::SIGNAL_ESTIMATOR => {
                Ok(Self(value))
            }
            _ => Err(MethodError::UnknownMethod(value)),
        }
    }
    /// Registry string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Identity of a convergence criterion; distinct variants are not substitutable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CriterionId {
    /// Norm of an equation residual.
    ResidualNorm,
    /// Width of a bracketing interval.
    BracketWidth,
    /// Norm of the last iterate step.
    StepNorm,
    /// Embedded local truncation error.
    EmbeddedLocalError,
    /// Statistical objective or loss.
    ObjectiveValue,
    /// Signal-grid resolution.
    SpectralResolution,
}

/// Stable precision identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrecisionId {
    /// IEEE-754 binary64.
    Binary64,
}

/// Stable refusal reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefusalId {
    /// Inputs were invalid.
    InvalidInput,
    /// The plan was unsupported.
    Unsupported,
    /// Admission would exceed resources.
    ResourceAdmission,
}

/// Positive, representable method work ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkLimit(u64);
impl WorkLimit {
    /// Constructs a positive limit.
    pub fn new(value: u64) -> Result<Self, MethodError> {
        (value > 0)
            .then_some(Self(value))
            .ok_or(MethodError::InvalidWorkLimit)
    }
    /// Raw work units.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Saturating charged-work counter paired with its admission ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkReceipt {
    charged: u64,
    limit: WorkLimit,
    saturated: bool,
}
impl WorkReceipt {
    /// Constructs a receipt, rejecting forged over-limit work.
    pub fn new(charged: u64, limit: WorkLimit, saturated: bool) -> Result<Self, MethodError> {
        if charged > limit.get() {
            return Err(MethodError::WorkExceeded);
        }
        Ok(Self {
            charged,
            limit,
            saturated,
        })
    }
    /// Empty receipt.
    pub fn empty(limit: WorkLimit) -> Self {
        Self {
            charged: 0,
            limit,
            saturated: false,
        }
    }
    /// Charges work without overflow and saturates at the limit.
    pub fn charge(self, units: u64) -> Self {
        let charged = self.charged.saturating_add(units).min(self.limit.get());
        Self {
            charged,
            limit: self.limit,
            saturated: charged == self.limit.get(),
        }
    }
    /// Charged units.
    pub fn charged(self) -> u64 {
        self.charged
    }
    /// Admitted ceiling.
    pub fn limit(self) -> WorkLimit {
        self.limit
    }
    /// Whether charging reached the ceiling.
    pub fn saturated(self) -> bool {
        self.saturated
    }
}

/// One criterion-specific finite, non-negative measure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorMeasure {
    criterion: CriterionId,
    value: f64,
}
impl ErrorMeasure {
    /// Validates a measure.
    pub fn new(criterion: CriterionId, value: f64) -> Result<Self, MethodError> {
        if !value.is_finite() || value < 0.0 {
            return Err(MethodError::InvalidMeasure);
        }
        Ok(Self { criterion, value })
    }
    /// Criterion identity.
    pub fn criterion(self) -> CriterionId {
        self.criterion
    }
    /// Measure value, preserving signed zero.
    pub fn value(self) -> f64 {
        self.value
    }
}

/// Requested criterion-specific tolerances.
#[derive(Clone, Debug, PartialEq)]
pub struct ToleranceSet(Vec<ErrorMeasure>);
impl ToleranceSet {
    /// Constructs a non-empty set with no duplicate criterion.
    pub fn new(values: impl IntoIterator<Item = ErrorMeasure>) -> Result<Self, MethodError> {
        let mut values: Vec<_> = values.into_iter().collect();
        if values.is_empty() {
            return Err(MethodError::EmptyToleranceSet);
        }
        values.sort_by_key(|v| v.criterion as u8);
        if values.windows(2).any(|v| v[0].criterion == v[1].criterion) {
            return Err(MethodError::ContradictoryEvidence);
        }
        Ok(Self(values))
    }
    /// Measures in stable criterion order.
    pub fn measures(&self) -> &[ErrorMeasure] {
        &self.0
    }
    /// Looks up a requested criterion.
    pub fn get(&self, criterion: CriterionId) -> Option<ErrorMeasure> {
        self.0.iter().copied().find(|v| v.criterion == criterion)
    }
}

/// Validated bounded plan shared by numerical algorithms.
#[derive(Clone, Debug, PartialEq)]
pub struct MethodPlan {
    method: MethodId,
    work_limit: WorkLimit,
    tolerances: ToleranceSet,
}
impl MethodPlan {
    /// Constructs a plan from already validated parts.
    pub fn new(method: MethodId, work_limit: WorkLimit, tolerances: ToleranceSet) -> Self {
        Self {
            method,
            work_limit,
            tolerances,
        }
    }
    /// Method identity.
    pub fn method(&self) -> &MethodId {
        &self.method
    }
    /// Work ceiling.
    pub fn work_limit(&self) -> WorkLimit {
        self.work_limit
    }
    /// Requested tolerances.
    pub fn tolerances(&self) -> &ToleranceSet {
        &self.tolerances
    }
}

/// Why execution stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Termination {
    /// A named requested criterion was achieved.
    Converged {
        /// Criterion that caused convergence.
        criterion: CriterionId,
    },
    /// Work admission stopped execution.
    WorkLimit,
    /// An iteration bound stopped execution.
    IterationLimit,
    /// Execution was externally interrupted.
    Interrupted,
    /// Execution was refused before numerical success.
    Refused {
        /// Stable refusal reason.
        reason: RefusalId,
    },
}

/// Stable identity of one execution environment and invocation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionIdentity {
    engine: String,
    implementation: String,
    invocation: String,
}
impl ExecutionIdentity {
    /// Validates non-empty, printable identity components.
    pub fn new(
        engine: impl Into<String>,
        implementation: impl Into<String>,
        invocation: impl Into<String>,
    ) -> Result<Self, MethodError> {
        let value = Self {
            engine: engine.into(),
            implementation: implementation.into(),
            invocation: invocation.into(),
        };
        if [&value.engine, &value.implementation, &value.invocation]
            .iter()
            .any(|s| s.is_empty() || s.chars().any(char::is_control))
        {
            return Err(MethodError::InvalidExecutionIdentity);
        }
        Ok(value)
    }
    /// Engine identity.
    pub fn engine(&self) -> &str {
        &self.engine
    }
    /// Implementation identity.
    pub fn implementation(&self) -> &str {
        &self.implementation
    }
    /// Invocation identity.
    pub fn invocation(&self) -> &str {
        &self.invocation
    }
}

/// Validated common evidence retained beside a domain report.
#[derive(Clone, Debug, PartialEq)]
pub struct MethodEvidence {
    method: MethodId,
    termination: Termination,
    work: WorkReceipt,
    requested: ToleranceSet,
    achieved: Vec<ErrorMeasure>,
    precision: PrecisionId,
    execution: ExecutionIdentity,
}
impl MethodEvidence {
    /// Validates common evidence and rejects forged convergence.
    pub fn new(
        method: MethodId,
        termination: Termination,
        work: WorkReceipt,
        requested: ToleranceSet,
        mut achieved: Vec<ErrorMeasure>,
        precision: PrecisionId,
        execution: ExecutionIdentity,
    ) -> Result<Self, MethodError> {
        achieved.sort_by_key(|v| v.criterion as u8);
        if achieved
            .windows(2)
            .any(|v| v[0].criterion == v[1].criterion)
        {
            return Err(MethodError::ContradictoryEvidence);
        }
        match &termination {
            Termination::Converged { criterion } => {
                let target = requested
                    .get(*criterion)
                    .ok_or(MethodError::ContradictoryEvidence)?;
                let actual = achieved
                    .iter()
                    .find(|v| v.criterion == *criterion)
                    .ok_or(MethodError::ContradictoryEvidence)?;
                if actual.value > target.value {
                    return Err(MethodError::ContradictoryEvidence);
                }
            }
            Termination::WorkLimit if !work.saturated => {
                return Err(MethodError::ContradictoryEvidence);
            }
            Termination::Refused { .. } if work.charged != 0 || !achieved.is_empty() => {
                return Err(MethodError::ContradictoryEvidence);
            }
            _ => {}
        }
        Ok(Self {
            method,
            termination,
            work,
            requested,
            achieved,
            precision,
            execution,
        })
    }
    /// Method identity.
    pub fn method(&self) -> &MethodId {
        &self.method
    }
    /// Termination reason.
    pub fn termination(&self) -> &Termination {
        &self.termination
    }
    /// Work receipt.
    pub fn work(&self) -> WorkReceipt {
        self.work
    }
    /// Requested tolerances.
    pub fn requested(&self) -> &ToleranceSet {
        &self.requested
    }
    /// Achieved measures.
    pub fn achieved(&self) -> &[ErrorMeasure] {
        &self.achieved
    }
    /// Precision.
    pub fn precision(&self) -> PrecisionId {
        self.precision
    }
    /// Execution identity.
    pub fn execution(&self) -> &ExecutionIdentity {
        &self.execution
    }
}
