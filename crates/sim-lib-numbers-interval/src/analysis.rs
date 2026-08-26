//! Ordering, threshold classification, verified roots, and evidence composition.

use super::*;

/// Definite ordering between disjoint certified intervals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertifiedOrdering {
    /// Entirely below.
    Below,
    /// Overlapping.
    Overlap,
    /// Entirely above.
    Above,
}
/// Compares two certified intervals.
pub fn compare(a: &CertifiedInterval, b: &CertifiedInterval) -> CertifiedOrdering {
    if a.upper < b.lower {
        CertifiedOrdering::Below
    } else if a.lower > b.upper {
        CertifiedOrdering::Above
    } else {
        CertifiedOrdering::Overlap
    }
}
/// Threshold decision result.
#[derive(Clone, Debug, PartialEq)]
pub enum ThresholdVerdict {
    /// Value is definitely below.
    Below,
    /// Value is definitely above.
    Above,
    /// Proof is insufficient, retaining the estimate.
    Unresolved(EstimateInterval),
}
/// Classifies only disjoint certified inputs; overlap remains unresolved.
pub fn classify_threshold(
    value: &CertifiedInterval,
    threshold: &CertifiedInterval,
) -> ThresholdVerdict {
    match compare(value, threshold) {
        CertifiedOrdering::Below => ThresholdVerdict::Below,
        CertifiedOrdering::Above => ThresholdVerdict::Above,
        CertifiedOrdering::Overlap => ThresholdVerdict::Unresolved(EstimateInterval {
            lower: value.lower.min(threshold.lower),
            upper: value.upper.max(threshold.upper),
        }),
    }
}
/// An uncertified threshold can never produce a definite verdict.
pub fn classify_estimate(value: EstimateInterval, threshold: EstimateInterval) -> ThresholdVerdict {
    ThresholdVerdict::Unresolved(EstimateInterval {
        lower: value.lower.min(threshold.lower),
        upper: value.upper.max(threshold.upper),
    })
}

/// Verified root conclusion.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedRoot {
    enclosure: CertifiedInterval,
    existence: bool,
    uniqueness: bool,
    method: RootMethod,
}
impl VerifiedRoot {
    /// Certified root enclosure.
    pub fn enclosure(&self) -> &CertifiedInterval {
        &self.enclosure
    }
    /// Whether existence was proved.
    pub fn existence(&self) -> bool {
        self.existence
    }
    /// Whether uniqueness was proved.
    pub fn uniqueness(&self) -> bool {
        self.uniqueness
    }
    /// Verification operator.
    pub fn method(&self) -> RootMethod {
        self.method
    }
}
/// Scalar verification operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootMethod {
    /// Interval Newton.
    IntervalNewton,
    /// One-dimensional Krawczyk operator.
    Krawczyk,
}
/// Applies scalar interval Newton to certified `f(m)` and derivative enclosures.
pub fn interval_newton(
    x: &CertifiedInterval,
    f_mid: &CertifiedInterval,
    derivative: &CertifiedInterval,
) -> Result<VerifiedRoot, IntervalError> {
    let m = x.midpoint()?;
    let image = m.sub(&f_mid.div(derivative)?)?;
    let enclosed = x.intersection(&image)?;
    let unique = image.lower > x.lower && image.upper < x.upper;
    let enclosure = CertifiedInterval::issue(
        enclosed.lower,
        enclosed.upper,
        evidence(
            CertificationKernelId::ScalarRootV1,
            vec![
                CertifiedInput::ContentId(x.content_id()),
                CertifiedInput::ContentId(f_mid.content_id()),
                CertifiedInput::ContentId(derivative.content_id()),
            ],
            "interval-newton-v1",
            if unique {
                "Newton image strictly interior: existence and uniqueness"
            } else {
                "Newton image intersects domain: existence not certified"
            },
        ),
    )?;
    Ok(VerifiedRoot {
        enclosure,
        existence: unique,
        uniqueness: unique,
        method: RootMethod::IntervalNewton,
    })
}
/// Applies the scalar Krawczyk operator with a certified point derivative inverse.
pub fn krawczyk(
    x: &CertifiedInterval,
    f_mid: &CertifiedInterval,
    derivative: &CertifiedInterval,
    center_inverse: &CertifiedInterval,
) -> Result<VerifiedRoot, IntervalError> {
    let m = x.midpoint()?;
    let one =
        RationalInterval::new(ExactRational::new(1, 1)?, ExactRational::new(1, 1)?)?.certify()?;
    let k = m.sub(&center_inverse.mul(f_mid)?)?.add(
        &one.sub(&center_inverse.mul(derivative)?)?
            .mul(&x.sub(&m)?)?,
    )?;
    let enclosed = x.intersection(&k)?;
    let unique = k.lower > x.lower && k.upper < x.upper;
    let enclosure = CertifiedInterval::issue(
        enclosed.lower,
        enclosed.upper,
        evidence(
            CertificationKernelId::ScalarRootV1,
            vec![
                CertifiedInput::ContentId(x.content_id()),
                CertifiedInput::ContentId(f_mid.content_id()),
                CertifiedInput::ContentId(derivative.content_id()),
                CertifiedInput::ContentId(center_inverse.content_id()),
            ],
            "krawczyk-v1",
            if unique {
                "Krawczyk image strictly interior: existence and uniqueness"
            } else {
                "Krawczyk image intersects domain: existence not certified"
            },
        ),
    )?;
    Ok(VerifiedRoot {
        enclosure,
        existence: unique,
        uniqueness: unique,
        method: RootMethod::Krawczyk,
    })
}

/// Explicit refusal for elementary functions outside reviewed kernels.
pub fn certify_elementary(
    name: &str,
    _input: &CertifiedInterval,
) -> Result<CertifiedInterval, IntervalError> {
    match name {
        "sqrt" => _input.sqrt(),
        _ => Err(IntervalError::UnsupportedElementary(name.into())),
    }
}

/// Certification failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntervalError {
    /// NaN has no ordered enclosure semantics.
    NaN,
    /// Endpoints do not form a nonempty interval.
    Empty,
    /// Exact denominator was zero.
    ZeroDenominator,
    /// Exact fixed-width arithmetic overflowed.
    ExactOverflow,
    /// Division interval contains zero.
    ZeroContainingDivisor,
    /// Operation requires finite bounds.
    Unbounded,
    /// Input is outside a reviewed kernel domain.
    Domain,
    /// No reviewed elementary kernel exists.
    UnsupportedElementary(String),
    /// IEEE indeterminate form such as zero times infinity.
    Indeterminate,
}
impl fmt::Display for IntervalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for IntervalError {}

pub(crate) fn evidence(
    kernel: CertificationKernelId,
    inputs: Vec<CertifiedInput>,
    trace: &str,
    method: &str,
) -> CertificationEvidence {
    let mut h = 0xcbf29ce484222325u64;
    for b in trace
        .bytes()
        .chain(method.bytes())
        .chain(format!("{kernel:?}{inputs:?}").bytes())
    {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3)
    }
    CertificationEvidence {
        kernel,
        inputs,
        trace_digest: format!("fnv1a64:{h:016x}"),
        method: method.into(),
    }
}
pub(crate) fn composed(
    a: &CertifiedInterval,
    b: &CertifiedInterval,
    op: &str,
    lo: f64,
    hi: f64,
) -> Result<CertifiedInterval, IntervalError> {
    CertifiedInterval::issue(
        lo,
        hi,
        evidence(
            CertificationKernelId::DirectedBinary64V1,
            vec![
                CertifiedInput::ContentId(a.content_id()),
                CertifiedInput::ContentId(b.content_id()),
            ],
            op,
            "directed IEEE-754 endpoint arithmetic",
        ),
    )
}
pub(crate) fn unary(
    a: &CertifiedInterval,
    op: &str,
    lo: f64,
    hi: f64,
) -> Result<CertifiedInterval, IntervalError> {
    CertifiedInterval::issue(
        lo,
        hi,
        evidence(
            CertificationKernelId::DirectedBinary64V1,
            vec![CertifiedInput::ContentId(a.content_id())],
            op,
            "directed IEEE-754 endpoint arithmetic",
        ),
    )
}
pub(crate) fn binary<F: Fn(f64, f64) -> f64>(
    a: &CertifiedInterval,
    b: &CertifiedInterval,
    op: &str,
    f: F,
    lo: (f64, f64),
    hi: (f64, f64),
) -> Result<CertifiedInterval, IntervalError> {
    let l = f(lo.0, lo.1);
    let h = f(hi.0, hi.1);
    if l.is_nan() || h.is_nan() {
        return Err(IntervalError::Indeterminate);
    };
    composed(a, b, op, down(l), up(h))
}
pub(crate) fn mul_bound(a: f64, b: f64) -> Result<f64, IntervalError> {
    let x = a * b;
    if x.is_nan() {
        Err(IntervalError::Indeterminate)
    } else {
        Ok(x)
    }
}
pub(crate) fn down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        x
    } else if x == 0.0 {
        -f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}
pub(crate) fn up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        x
    } else if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}
pub(crate) fn canonical_zero_lower(x: f64) -> f64 {
    if x == 0.0 { -0.0 } else { x }
}
pub(crate) fn canonical_zero_upper(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}
