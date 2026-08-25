#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Sealed interval certification: evidence may be inspected, never imported as authority.

use sim_kernel::{
    AbiVersion, Datum, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use std::{error::Error, fmt};

/// A non-authoritative numerical spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimateInterval {
    lower: f64,
    upper: f64,
}
impl EstimateInterval {
    /// Validates an ordered, non-NaN estimate. Infinities are allowed.
    pub fn new(lower: f64, upper: f64) -> Result<Self, IntervalError> {
        if lower.is_nan() || upper.is_nan() {
            return Err(IntervalError::NaN);
        }
        if lower > upper {
            return Err(IntervalError::Empty);
        }
        Ok(Self { lower, upper })
    }
    /// Lower estimate.
    pub fn lower(self) -> f64 {
        self.lower
    }
    /// Upper estimate.
    pub fn upper(self) -> f64 {
        self.upper
    }
}

/// Stable identity of a reviewed certification kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CertificationKernelId {
    /// Exact rational-to-binary64 enclosure kernel, version one.
    RationalBinary64V1,
    /// Directed IEEE-754 binary64 arithmetic kernel, version one.
    DirectedBinary64V1,
    /// Reviewed nonnegative square-root kernel, version one.
    SqrtBinary64V1,
    /// Scalar interval Newton/Krawczyk verifier, version one.
    ScalarRootV1,
}

/// Exact or content-addressed inputs retained by a certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CertifiedInput {
    /// Exact signed rational input.
    Rational {
        /// Numerator.
        numerator: i128,
        /// Positive denominator.
        denominator: i128,
    },
    /// Exact IEEE-754 bit pattern.
    Binary64(u64),
    /// Content identity of an earlier certificate or immutable input.
    ContentId(String),
}

/// Reviewable evidence from the issuing method.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertificationEvidence {
    kernel: CertificationKernelId,
    inputs: Vec<CertifiedInput>,
    trace_digest: String,
    method: String,
}
impl CertificationEvidence {
    /// Reviewed issuer.
    pub fn kernel(&self) -> CertificationKernelId {
        self.kernel
    }
    /// Exact inputs or their immutable identities.
    pub fn inputs(&self) -> &[CertifiedInput] {
        &self.inputs
    }
    /// Canonical operation-trace digest.
    pub fn trace_digest(&self) -> &str {
        &self.trace_digest
    }
    /// Human-reviewable method evidence.
    pub fn method(&self) -> &str {
        &self.method
    }
}

/// A mathematical enclosure. Its representation and issuer are private.
///
/// Estimate promotion and field construction do not compile:
/// ```compile_fail
/// use sim_lib_numbers_interval::{CertifiedInterval, EstimateInterval};
/// let estimate = EstimateInterval::new(0.0, 1.0).unwrap();
/// let _: CertifiedInterval = estimate.into();
/// ```
/// ```compile_fail
/// use sim_lib_numbers_interval::CertifiedInterval;
/// let _ = CertifiedInterval { lower: 0.0, upper: 1.0, evidence: panic!() };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedInterval {
    lower: f64,
    upper: f64,
    evidence: CertificationEvidence,
}
impl CertifiedInterval {
    fn issue(
        lower: f64,
        upper: f64,
        evidence: CertificationEvidence,
    ) -> Result<Self, IntervalError> {
        if lower.is_nan() || upper.is_nan() {
            return Err(IntervalError::NaN);
        }
        if lower > upper {
            return Err(IntervalError::Empty);
        }
        Ok(Self {
            lower: canonical_zero_lower(lower),
            upper: canonical_zero_upper(upper),
            evidence,
        })
    }
    /// Lower certified endpoint.
    pub fn lower(&self) -> f64 {
        self.lower
    }
    /// Upper certified endpoint.
    pub fn upper(&self) -> f64 {
        self.upper
    }
    /// Issuance evidence.
    pub fn evidence(&self) -> &CertificationEvidence {
        &self.evidence
    }
    /// Content identity used when composing certificates.
    pub fn content_id(&self) -> String {
        format!(
            "ci-v1:{:016x}:{:016x}:{}",
            self.lower.to_bits(),
            self.upper.to_bits(),
            self.evidence.trace_digest
        )
    }
    /// Directed addition.
    pub fn add(&self, rhs: &Self) -> Result<Self, IntervalError> {
        binary(
            self,
            rhs,
            "add",
            |a, b| a + b,
            (self.lower, rhs.lower),
            (self.upper, rhs.upper),
        )
    }
    /// Directed subtraction.
    pub fn sub(&self, rhs: &Self) -> Result<Self, IntervalError> {
        binary(
            self,
            rhs,
            "sub",
            |a, b| a - b,
            (self.lower, rhs.upper),
            (self.upper, rhs.lower),
        )
    }
    /// Directed multiplication.
    pub fn mul(&self, rhs: &Self) -> Result<Self, IntervalError> {
        let xs = [
            mul_bound(self.lower, rhs.lower)?,
            mul_bound(self.lower, rhs.upper)?,
            mul_bound(self.upper, rhs.lower)?,
            mul_bound(self.upper, rhs.upper)?,
        ];
        composed(
            self,
            rhs,
            "mul",
            down(xs.iter().copied().fold(f64::INFINITY, f64::min)),
            up(xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        )
    }
    /// Directed division, refusing zero-containing divisors.
    pub fn div(&self, rhs: &Self) -> Result<Self, IntervalError> {
        if rhs.lower <= 0.0 && rhs.upper >= 0.0 {
            return Err(IntervalError::ZeroContainingDivisor);
        }
        let reciprocal = Self::issue(
            down(1.0 / rhs.upper),
            up(1.0 / rhs.lower),
            evidence(
                CertificationKernelId::DirectedBinary64V1,
                vec![CertifiedInput::ContentId(rhs.content_id())],
                "reciprocal",
                "directed reciprocal; divisor excludes zero",
            ),
        )?;
        self.mul(&reciprocal)
    }
    /// Set intersection.
    pub fn intersection(&self, rhs: &Self) -> Result<Self, IntervalError> {
        composed(
            self,
            rhs,
            "intersection",
            self.lower.max(rhs.lower),
            self.upper.min(rhs.upper),
        )
    }
    /// Convex hull.
    pub fn hull(&self, rhs: &Self) -> Self {
        composed(
            self,
            rhs,
            "hull",
            self.lower.min(rhs.lower),
            self.upper.max(rhs.upper),
        )
        .expect("hull is ordered")
    }
    /// Directed midpoint enclosure, avoiding overflow.
    pub fn midpoint(&self) -> Result<Self, IntervalError> {
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return Err(IntervalError::Unbounded);
        }
        let m = self.lower / 2.0 + self.upper / 2.0;
        unary(self, "midpoint", down(m), up(m))
    }
    /// Directed width enclosure.
    pub fn width(&self) -> Result<Self, IntervalError> {
        unary(self, "width", 0.0, up(self.upper - self.lower))
    }
    /// Reviewed square root; negative inputs are outside the kernel domain.
    pub fn sqrt(&self) -> Result<Self, IntervalError> {
        if self.lower < 0.0 {
            return Err(IntervalError::Domain);
        }
        Self::issue(
            down(self.lower.sqrt()).max(0.0),
            up(self.upper.sqrt()),
            evidence(
                CertificationKernelId::SqrtBinary64V1,
                vec![CertifiedInput::ContentId(self.content_id())],
                "sqrt-v1",
                "IEEE sqrt is correctly rounded; endpoints stepped outward",
            ),
        )
    }
    /// Canonical inspection-only datum. There is intentionally no inverse decoder.
    pub fn to_datum(&self) -> Datum {
        Datum::Node {
            tag: Symbol::new("numbers-interval/certified-inspection-v1"),
            fields: vec![
                (
                    Symbol::new("lower-bits"),
                    Datum::String(format!("{:016x}", self.lower.to_bits())),
                ),
                (
                    Symbol::new("upper-bits"),
                    Datum::String(format!("{:016x}", self.upper.to_bits())),
                ),
                (
                    Symbol::new("kernel"),
                    Datum::String(format!("{:?}", self.evidence.kernel)),
                ),
                (
                    Symbol::new("trace-digest"),
                    Datum::String(self.evidence.trace_digest.clone()),
                ),
            ],
        }
    }
}

/// Reduced exact rational.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactRational {
    numerator: i128,
    denominator: i128,
}
impl ExactRational {
    /// Constructs a reduced rational using checked integer arithmetic.
    pub fn new(numerator: i128, denominator: i128) -> Result<Self, IntervalError> {
        if denominator == 0 {
            return Err(IntervalError::ZeroDenominator);
        }
        let negative = (numerator < 0) ^ (denominator < 0);
        let n = numerator.unsigned_abs();
        let d = denominator.unsigned_abs();
        let g = gcd(n, d);
        let nr = i128::try_from(n / g).map_err(|_| IntervalError::ExactOverflow)?;
        let dr = i128::try_from(d / g).map_err(|_| IntervalError::ExactOverflow)?;
        Ok(Self {
            numerator: if negative { -nr } else { nr },
            denominator: dr,
        })
    }
    /// Numerator.
    pub fn numerator(self) -> i128 {
        self.numerator
    }
    /// Positive denominator.
    pub fn denominator(self) -> i128 {
        self.denominator
    }
    fn add(self, o: Self) -> Result<Self, IntervalError> {
        Self::new(
            self.numerator
                .checked_mul(o.denominator)
                .and_then(|x| {
                    o.numerator
                        .checked_mul(self.denominator)
                        .and_then(|y| x.checked_add(y))
                })
                .ok_or(IntervalError::ExactOverflow)?,
            self.denominator
                .checked_mul(o.denominator)
                .ok_or(IntervalError::ExactOverflow)?,
        )
    }
    fn sub(self, o: Self) -> Result<Self, IntervalError> {
        Self::new(
            self.numerator
                .checked_mul(o.denominator)
                .and_then(|x| {
                    o.numerator
                        .checked_mul(self.denominator)
                        .and_then(|y| x.checked_sub(y))
                })
                .ok_or(IntervalError::ExactOverflow)?,
            self.denominator
                .checked_mul(o.denominator)
                .ok_or(IntervalError::ExactOverflow)?,
        )
    }
    fn mul(self, o: Self) -> Result<Self, IntervalError> {
        Self::new(
            self.numerator
                .checked_mul(o.numerator)
                .ok_or(IntervalError::ExactOverflow)?,
            self.denominator
                .checked_mul(o.denominator)
                .ok_or(IntervalError::ExactOverflow)?,
        )
    }
    fn div(self, o: Self) -> Result<Self, IntervalError> {
        if o.numerator == 0 {
            return Err(IntervalError::ZeroContainingDivisor);
        }
        Self::new(
            self.numerator
                .checked_mul(o.denominator)
                .ok_or(IntervalError::ExactOverflow)?,
            self.denominator
                .checked_mul(o.numerator)
                .ok_or(IntervalError::ExactOverflow)?,
        )
    }
    fn value(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r
    }
    a.max(1)
}

/// An exact rational interval, used as an independent enclosure oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RationalInterval {
    lower: ExactRational,
    upper: ExactRational,
}
impl RationalInterval {
    /// Constructs an ordered exact interval.
    pub fn new(lower: ExactRational, upper: ExactRational) -> Result<Self, IntervalError> {
        if cmp_rat(lower, upper).is_gt() {
            Err(IntervalError::Empty)
        } else {
            Ok(Self { lower, upper })
        }
    }
    /// Lower exact endpoint.
    pub fn lower(self) -> ExactRational {
        self.lower
    }
    /// Upper exact endpoint.
    pub fn upper(self) -> ExactRational {
        self.upper
    }
    /// Exact addition.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: Self) -> Result<Self, IntervalError> {
        Self::new(self.lower.add(o.lower)?, self.upper.add(o.upper)?)
    }
    /// Exact subtraction.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, o: Self) -> Result<Self, IntervalError> {
        Self::new(self.lower.sub(o.upper)?, self.upper.sub(o.lower)?)
    }
    /// Exact multiplication.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, o: Self) -> Result<Self, IntervalError> {
        let x = [
            self.lower.mul(o.lower)?,
            self.lower.mul(o.upper)?,
            self.upper.mul(o.lower)?,
            self.upper.mul(o.upper)?,
        ];
        Self::new(
            *x.iter().min_by(|a, b| cmp_rat(**a, **b)).unwrap(),
            *x.iter().max_by(|a, b| cmp_rat(**a, **b)).unwrap(),
        )
    }
    /// Exact division, refusing a divisor containing zero.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, o: Self) -> Result<Self, IntervalError> {
        let z = ExactRational::new(0, 1)?;
        if !cmp_rat(o.lower, z).is_gt() && !cmp_rat(o.upper, z).is_lt() {
            return Err(IntervalError::ZeroContainingDivisor);
        };
        self.mul(Self::new(
            ExactRational::new(o.upper.denominator, o.upper.numerator)?,
            ExactRational::new(o.lower.denominator, o.lower.numerator)?,
        )?)
    }
    /// Exact intersection.
    pub fn intersection(self, o: Self) -> Result<Self, IntervalError> {
        Self::new(
            if cmp_rat(self.lower, o.lower).is_gt() {
                self.lower
            } else {
                o.lower
            },
            if cmp_rat(self.upper, o.upper).is_lt() {
                self.upper
            } else {
                o.upper
            },
        )
    }
    /// Exact hull.
    pub fn hull(self, o: Self) -> Self {
        Self {
            lower: if cmp_rat(self.lower, o.lower).is_lt() {
                self.lower
            } else {
                o.lower
            },
            upper: if cmp_rat(self.upper, o.upper).is_gt() {
                self.upper
            } else {
                o.upper
            },
        }
    }
    /// Exact midpoint.
    pub fn midpoint(self) -> Result<ExactRational, IntervalError> {
        self.lower.add(self.upper)?.div(ExactRational::new(2, 1)?)
    }
    /// Exact width.
    pub fn width(self) -> Result<ExactRational, IntervalError> {
        self.upper.sub(self.lower)
    }
    /// Issues a binary64 enclosure from exact endpoints.
    pub fn certify(self) -> Result<CertifiedInterval, IntervalError> {
        CertifiedInterval::issue(
            down(self.lower.value()),
            up(self.upper.value()),
            evidence(
                CertificationKernelId::RationalBinary64V1,
                vec![rat_input(self.lower), rat_input(self.upper)],
                "rational-binary64-v1",
                "exact rational endpoints converted then stepped outward",
            ),
        )
    }
}
fn cmp_rat(a: ExactRational, b: ExactRational) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.numerator.signum(), b.numerator.signum()) {
        (x, y) if x != y => x.cmp(&y),
        (0, 0) => Ordering::Equal,
        (-1, -1) => cmp_positive(
            a.numerator.unsigned_abs(),
            a.denominator as u128,
            b.numerator.unsigned_abs(),
            b.denominator as u128,
        )
        .reverse(),
        _ => cmp_positive(
            a.numerator.unsigned_abs(),
            a.denominator as u128,
            b.numerator.unsigned_abs(),
            b.denominator as u128,
        ),
    }
}
fn cmp_positive(mut an: u128, mut ad: u128, mut bn: u128, mut bd: u128) -> std::cmp::Ordering {
    let mut reversed = false;
    loop {
        let (aq, ar) = (an / ad, an % ad);
        let (bq, br) = (bn / bd, bn % bd);
        if aq != bq {
            let o = aq.cmp(&bq);
            return if reversed { o.reverse() } else { o };
        }
        match (ar, br) {
            (0, 0) => return std::cmp::Ordering::Equal,
            (0, _) => {
                return if reversed {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                };
            }
            (_, 0) => {
                return if reversed {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
            }
            _ => {
                (an, ad, bn, bd) = (ad, ar, bd, br);
                reversed = !reversed;
            }
        }
    }
}
fn rat_input(r: ExactRational) -> CertifiedInput {
    CertifiedInput::Rational {
        numerator: r.numerator,
        denominator: r.denominator,
    }
}

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

fn evidence(
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
fn composed(
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
fn unary(
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
fn binary<F: Fn(f64, f64) -> f64>(
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
fn mul_bound(a: f64, b: f64) -> Result<f64, IntervalError> {
    let x = a * b;
    if x.is_nan() {
        Err(IntervalError::Indeterminate)
    } else {
        Ok(x)
    }
}
fn down(x: f64) -> f64 {
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
fn up(x: f64) -> f64 {
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
fn canonical_zero_lower(x: f64) -> f64 {
    if x == 0.0 { -0.0 } else { x }
}
fn canonical_zero_upper(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

/// Loadable inspection surface; it does not export a certificate constructor.
#[derive(Default)]
pub struct IntervalLib;
impl Lib for IntervalLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "interval"),
            version: Version(env!("CARGO_PKG_VERSION").into()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![],
            capabilities: vec![],
            exports: vec![Export::Value {
                symbol: interval_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(interval_schema_symbol(),cx.factory().string("sealed estimate certified rational directed-binary64 refusal interval-newton krawczyk threshold inspection-only".into())?)
    }
}
/// Runtime schema symbol.
pub fn interval_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/interval", "schema")
}
/// Embedded recipes.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
