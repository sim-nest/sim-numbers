//! Certified floating-point and exact-rational interval arithmetic.

use super::*;

/// A non-authoritative numerical spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimateInterval {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
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
    pub(crate) kernel: CertificationKernelId,
    pub(crate) inputs: Vec<CertifiedInput>,
    pub(crate) trace_digest: String,
    pub(crate) method: String,
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
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) evidence: CertificationEvidence,
}
impl CertifiedInterval {
    pub(crate) fn issue(
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
    pub(crate) fn value(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}
pub(crate) fn gcd(mut a: u128, mut b: u128) -> u128 {
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
pub(crate) fn cmp_rat(a: ExactRational, b: ExactRational) -> std::cmp::Ordering {
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
pub(crate) fn cmp_positive(
    mut an: u128,
    mut ad: u128,
    mut bn: u128,
    mut bd: u128,
) -> std::cmp::Ordering {
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
pub(crate) fn rat_input(r: ExactRational) -> CertifiedInput {
    CertifiedInput::Rational {
        numerator: r.numerator,
        denominator: r.denominator,
    }
}
