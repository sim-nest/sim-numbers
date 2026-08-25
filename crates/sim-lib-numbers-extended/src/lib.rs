#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Normalized double-double arithmetic and the `numbers/extended` runtime
//! domain. A value is an unevaluated sum of two binary64 components with
//! `|lo| <= 0.5 ulp(hi)` for finite nonzero results.
//!
//! Addition and multiplication are built from Knuth/Dekker error-free
//! transforms (`two_sum`, `quick_two_sum`, and an FMA `two_prod`). Under
//! round-to-nearest binary64 arithmetic, their residuals are exact. Division
//! and square root use two correction steps. The elementary functions perform
//! explicit argument reduction followed by double-double Taylor evaluation.
//! Checked specimens require errors below 32 double-double ulps on their stated
//! compact domains; no correct-rounding claim is made.

use core::{cmp::Ordering, fmt, ops};
use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Export, Factory, Lib, LibManifest, LibTarget, Linker, NumberDomain,
    NumberLiteral, Object, PromotionRule, Result, Symbol, Value, Version,
};
use sim_lib_numbers_core::{RealScalar, domains};

/// Recipes embedded for runtime discovery.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

/// A normalized sum of two binary64 components.
#[derive(Clone, Copy, Default)]
pub struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    /// Additive identity.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    /// Multiplicative identity.
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    /// Approximate unit roundoff of the two-component format.
    pub const EPSILON: Self = Self {
        hi: 2.465_190_328_815_662e-32,
        lo: 0.0,
    };
    /// Circular constant to double-double precision.
    pub const PI: Self = Self {
        hi: core::f64::consts::PI,
        lo: 1.224_646_799_147_353_2e-16,
    };
    /// Natural logarithm of two to double-double precision.
    pub const LN_2: Self = Self {
        hi: core::f64::consts::LN_2,
        lo: 2.319_046_813_846_299_6e-17,
    };

    /// Constructs and normalizes two components. Non-finite high components
    /// are canonicalized with a zero low component; a non-finite low component
    /// produces NaN.
    pub fn new(hi: f64, lo: f64) -> Self {
        if !hi.is_finite() {
            return Self { hi, lo: 0.0 };
        }
        if !lo.is_finite() {
            return Self {
                hi: f64::NAN,
                lo: 0.0,
            };
        }
        let (hi, lo) = two_sum(hi, lo);
        Self { hi, lo }
    }

    /// Constructs exactly from one binary64 value.
    pub const fn from_f64_exact(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// Constructs an integer without first rounding the complete value through
    /// binary64. Each base-2 digit is accumulated in double-double arithmetic.
    pub fn from_i128(value: i128) -> Self {
        let negative = value.is_negative();
        let mut magnitude = value.unsigned_abs();
        let mut place = Self::ONE;
        let mut result = Self::ZERO;
        while magnitude != 0 {
            if magnitude & 1 != 0 {
                result = result + place;
            }
            magnitude >>= 1;
            if magnitude != 0 {
                place = place + place;
            }
        }
        if negative { -result } else { result }
    }

    /// Constructs the nearest double-double quotient of two exact `i128`
    /// components. A zero denominator is rejected.
    pub fn from_ratio_i128(numerator: i128, denominator: i128) -> Option<Self> {
        (denominator != 0).then(|| Self::from_i128(numerator) / Self::from_i128(denominator))
    }

    /// Reconstructs an exact component identity after validating normalization.
    pub fn from_bits(hi: u64, lo: u64) -> Option<Self> {
        let raw = Self {
            hi: f64::from_bits(hi),
            lo: f64::from_bits(lo),
        };
        let normalized = Self::new(raw.hi, raw.lo);
        (normalized.hi.to_bits() == hi && normalized.lo.to_bits() == lo).then_some(raw)
    }

    /// Returns exact component bits, suitable for lossless records and codecs.
    pub fn to_bits(self) -> (u64, u64) {
        (self.hi.to_bits(), self.lo.to_bits())
    }
    /// High component.
    pub fn hi(self) -> f64 {
        self.hi
    }
    /// Low component.
    pub fn lo(self) -> f64 {
        self.lo
    }
    /// Nearest binary64 projection.
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
    /// True for finite component pairs.
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }
    /// True for NaN.
    pub fn is_nan(self) -> bool {
        self.hi.is_nan()
    }
    /// Absolute value.
    pub fn abs(self) -> Self {
        if self.hi.is_sign_negative() {
            -self
        } else {
            self
        }
    }

    /// Canonical lossless text: `dd:<hi-bits>:<lo-bits>`.
    pub fn canonical(self) -> String {
        format!("dd:{:016x}:{:016x}", self.hi.to_bits(), self.lo.to_bits())
    }

    /// Parses canonical lossless component text.
    pub fn parse_canonical(text: &str) -> Option<Self> {
        let mut parts = text.split(':');
        if parts.next()? != "dd" {
            return None;
        }
        let hi = u64::from_str_radix(parts.next()?, 16).ok()?;
        let lo = u64::from_str_radix(parts.next()?, 16).ok()?;
        if parts.next().is_some() {
            return None;
        }
        Self::from_bits(hi, lo)
    }

    /// Square root with two Newton corrections.
    pub fn sqrt(self) -> Self {
        if self.hi < 0.0 || self.is_nan() {
            return Self::from_f64_exact(f64::NAN);
        }
        if self.hi == 0.0 || self.hi.is_infinite() {
            return Self::from_f64_exact(self.hi.sqrt());
        }
        let mut x = Self::from_f64_exact(self.hi.sqrt());
        x = (x + self / x) * Self::from_f64_exact(0.5);
        x = (x + self / x) * Self::from_f64_exact(0.5);
        x
    }

    /// Exponential. Reduction chooses `k = round(x/ln(2))`, evaluates a
    /// 34-term Taylor polynomial on `[-ln(2)/2, ln(2)/2]`, then scales by 2^k.
    pub fn exp(self) -> Self {
        if self.is_nan() {
            return self;
        }
        if self.hi > 709.782_712_893_384 {
            return Self::from_f64_exact(f64::INFINITY);
        }
        if self.hi < -745.133_219_101_941_1 {
            return Self::ZERO;
        }
        let k = (self / Self::LN_2).to_f64().round() as i32;
        let r = self - Self::LN_2 * Self::from_f64_exact(k as f64);
        let mut term = Self::ONE;
        let mut sum = Self::ONE;
        for n in 1..=34 {
            term = term * r / Self::from_f64_exact(n as f64);
            sum = sum + term;
        }
        sum * Self::from_f64_exact((2.0f64).powi(k))
    }

    /// Natural logarithm. The binary64 logarithm supplies range reduction and
    /// three Newton corrections solve `exp(y)=self` in double-double.
    pub fn ln(self) -> Self {
        if self.hi <= 0.0 || self.is_nan() {
            return Self::from_f64_exact(self.hi.ln());
        }
        let mut y = Self::from_f64_exact(self.to_f64().ln());
        for _ in 0..3 {
            y = y + self / y.exp() - Self::ONE;
        }
        y
    }

    fn sin_cos_reduced(self) -> (Self, Self) {
        let half_pi = Self::PI * Self::from_f64_exact(0.5);
        let q = (self / half_pi).to_f64().round() as i64;
        let r = self - half_pi * Self::from_f64_exact(q as f64);
        let rr = r * r;
        let mut sin = r;
        let mut st = r;
        let mut cos = Self::ONE;
        let mut ct = Self::ONE;
        for n in 1..=18 {
            st = -st * rr / Self::from_f64_exact((2 * n * (2 * n + 1)) as f64);
            sin = sin + st;
            ct = -ct * rr / Self::from_f64_exact(((2 * n - 1) * (2 * n)) as f64);
            cos = cos + ct;
        }
        match q.rem_euclid(4) {
            0 => (sin, cos),
            1 => (cos, -sin),
            2 => (-sin, -cos),
            _ => (-cos, sin),
        }
    }
    /// Sine with quadrant reduction by the split double-double pi constant.
    pub fn sin(self) -> Self {
        self.sin_cos_reduced().0
    }
    /// Cosine with quadrant reduction by the split double-double pi constant.
    pub fn cos(self) -> Self {
        self.sin_cos_reduced().1
    }
    /// Integer power by exponentiation by squaring.
    pub fn powi(self, mut n: i32) -> Self {
        if n < 0 {
            return Self::ONE / self.powi(-n);
        }
        let mut base = self;
        let mut out = Self::ONE;
        while n != 0 {
            if n & 1 != 0 {
                out = out * base;
            }
            base = base * base;
            n >>= 1;
        }
        out
    }
}

fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl ops::Add for DoubleDouble {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        if !self.is_finite() || !rhs.is_finite() {
            return Self::from_f64_exact(self.to_f64() + rhs.to_f64());
        }
        let (mut hi, mut lo) = two_sum(self.hi, rhs.hi);
        let (tail, tail_error) = two_sum(self.lo, rhs.lo);
        lo += tail;
        (hi, lo) = quick_two_sum(hi, lo);
        lo += tail_error;
        (hi, lo) = quick_two_sum(hi, lo);
        Self::new(hi, lo)
    }
}
impl ops::Sub for DoubleDouble {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}
impl ops::Neg for DoubleDouble {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}
impl ops::Mul for DoubleDouble {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if !self.is_finite() || !rhs.is_finite() {
            return Self::from_f64_exact(self.to_f64() * rhs.to_f64());
        }
        let (p, e) = two_prod(self.hi, rhs.hi);
        let e = e + self.hi * rhs.lo + self.lo * rhs.hi + self.lo * rhs.lo;
        let (hi, lo) = quick_two_sum(p, e);
        Self::new(hi, lo)
    }
}
impl ops::Div for DoubleDouble {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if !self.is_finite() || !rhs.is_finite() || rhs.hi == 0.0 {
            return Self::from_f64_exact(self.to_f64() / rhs.to_f64());
        }
        let q1 = self.hi / rhs.hi;
        let q1d = Self::from_f64_exact(q1);
        let r = self - rhs * q1d;
        let q2 = r.hi / rhs.hi;
        let q = Self::new(q1, q2);
        let r2 = self - rhs * q;
        Self::new(q.hi, q.lo + r2.hi / rhs.hi)
    }
}
impl PartialEq for DoubleDouble {
    fn eq(&self, rhs: &Self) -> bool {
        self.hi == rhs.hi && self.lo == rhs.lo
    }
}
impl PartialOrd for DoubleDouble {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&rhs.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&rhs.lo),
            other => Some(other),
        }
    }
}
impl fmt::Debug for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DoubleDouble")
            .field("hi", &self.hi)
            .field("lo", &self.lo)
            .finish()
    }
}
impl fmt::Display for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.canonical())
    }
}

impl RealScalar for DoubleDouble {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    const EPSILON: Self = Self::EPSILON;
    const MIN_POSITIVE: Self = Self {
        hi: f64::MIN_POSITIVE,
        lo: 0.0,
    };
    const MAX: Self = Self {
        hi: f64::MAX,
        lo: 0.0,
    };
    fn from_f64(v: f64) -> Option<Self> {
        v.is_finite().then_some(Self::from_f64_exact(v))
    }
    fn to_f64(self) -> f64 {
        self.to_f64()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn mul_add(self, m: Self, a: Self) -> Self {
        self * m + a
    }
    fn add(self, r: Self) -> Self {
        self + r
    }
    fn sub(self, r: Self) -> Self {
        self - r
    }
    fn mul(self, r: Self) -> Self {
        self * r
    }
    fn div(self, r: Self) -> Self {
        self / r
    }
}

/// Canonical runtime domain symbol.
pub fn number_domain() -> Symbol {
    domains::extended()
}

#[sim_citizen_derive::non_citizen(
    reason = "number-domain marker; reconstruct by loading its library",
    kind = "marker",
    descriptor = "numbers/extended"
)]
/// Runtime marker for the extended domain.
pub struct ExtendedNumberDomain;
impl NumberDomain for ExtendedNumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }
    fn parse_literal(&self, cx: &mut sim_kernel::Cx, text: &str) -> Result<Option<Value>> {
        let Some(v) = DoubleDouble::parse_canonical(text) else {
            return Ok(None);
        };
        cx.factory()
            .number_literal(number_domain(), v.canonical())
            .map(Some)
    }
    fn encode_literal(
        &self,
        cx: &mut sim_kernel::Cx,
        value: Value,
    ) -> Result<Option<NumberLiteral>> {
        match value.object().as_expr(cx)? {
            sim_kernel::Expr::Number(n) if n.domain == number_domain() => Ok(Some(n)),
            _ => Ok(None),
        }
    }
    fn promotions(&self) -> Vec<PromotionRule> {
        promotion_rules()
    }
}
impl Object for ExtendedNumberDomain {
    fn display(&self, _: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/extended>".into())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl sim_kernel::ObjectCompat for ExtendedNumberDomain {
    fn class(&self, cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        sim_lib_numbers_core::number_domain_class_stub(cx)
    }
    fn as_expr(&self, _: &mut sim_kernel::Cx) -> Result<sim_kernel::Expr> {
        Ok(sim_kernel::Expr::Symbol(number_domain()))
    }
    fn as_number_domain(&self) -> Option<&dyn NumberDomain> {
        Some(self)
    }
}

fn promote_to_extended(cx: &mut sim_kernel::Cx, n: NumberLiteral) -> Result<Value> {
    let value = if n.domain == domains::f64() {
        n.canonical
            .parse::<f64>()
            .ok()
            .map(DoubleDouble::from_f64_exact)
    } else if n.domain == domains::rational() {
        let (a, b) = n.canonical.split_once('/').unwrap_or((&n.canonical, "1"));
        match (a.parse::<i128>(), b.parse::<i128>()) {
            (Ok(a), Ok(b)) => DoubleDouble::from_ratio_i128(a, b),
            _ => None,
        }
    } else {
        None
    };
    let value = value.ok_or_else(|| {
        sim_kernel::Error::Message(format!(
            "cannot promote {} to numbers/extended",
            n.canonical
        ))
    })?;
    cx.factory()
        .number_literal(number_domain(), value.canonical())
}
fn promotion_rules() -> Vec<PromotionRule> {
    vec![
        PromotionRule {
            from_domain: domains::f64(),
            to_domain: number_domain(),
            cost: 1,
            convert: promote_to_extended,
        },
        PromotionRule {
            from_domain: domains::rational(),
            to_domain: number_domain(),
            cost: 2,
            convert: promote_to_extended,
        },
    ]
}

/// Loadable runtime library registering the domain and its deliberate f64 and rational edges.
pub struct ExtendedNumbersLib;
impl ExtendedNumbersLib {
    /// Constructs the stateless installer.
    pub fn new() -> Self {
        Self
    }
}
impl Default for ExtendedNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}
impl Lib for ExtendedNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: number_domain(),
            version: Version(env!("CARGO_PKG_VERSION").into()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![],
            capabilities: vec![],
            exports: vec![Export::NumberDomain {
                symbol: number_domain(),
                number_domain_id: None,
            }],
        }
    }
    fn load(&self, _: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(ExtendedNumberDomain))
                .expect("box domain"),
        )?;
        for r in promotion_rules() {
            linker.promotion_rule(r);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
