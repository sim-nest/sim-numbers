use core::fmt;

/// Scalar operations required by quantity algebra and exact unit conversion.
///
/// Installed number domains can implement this trait without exposing their
/// representation. `mul_ratio` and `add_ratio` are the critical boundary:
/// conversions remain inside the scalar domain and never detour through f64.
pub trait Scalar: Clone + PartialEq + fmt::Debug {
    /// Domain-specific arithmetic error.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Adds two same-domain values.
    fn add(&self, rhs: &Self) -> Result<Self, Self::Error>;
    /// Subtracts two same-domain values.
    fn sub(&self, rhs: &Self) -> Result<Self, Self::Error>;
    /// Multiplies two same-domain values.
    fn mul(&self, rhs: &Self) -> Result<Self, Self::Error>;
    /// Divides two same-domain values.
    fn div(&self, rhs: &Self) -> Result<Self, Self::Error>;
    /// Multiplies by an exact rational constant in this domain.
    fn mul_ratio(&self, numerator: i128, denominator: i128) -> Result<Self, Self::Error>;
    /// Adds an exact rational constant in this domain.
    fn add_ratio(&self, numerator: i128, denominator: i128) -> Result<Self, Self::Error>;
    /// Raises the scalar to an exact rational power when admitted by the domain.
    fn pow_ratio(&self, numerator: i16, denominator: u16) -> Result<Self, Self::Error>;
}

/// A small exact rational scalar useful for fixtures, recipes, and exact
/// conversion oracles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactScalar {
    numerator: i128,
    denominator: i128,
}

/// Exact scalar arithmetic failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactScalarError(&'static str);

impl fmt::Display for ExactScalarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for ExactScalarError {}

impl ExactScalar {
    /// Constructs a reduced exact rational.
    pub fn new(numerator: i128, denominator: i128) -> Result<Self, ExactScalarError> {
        if denominator == 0 {
            return Err(ExactScalarError("zero denominator"));
        }
        let sign = if denominator < 0 { -1 } else { 1 };
        let n = numerator
            .checked_mul(sign)
            .ok_or(ExactScalarError("scalar overflow"))?;
        let d = denominator.abs();
        let gcd = gcd128(n.unsigned_abs(), d as u128) as i128;
        Ok(Self {
            numerator: n / gcd,
            denominator: d / gcd,
        })
    }

    /// Returns the reduced numerator.
    pub const fn numerator(self) -> i128 {
        self.numerator
    }
    /// Returns the positive reduced denominator.
    pub const fn denominator(self) -> i128 {
        self.denominator
    }
}

impl From<i64> for ExactScalar {
    fn from(value: i64) -> Self {
        Self {
            numerator: value as i128,
            denominator: 1,
        }
    }
}

impl Scalar for ExactScalar {
    type Error = ExactScalarError;
    fn add(&self, rhs: &Self) -> Result<Self, Self::Error> {
        Self::new(
            self.numerator
                .checked_mul(rhs.denominator)
                .and_then(|a| {
                    rhs.numerator
                        .checked_mul(self.denominator)
                        .and_then(|b| a.checked_add(b))
                })
                .ok_or(ExactScalarError("scalar overflow"))?,
            self.denominator
                .checked_mul(rhs.denominator)
                .ok_or(ExactScalarError("scalar overflow"))?,
        )
    }
    fn sub(&self, rhs: &Self) -> Result<Self, Self::Error> {
        self.add(&Self {
            numerator: -rhs.numerator,
            denominator: rhs.denominator,
        })
    }
    fn mul(&self, rhs: &Self) -> Result<Self, Self::Error> {
        Self::new(
            self.numerator
                .checked_mul(rhs.numerator)
                .ok_or(ExactScalarError("scalar overflow"))?,
            self.denominator
                .checked_mul(rhs.denominator)
                .ok_or(ExactScalarError("scalar overflow"))?,
        )
    }
    fn div(&self, rhs: &Self) -> Result<Self, Self::Error> {
        Self::new(
            self.numerator
                .checked_mul(rhs.denominator)
                .ok_or(ExactScalarError("scalar overflow"))?,
            self.denominator
                .checked_mul(rhs.numerator)
                .ok_or(ExactScalarError("scalar overflow"))?,
        )
    }
    fn mul_ratio(&self, n: i128, d: i128) -> Result<Self, Self::Error> {
        self.mul(&Self::new(n, d)?)
    }
    fn add_ratio(&self, n: i128, d: i128) -> Result<Self, Self::Error> {
        self.add(&Self::new(n, d)?)
    }
    fn pow_ratio(&self, n: i16, d: u16) -> Result<Self, Self::Error> {
        if d != 1 {
            return Err(ExactScalarError("non-integral exact scalar power"));
        }
        let power = i32::from(n);
        let magnitude = power.unsigned_abs();
        let num = self
            .numerator
            .checked_pow(magnitude)
            .ok_or(ExactScalarError("scalar overflow"))?;
        let den = self
            .denominator
            .checked_pow(magnitude)
            .ok_or(ExactScalarError("scalar overflow"))?;
        if power < 0 {
            Self::new(den, num)
        } else {
            Self::new(num, den)
        }
    }
}

impl Scalar for f64 {
    type Error = ExactScalarError;
    fn add(&self, rhs: &Self) -> Result<Self, Self::Error> {
        Ok(*self + *rhs)
    }
    fn sub(&self, rhs: &Self) -> Result<Self, Self::Error> {
        Ok(*self - *rhs)
    }
    fn mul(&self, rhs: &Self) -> Result<Self, Self::Error> {
        Ok(*self * *rhs)
    }
    fn div(&self, rhs: &Self) -> Result<Self, Self::Error> {
        Ok(*self / *rhs)
    }
    fn mul_ratio(&self, n: i128, d: i128) -> Result<Self, Self::Error> {
        Ok(*self * n as f64 / d as f64)
    }
    fn add_ratio(&self, n: i128, d: i128) -> Result<Self, Self::Error> {
        Ok(*self + n as f64 / d as f64)
    }
    fn pow_ratio(&self, n: i16, d: u16) -> Result<Self, Self::Error> {
        Ok(self.powf(f64::from(n) / f64::from(d)))
    }
}

fn gcd128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}
