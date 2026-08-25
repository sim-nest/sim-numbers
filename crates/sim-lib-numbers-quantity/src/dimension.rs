use core::fmt;

/// The seven SI base dimensions, deliberately owned outside the kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum BaseDimension {
    /// Time.
    Time = 0,
    /// Length.
    Length = 1,
    /// Mass.
    Mass = 2,
    /// Electric current.
    Current = 3,
    /// Thermodynamic temperature.
    Temperature = 4,
    /// Amount of substance.
    Amount = 5,
    /// Luminous intensity.
    LuminousIntensity = 6,
}

/// Maximum absolute numerator or denominator admitted for an exponent.
pub const MAX_EXPONENT_MAGNITUDE: i32 = 1024;

/// A normalized, bounded, exact rational exponent.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Exponent {
    numerator: i16,
    denominator: u16,
}

/// Invalid dimension or exponent construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DimensionError {
    /// A rational exponent had a zero denominator.
    ZeroDenominator,
    /// A normalized exponent exceeded the resource bound.
    ExponentOutOfBounds,
}

impl fmt::Display for DimensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator => f.write_str("dimension exponent denominator must not be zero"),
            Self::ExponentOutOfBounds => write!(
                f,
                "dimension exponent exceeds magnitude bound {MAX_EXPONENT_MAGNITUDE}"
            ),
        }
    }
}

impl std::error::Error for DimensionError {}

impl Exponent {
    /// The additive identity exponent.
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };

    /// Constructs and reduces an exact exponent.
    pub fn new(numerator: i32, denominator: i32) -> Result<Self, DimensionError> {
        if denominator == 0 {
            return Err(DimensionError::ZeroDenominator);
        }
        let sign = if denominator < 0 { -1 } else { 1 };
        let mut n = numerator
            .checked_mul(sign)
            .ok_or(DimensionError::ExponentOutOfBounds)?;
        let mut d = denominator.abs();
        if n == 0 {
            return Ok(Self::ZERO);
        }
        let divisor = gcd(n.unsigned_abs(), d as u32) as i32;
        n /= divisor;
        d /= divisor;
        if n.abs() > MAX_EXPONENT_MAGNITUDE || d > MAX_EXPONENT_MAGNITUDE {
            return Err(DimensionError::ExponentOutOfBounds);
        }
        Ok(Self {
            numerator: n as i16,
            denominator: d as u16,
        })
    }

    /// Returns the reduced numerator.
    pub const fn numerator(self) -> i16 {
        self.numerator
    }

    /// Returns the positive reduced denominator.
    pub const fn denominator(self) -> u16 {
        self.denominator
    }

    fn add(self, rhs: Self) -> Result<Self, DimensionError> {
        Self::new(
            i32::from(self.numerator) * i32::from(rhs.denominator)
                + i32::from(rhs.numerator) * i32::from(self.denominator),
            i32::from(self.denominator) * i32::from(rhs.denominator),
        )
    }

    fn scale(self, power: Self) -> Result<Self, DimensionError> {
        Self::new(
            i32::from(self.numerator) * i32::from(power.numerator),
            i32::from(self.denominator) * i32::from(power.denominator),
        )
    }
}

impl fmt::Display for Exponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

/// A canonical exponent vector in SI base-dimension order.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Dimension([Exponent; 7]);

impl Dimension {
    /// The dimensionless identity.
    pub const DIMENSIONLESS: Self = Self([Exponent::ZERO; 7]);

    /// Constructs a dimension from the canonical SI-ordered vector.
    pub const fn from_exponents(exponents: [Exponent; 7]) -> Self {
        Self(exponents)
    }

    /// Constructs one SI base dimension.
    pub fn base(base: BaseDimension) -> Self {
        let mut values = [Exponent::ZERO; 7];
        values[base as usize] = Exponent {
            numerator: 1,
            denominator: 1,
        };
        Self(values)
    }

    /// Returns the canonical exponent vector.
    pub const fn exponents(&self) -> &[Exponent; 7] {
        &self.0
    }

    /// Returns whether all exponents are zero.
    pub fn is_dimensionless(&self) -> bool {
        self == &Self::DIMENSIONLESS
    }

    /// Multiplies dimensions by adding exact exponents.
    pub fn product(&self, rhs: &Self) -> Result<Self, DimensionError> {
        let mut out = [Exponent::ZERO; 7];
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = self.0[index].add(rhs.0[index])?;
        }
        Ok(Self(out))
    }

    /// Divides dimensions by subtracting exact exponents.
    pub fn quotient(&self, rhs: &Self) -> Result<Self, DimensionError> {
        self.product(&rhs.power(Exponent::new(-1, 1)?)?)
    }

    /// Raises every exponent to an exact rational power.
    pub fn power(&self, power: Exponent) -> Result<Self, DimensionError> {
        let mut out = [Exponent::ZERO; 7];
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = self.0[index].scale(power)?;
        }
        Ok(Self(out))
    }

    /// Stable canonical text used by content identifiers and codecs.
    pub fn canonical(&self) -> String {
        const NAMES: [&str; 7] = [
            "time",
            "length",
            "mass",
            "current",
            "temperature",
            "amount",
            "luminous-intensity",
        ];
        self.0
            .iter()
            .zip(NAMES)
            .filter(|(e, _)| **e != Exponent::ZERO)
            .map(|(e, n)| format!("{n}^{e}"))
            .collect::<Vec<_>>()
            .join("*")
            .pipe(|s| if s.is_empty() { "1".to_owned() } else { s })
    }
}

trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}
