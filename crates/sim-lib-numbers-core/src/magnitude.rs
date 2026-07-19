//! Shared magnitude limits for arbitrary-precision number libraries.
//!
//! These helpers keep exact-number domains finite by default while letting
//! callers thread a different limit through their own numeric budget surfaces.

use sim_kernel::{Error, Result};

/// Default maximum bit length accepted for arbitrary-precision literals and
/// results.
pub const DEFAULT_MAX_ARBITRARY_MAGNITUDE_BITS: u64 = 262_144;

/// A finite ceiling for arbitrary-precision literal and result magnitude.
///
/// The limit is expressed in binary bits so domains with native bit-length
/// metadata can enforce it directly. Decimal literal parsers can use
/// [`MagnitudeLimit::check_decimal_digits`] to conservatively estimate the same
/// budget before allocating big integer storage.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_core::MagnitudeLimit;
///
/// let limit = MagnitudeLimit::new(128);
/// assert!(limit.check_bits("integer result", 127).is_ok());
/// assert!(limit.check_decimal_digits("decimal literal", 1_000).is_err());
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MagnitudeLimit {
    max_bits: u64,
}

impl MagnitudeLimit {
    /// Creates a finite arbitrary-precision magnitude limit.
    pub const fn new(max_bits: u64) -> Self {
        Self { max_bits }
    }

    /// Returns the default arbitrary-precision magnitude limit.
    pub const fn default_arbitrary_precision() -> Self {
        Self::new(DEFAULT_MAX_ARBITRARY_MAGNITUDE_BITS)
    }

    /// Returns the maximum allowed magnitude in binary bits.
    pub const fn max_bits(self) -> u64 {
        self.max_bits
    }

    /// Checks a known binary bit length against this limit.
    pub fn check_bits(self, context: &str, estimated_bits: u64) -> Result<()> {
        if estimated_bits > self.max_bits {
            return Err(Error::Eval(format!(
                "{context} magnitude too large: estimated {estimated_bits} bits exceeds limit {}",
                self.max_bits
            )));
        }
        Ok(())
    }

    /// Checks a decimal digit count against this limit, returning the estimated
    /// binary bit length when it fits.
    pub fn check_decimal_digits(self, context: &str, digits: usize) -> Result<u64> {
        let estimated_bits = decimal_digits_to_bits_ceil(digits);
        self.check_bits(context, estimated_bits)?;
        Ok(estimated_bits)
    }

    /// Returns the largest decimal digit count conservatively accepted by this
    /// bit budget.
    pub fn max_decimal_digits(self) -> usize {
        self.max_bits
            .saturating_mul(1000)
            .checked_div(3322)
            .unwrap_or(0)
            .try_into()
            .unwrap_or(usize::MAX)
    }
}

impl Default for MagnitudeLimit {
    fn default() -> Self {
        Self::default_arbitrary_precision()
    }
}

/// Conservatively estimates the binary bit length needed for a decimal digit
/// count.
///
/// The estimate rounds up `digits * log2(10)` with a fixed rational
/// approximation. It is intentionally conservative for budget checks that must
/// happen before parsing the decimal text into a big integer.
pub fn decimal_digits_to_bits_ceil(digits: usize) -> u64 {
    let digits = u64::try_from(digits).unwrap_or(u64::MAX);
    if digits == 0 {
        return 0;
    }
    digits.saturating_mul(3322).saturating_add(999) / 1000
}
