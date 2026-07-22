//! Rational literal and finite decimal parsing with magnitude checks.

use num_bigint::{BigInt, Sign};
use sim_kernel::{Error, Result};
use sim_lib_numbers_core::{MagnitudeLimit, decimal_digits_to_bits_ceil};

/// Parses a `num/den` rational literal into a reduced numerator/denominator
/// `BigInt` pair, returning `None` on malformed text, a zero denominator, or a
/// magnitude-budget error. Use [`parse_rational_parts_checked`] when callers
/// need the budget error.
///
/// The result is normalized: the sign is carried by the numerator and the parts
/// share no common factor.
///
/// # Examples
///
/// ```
/// use num_bigint::BigInt;
/// use sim_lib_numbers_rational::parse_rational_parts;
///
/// assert_eq!(
///     parse_rational_parts("6/8"),
///     Some((BigInt::from(3), BigInt::from(4)))
/// );
/// assert_eq!(parse_rational_parts("1/0"), None);
/// ```
pub fn parse_rational_parts(text: &str) -> Option<(BigInt, BigInt)> {
    parse_rational_parts_checked(text).ok().flatten()
}

/// Parses a `num/den` rational literal like [`parse_rational_parts`], returning
/// an error when the literal exceeds the arbitrary-precision magnitude budget.
pub fn parse_rational_parts_checked(text: &str) -> Result<Option<(BigInt, BigInt)>> {
    let Some((num_text, den_text)) = text.split_once('/') else {
        return Ok(None);
    };
    let Some(num) = parse_limited_bigint_component(num_text, "rational numerator")? else {
        return Ok(None);
    };
    let Some(den) = parse_limited_bigint_component(den_text, "rational denominator")? else {
        return Ok(None);
    };
    normalize_bigint_rational_checked(num, den)
}

/// Converts a finite decimal string (no exponent) into an exact reduced
/// numerator/denominator `BigInt` pair, the basis of the f64 -> rational
/// promotion. Returns `None` on empty, exponential, malformed, or overly
/// precise input. Use [`f64_decimal_to_rational_checked`] when callers need the
/// budget error.
///
/// # Examples
///
/// ```
/// use num_bigint::BigInt;
/// use sim_lib_numbers_rational::f64_decimal_to_rational;
///
/// assert_eq!(
///     f64_decimal_to_rational("0.25"),
///     Some((BigInt::from(1), BigInt::from(4)))
/// );
/// assert_eq!(f64_decimal_to_rational("1e3"), None);
/// ```
pub fn f64_decimal_to_rational(text: &str) -> Option<(BigInt, BigInt)> {
    f64_decimal_to_rational_checked(text).ok().flatten()
}

/// Converts a finite decimal string like [`f64_decimal_to_rational`], returning
/// an error when the decimal is too precise for the arbitrary-precision
/// magnitude budget.
pub fn f64_decimal_to_rational_checked(text: &str) -> Result<Option<(BigInt, BigInt)>> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.contains(['e', 'E']) {
        return Ok(None);
    }
    let negative = trimmed.starts_with('-');
    let unsigned = trimmed.strip_prefix(['-', '+']).unwrap_or(trimmed);
    let (whole_text, fractional_text) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole_text.is_empty() && fractional_text.is_empty() {
        return Ok(None);
    }
    check_decimal_literal_scale(fractional_text.len())?;
    let whole = if whole_text.is_empty() {
        BigInt::from(0_u8)
    } else {
        let Some(whole) = parse_limited_bigint_component(whole_text, "decimal literal whole")?
        else {
            return Ok(None);
        };
        whole
    };
    let scale_power: u32 = fractional_text.len().try_into().map_err(|_| {
        Error::Eval("decimal literal too precise: fractional scale exceeds u32".to_owned())
    })?;
    let scale = BigInt::from(10_u8).pow(scale_power);
    check_bigint_magnitude("decimal literal scale", &scale)?;
    let fractional = if fractional_text.is_empty() {
        BigInt::from(0_u8)
    } else {
        let Some(fractional) =
            parse_limited_bigint_component(fractional_text, "decimal literal fraction")?
        else {
            return Ok(None);
        };
        fractional
    };
    let mut numerator = whole * scale.clone() + fractional;
    if negative {
        numerator = -numerator;
    }
    normalize_bigint_rational_checked(numerator, scale)
}

fn normalize_bigint_rational(num: BigInt, den: BigInt) -> Option<(BigInt, BigInt)> {
    if den == BigInt::from(0_u8) {
        return None;
    }
    let sign = if den.sign() == Sign::Minus {
        -BigInt::from(1_u8)
    } else {
        BigInt::from(1_u8)
    };
    let num = num * sign.clone();
    let den = den * sign;
    let gcd = gcd_bigint(num.clone(), den.clone());
    Some((num / gcd.clone(), den / gcd))
}

fn normalize_bigint_rational_checked(num: BigInt, den: BigInt) -> Result<Option<(BigInt, BigInt)>> {
    let Some((num, den)) = normalize_bigint_rational(num, den) else {
        return Ok(None);
    };
    check_bigint_magnitude("rational numerator", &num)?;
    check_bigint_magnitude("rational denominator", &den)?;
    Ok(Some((num, den)))
}

fn parse_limited_bigint_component(text: &str, context: &str) -> Result<Option<BigInt>> {
    if text.is_empty() {
        return Ok(None);
    }
    check_decimal_digits(context, text)?;
    let Some(value) = text.parse::<BigInt>().ok() else {
        return Ok(None);
    };
    check_bigint_magnitude(context, &value)?;
    Ok(Some(value))
}

fn check_decimal_literal_scale(digits: usize) -> Result<()> {
    let estimated_bits = decimal_digits_to_bits_ceil(digits);
    let limit = magnitude_limit();
    if estimated_bits > limit.max_bits() {
        return Err(Error::Eval(format!(
            "decimal literal too precise: fractional scale estimated {estimated_bits} bits exceeds limit {}",
            limit.max_bits()
        )));
    }
    Ok(())
}

fn check_decimal_digits(context: &str, text: &str) -> Result<()> {
    magnitude_limit()
        .check_decimal_digits(context, decimal_digit_count(text))
        .map(|_| ())
}

fn check_bigint_magnitude(context: &str, value: &BigInt) -> Result<()> {
    magnitude_limit().check_bits(context, value.bits())
}

fn magnitude_limit() -> MagnitudeLimit {
    MagnitudeLimit::default_arbitrary_precision()
}

fn decimal_digit_count(text: &str) -> usize {
    text.bytes().filter(|byte| byte.is_ascii_digit()).count()
}

fn gcd_bigint(mut left: BigInt, mut right: BigInt) -> BigInt {
    left = bigint_abs(left);
    right = bigint_abs(right);
    while right != BigInt::from(0_u8) {
        let next = left % right.clone();
        left = right;
        right = next;
    }
    if left == BigInt::from(0_u8) {
        BigInt::from(1_u8)
    } else {
        left
    }
}

fn bigint_abs(value: BigInt) -> BigInt {
    if value.sign() == Sign::Minus {
        -value
    } else {
        value
    }
}
