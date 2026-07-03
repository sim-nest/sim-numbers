//! Reduced rational arithmetic rules, literal/decimal parsing into
//! numerator/denominator pairs, and the promotions to and from the integer and
//! `f64` domains, in both literal and value form.

use num_bigint::{BigInt, Sign};
use sim_kernel::{Cx, Error, NumberLiteral, Result, Value};

use super::domain::{f64_domain, number_domain};
use super::integer::{parse_integer_literal, parse_integer_value};
use super::value::{Rational, expect_rational_parts, make_reduced_rational};

pub(crate) type RationalRuleFn = fn(&mut Cx, NumberLiteral, NumberLiteral) -> Result<Value>;
pub(crate) type ValueRuleFn = fn(&mut Cx, Value, Value) -> Result<Value>;

/// Parses a `num/den` rational literal into a reduced numerator/denominator
/// `BigInt` pair, returning `None` on malformed text or a zero denominator.
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
    let (num_text, den_text) = text.split_once('/')?;
    let num = num_text.parse::<BigInt>().ok()?;
    let den = den_text.parse::<BigInt>().ok()?;
    normalize_bigint_rational(num, den)
}

/// Converts a finite decimal string (no exponent) into an exact reduced
/// numerator/denominator `BigInt` pair, the basis of the f64 -> rational
/// promotion. Returns `None` on empty, exponential, or malformed input.
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
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.contains(['e', 'E']) {
        return None;
    }
    let negative = trimmed.starts_with('-');
    let unsigned = trimmed.strip_prefix(['-', '+']).unwrap_or(trimmed);
    let (whole_text, fractional_text) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole_text.is_empty() && fractional_text.is_empty() {
        return None;
    }
    let whole = if whole_text.is_empty() {
        BigInt::from(0_u8)
    } else {
        whole_text.parse::<BigInt>().ok()?
    };
    let scale = BigInt::from(10_u8).pow(fractional_text.len() as u32);
    let fractional = if fractional_text.is_empty() {
        BigInt::from(0_u8)
    } else {
        fractional_text.parse::<BigInt>().ok()?
    };
    let mut numerator = whole * scale.clone() + fractional;
    if negative {
        numerator = -numerator;
    }
    normalize_bigint_rational(numerator, scale)
}

pub(crate) fn rational_add_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = literal_value(cx, left)?;
    let right = literal_value(cx, right)?;
    rational_add_value_rule(cx, left, right)
}

pub(crate) fn rational_sub_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = literal_value(cx, left)?;
    let right = literal_value(cx, right)?;
    rational_sub_value_rule(cx, left, right)
}

pub(crate) fn rational_mul_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = literal_value(cx, left)?;
    let right = literal_value(cx, right)?;
    rational_mul_value_rule(cx, left, right)
}

pub(crate) fn rational_div_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = literal_value(cx, left)?;
    let right = literal_value(cx, right)?;
    rational_div_value_rule(cx, left, right)
}

pub(crate) fn rational_pow_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = literal_value(cx, left)?;
    let right = literal_value(cx, right)?;
    rational_pow_value_rule(cx, left, right)
}

pub(crate) fn rational_neg_rule(cx: &mut Cx, operand: NumberLiteral) -> Result<Value> {
    let operand = literal_value(cx, operand)?;
    rational_neg_value_rule(cx, operand)
}

pub(crate) fn rational_sum_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let operands = operands
        .into_iter()
        .map(|operand| literal_value(cx, operand))
        .collect::<Result<Vec<_>>>()?;
    rational_sum_value_rule(cx, operands)
}

pub(crate) fn rational_product_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let operands = operands
        .into_iter()
        .map(|operand| literal_value(cx, operand))
        .collect::<Result<Vec<_>>>()?;
    rational_product_value_rule(cx, operands)
}

pub(crate) fn rational_add_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_rational_parts(cx, left, "left")?;
    let right = expect_rational_parts(cx, right, "right")?;
    let left_scaled = mul(cx, left.num.clone(), right.den.clone())?;
    let right_scaled = mul(cx, right.num, left.den.clone())?;
    let numerator = add(cx, left_scaled, right_scaled)?;
    let denominator = mul(cx, left.den, right.den)?;
    make_reduced_rational(cx, numerator, denominator)
}

pub(crate) fn rational_sub_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_rational_parts(cx, left, "left")?;
    let right = expect_rational_parts(cx, right, "right")?;
    let left_scaled = mul(cx, left.num.clone(), right.den.clone())?;
    let right_scaled = mul(cx, right.num, left.den.clone())?;
    let numerator = sub(cx, left_scaled, right_scaled)?;
    let denominator = mul(cx, left.den, right.den)?;
    make_reduced_rational(cx, numerator, denominator)
}

pub(crate) fn rational_mul_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_rational_parts(cx, left, "left")?;
    let right = expect_rational_parts(cx, right, "right")?;
    let numerator = mul(cx, left.num, right.num)?;
    let denominator = mul(cx, left.den, right.den)?;
    make_reduced_rational(cx, numerator, denominator)
}

pub(crate) fn rational_div_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_rational_parts(cx, left, "left")?;
    let right = expect_rational_parts(cx, right, "right")?;
    let numerator = mul(cx, left.num, right.den)?;
    let denominator = mul(cx, left.den, right.num)?;
    make_reduced_rational(cx, numerator, denominator)
}

pub(crate) fn rational_pow_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_rational_parts(cx, left, "left")?;
    let exponent = expect_integer_exponent(cx, right)?;
    let negative = exponent.sign() == Sign::Minus;
    let exponent = bigint_abs(exponent).to_string();
    let exponent = parse_integer_value(cx, &exponent)?;
    let (num, den) = if negative {
        (left.den, left.num)
    } else {
        (left.num, left.den)
    };
    let numerator = pow(cx, num, exponent.clone())?;
    let denominator = pow(cx, den, exponent)?;
    make_reduced_rational(cx, numerator, denominator)
}

pub(crate) fn rational_neg_value_rule(cx: &mut Cx, operand: Value) -> Result<Value> {
    let operand = expect_rational_parts(cx, operand, "operand")?;
    let numerator =
        cx.apply_value_number_unary_op(&sim_kernel::Symbol::qualified("math", "neg"), operand.num)?;
    make_reduced_rational(cx, numerator, operand.den)
}

pub(crate) fn rational_sum_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let zero = parse_integer_value(cx, "0")?;
    let one = parse_integer_value(cx, "1")?;
    let mut acc = make_reduced_rational(cx, zero, one)?;
    for operand in operands {
        acc = rational_add_value_rule(cx, acc, operand)?;
    }
    Ok(acc)
}

pub(crate) fn rational_product_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let one_num = parse_integer_value(cx, "1")?;
    let one_den = parse_integer_value(cx, "1")?;
    let mut acc = make_reduced_rational(cx, one_num, one_den)?;
    for operand in operands {
        acc = rational_mul_value_rule(cx, acc, operand)?;
    }
    Ok(acc)
}

pub(crate) fn promote_integer_literal_to_rational(
    _cx: &mut Cx,
    number: NumberLiteral,
) -> Result<NumberLiteral> {
    let value = parse_integer_literal(&number)?;
    Ok(NumberLiteral {
        domain: number_domain(),
        canonical: format!("{value}/1"),
    })
}

pub(crate) fn promote_integer_value_to_rational(cx: &mut Cx, value: Value) -> Result<Value> {
    let literal = expect_value_literal(cx, value, "integer-to-rational promotion")?;
    let promoted = promote_integer_literal_to_rational(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

pub(crate) fn promote_f64_literal_to_rational(
    _cx: &mut Cx,
    number: NumberLiteral,
) -> Result<NumberLiteral> {
    if number.domain != f64_domain() {
        return Err(Error::Eval(format!(
            "expected number domain {} for promotion, found {}",
            f64_domain(),
            number.domain
        )));
    }
    let (num, den) = f64_decimal_to_rational(&number.canonical).ok_or_else(|| {
        Error::Eval(format!(
            "could not promote f64 literal to rational: {}",
            number.canonical
        ))
    })?;
    Ok(NumberLiteral {
        domain: number_domain(),
        canonical: format!("{num}/{den}"),
    })
}

pub(crate) fn promote_f64_value_to_rational(cx: &mut Cx, value: Value) -> Result<Value> {
    let literal = expect_value_literal(cx, value, "f64-to-rational promotion")?;
    let promoted = promote_f64_literal_to_rational(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

pub(crate) fn promote_rational_literal_to_f64(
    _cx: &mut Cx,
    number: NumberLiteral,
) -> Result<NumberLiteral> {
    if number.domain != number_domain() {
        return Err(Error::Eval(format!(
            "expected number domain {} for promotion, found {}",
            number_domain(),
            number.domain
        )));
    }
    let (num, den) = parse_rational_parts(&number.canonical).ok_or_else(|| {
        Error::Eval(format!(
            "invalid rational literal for f64 promotion: {}",
            number.canonical
        ))
    })?;
    let num = num.to_string().parse::<f64>().map_err(|err| {
        Error::Eval(format!(
            "could not convert rational numerator to f64: {}",
            err
        ))
    })?;
    let den = den.to_string().parse::<f64>().map_err(|err| {
        Error::Eval(format!(
            "could not convert rational denominator to f64: {}",
            err
        ))
    })?;
    Ok(NumberLiteral {
        domain: f64_domain(),
        canonical: format!("{}", num / den),
    })
}

pub(crate) fn promote_rational_value_to_f64(cx: &mut Cx, value: Value) -> Result<Value> {
    if let Some(literal) = cx
        .number_value_ref(value.clone())?
        .and_then(|number| number.literal)
    {
        let promoted = promote_rational_literal_to_f64(cx, literal)?;
        return cx
            .factory()
            .number_literal(promoted.domain, promoted.canonical);
    }

    let rational = expect_rational_parts(cx, value, "operand")?;
    let num_literal = expect_value_literal(cx, rational.num, "rational numerator")?;
    let den_literal = expect_value_literal(cx, rational.den, "rational denominator")?;
    let num = parse_integer_literal(&num_literal)?
        .to_string()
        .parse::<f64>()
        .map_err(|err| {
            Error::Eval(format!(
                "could not convert rational numerator to f64: {err}"
            ))
        })?;
    let den = parse_integer_literal(&den_literal)?
        .to_string()
        .parse::<f64>()
        .map_err(|err| {
            Error::Eval(format!(
                "could not convert rational denominator to f64: {err}"
            ))
        })?;
    cx.factory()
        .number_literal(f64_domain(), format!("{}", num / den))
}

fn literal_value(cx: &mut Cx, literal: NumberLiteral) -> Result<Value> {
    if literal.domain != number_domain() {
        return Err(Error::Eval(format!(
            "expected rational literal domain {}, found {}",
            number_domain(),
            literal.domain
        )));
    }
    cx.factory()
        .number_literal(literal.domain, literal.canonical)
}

fn expect_value_literal(cx: &mut Cx, value: Value, context: &str) -> Result<NumberLiteral> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(Error::Eval(format!("{context} expected a number value")));
    };
    number.literal.ok_or_else(|| {
        Error::Eval(format!(
            "{context} in {} does not have a canonical literal form",
            number.domain
        ))
    })
}

fn expect_integer_exponent(cx: &mut Cx, value: Value) -> Result<BigInt> {
    let Rational { num, den } = expect_rational_parts(cx, value, "right")?;
    let den_ref = cx.number_value_ref(den)?.ok_or_else(|| {
        Error::Eval("rational exponent lost denominator numeric identity".to_owned())
    })?;
    let den_literal = den_ref.literal.ok_or_else(|| {
        Error::Eval("rational exponent denominator requires canonical integer form".to_owned())
    })?;
    if parse_integer_literal(&den_literal)? != BigInt::from(1_u8) {
        return Err(Error::Eval(
            "pow exponent must be an integer-valued rational".to_owned(),
        ));
    }
    let num_ref = cx.number_value_ref(num)?.ok_or_else(|| {
        Error::Eval("rational exponent lost numerator numeric identity".to_owned())
    })?;
    let num_literal = num_ref.literal.ok_or_else(|| {
        Error::Eval("rational exponent numerator requires canonical integer form".to_owned())
    })?;
    parse_integer_literal(&num_literal)
}

fn add(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&sim_kernel::Symbol::qualified("math", "add"), left, right)
}

fn sub(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&sim_kernel::Symbol::qualified("math", "sub"), left, right)
}

fn mul(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&sim_kernel::Symbol::qualified("math", "mul"), left, right)
}

fn pow(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&sim_kernel::Symbol::qualified("math", "pow"), left, right)
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
