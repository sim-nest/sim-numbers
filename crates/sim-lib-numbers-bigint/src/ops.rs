//! Exact bigint arithmetic rules, canonicalization, and the promotions into
//! `rational` (and the integer-to-bigint widening), in both literal and value
//! form.

use std::cmp::Ordering;

use num_bigint::BigInt;
use sim_kernel::{Cx, NumberLiteral, Result, Value};
use sim_lib_numbers_core::{MagnitudeLimit, domains};

use crate::implementation::{number_domain, rational_domain};

pub(crate) type BigIntRuleFn = fn(&mut Cx, NumberLiteral, NumberLiteral) -> Result<Value>;
pub(crate) type ValueRuleFn = fn(&mut Cx, Value, Value) -> Result<Value>;

pub(crate) fn bigint_add_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    binary_bigint_rule(cx, "add result", left, right, |left, right| left + right)
}

pub(crate) fn bigint_sub_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    binary_bigint_rule(cx, "sub result", left, right, |left, right| left - right)
}

pub(crate) fn bigint_mul_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    binary_bigint_rule(cx, "mul result", left, right, |left, right| left * right)
}

pub(crate) fn bigint_div_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let right_value = parse_bigint_literal(right, "right")?;
    if right_value == BigInt::from(0_u8) {
        return Err(sim_kernel::Error::Eval(
            "div divided by zero in bigint arithmetic".to_owned(),
        ));
    }
    let left_value = parse_bigint_literal(left, "left")?;
    let result = left_value / right_value;
    check_bigint_magnitude("div result", &result)?;
    cx.factory()
        .number_literal(number_domain(), result.to_string())
}

pub(crate) fn bigint_rem_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let right_value = parse_bigint_literal(right, "right")?;
    if right_value == BigInt::from(0_u8) {
        return Err(sim_kernel::Error::Eval(
            "rem divided by zero in bigint arithmetic".to_owned(),
        ));
    }
    let left_value = parse_bigint_literal(left, "left")?;
    let result = left_value % right_value;
    check_bigint_magnitude("rem result", &result)?;
    cx.factory()
        .number_literal(number_domain(), result.to_string())
}

pub(crate) fn bigint_pow_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let base = parse_bigint_literal(left, "left")?;
    let exponent = parse_bigint_literal(right, "right")?;
    let exponent_u32: u32 = exponent.try_into().map_err(|_| {
        sim_kernel::Error::Eval("pow exponent must be a nonnegative integer".to_owned())
    })?;
    magnitude_limit().check_bits("pow result", estimate_pow_bits(&base, exponent_u32))?;
    let result = base.pow(exponent_u32);
    check_bigint_magnitude("pow result", &result)?;
    cx.factory()
        .number_literal(number_domain(), result.to_string())
}

pub(crate) fn bigint_cmp_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = parse_bigint_literal(left, "left")?;
    let right = parse_bigint_literal(right, "right")?;
    comparison_value(cx, left.cmp(&right))
}

pub(crate) fn bigint_neg_rule(cx: &mut Cx, operand: NumberLiteral) -> Result<Value> {
    let result = -parse_bigint_literal(operand, "operand")?;
    check_bigint_magnitude("neg result", &result)?;
    cx.factory()
        .number_literal(number_domain(), result.to_string())
}

pub(crate) fn bigint_sum_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = BigInt::from(0_u8);
    for operand in operands {
        acc += parse_bigint_literal(operand, "operand")?;
        check_bigint_magnitude("sum result", &acc)?;
    }
    cx.factory()
        .number_literal(number_domain(), acc.to_string())
}

pub(crate) fn bigint_product_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = BigInt::from(1_u8);
    for operand in operands {
        acc *= parse_bigint_literal(operand, "operand")?;
        check_bigint_magnitude("product result", &acc)?;
    }
    cx.factory()
        .number_literal(number_domain(), acc.to_string())
}

pub(crate) fn bigint_add_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    bigint_add_rule(cx, left, right)
}

pub(crate) fn bigint_sub_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    bigint_sub_rule(cx, left, right)
}

pub(crate) fn bigint_mul_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    bigint_mul_rule(cx, left, right)
}

pub(crate) fn bigint_div_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    bigint_div_rule(cx, left, right)
}

pub(crate) fn bigint_rem_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    bigint_rem_rule(cx, left, right)
}

pub(crate) fn bigint_pow_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    bigint_pow_rule(cx, left, right)
}

pub(crate) fn bigint_cmp_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    bigint_cmp_rule(cx, left, right)
}

pub(crate) fn bigint_neg_value_rule(cx: &mut Cx, operand: Value) -> Result<Value> {
    let operand = expect_domain_literal(cx, operand, "operand")?;
    bigint_neg_rule(cx, operand)
}

pub(crate) fn bigint_sum_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    bigint_sum_rule(cx, operands)
}

pub(crate) fn bigint_product_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    bigint_product_rule(cx, operands)
}

pub(crate) fn promote_integer_to_bigint(
    _cx: &mut Cx,
    number: NumberLiteral,
) -> Result<NumberLiteral> {
    Ok(NumberLiteral {
        domain: number_domain(),
        canonical: canonical_bigint(&number.canonical)?,
    })
}

pub(crate) fn promote_bigint_to_rational(
    _cx: &mut Cx,
    number: NumberLiteral,
) -> Result<NumberLiteral> {
    Ok(NumberLiteral {
        domain: rational_domain(),
        canonical: format!("{}/1", canonical_bigint(&number.canonical)?),
    })
}

fn binary_bigint_rule(
    cx: &mut Cx,
    context: &str,
    left: NumberLiteral,
    right: NumberLiteral,
    apply: impl FnOnce(BigInt, BigInt) -> BigInt,
) -> Result<Value> {
    let result = apply(
        parse_bigint_literal(left, "left")?,
        parse_bigint_literal(right, "right")?,
    );
    check_bigint_magnitude(context, &result)?;
    cx.factory()
        .number_literal(number_domain(), result.to_string())
}

fn parse_bigint_literal(number: NumberLiteral, side: &str) -> Result<BigInt> {
    if number.domain != number_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    precheck_bigint_decimal_text(&format!("{side} operand"), &number.canonical)?;
    let value = number.canonical.parse::<BigInt>().map_err(|err| {
        sim_kernel::Error::Eval(format!(
            "{side} operand was not a valid bigint literal: {}",
            err
        ))
    })?;
    check_bigint_magnitude(&format!("{side} operand"), &value)?;
    Ok(value)
}

fn comparison_value(cx: &mut Cx, ordering: Ordering) -> Result<Value> {
    let value = match ordering {
        Ordering::Less => "-1",
        Ordering::Equal => "0",
        Ordering::Greater => "1",
    };
    cx.factory()
        .number_literal(domains::i64(), value.to_owned())
}

pub(crate) fn canonical_bigint(text: &str) -> Result<String> {
    precheck_bigint_decimal_text("bigint literal", text)?;
    let value = text
        .parse::<BigInt>()
        .map_err(|err| sim_kernel::Error::Eval(format!("invalid bigint literal: {}", err)))?;
    check_bigint_magnitude("bigint literal", &value)?;
    Ok(value.to_string())
}

fn magnitude_limit() -> MagnitudeLimit {
    MagnitudeLimit::default_arbitrary_precision()
}

pub(crate) fn precheck_bigint_decimal_text(context: &str, text: &str) -> Result<()> {
    magnitude_limit()
        .check_decimal_digits(context, decimal_digit_count(text))
        .map(|_| ())
}

fn check_bigint_magnitude(context: &str, value: &BigInt) -> Result<()> {
    magnitude_limit().check_bits(context, value.bits())
}

fn estimate_pow_bits(base: &BigInt, exponent: u32) -> u64 {
    if exponent == 0 {
        return 1;
    }
    let zero = BigInt::from(0_u8);
    let one = BigInt::from(1_u8);
    let negative_one = -one.clone();
    if base == &zero || base == &one || base == &negative_one {
        return 1;
    }
    base.bits().max(1).saturating_mul(u64::from(exponent))
}

fn decimal_digit_count(text: &str) -> usize {
    text.bytes().filter(|byte| byte.is_ascii_digit()).count()
}

fn expect_domain_literal(cx: &mut Cx, value: Value, side: &str) -> Result<NumberLiteral> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found non-number",
            number_domain()
        )));
    };
    if number.domain != number_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    number.literal.ok_or_else(|| {
        sim_kernel::Error::Eval(format!(
            "{side} operand in {} does not have a canonical literal form",
            number_domain()
        ))
    })
}

fn expect_domain_literals(
    cx: &mut Cx,
    values: Vec<Value>,
    side: &str,
) -> Result<Vec<NumberLiteral>> {
    values
        .into_iter()
        .map(|value| expect_domain_literal(cx, value, side))
        .collect()
}
