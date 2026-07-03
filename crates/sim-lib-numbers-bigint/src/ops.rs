//! Exact bigint arithmetic rules, canonicalization, and the promotions into
//! `rational` (and the integer-to-bigint widening), in both literal and value
//! form.

use std::cmp::Ordering;

use num_bigint::BigInt;
use sim_kernel::{Cx, NumberLiteral, Result, Value};
use sim_lib_numbers_core::domains;

use crate::implementation::{number_domain, rational_domain};

pub(crate) type BigIntRuleFn = fn(&mut Cx, NumberLiteral, NumberLiteral) -> Result<Value>;
pub(crate) type ValueRuleFn = fn(&mut Cx, Value, Value) -> Result<Value>;

pub(crate) fn bigint_add_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    binary_bigint_rule(cx, left, right, |left, right| left + right)
}

pub(crate) fn bigint_sub_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    binary_bigint_rule(cx, left, right, |left, right| left - right)
}

pub(crate) fn bigint_mul_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    binary_bigint_rule(cx, left, right, |left, right| left * right)
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
    cx.factory()
        .number_literal(number_domain(), (left_value / right_value).to_string())
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
    cx.factory()
        .number_literal(number_domain(), (left_value % right_value).to_string())
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
    cx.factory()
        .number_literal(number_domain(), base.pow(exponent_u32).to_string())
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
    cx.factory().number_literal(
        number_domain(),
        (-parse_bigint_literal(operand, "operand")?).to_string(),
    )
}

pub(crate) fn bigint_sum_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = BigInt::from(0_u8);
    for operand in operands {
        acc += parse_bigint_literal(operand, "operand")?;
    }
    cx.factory()
        .number_literal(number_domain(), acc.to_string())
}

pub(crate) fn bigint_product_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = BigInt::from(1_u8);
    for operand in operands {
        acc *= parse_bigint_literal(operand, "operand")?;
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
    left: NumberLiteral,
    right: NumberLiteral,
    apply: impl FnOnce(BigInt, BigInt) -> BigInt,
) -> Result<Value> {
    cx.factory().number_literal(
        number_domain(),
        apply(
            parse_bigint_literal(left, "left")?,
            parse_bigint_literal(right, "right")?,
        )
        .to_string(),
    )
}

fn parse_bigint_literal(number: NumberLiteral, side: &str) -> Result<BigInt> {
    if number.domain != number_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    number.canonical.parse::<BigInt>().map_err(|err| {
        sim_kernel::Error::Eval(format!(
            "{side} operand was not a valid bigint literal: {}",
            err
        ))
    })
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
    text.parse::<BigInt>()
        .map(|value| value.to_string())
        .map_err(|err| sim_kernel::Error::Eval(format!("invalid bigint literal: {}", err)))
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
