//! Checked i64 arithmetic rules and the promotions from i64 into the `f64` and
//! `rational` domains, in both literal and value form.

use std::cmp::Ordering;

use num_bigint::BigInt;
use sim_kernel::{Cx, NumberLiteral, Result, Value};
use sim_lib_numbers_core::domains;

use super::{f64_domain, number_domain, rational_domain};

pub(crate) type I64RuleFn = fn(&mut Cx, NumberLiteral, NumberLiteral) -> Result<Value>;
pub(crate) type ValueRuleFn = fn(&mut Cx, Value, Value) -> Result<Value>;

pub(crate) fn i64_add_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    i64_rule(
        cx,
        left,
        right,
        |left, right| left.checked_add(right),
        "add",
    )
}

pub(crate) fn i64_sub_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    i64_rule(
        cx,
        left,
        right,
        |left, right| left.checked_sub(right),
        "sub",
    )
}

pub(crate) fn i64_mul_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    i64_rule(
        cx,
        left,
        right,
        |left, right| left.checked_mul(right),
        "mul",
    )
}

pub(crate) fn i64_div_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = parse_i64_literal(left, "left")?;
    let right = parse_i64_literal(right, "right")?;
    if right == 0 {
        return Err(sim_kernel::Error::Eval(
            "div divided by zero in i64 arithmetic".to_owned(),
        ));
    }
    if rational_loaded(cx) {
        return cx.factory().number_literal(
            rational_domain(),
            canonical_rational(left as i128, right as i128),
        );
    }
    match left.checked_div(right) {
        Some(out) => cx
            .factory()
            .number_literal(number_domain(), out.to_string()),
        None => widen_i64_binary(cx, left, right, "div"),
    }
}

pub(crate) fn i64_rem_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    i64_rule(
        cx,
        left,
        right,
        |left, right| left.checked_rem(right),
        "rem",
    )
}

pub(crate) fn i64_pow_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let base = parse_i64_literal(left, "left")?;
    let exponent = parse_i64_literal(right, "right")?;
    let exponent_u32 = u32::try_from(exponent)
        .map_err(|_| sim_kernel::Error::Eval("pow exponent must be nonnegative".to_owned()))?;
    if let Some(out) = base.checked_pow(exponent_u32) {
        return cx
            .factory()
            .number_literal(number_domain(), out.to_string());
    }
    let big = BigInt::from(base).pow(exponent_u32);
    if bigint_loaded(cx) {
        return cx
            .factory()
            .number_literal(bigint_domain(), big.to_string());
    }
    if i128_loaded(cx)
        && let Ok(out) = i128::try_from(big)
    {
        return cx.factory().number_literal(i128_domain(), out.to_string());
    }
    Err(sim_kernel::Error::Eval(
        "pow overflowed i64 arithmetic".to_owned(),
    ))
}

pub(crate) fn i64_cmp_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = parse_i64_literal(left, "left")?;
    let right = parse_i64_literal(right, "right")?;
    comparison_value(cx, left.cmp(&right))
}

pub(crate) fn i64_neg_rule(cx: &mut Cx, operand: NumberLiteral) -> Result<Value> {
    let operand = parse_i64_literal(operand, "operand")?;
    let out = operand.checked_neg().ok_or_else(|| {
        sim_kernel::Error::Eval("neg overflowed or divided by zero in i64 arithmetic".to_owned())
    })?;
    cx.factory()
        .number_literal(number_domain(), out.to_string())
}

pub(crate) fn i64_sum_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = 0_i64;
    for operand in operands {
        let operand = parse_i64_literal(operand, "operand")?;
        acc = acc.checked_add(operand).ok_or_else(|| {
            sim_kernel::Error::Eval(
                "sum overflowed or divided by zero in i64 arithmetic".to_owned(),
            )
        })?;
    }
    cx.factory()
        .number_literal(number_domain(), acc.to_string())
}

pub(crate) fn i64_product_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = 1_i64;
    for operand in operands {
        let operand = parse_i64_literal(operand, "operand")?;
        acc = acc.checked_mul(operand).ok_or_else(|| {
            sim_kernel::Error::Eval(
                "product overflowed or divided by zero in i64 arithmetic".to_owned(),
            )
        })?;
    }
    cx.factory()
        .number_literal(number_domain(), acc.to_string())
}

pub(crate) fn i64_add_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    i64_add_rule(cx, left, right)
}

pub(crate) fn i64_sub_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    i64_sub_rule(cx, left, right)
}

pub(crate) fn i64_mul_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    i64_mul_rule(cx, left, right)
}

pub(crate) fn i64_div_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    i64_div_rule(cx, left, right)
}

pub(crate) fn i64_rem_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    i64_rem_rule(cx, left, right)
}

pub(crate) fn i64_pow_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    i64_pow_rule(cx, left, right)
}

pub(crate) fn i64_cmp_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    i64_cmp_rule(cx, left, right)
}

pub(crate) fn i64_neg_value_rule(cx: &mut Cx, operand: Value) -> Result<Value> {
    let operand = expect_domain_literal(cx, operand, "operand")?;
    i64_neg_rule(cx, operand)
}

pub(crate) fn i64_sum_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    i64_sum_rule(cx, operands)
}

pub(crate) fn i64_product_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    i64_product_rule(cx, operands)
}

fn i64_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
    apply: impl FnOnce(i64, i64) -> Option<i64>,
    name: &str,
) -> Result<Value> {
    let left = parse_i64_literal(left, "left")?;
    let right = parse_i64_literal(right, "right")?;
    match apply(left, right) {
        Some(out) => cx
            .factory()
            .number_literal(number_domain(), out.to_string()),
        None => widen_i64_binary(cx, left, right, name),
    }
}

fn parse_i64_literal(number: NumberLiteral, side: &str) -> Result<i64> {
    if number.domain != number_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    number.canonical.parse::<i64>().map_err(|err| {
        sim_kernel::Error::Eval(format!(
            "{side} operand was not a valid i64 literal: {}",
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
        .number_literal(number_domain(), value.to_owned())
}

pub(crate) fn promote_i64_to_f64(_cx: &mut Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    let value = parse_i64_literal(number, "operand")?;
    Ok(NumberLiteral {
        domain: f64_domain(),
        canonical: value.to_string(),
    })
}

pub(crate) fn promote_i64_to_rational(
    _cx: &mut Cx,
    number: NumberLiteral,
) -> Result<NumberLiteral> {
    let value = parse_i64_literal(number, "operand")?;
    Ok(NumberLiteral {
        domain: rational_domain(),
        canonical: format!("{value}/1"),
    })
}

pub(crate) fn promote_i64_value_to_f64(cx: &mut Cx, value: Value) -> Result<Value> {
    let number = expect_domain_literal(cx, value, "operand")?;
    let promoted = promote_i64_to_f64(cx, number)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

pub(crate) fn promote_i64_value_to_rational(cx: &mut Cx, value: Value) -> Result<Value> {
    let number = expect_domain_literal(cx, value, "operand")?;
    let promoted = promote_i64_to_rational(cx, number)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
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

fn widen_i64_binary(cx: &mut Cx, left: i64, right: i64, name: &str) -> Result<Value> {
    let left_i128 = i128::from(left);
    let right_i128 = i128::from(right);
    let out_i128 = match name {
        "add" => left_i128 + right_i128,
        "sub" => left_i128 - right_i128,
        "mul" => left_i128 * right_i128,
        "div" => {
            if right == 0 {
                return Err(sim_kernel::Error::Eval(
                    "div divided by zero in i64 arithmetic".to_owned(),
                ));
            }
            left_i128 / right_i128
        }
        "rem" => {
            if right == 0 {
                return Err(sim_kernel::Error::Eval(
                    "rem divided by zero in i64 arithmetic".to_owned(),
                ));
            }
            left_i128 % right_i128
        }
        _ => {
            return Err(sim_kernel::Error::Eval(format!(
                "{name} overflowed i64 arithmetic"
            )));
        }
    };
    if bigint_loaded(cx) {
        let out = match name {
            "add" => BigInt::from(left) + BigInt::from(right),
            "sub" => BigInt::from(left) - BigInt::from(right),
            "mul" => BigInt::from(left) * BigInt::from(right),
            "rem" => BigInt::from(left) % BigInt::from(right),
            _ => BigInt::from(out_i128),
        };
        return cx
            .factory()
            .number_literal(bigint_domain(), out.to_string());
    }
    if i128_loaded(cx) {
        return cx
            .factory()
            .number_literal(i128_domain(), out_i128.to_string());
    }
    Err(sim_kernel::Error::Eval(format!(
        "{name} overflowed i64 arithmetic"
    )))
}

fn bigint_domain() -> sim_kernel::Symbol {
    domains::bigint()
}

fn i128_domain() -> sim_kernel::Symbol {
    domains::i128()
}

fn rational_loaded(cx: &Cx) -> bool {
    cx.registry()
        .number_domain_by_symbol(&rational_domain())
        .is_some()
}

fn bigint_loaded(cx: &Cx) -> bool {
    cx.registry()
        .number_domain_by_symbol(&bigint_domain())
        .is_some()
}

fn i128_loaded(cx: &Cx) -> bool {
    cx.registry()
        .number_domain_by_symbol(&i128_domain())
        .is_some()
}

fn canonical_rational(numerator: i128, denominator: i128) -> String {
    let sign = if denominator < 0 { -1 } else { 1 };
    let numerator = numerator * sign;
    let denominator = denominator * sign;
    let gcd = gcd_i128(numerator, denominator);
    format!("{}/{}", numerator / gcd, denominator / gcd)
}

fn gcd_i128(mut left: i128, mut right: i128) -> i128 {
    left = left.abs();
    right = right.abs();
    while right != 0 {
        let rem = left % right;
        left = right;
        right = rem;
    }
    left.max(1)
}
