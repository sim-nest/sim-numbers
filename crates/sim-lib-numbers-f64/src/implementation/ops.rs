//! f64 arithmetic rules and canonical-form rendering, in both literal and
//! value form.

use std::cmp::Ordering;

use sim_kernel::{Cx, Error, NumberLiteral, Result, Value};
use sim_lib_numbers_core::domains;

use super::number_domain;

pub(crate) type F64RuleFn = fn(&mut Cx, NumberLiteral, NumberLiteral) -> Result<Value>;
pub(crate) type ValueRuleFn = fn(&mut Cx, Value, Value) -> Result<Value>;

pub(crate) fn canonical_f64(text: &str) -> String {
    match text.parse::<f64>() {
        Ok(value) => {
            let rendered = value.to_string();
            if rendered == "-0" {
                "0".to_owned()
            } else {
                rendered
            }
        }
        Err(_) => text.to_owned(),
    }
}

fn canonical_f64_value(value: f64) -> String {
    let rendered = value.to_string();
    if rendered == "-0" {
        "0".to_owned()
    } else {
        rendered
    }
}

pub(crate) fn f64_add_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    f64_rule(cx, left, right, |left, right| left + right)
}

pub(crate) fn f64_sub_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    f64_rule(cx, left, right, |left, right| left - right)
}

pub(crate) fn f64_mul_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    f64_rule(cx, left, right, |left, right| left * right)
}

pub(crate) fn f64_div_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    f64_rule(cx, left, right, |left, right| left / right)
}

pub(crate) fn f64_cmp_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let left = parse_f64_literal(left, "left")?;
    let right = parse_f64_literal(right, "right")?;
    let ordering = left
        .partial_cmp(&right)
        .ok_or_else(|| Error::Eval("cmp cannot order NaN in f64 arithmetic".to_owned()))?;
    comparison_value(cx, ordering)
}

pub(crate) fn f64_neg_rule(cx: &mut Cx, operand: NumberLiteral) -> Result<Value> {
    let operand = parse_f64_literal(operand, "operand")?;
    cx.factory()
        .number_literal(number_domain(), canonical_f64_value(-operand))
}

pub(crate) fn f64_sum_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = 0.0_f64;
    for operand in operands {
        acc += parse_f64_literal(operand, "operand")?;
    }
    cx.factory()
        .number_literal(number_domain(), canonical_f64_value(acc))
}

pub(crate) fn f64_product_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = 1.0_f64;
    for operand in operands {
        acc *= parse_f64_literal(operand, "operand")?;
    }
    cx.factory()
        .number_literal(number_domain(), canonical_f64_value(acc))
}

pub(crate) fn f64_add_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f64_add_rule(cx, left, right)
}

pub(crate) fn f64_sub_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f64_sub_rule(cx, left, right)
}

pub(crate) fn f64_mul_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f64_mul_rule(cx, left, right)
}

pub(crate) fn f64_div_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f64_div_rule(cx, left, right)
}

pub(crate) fn f64_cmp_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f64_cmp_rule(cx, left, right)
}

pub(crate) fn f64_neg_value_rule(cx: &mut Cx, operand: Value) -> Result<Value> {
    let operand = expect_domain_literal(cx, operand, "operand")?;
    f64_neg_rule(cx, operand)
}

pub(crate) fn f64_sum_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    f64_sum_rule(cx, operands)
}

pub(crate) fn f64_product_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    f64_product_rule(cx, operands)
}

fn f64_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
    apply: impl FnOnce(f64, f64) -> f64,
) -> Result<Value> {
    let left = parse_f64_literal(left, "left")?;
    let right = parse_f64_literal(right, "right")?;
    cx.factory()
        .number_literal(number_domain(), canonical_f64_value(apply(left, right)))
}

fn parse_f64_literal(number: NumberLiteral, side: &str) -> Result<f64> {
    if number.domain != number_domain() {
        return Err(Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    number.canonical.parse::<f64>().map_err(|err| {
        Error::Eval(format!(
            "{side} operand was not a valid f64 literal: {}",
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

fn expect_domain_literal(cx: &mut Cx, value: Value, side: &str) -> Result<NumberLiteral> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(Error::Eval(format!(
            "{side} operand expected number domain {}, found non-number",
            number_domain()
        )));
    };
    if number.domain != number_domain() {
        return Err(Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    number.literal.ok_or_else(|| {
        Error::Eval(format!(
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
