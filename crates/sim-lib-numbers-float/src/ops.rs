//! f32 arithmetic rules, canonical-form rendering, and the promotion from f32
//! into `f64`, in both literal and value form.

use sim_kernel::{Cx, Error, NumberLiteral, Result, Value};

use crate::implementation::{f64_domain, number_domain};

pub(crate) type F32RuleFn = fn(&mut Cx, NumberLiteral, NumberLiteral) -> Result<Value>;
pub(crate) type ValueRuleFn = fn(&mut Cx, Value, Value) -> Result<Value>;

pub(crate) fn canonical_f32(text: &str) -> String {
    match text.parse::<f32>() {
        Ok(value) => canonical_f32_value(value),
        Err(_) => text.to_owned(),
    }
}

fn canonical_f32_value(value: f32) -> String {
    let rendered = value.to_string();
    if rendered == "-0" {
        "0".to_owned()
    } else {
        rendered
    }
}

pub(crate) fn promote_f32_to_f64(_cx: &mut Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    Ok(NumberLiteral {
        domain: f64_domain(),
        canonical: canonical_f32_value(parse_f32_literal(number, "operand")?).to_owned(),
    })
}

pub(crate) fn promote_f32_value_to_f64(cx: &mut Cx, value: Value) -> Result<Value> {
    let literal = expect_domain_literal(cx, value, "operand")?;
    let promoted = promote_f32_to_f64(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

pub(crate) fn f32_add_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    f32_rule(cx, left, right, |left, right| left + right)
}

pub(crate) fn f32_sub_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    f32_rule(cx, left, right, |left, right| left - right)
}

pub(crate) fn f32_mul_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    f32_rule(cx, left, right, |left, right| left * right)
}

pub(crate) fn f32_div_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    f32_rule(cx, left, right, |left, right| left / right)
}

pub(crate) fn f32_neg_rule(cx: &mut Cx, operand: NumberLiteral) -> Result<Value> {
    cx.factory().number_literal(
        number_domain(),
        canonical_f32_value(-parse_f32_literal(operand, "operand")?),
    )
}

pub(crate) fn f32_sum_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = 0.0_f32;
    for operand in operands {
        acc += parse_f32_literal(operand, "operand")?;
    }
    cx.factory()
        .number_literal(number_domain(), canonical_f32_value(acc))
}

pub(crate) fn f32_product_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = 1.0_f32;
    for operand in operands {
        acc *= parse_f32_literal(operand, "operand")?;
    }
    cx.factory()
        .number_literal(number_domain(), canonical_f32_value(acc))
}

pub(crate) fn f32_add_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f32_add_rule(cx, left, right)
}

pub(crate) fn f32_sub_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f32_sub_rule(cx, left, right)
}

pub(crate) fn f32_mul_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f32_mul_rule(cx, left, right)
}

pub(crate) fn f32_div_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    f32_div_rule(cx, left, right)
}

pub(crate) fn f32_neg_value_rule(cx: &mut Cx, operand: Value) -> Result<Value> {
    let operand = expect_domain_literal(cx, operand, "operand")?;
    f32_neg_rule(cx, operand)
}

pub(crate) fn f32_sum_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    f32_sum_rule(cx, operands)
}

pub(crate) fn f32_product_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    f32_product_rule(cx, operands)
}

fn f32_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
    apply: impl FnOnce(f32, f32) -> f32,
) -> Result<Value> {
    cx.factory().number_literal(
        number_domain(),
        canonical_f32_value(apply(
            parse_f32_literal(left, "left")?,
            parse_f32_literal(right, "right")?,
        )),
    )
}

fn parse_f32_literal(number: NumberLiteral, side: &str) -> Result<f32> {
    if number.domain != number_domain() {
        return Err(Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    number.canonical.parse::<f32>().map_err(|err| {
        Error::Eval(format!(
            "{side} operand was not a valid f32 literal: {}",
            err
        ))
    })
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
