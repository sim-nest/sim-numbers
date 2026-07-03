//! Complex arithmetic rules, literal parsing and canonicalization, and the
//! promotions registering `f64`, `i64`, and `rational` into the complex domain.

use sim_kernel::{Cx, Expr, Linker, NumberLiteral, PromotionRule, Result, Value};
use sim_lib_numbers_core::domains;

use super::literal::{f64_domain, i64_domain, number_domain, rational_domain};

pub(super) type ComplexRuleFn = fn(&mut Cx, NumberLiteral, NumberLiteral) -> Result<Value>;
pub(super) type ValueRuleFn = fn(&mut Cx, Value, Value) -> Result<Value>;

pub fn register_promotions(linker: &mut Linker<'_>) {
    for rule in [
        PromotionRule {
            from_domain: f64_domain(),
            to_domain: number_domain(),
            cost: 1,
            convert: promote_f64_to_complex,
        },
        PromotionRule {
            from_domain: i64_domain(),
            to_domain: number_domain(),
            cost: 1,
            convert: promote_i64_to_complex,
        },
        PromotionRule {
            from_domain: rational_domain(),
            to_domain: number_domain(),
            cost: 1,
            convert: promote_rational_to_complex,
        },
    ] {
        linker.promotion_rule(rule);
    }
}

pub(super) fn complex_add_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    complex_rule(
        cx,
        left,
        right,
        |left, right| Some((left.0 + right.0, left.1 + right.1)),
        "add",
    )
}

pub(super) fn complex_sub_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    complex_rule(
        cx,
        left,
        right,
        |left, right| Some((left.0 - right.0, left.1 - right.1)),
        "sub",
    )
}

pub(super) fn complex_mul_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    complex_rule(
        cx,
        left,
        right,
        |left, right| {
            Some((
                left.0 * right.0 - left.1 * right.1,
                left.0 * right.1 + left.1 * right.0,
            ))
        },
        "mul",
    )
}

pub(super) fn complex_div_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    complex_rule(
        cx,
        left,
        right,
        |left, right| {
            let denominator = right.0 * right.0 + right.1 * right.1;
            if denominator == 0.0 {
                return None;
            }
            Some((
                (left.0 * right.0 + left.1 * right.1) / denominator,
                (left.1 * right.0 - left.0 * right.1) / denominator,
            ))
        },
        "div",
    )
}

pub(super) fn complex_neg_rule(cx: &mut Cx, operand: NumberLiteral) -> Result<Value> {
    let operand = parse_complex_number(operand, "operand")?;
    cx.factory()
        .number_literal(number_domain(), canonical_complex(-operand.0, -operand.1))
}

pub(super) fn complex_sum_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = (0.0_f64, 0.0_f64);
    for operand in operands {
        let operand = parse_complex_number(operand, "operand")?;
        acc = (acc.0 + operand.0, acc.1 + operand.1);
    }
    cx.factory()
        .number_literal(number_domain(), canonical_complex(acc.0, acc.1))
}

pub(super) fn complex_product_rule(cx: &mut Cx, operands: Vec<NumberLiteral>) -> Result<Value> {
    let mut acc = (1.0_f64, 0.0_f64);
    for operand in operands {
        let operand = parse_complex_number(operand, "operand")?;
        acc = (
            acc.0 * operand.0 - acc.1 * operand.1,
            acc.0 * operand.1 + acc.1 * operand.0,
        );
    }
    cx.factory()
        .number_literal(number_domain(), canonical_complex(acc.0, acc.1))
}

pub(crate) fn complex_add_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    complex_add_rule(cx, left, right)
}

pub(crate) fn complex_sub_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    complex_sub_rule(cx, left, right)
}

pub(crate) fn complex_mul_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    complex_mul_rule(cx, left, right)
}

pub(crate) fn complex_div_value_rule(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_domain_literal(cx, left, "left")?;
    let right = expect_domain_literal(cx, right, "right")?;
    complex_div_rule(cx, left, right)
}

pub(crate) fn complex_neg_value_rule(cx: &mut Cx, operand: Value) -> Result<Value> {
    let operand = expect_domain_literal(cx, operand, "operand")?;
    complex_neg_rule(cx, operand)
}

pub(crate) fn complex_sum_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    complex_sum_rule(cx, operands)
}

pub(crate) fn complex_product_value_rule(cx: &mut Cx, operands: Vec<Value>) -> Result<Value> {
    let operands = expect_domain_literals(cx, operands, "operand")?;
    complex_product_rule(cx, operands)
}

fn complex_rule(
    cx: &mut Cx,
    left: NumberLiteral,
    right: NumberLiteral,
    apply: impl FnOnce((f64, f64), (f64, f64)) -> Option<(f64, f64)>,
    name: &str,
) -> Result<Value> {
    let left = parse_complex_number(left, "left")?;
    let right = parse_complex_number(right, "right")?;
    let out = apply(left, right).ok_or_else(|| {
        sim_kernel::Error::Eval(format!(
            "{name} failed in complex arithmetic (overflow or divide by zero)"
        ))
    })?;
    cx.factory()
        .number_literal(number_domain(), canonical_complex(out.0, out.1))
}

fn parse_complex_number(number: NumberLiteral, side: &str) -> Result<(f64, f64)> {
    if number.domain != number_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    parse_complex_literal(&number.canonical).ok_or_else(|| {
        sim_kernel::Error::Eval(format!(
            "{side} operand was not a valid complex literal: {}",
            number.canonical
        ))
    })
}

/// Parses a canonical `a+bi` complex literal into its `(real, imag)` parts,
/// returning `None` when the text is not a well-formed complex literal.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_complex::parse_complex_literal;
///
/// assert_eq!(parse_complex_literal("3+4i"), Some((3.0, 4.0)));
/// assert_eq!(parse_complex_literal("1.5-2.25i"), Some((1.5, -2.25)));
/// assert_eq!(parse_complex_literal("not-complex"), None);
/// ```
pub fn parse_complex_literal(text: &str) -> Option<(f64, f64)> {
    let trimmed = text.trim();
    let imag_text = trimmed.strip_suffix('i')?;
    let split = imag_text
        .as_bytes()
        .get(1..)?
        .iter()
        .rposition(|byte| *byte == b'+' || *byte == b'-')
        .map(|index| index + 1)?;
    let (real_text, imag_with_sign) = imag_text.split_at(split);
    let real = real_text.parse::<f64>().ok()?;
    let imag = imag_with_sign.parse::<f64>().ok()?;
    Some((real, imag))
}

/// Renders `(real, imag)` parts into the canonical `a+bi` literal form, fixing
/// the imaginary sign and normalizing negative zero via [`canonical_f64`].
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_complex::canonical_complex;
///
/// assert_eq!(canonical_complex(3.0, 4.0), "3+4i");
/// assert_eq!(canonical_complex(1.5, -2.25), "1.5-2.25i");
/// ```
pub fn canonical_complex(real: f64, imag: f64) -> String {
    let real = canonical_f64(real);
    let imag = canonical_f64(imag);
    if imag.starts_with('-') {
        format!("{real}{imag}i")
    } else {
        format!("{real}+{imag}i")
    }
}

/// Renders an `f64` part for inclusion in a complex literal, normalizing the
/// `-0` rendering to `0`.
pub fn canonical_f64(value: f64) -> String {
    let rendered = value.to_string();
    if rendered == "-0" {
        "0".to_owned()
    } else {
        rendered
    }
}

fn promote_f64_to_complex(_cx: &mut Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    if number.domain != f64_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "expected number domain {} for promotion, found {}",
            f64_domain(),
            number.domain
        )));
    }
    let value = number.canonical.parse::<f64>().map_err(|err| {
        sim_kernel::Error::Eval(format!("could not promote f64 literal to complex: {}", err))
    })?;
    Ok(NumberLiteral {
        domain: number_domain(),
        canonical: canonical_complex(value, 0.0),
    })
}

fn promote_i64_to_complex(_cx: &mut Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    if number.domain != i64_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "expected number domain {} for promotion, found {}",
            i64_domain(),
            number.domain
        )));
    }
    let value = number.canonical.parse::<i64>().map_err(|err| {
        sim_kernel::Error::Eval(format!("could not promote i64 literal to complex: {}", err))
    })?;
    Ok(NumberLiteral {
        domain: number_domain(),
        canonical: canonical_complex(value as f64, 0.0),
    })
}

fn promote_rational_to_complex(_cx: &mut Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    if number.domain != rational_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "expected number domain {} for promotion, found {}",
            rational_domain(),
            number.domain
        )));
    }
    let value = parse_rational_as_f64(&number.canonical).ok_or_else(|| {
        sim_kernel::Error::Eval(format!(
            "could not promote rational literal to complex: {}",
            number.canonical
        ))
    })?;
    Ok(NumberLiteral {
        domain: number_domain(),
        canonical: canonical_complex(value, 0.0),
    })
}

pub(crate) fn promote_f64_value_to_complex(cx: &mut Cx, value: Value) -> Result<Value> {
    let number = expect_number_literal(cx, value, f64_domain(), "operand")?;
    let promoted = promote_f64_to_complex(cx, number)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

pub(crate) fn promote_i64_value_to_complex(cx: &mut Cx, value: Value) -> Result<Value> {
    let number = expect_number_literal(cx, value, i64_domain(), "operand")?;
    let promoted = promote_i64_to_complex(cx, number)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

pub(crate) fn promote_rational_value_to_complex(cx: &mut Cx, value: Value) -> Result<Value> {
    let Some(number) = cx.number_value_ref(value.clone())? else {
        return Err(sim_kernel::Error::Eval(format!(
            "operand expected number domain {}, found non-number",
            rational_domain()
        )));
    };
    if number.domain != rational_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "operand expected number domain {}, found {}",
            rational_domain(),
            number.domain
        )));
    }
    if let Some(literal) = number.literal {
        let promoted = promote_rational_to_complex(cx, literal)?;
        return cx
            .factory()
            .number_literal(promoted.domain, promoted.canonical);
    }

    let value = noncompact_rational_as_f64(cx, value)?;
    cx.factory()
        .number_literal(number_domain(), canonical_complex(value, 0.0))
}

/// Parses a `num/den` rational literal to its `f64` value for the rational ->
/// complex promotion, returning `None` on malformed text or a zero denominator.
pub fn parse_rational_as_f64(text: &str) -> Option<f64> {
    let (numerator, denominator) = text.split_once('/')?;
    let numerator = numerator.parse::<f64>().ok()?;
    let denominator = denominator.parse::<f64>().ok()?;
    if denominator == 0.0 {
        return None;
    }
    Some(numerator / denominator)
}

fn expect_domain_literal(cx: &mut Cx, value: Value, side: &str) -> Result<NumberLiteral> {
    expect_number_literal(cx, value, number_domain(), side)
}

fn expect_number_literal(
    cx: &mut Cx,
    value: Value,
    expected_domain: sim_kernel::Symbol,
    side: &str,
) -> Result<NumberLiteral> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found non-number",
            expected_domain
        )));
    };
    if number.domain != expected_domain {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            expected_domain, number.domain
        )));
    }
    number.literal.ok_or_else(|| {
        sim_kernel::Error::Eval(format!(
            "{side} operand in {} does not have a canonical literal form",
            expected_domain
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

fn noncompact_rational_as_f64(cx: &mut Cx, value: Value) -> Result<f64> {
    let Expr::Extension { tag, payload } = value.object().as_expr(cx)? else {
        return Err(sim_kernel::Error::Eval(
            "operand rational value does not have a complex-compatible surface".to_owned(),
        ));
    };
    if tag != domains::rational_value_class() {
        return Err(sim_kernel::Error::Eval(format!(
            "operand expected rational value extension, found {}",
            tag
        )));
    }
    let Expr::Vector(parts) = payload.as_ref() else {
        return Err(sim_kernel::Error::Eval(
            "operand rational extension payload must be a vector".to_owned(),
        ));
    };
    let [num, den] = parts.as_slice() else {
        return Err(sim_kernel::Error::Eval(
            "operand rational extension must contain numerator and denominator".to_owned(),
        ));
    };
    let num = number_expr_as_f64(num, "numerator")?;
    let den = number_expr_as_f64(den, "denominator")?;
    if den == 0.0 {
        return Err(sim_kernel::Error::Eval(
            "operand rational denominator must not be zero".to_owned(),
        ));
    }
    Ok(num / den)
}

fn number_expr_as_f64(expr: &Expr, side: &str) -> Result<f64> {
    let Expr::Number(number) = expr else {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} expected numeric literal in rational extension"
        )));
    };
    number.canonical.parse::<f64>().map_err(|err| {
        sim_kernel::Error::Eval(format!(
            "{side} could not be converted to f64 for complex promotion: {}",
            err
        ))
    })
}
