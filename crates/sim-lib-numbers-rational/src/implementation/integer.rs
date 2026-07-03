//! Integer-component helpers for rationals: parsing and validating numerator
//! and denominator values, recognizing integer domains, and compacting an
//! integer-valued rational back to its canonical form.

use num_bigint::BigInt;
use sim_kernel::{Cx, Error, NumberLiteral, Result, Symbol, Value};

pub(crate) fn parse_integer_value(cx: &mut Cx, text: &str) -> Result<Value> {
    let literal = cx.parse_number_literal(text)?.ok_or_else(|| {
        Error::Eval(format!(
            "could not parse integer literal component {}",
            text
        ))
    })?;
    if !is_integer_domain(&literal.domain) {
        return Err(Error::Eval(format!(
            "rational component {} parsed as non-integer domain {}",
            text, literal.domain
        )));
    }
    cx.factory()
        .number_literal(literal.domain, literal.canonical)
}

pub(crate) fn compact_canonical(cx: &mut Cx, num: &Value, den: &Value) -> Result<Option<String>> {
    let Some(num_ref) = cx.number_value_ref(num.clone())? else {
        return Ok(None);
    };
    let Some(den_ref) = cx.number_value_ref(den.clone())? else {
        return Ok(None);
    };
    if num_ref.domain != den_ref.domain || !is_integer_domain(&num_ref.domain) {
        return Ok(None);
    }
    let Some(num_literal) = num_ref.literal else {
        return Ok(None);
    };
    let Some(den_literal) = den_ref.literal else {
        return Ok(None);
    };
    let num_big = parse_integer_literal(&num_literal)?;
    let den_big = parse_integer_literal(&den_literal)?;
    if den_big == BigInt::from(0_u8) {
        return Ok(None);
    }
    Ok(Some(format!("{}/{}", num_big, den_big)))
}

pub(crate) fn is_integer_domain(domain: &Symbol) -> bool {
    matches!(
        domain.to_string().as_str(),
        "numbers/i8"
            | "numbers/u8"
            | "numbers/i16"
            | "numbers/u16"
            | "numbers/i32"
            | "numbers/u32"
            | "numbers/i64"
            | "numbers/u64"
            | "numbers/i128"
            | "numbers/u128"
            | "numbers/isize"
            | "numbers/usize"
            | "numbers/bigint"
    )
}

pub(crate) fn parse_integer_literal(number: &NumberLiteral) -> Result<BigInt> {
    if !is_integer_domain(&number.domain) {
        return Err(Error::Eval(format!(
            "expected integer number domain, found {}",
            number.domain
        )));
    }
    number.canonical.parse::<BigInt>().map_err(|err| {
        Error::Eval(format!(
            "invalid integer canonical literal {}: {}",
            number.canonical, err
        ))
    })
}
