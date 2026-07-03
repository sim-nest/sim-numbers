#![allow(missing_docs)]

use crate::number_domain;

#[sim::sim_lib(id = "numbers/f64", version = "0.1.0", native_export = true)]
mod f64_native {
    use super::*;
    #[allow(unused_imports)]
    use sim::{
        kernel::{Error, Expr, NumberLiteral, Result},
        sim_number_domain,
    };

    #[sim_number_domain(
        symbol = "numbers/f64",
        parse = "parse_f64_native",
        encode = "encode_f64_native",
        add = "add_f64_native",
        sub = "sub_f64_native",
        mul = "mul_f64_native",
        div = "div_f64_native",
        neg = "neg_f64_native",
        sum = "sum_f64_native",
        product = "product_f64_native"
    )]
    pub fn f64_domain() {}

    pub fn parse_f64_native(text: String) -> Result<Option<Expr>> {
        if text.parse::<f64>().is_err() {
            return Ok(None);
        }
        Ok(Some(number_expr(canonical_f64_text(&text))))
    }

    pub fn encode_f64_native(expr: Expr) -> Result<Option<Expr>> {
        match expr {
            Expr::Number(number) if number.domain == number_domain() => {
                Ok(Some(Expr::Number(number)))
            }
            _ => Ok(None),
        }
    }

    pub fn add_f64_native(left: Expr, right: Expr) -> Result<Expr> {
        binary_f64(left, right, |left, right| left + right)
    }

    pub fn sub_f64_native(left: Expr, right: Expr) -> Result<Expr> {
        binary_f64(left, right, |left, right| left - right)
    }

    pub fn mul_f64_native(left: Expr, right: Expr) -> Result<Expr> {
        binary_f64(left, right, |left, right| left * right)
    }

    pub fn div_f64_native(left: Expr, right: Expr) -> Result<Expr> {
        binary_f64(left, right, |left, right| left / right)
    }

    pub fn neg_f64_native(operand: Expr) -> Result<Expr> {
        Ok(number_expr(canonical_f64_value(-expect_f64(
            operand, "operand",
        )?)))
    }

    pub fn sum_f64_native(operands: Vec<Expr>) -> Result<Expr> {
        let mut acc = 0.0_f64;
        for operand in operands {
            acc += expect_f64(operand, "operand")?;
        }
        Ok(number_expr(canonical_f64_value(acc)))
    }

    pub fn product_f64_native(operands: Vec<Expr>) -> Result<Expr> {
        let mut acc = 1.0_f64;
        for operand in operands {
            acc *= expect_f64(operand, "operand")?;
        }
        Ok(number_expr(canonical_f64_value(acc)))
    }

    fn binary_f64(left: Expr, right: Expr, apply: impl FnOnce(f64, f64) -> f64) -> Result<Expr> {
        let left = expect_f64(left, "left")?;
        let right = expect_f64(right, "right")?;
        Ok(number_expr(canonical_f64_value(apply(left, right))))
    }

    fn expect_f64(expr: Expr, side: &str) -> Result<f64> {
        let Expr::Number(number) = expr else {
            return Err(Error::TypeMismatch {
                expected: "number",
                found: "non-number",
            });
        };
        if number.domain != number_domain() {
            return Err(Error::Eval(format!(
                "{side} operand expected number domain {}, found {}",
                number_domain(),
                number.domain
            )));
        }
        number.canonical.parse::<f64>().map_err(|err| {
            Error::Eval(format!("{side} operand was not a valid f64 literal: {err}"))
        })
    }

    fn number_expr(canonical: String) -> Expr {
        Expr::Number(NumberLiteral {
            domain: number_domain(),
            canonical,
        })
    }

    fn canonical_f64_text(text: &str) -> String {
        match text.parse::<f64>() {
            Ok(value) => canonical_f64_value(value),
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
}
