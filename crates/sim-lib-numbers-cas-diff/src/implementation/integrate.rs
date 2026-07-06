//! Symbolic integration of `CasExpr` trees: the recursive `integrate` rules
//! covering the builtin operators and power-of-variable cases.

use sim_kernel::{Cx, Error, Result, Symbol, Value};
use sim_lib_numbers_cas::{CasExpr, simplify_expr};
use sim_lib_numbers_core::domains;

use super::diff::math;

/// The `integrate-sym` symbol: the symbolic integration entry point.
pub fn integrate_sym_symbol() -> Symbol {
    Symbol::new("integrate-sym")
}

/// Integrate a [`CasExpr`] with respect to `var`, returning a simplified
/// antiderivative tree (without constant of integration).
///
/// Covers the builtin arithmetic operators and power-of-variable cases; inputs
/// outside the supported set produce an error.
pub fn integrate_cas(cx: &mut Cx, expr: &CasExpr, var: &Symbol) -> Result<CasExpr> {
    let integral = match expr {
        CasExpr::Num(value) => op(
            math("mul"),
            vec![CasExpr::Num(value.clone()), CasExpr::Var(var.clone())],
        ),
        CasExpr::Var(symbol) if symbol == var => integrate_power_of_var(cx, 1, var)?,
        CasExpr::Var(symbol) => op(
            math("mul"),
            vec![CasExpr::Var(symbol.clone()), CasExpr::Var(var.clone())],
        ),
        CasExpr::Op(operator, args) if *operator == math("add") => {
            op(math("add"), integrate_all(cx, args, var)?)
        }
        CasExpr::Op(operator, args) if *operator == math("sub") => {
            op(math("sub"), integrate_all(cx, args, var)?)
        }
        CasExpr::Op(operator, args) if *operator == math("mul") => integrate_mul(cx, args, var)?,
        CasExpr::Op(operator, args) if *operator == math("pow") => integrate_pow(cx, args, var)?,
        _ => {
            return Err(Error::Eval(format!(
                "{} only supports constants, sums, scalar products, and powers of the integration variable",
                integrate_sym_symbol()
            )));
        }
    };
    simplify_expr(cx, integral)
}

fn integrate_all(cx: &mut Cx, args: &[CasExpr], var: &Symbol) -> Result<Vec<CasExpr>> {
    args.iter().map(|arg| integrate_cas(cx, arg, var)).collect()
}

fn integrate_mul(cx: &mut Cx, args: &[CasExpr], var: &Symbol) -> Result<CasExpr> {
    let mut constant = Vec::new();
    let mut variable = Vec::new();
    for arg in args {
        if depends_on(arg, var) {
            variable.push(arg.clone());
        } else {
            constant.push(arg.clone());
        }
    }
    if variable.len() != 1 {
        return Err(Error::Eval(format!(
            "{} only handles products with exactly one variable-dependent factor",
            integrate_sym_symbol()
        )));
    }
    let mut out = constant;
    out.push(integrate_cas(cx, &variable[0], var)?);
    Ok(op(math("mul"), out))
}

fn integrate_pow(cx: &mut Cx, args: &[CasExpr], var: &Symbol) -> Result<CasExpr> {
    let [base, exponent] = two_args(args)?;
    if !matches!(base, CasExpr::Var(symbol) if symbol == var) {
        return Err(Error::Eval(format!(
            "{} only supports powers of the integration variable",
            integrate_sym_symbol()
        )));
    }
    let exponent = literal_i64(cx, exponent)?.ok_or_else(|| {
        Error::Eval(format!(
            "{} only supports integer exponents for symbolic powers",
            integrate_sym_symbol()
        ))
    })?;
    integrate_power_of_var(cx, exponent, var)
}

fn integrate_power_of_var(cx: &mut Cx, exponent: i64, var: &Symbol) -> Result<CasExpr> {
    if exponent == -1 {
        return Ok(op(Symbol::new("ln"), vec![CasExpr::Var(var.clone())]));
    }
    let next = exponent.checked_add(1).ok_or_else(|| {
        Error::Eval(format!(
            "{} exponent {exponent} overflows when raised for integration",
            integrate_sym_symbol()
        ))
    })?;
    Ok(op(
        math("mul"),
        vec![
            CasExpr::Num(rational_constant(cx, 1, next)?),
            op(
                math("pow"),
                vec![
                    CasExpr::Var(var.clone()),
                    CasExpr::Num(integer_constant(cx, next)?),
                ],
            ),
        ],
    ))
}

fn depends_on(expr: &CasExpr, var: &Symbol) -> bool {
    match expr {
        CasExpr::Num(_) => false,
        CasExpr::Var(symbol) => symbol == var,
        CasExpr::Op(_, args) => args.iter().any(|arg| depends_on(arg, var)),
    }
}

fn literal_i64(cx: &mut Cx, expr: &CasExpr) -> Result<Option<i64>> {
    let CasExpr::Num(value) = expr else {
        return Ok(None);
    };
    let display = value.object().display(cx)?;
    Ok(display.parse::<i64>().ok())
}

fn integer_constant(cx: &mut Cx, value: i64) -> Result<Value> {
    if cx
        .registry()
        .number_domain_by_symbol(&domains::i64())
        .is_some()
    {
        return cx
            .factory()
            .number_literal(domains::i64(), value.to_string());
    }
    cx.factory()
        .number_literal(domains::f64(), format!("{value}.0"))
}

fn rational_constant(cx: &mut Cx, num: i64, den: i64) -> Result<Value> {
    if cx
        .registry()
        .number_domain_by_symbol(&domains::rational())
        .is_some()
    {
        return cx
            .factory()
            .number_literal(domains::rational(), format!("{num}/{den}"));
    }
    let value = num as f64 / den as f64;
    cx.factory()
        .number_literal(domains::f64(), value.to_string())
}

fn two_args(args: &[CasExpr]) -> Result<[&CasExpr; 2]> {
    let [left, right] = args else {
        return Err(Error::Eval(format!(
            "{} expects exactly two operands",
            math("pow")
        )));
    };
    Ok([left, right])
}

use super::diff::op;
