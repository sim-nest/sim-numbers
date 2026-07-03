//! Symbolic differentiation of `CasExpr` trees: the recursive `diff` rules for
//! the builtin operators, falling through to the extensible rule registry.

use sim_kernel::{Cx, Error, Result, Symbol, Value};
use sim_lib_numbers_cas::{CasExpr, simplify_expr};
use sim_lib_numbers_core::domains;

use super::registry::apply_registered_rule;

/// The `diff` symbol: the symbolic differentiation entry point.
pub fn diff_symbol() -> Symbol {
    Symbol::new("diff")
}

/// Differentiate a [`CasExpr`] with respect to `var`, returning a simplified
/// derivative tree.
///
/// Builtin operators (`+`, `-`, `*`, `/`, `^`, and the standard trig functions)
/// use their hard-coded rules; unrecognized operators fall through to the
/// extensible differentiation-rule registry.
pub fn diff_cas(cx: &mut Cx, expr: &CasExpr, var: &Symbol) -> Result<CasExpr> {
    let derivative = match expr {
        CasExpr::Num(_) => zero(cx)?,
        CasExpr::Var(symbol) if symbol == var => one(cx)?,
        CasExpr::Var(_) => zero(cx)?,
        CasExpr::Op(operator, args) if *operator == math("add") => {
            op(math("add"), diff_all(cx, args, var)?)
        }
        CasExpr::Op(operator, args) if *operator == math("sub") => diff_sub(cx, args, var)?,
        CasExpr::Op(operator, args) if *operator == math("mul") => diff_mul(cx, args, var)?,
        CasExpr::Op(operator, args) if *operator == math("div") => diff_div(cx, args, var)?,
        CasExpr::Op(operator, args) if *operator == math("pow") => diff_pow(cx, args, var)?,
        CasExpr::Op(operator, args) if *operator == Symbol::new("sin") => {
            chain_rule(cx, Symbol::new("cos"), args, var)?
        }
        CasExpr::Op(operator, args) if *operator == Symbol::new("cos") => {
            let [arg] = one_arg(args, operator)?;
            op(
                math("mul"),
                vec![
                    neg_one(cx)?,
                    diff_cas(cx, arg, var)?,
                    op(Symbol::new("sin"), vec![arg.clone()]),
                ],
            )
        }
        CasExpr::Op(operator, args) if *operator == Symbol::new("tan") => {
            let [arg] = one_arg(args, operator)?;
            op(
                math("div"),
                vec![
                    diff_cas(cx, arg, var)?,
                    op(
                        math("pow"),
                        vec![
                            op(Symbol::new("cos"), vec![arg.clone()]),
                            CasExpr::Num(number_constant(cx, "2")?),
                        ],
                    ),
                ],
            )
        }
        CasExpr::Op(operator, args) if *operator == Symbol::new("ln") => {
            let [arg] = one_arg(args, operator)?;
            op(math("div"), vec![diff_cas(cx, arg, var)?, arg.clone()])
        }
        CasExpr::Op(operator, args) if *operator == Symbol::new("exp") => {
            chain_rule(cx, Symbol::new("exp"), args, var)?
        }
        CasExpr::Op(operator, args) => {
            if let Some(custom) = apply_registered_rule(operator, args, var) {
                custom
            } else {
                op(
                    diff_symbol(),
                    vec![
                        CasExpr::Op(operator.clone(), args.clone()),
                        CasExpr::Var(var.clone()),
                    ],
                )
            }
        }
    };
    simplify_expr(cx, derivative)
}

fn diff_all(cx: &mut Cx, args: &[CasExpr], var: &Symbol) -> Result<Vec<CasExpr>> {
    args.iter().map(|arg| diff_cas(cx, arg, var)).collect()
}

fn diff_sub(cx: &mut Cx, args: &[CasExpr], var: &Symbol) -> Result<CasExpr> {
    match args {
        [] => Err(Error::Eval(
            "cannot differentiate an empty subtraction".to_owned(),
        )),
        [arg] => Ok(op(math("mul"), vec![neg_one(cx)?, diff_cas(cx, arg, var)?])),
        _ => Ok(op(math("sub"), diff_all(cx, args, var)?)),
    }
}

fn diff_mul(cx: &mut Cx, args: &[CasExpr], var: &Symbol) -> Result<CasExpr> {
    if args.is_empty() {
        return Err(Error::Eval(
            "cannot differentiate an empty multiplication".to_owned(),
        ));
    }
    let mut terms = Vec::with_capacity(args.len());
    for (index, _) in args.iter().enumerate() {
        let mut factors = Vec::with_capacity(args.len());
        for (offset, arg) in args.iter().enumerate() {
            if index == offset {
                factors.push(diff_cas(cx, arg, var)?);
            } else {
                factors.push(arg.clone());
            }
        }
        terms.push(op(math("mul"), factors));
    }
    Ok(op(math("add"), terms))
}

fn diff_div(cx: &mut Cx, args: &[CasExpr], var: &Symbol) -> Result<CasExpr> {
    match args {
        [] => Err(Error::Eval(
            "cannot differentiate an empty division".to_owned(),
        )),
        [arg] => Ok(op(
            math("div"),
            vec![
                op(math("mul"), vec![neg_one(cx)?, diff_cas(cx, arg, var)?]),
                op(
                    math("pow"),
                    vec![arg.clone(), CasExpr::Num(number_constant(cx, "2")?)],
                ),
            ],
        )),
        [left, right] => {
            let left_diff = diff_cas(cx, left, var)?;
            let right_diff = diff_cas(cx, right, var)?;
            Ok(op(
                math("div"),
                vec![
                    op(
                        math("sub"),
                        vec![
                            op(math("mul"), vec![left_diff, right.clone()]),
                            op(math("mul"), vec![left.clone(), right_diff]),
                        ],
                    ),
                    op(
                        math("pow"),
                        vec![right.clone(), CasExpr::Num(number_constant(cx, "2")?)],
                    ),
                ],
            ))
        }
        [head, tail @ ..] => diff_div(cx, &[head.clone(), op(math("mul"), tail.to_vec())], var),
    }
}

fn diff_pow(cx: &mut Cx, args: &[CasExpr], var: &Symbol) -> Result<CasExpr> {
    let [base, exponent] = two_args(args, &math("pow"))?;
    let base_diff = diff_cas(cx, base, var)?;
    if let CasExpr::Num(value) = exponent
        && let Some(decremented) = decrement_value(cx, value)?
    {
        return Ok(op(
            math("mul"),
            vec![
                CasExpr::Num(value.clone()),
                op(math("pow"), vec![base.clone(), CasExpr::Num(decremented)]),
                base_diff,
            ],
        ));
    }
    let exponent_diff = diff_cas(cx, exponent, var)?;
    Ok(op(
        math("mul"),
        vec![
            op(math("pow"), vec![base.clone(), exponent.clone()]),
            op(
                math("add"),
                vec![
                    op(
                        math("mul"),
                        vec![exponent_diff, op(Symbol::new("ln"), vec![base.clone()])],
                    ),
                    op(
                        math("div"),
                        vec![
                            op(math("mul"), vec![exponent.clone(), base_diff]),
                            base.clone(),
                        ],
                    ),
                ],
            ),
        ],
    ))
}

fn chain_rule(cx: &mut Cx, outer: Symbol, args: &[CasExpr], var: &Symbol) -> Result<CasExpr> {
    let [arg] = one_arg(args, &outer)?;
    Ok(op(
        math("mul"),
        vec![diff_cas(cx, arg, var)?, op(outer, vec![arg.clone()])],
    ))
}

fn one_arg<'a>(args: &'a [CasExpr], operator: &Symbol) -> Result<[&'a CasExpr; 1]> {
    let [arg] = args else {
        return Err(Error::Eval(format!(
            "{operator} expects exactly one CAS operand"
        )));
    };
    Ok([arg])
}

fn two_args<'a>(args: &'a [CasExpr], operator: &Symbol) -> Result<[&'a CasExpr; 2]> {
    let [left, right] = args else {
        return Err(Error::Eval(format!(
            "{operator} expects exactly two CAS operands"
        )));
    };
    Ok([left, right])
}

fn zero(cx: &mut Cx) -> Result<CasExpr> {
    Ok(CasExpr::Num(number_constant(cx, "0")?))
}

fn one(cx: &mut Cx) -> Result<CasExpr> {
    Ok(CasExpr::Num(number_constant(cx, "1")?))
}

fn neg_one(cx: &mut Cx) -> Result<CasExpr> {
    Ok(CasExpr::Num(number_constant(cx, "-1")?))
}

fn number_constant(cx: &mut Cx, canonical: &str) -> Result<Value> {
    if cx
        .registry()
        .number_domain_by_symbol(&domains::i64())
        .is_some()
    {
        return cx
            .factory()
            .number_literal(domains::i64(), canonical.to_owned());
    }
    if cx
        .registry()
        .number_domain_by_symbol(&domains::f64())
        .is_some()
    {
        let canonical = if canonical == "-1" {
            "-1.0".to_owned()
        } else {
            format!("{canonical}.0")
        };
        return cx.factory().number_literal(domains::f64(), canonical);
    }
    Err(Error::Eval(
        "CAS differentiation requires a loaded integer or f64 number domain".to_owned(),
    ))
}

fn decrement_value(cx: &mut Cx, value: &Value) -> Result<Option<Value>> {
    if literal_number(cx, value)?.is_none() {
        return Ok(None);
    }
    let one = number_constant(cx, "1")?;
    let decremented = cx.apply_value_number_binary_op(&math("sub"), value.clone(), one)?;
    Ok(cx
        .number_value_ref(decremented.clone())?
        .and_then(|number| number.literal)
        .map(|_| decremented))
}

use sim_lib_numbers_cas::literal_number;

pub(crate) fn math(name: &str) -> Symbol {
    Symbol::qualified("math", name)
}

pub(crate) fn op(operator: Symbol, args: Vec<CasExpr>) -> CasExpr {
    CasExpr::Op(operator, args)
}
