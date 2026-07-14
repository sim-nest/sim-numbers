//! Algebraic simplification of `CasExpr` trees, and the conversions from
//! runtime values and surface `Expr` into the symbolic form the simplifier
//! operates on.

use sim_kernel::{CanonicalKey, Cx, Error, Expr, NumberLiteral, Result, Symbol, Value};
use sim_lib_numbers_core::domains;

use super::value::{CasExpr, CasValue, cas_expr_to_surface_expr, cas_expr_to_value};

pub(crate) fn simplify_value(cx: &mut Cx, value: Value) -> Result<Value> {
    let expr = value_to_cas_expr(cx, value)?;
    let simplified = simplify_expr(cx, expr)?;
    cas_expr_to_value(cx, simplified)
}

/// Convert a runtime [`Value`] into a symbolic [`CasExpr`].
///
/// Symbolic CAS values are unwrapped directly, concrete numbers wrap as
/// [`CasExpr::Num`], and other values are parsed from their surface `Expr` via
/// [`expr_to_cas_expr`]. Inputs that are neither numeric nor symbolic error.
///
/// [`CasExpr::Num`]: crate::CasExpr::Num
pub fn value_to_cas_expr(cx: &mut Cx, value: Value) -> Result<CasExpr> {
    if let Some(cas) = value.object().downcast_ref::<CasValue>() {
        return Ok(cas.expr().clone());
    }
    if cx.number_value_ref(value.clone())?.is_some() {
        return CasExpr::num(cx, value);
    }
    let expr = value.object().as_expr(cx)?;
    if let Some(parsed) = expr_to_cas_expr(cx, &expr)? {
        return Ok(parsed);
    }
    Err(Error::Eval(format!(
        "expected a numeric or symbolic CAS input, found {}",
        value.object().display(cx)?
    )))
}

/// Parse a surface [`Expr`] into a symbolic [`CasExpr`], if it is CAS-shaped.
///
/// Numbers, symbols (and quoted symbols), operator calls, and canonical Lisp
/// lists map to their algebraic forms; anything else yields `None`.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, Expr, NoopEvalPolicy, Symbol};
/// use sim_lib_numbers_cas::{expr_to_cas_expr, CasExpr};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// let parsed = expr_to_cas_expr(&mut cx, &Expr::Symbol(Symbol::new("y"))).unwrap();
/// assert!(matches!(parsed, Some(CasExpr::Var(_))));
///
/// let not_cas = expr_to_cas_expr(&mut cx, &Expr::String("hi".to_owned())).unwrap();
/// assert!(not_cas.is_none());
/// ```
pub fn expr_to_cas_expr(cx: &mut Cx, expr: &Expr) -> Result<Option<CasExpr>> {
    Ok(match expr {
        Expr::Number(number) => {
            let value = cx
                .factory()
                .number_literal(number.domain.clone(), number.canonical.clone())?;
            Some(CasExpr::num(cx, value)?)
        }
        Expr::Symbol(symbol) => Some(CasExpr::Var(symbol.clone())),
        Expr::Call { operator, args } => {
            let Expr::Symbol(operator) = operator.as_ref() else {
                return Ok(None);
            };
            let args = args
                .iter()
                .map(|arg| {
                    expr_to_cas_expr(cx, arg)?.ok_or_else(|| {
                        Error::Eval(format!("cannot parse {:?} as a CAS operand", arg))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Some(CasExpr::Op(normalize_operator(operator), args))
        }
        Expr::Quote {
            mode: sim_kernel::QuoteMode::Quote,
            expr,
        } => match expr.as_ref() {
            Expr::Symbol(symbol) => Some(CasExpr::Var(symbol.clone())),
            _ => None,
        },
        Expr::List(items) => parse_surface_list(cx, items)?,
        _ => None,
    })
}

/// Simplify a [`CasExpr`] tree, folding constants and applying the per-operator
/// algebraic rules (associative flattening, identity/zero elimination, and so
/// on) for `math/add`, `math/mul`, and `math/pow`.
pub fn simplify_expr(cx: &mut Cx, expr: CasExpr) -> Result<CasExpr> {
    match expr {
        CasExpr::Num(value) => CasExpr::num(cx, value),
        CasExpr::Var(symbol) => Ok(CasExpr::Var(symbol)),
        CasExpr::Op(operator, args) => {
            let args = args
                .into_iter()
                .map(|arg| simplify_expr(cx, arg))
                .collect::<Result<Vec<_>>>()?;
            match operator.to_string().as_str() {
                "math/add" => simplify_add(cx, operator, args),
                "math/mul" => simplify_mul(cx, operator, args),
                "math/pow" => simplify_pow(cx, operator, args),
                _ => simplify_generic(cx, operator, args),
            }
        }
    }
}

fn simplify_add(cx: &mut Cx, operator: Symbol, args: Vec<CasExpr>) -> Result<CasExpr> {
    let mut flat = Vec::new();
    for arg in args {
        match arg {
            CasExpr::Op(inner, nested) if inner == operator => flat.extend(nested),
            other => flat.push(other),
        }
    }

    let has_symbolic = flat.iter().any(|arg| !matches!(arg, CasExpr::Num(_)));
    let (foldable, mut others): (Vec<_>, Vec<_>) = flat
        .into_iter()
        .partition(|arg| foldable_numeric(cx, arg).unwrap_or(false));
    if has_symbolic && foldable.len() > 1 {
        let folded = fold_numeric_values(cx, &operator, foldable)?;
        others.push(CasExpr::num(cx, folded)?);
    } else {
        others.splice(0..0, foldable);
    }

    let mut reduced = Vec::new();
    for arg in others {
        if let CasExpr::Num(value) = &arg
            && is_literal_zero(cx, value)?
        {
            continue;
        }
        reduced.push(arg);
    }
    finalize_commutative(cx, operator, reduced)
}

fn simplify_mul(cx: &mut Cx, operator: Symbol, args: Vec<CasExpr>) -> Result<CasExpr> {
    let mut flat = Vec::new();
    for arg in args {
        match arg {
            CasExpr::Op(inner, nested) if inner == operator => flat.extend(nested),
            other => flat.push(other),
        }
    }

    for arg in &flat {
        if let CasExpr::Num(value) = arg
            && is_literal_zero(cx, value)?
        {
            return CasExpr::num(cx, value.clone());
        }
    }

    let has_symbolic = flat.iter().any(|arg| !matches!(arg, CasExpr::Num(_)));
    let (foldable, mut others): (Vec<_>, Vec<_>) = flat
        .into_iter()
        .partition(|arg| foldable_numeric(cx, arg).unwrap_or(false));
    if has_symbolic && foldable.len() > 1 {
        let folded = fold_numeric_values(cx, &operator, foldable)?;
        others.push(CasExpr::num(cx, folded)?);
    } else {
        others.splice(0..0, foldable);
    }

    let mut reduced = Vec::new();
    for arg in others {
        if let CasExpr::Num(value) = &arg
            && is_literal_one(cx, value)?
        {
            continue;
        }
        reduced.push(arg);
    }
    finalize_commutative(cx, operator, reduced)
}

fn simplify_pow(cx: &mut Cx, operator: Symbol, args: Vec<CasExpr>) -> Result<CasExpr> {
    let [base, exponent] = args.as_slice() else {
        return Ok(CasExpr::Op(operator, args));
    };
    if let CasExpr::Num(value) = exponent {
        if is_literal_zero(cx, value)? {
            let one = number_constant(cx, "1")?;
            return CasExpr::num(cx, one);
        }
        if is_literal_one(cx, value)? {
            return Ok(base.clone());
        }
    }
    if let (CasExpr::Num(base_value), CasExpr::Num(exponent_value)) = (base, exponent)
        && is_literal_zero(cx, base_value)?
        && is_positive_literal(cx, exponent_value)?
    {
        return CasExpr::num(cx, base_value.clone());
    }
    simplify_generic(cx, operator, args)
}

fn simplify_generic(cx: &mut Cx, operator: Symbol, args: Vec<CasExpr>) -> Result<CasExpr> {
    let all_foldable = args
        .iter()
        .all(|arg| foldable_numeric(cx, arg).unwrap_or(false));
    if all_foldable && !args.is_empty() {
        let folded = fold_numeric_values(cx, &operator, args)?;
        return CasExpr::num(cx, folded);
    }
    Ok(CasExpr::Op(operator, args))
}

fn finalize_commutative(cx: &mut Cx, operator: Symbol, mut args: Vec<CasExpr>) -> Result<CasExpr> {
    match args.len() {
        0 => {
            let value = number_constant(
                cx,
                if operator == Symbol::qualified("math", "add") {
                    "0"
                } else {
                    "1"
                },
            )?;
            CasExpr::num(cx, value)
        }
        1 => Ok(args.pop().unwrap()),
        _ => {
            // Lower every sort key up front so a fallible lowering propagates as
            // an `Err` instead of panicking inside the comparator.
            let mut keyed = args
                .into_iter()
                .map(|arg| Ok((cas_sort_key(cx, &arg)?, arg)))
                .collect::<Result<Vec<_>>>()?;
            keyed.sort_by(|(left, _), (right, _)| left.cmp(right));
            Ok(CasExpr::Op(
                operator,
                keyed.into_iter().map(|(_, arg)| arg).collect(),
            ))
        }
    }
}

fn foldable_numeric(cx: &mut Cx, expr: &CasExpr) -> Result<bool> {
    let CasExpr::Num(value) = expr else {
        return Ok(false);
    };
    Ok(literal_number(cx, value)?.is_some())
}

fn fold_numeric_values(cx: &mut Cx, operator: &Symbol, exprs: Vec<CasExpr>) -> Result<Value> {
    let mut values = exprs
        .into_iter()
        .map(|expr| match expr {
            CasExpr::Num(value) => Ok(value),
            _ => Err(Error::Eval(
                "numeric fold encountered a non-numeric CAS operand".to_owned(),
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut acc = values.remove(0);
    for value in values {
        acc = cx.apply_value_number_binary_op(operator, acc, value)?;
    }
    Ok(acc)
}

/// The `NumberLiteral` a CAS value carries, if it is a number value.
pub fn literal_number(cx: &mut Cx, value: &Value) -> Result<Option<NumberLiteral>> {
    Ok(cx
        .number_value_ref(value.clone())?
        .and_then(|number| number.literal))
}

fn is_literal_zero(cx: &mut Cx, value: &Value) -> Result<bool> {
    Ok(matches!(
        literal_number(cx, value)?
            .as_ref()
            .map(|number| number.canonical.as_str()),
        Some("0" | "0.0" | "0/1")
    ))
}

fn is_literal_one(cx: &mut Cx, value: &Value) -> Result<bool> {
    Ok(matches!(
        literal_number(cx, value)?
            .as_ref()
            .map(|number| number.canonical.as_str()),
        Some("1" | "1.0" | "1/1")
    ))
}

fn is_positive_literal(cx: &mut Cx, value: &Value) -> Result<bool> {
    let Some(number) = literal_number(cx, value)? else {
        return Ok(false);
    };
    if let Ok(integer) = number.canonical.parse::<i128>() {
        return Ok(integer > 0);
    }
    if let Ok(float) = number.canonical.parse::<f64>() {
        return Ok(float > 0.0);
    }
    Ok(false)
}

fn number_constant(cx: &mut Cx, canonical: &str) -> Result<Value> {
    let domain = if cx
        .registry()
        .number_domain_by_symbol(&domains::i64())
        .is_some()
    {
        domains::i64()
    } else if cx
        .registry()
        .number_domain_by_symbol(&domains::f64())
        .is_some()
    {
        domains::f64()
    } else {
        domains::i64()
    };
    cx.factory().number_literal(domain, canonical.to_owned())
}

fn cas_sort_key(cx: &mut Cx, expr: &CasExpr) -> Result<(u8, CanonicalKey)> {
    let rank = match expr {
        CasExpr::Num(_) => 0,
        CasExpr::Var(_) => 1,
        CasExpr::Op(_, _) => 2,
    };
    Ok((rank, cas_expr_to_surface_expr(cx, expr)?.canonical_key()))
}

fn parse_surface_list(cx: &mut Cx, items: &[Expr]) -> Result<Option<CasExpr>> {
    let Some((head, tail)) = items.split_first() else {
        return Ok(None);
    };
    let Expr::Symbol(operator) = head else {
        return Ok(None);
    };
    let Some(operator) = canonical_operator(operator) else {
        return Ok(None);
    };
    let args = tail
        .iter()
        .map(|expr| {
            expr_to_cas_expr(cx, expr)?.ok_or_else(|| {
                Error::Eval("CAS surface list contained a non-CAS argument".to_owned())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(CasExpr::Op(operator, args)))
}

fn normalize_operator(operator: &Symbol) -> Symbol {
    match operator.to_string().as_str() {
        "+" | "math/add" => Symbol::qualified("math", "add"),
        "-" | "math/sub" => Symbol::qualified("math", "sub"),
        "*" | "math/mul" => Symbol::qualified("math", "mul"),
        "/" | "math/div" => Symbol::qualified("math", "div"),
        "^" | "math/pow" => Symbol::qualified("math", "pow"),
        _ => operator.clone(),
    }
}

fn canonical_operator(symbol: &Symbol) -> Option<Symbol> {
    Some(match (symbol.namespace.as_deref(), symbol.name.as_ref()) {
        (Some("math"), "add" | "sub" | "mul" | "div" | "pow") => symbol.clone(),
        (None, "+") => Symbol::qualified("math", "add"),
        (None, "-") => Symbol::qualified("math", "sub"),
        (None, "*") => Symbol::qualified("math", "mul"),
        (None, "/") => Symbol::qualified("math", "div"),
        (None, "^") => Symbol::qualified("math", "pow"),
        _ => return None,
    })
}
