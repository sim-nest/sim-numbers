//! Algebraic simplification of `CasExpr` trees, and the conversions from
//! runtime values and surface `Expr` into the symbolic form the simplifier
//! operates on.

use sim_codec::lower_operator_nodes;
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
/// Numbers, symbols (and quoted symbols), operator nodes, symbol-headed calls,
/// and symbol-headed lists map to their algebraic forms; anything else yields
/// `None`.
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
    let lowered = lower_operator_nodes(expr.clone());
    parse_cas_expr_lowered(cx, &lowered)
}

fn parse_cas_expr_lowered(cx: &mut Cx, expr: &Expr) -> Result<Option<CasExpr>> {
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
            let Some(args) = parse_cas_args(cx, args)? else {
                return Ok(None);
            };
            Some(CasExpr::Op(normalize_operator(operator, args.len()), args))
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
///
/// The simplifier assumes finite algebraic operands for the additive identity,
/// multiplicative identity, and zero-product rules. Exponentiation keeps the
/// indeterminate literal form `0^0` unfolded instead of rewriting it to `1`.
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

    let (foldable, mut others) = partition_foldable_numeric(cx, flat)?;
    if !foldable.is_empty() && (others.is_empty() || foldable.len() > 1) {
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

    let (foldable, mut others) = partition_foldable_numeric(cx, flat)?;
    if !foldable.is_empty() && (others.is_empty() || foldable.len() > 1) {
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
            if let CasExpr::Num(base_value) = base
                && is_literal_zero(cx, base_value)?
            {
                return Ok(CasExpr::Op(operator, args));
            }
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
    let all_foldable = all_foldable_numeric(cx, &args)?;
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

fn partition_foldable_numeric(
    cx: &mut Cx,
    args: Vec<CasExpr>,
) -> Result<(Vec<CasExpr>, Vec<CasExpr>)> {
    let mut foldable = Vec::new();
    let mut others = Vec::new();
    for arg in args {
        if foldable_numeric(cx, &arg)? {
            foldable.push(arg);
        } else {
            others.push(arg);
        }
    }
    Ok((foldable, others))
}

fn all_foldable_numeric(cx: &mut Cx, args: &[CasExpr]) -> Result<bool> {
    for arg in args {
        if !foldable_numeric(cx, arg)? {
            return Ok(false);
        }
    }
    Ok(true)
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
    let Some(args) = parse_cas_args(cx, tail)? else {
        return Ok(None);
    };
    let operator = normalize_operator(operator, args.len());
    Ok(Some(CasExpr::Op(operator, args)))
}

fn parse_cas_args(cx: &mut Cx, exprs: &[Expr]) -> Result<Option<Vec<CasExpr>>> {
    let args = exprs
        .iter()
        .map(|expr| parse_cas_expr_lowered(cx, expr))
        .collect::<Result<Vec<_>>>()?;
    Ok(args.into_iter().collect::<Option<Vec<_>>>())
}

fn normalize_operator(operator: &Symbol, arity: usize) -> Symbol {
    match (operator.namespace.as_deref(), operator.name.as_ref(), arity) {
        (None, "+", _) | (Some("math"), "add", _) => Symbol::qualified("math", "add"),
        (None, "-", 1) | (Some("math"), "neg", _) => Symbol::qualified("math", "neg"),
        (None, "-", _) | (Some("math"), "sub", _) => Symbol::qualified("math", "sub"),
        (None, "*", _) | (Some("math"), "mul", _) => Symbol::qualified("math", "mul"),
        (None, "/", _) | (Some("math"), "div", _) => Symbol::qualified("math", "div"),
        (None, "%", _) | (Some("math"), "rem", _) => Symbol::qualified("math", "rem"),
        (None, "^", _) | (Some("math"), "pow", _) => Symbol::qualified("math", "pow"),
        _ => operator.clone(),
    }
}
