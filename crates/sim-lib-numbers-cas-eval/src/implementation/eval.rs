//! Evaluation of `CasExpr` trees against an environment, in numeric or
//! symbolic mode, plus the surface `Expr`/`CasExpr` conversions it relies on.

use sim_kernel::{Args, Cx, Env, Error, Expr, Result, Symbol, Value};
use sim_lib_numbers_cas::{
    CasExpr, cas_expr_to_surface_expr, cas_expr_to_value, expr_to_cas_expr, simplify_expr,
    value_to_cas_expr,
};
use sim_lib_numbers_core::domains;

/// The `eval-cas` symbol: the CAS evaluation entry point.
pub fn eval_cas_symbol() -> Symbol {
    Symbol::new("eval-cas")
}

/// Lower a [`CasExpr`] to a surface [`Expr`].
pub fn cas_to_expr(cx: &mut Cx, expr: &CasExpr) -> Result<Expr> {
    cas_expr_to_surface_expr(cx, expr)
}

/// Parse a surface [`Expr`] into a [`CasExpr`], erroring if it is not
/// CAS-shaped.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, Expr, NoopEvalPolicy, Symbol};
/// use sim_lib_numbers_cas_eval::{cas_to_expr, expr_to_cas};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// let surface = Expr::Symbol(Symbol::new("y"));
/// let tree = expr_to_cas(&mut cx, &surface).unwrap();
/// assert_eq!(cas_to_expr(&mut cx, &tree).unwrap(), surface);
/// ```
pub fn expr_to_cas(cx: &mut Cx, expr: &Expr) -> Result<CasExpr> {
    expr_to_cas_expr(cx, expr)?
        .ok_or_else(|| Error::Eval(format!("cannot parse {:?} as a CAS expression", expr)))
}

/// Numerically evaluate a [`CasExpr`] against `env`.
///
/// Variables must be bound in `env`; an unbound variable is an error. Operators
/// dispatch through the runtime's registered number functions.
pub fn eval_cas(cx: &mut Cx, expr: &CasExpr, env: &Env) -> Result<Value> {
    eval_cas_with_mode(cx, expr, env, false)
}

/// Symbolically evaluate a [`CasExpr`] against `env`.
///
/// Like [`eval_cas`], but variables left unbound in `env` survive as symbolic
/// CAS values and operators over them fold through the CAS simplifier rather
/// than erroring.
pub fn eval_cas_symbolic(cx: &mut Cx, expr: &CasExpr, env: &Env) -> Result<Value> {
    eval_cas_with_mode(cx, expr, env, true)
}

fn eval_cas_with_mode(cx: &mut Cx, expr: &CasExpr, env: &Env, symbolic: bool) -> Result<Value> {
    match expr {
        CasExpr::Num(value) => Ok(value.clone()),
        CasExpr::Var(symbol) => match env.get(symbol) {
            Some(value) => Ok(value),
            None if symbolic => cas_expr_to_value(cx, CasExpr::Var(symbol.clone())),
            None => Err(Error::Eval(format!("unbound CAS variable {}", symbol))),
        },
        CasExpr::Op(operator, args) => {
            let values = args
                .iter()
                .map(|arg| eval_cas_with_mode(cx, arg, env, symbolic))
                .collect::<Result<Vec<_>>>()?;
            if symbolic && contains_symbolic_value(cx, &values)? {
                return apply_symbolic(cx, operator.clone(), values);
            }
            cx.call_function(operator, Args::new(values))
        }
    }
}

fn is_symbolic_value(cx: &mut Cx, value: &Value) -> Result<bool> {
    Ok(matches!(
        cx.number_value_ref(value.clone())?,
        Some(number) if number.domain == domains::cas()
    ))
}

fn contains_symbolic_value(cx: &mut Cx, values: &[Value]) -> Result<bool> {
    for value in values {
        if is_symbolic_value(cx, value)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn apply_symbolic(cx: &mut Cx, operator: Symbol, values: Vec<Value>) -> Result<Value> {
    let mut args = Vec::with_capacity(values.len());
    for value in values {
        args.push(value_to_cas_expr(cx, value)?);
    }
    let simplified = simplify_expr(cx, CasExpr::Op(operator, args))?;
    cas_expr_to_value(cx, simplified)
}
