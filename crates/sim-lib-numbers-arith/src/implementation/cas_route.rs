//! Routing of arithmetic to the CAS: deciding when symbolic operands require
//! `cas/simplify`, and coercing symbol-like arguments into `cas/var` values
//! when the CAS library is loaded.

use sim_kernel::{Args, Cx, Expr, Result, Symbol, Value};
use sim_lib_numbers_core::domains;

pub(super) fn coerce_arith_argument(cx: &mut Cx, value: Value) -> Result<Value> {
    if cx.number_value_ref(value.clone())?.is_some() {
        return Ok(value);
    }
    let cas_var = Symbol::qualified("cas", "var");
    if cx.registry().function_by_symbol(&cas_var).is_some() && matches_symbolish(cx, &value)? {
        return cx.call_function(&cas_var, Args::new(vec![value]));
    }
    Ok(value)
}

pub(super) fn should_route_via_cas(cx: &mut Cx, values: &[Value]) -> Result<bool> {
    if cx
        .registry()
        .function_by_symbol(&Symbol::qualified("cas", "simplify"))
        .is_none()
    {
        return Ok(false);
    }
    for value in values {
        let Some(number) = cx.number_value_ref(value.clone())? else {
            continue;
        };
        if number.domain == domains::cas() || number.domain == domains::continued_fraction() {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn route_via_cas(
    cx: &mut Cx,
    operator: Symbol,
    values: &[Value],
) -> Result<Option<Value>> {
    let mut iter = values.iter();
    let Some(first) = iter.next() else {
        return Ok(None);
    };
    let Some(second) = iter.next() else {
        return Ok(None);
    };
    let mut acc = cas_binary_value(cx, operator.clone(), first, second)?;
    for value in iter {
        acc = cas_binary_value(cx, operator.clone(), &acc, value)?;
    }
    Ok(Some(acc))
}

fn matches_symbolish(cx: &mut Cx, value: &Value) -> Result<bool> {
    Ok(match value.object().as_expr(cx)? {
        Expr::Symbol(_) => true,
        Expr::Quote {
            mode: sim_kernel::QuoteMode::Quote,
            expr,
        } => matches!(*expr, Expr::Symbol(_)),
        _ => false,
    })
}

fn cas_binary_value(cx: &mut Cx, operator: Symbol, left: &Value, right: &Value) -> Result<Value> {
    let expr = Expr::List(vec![
        Expr::Symbol(operator),
        left.object().as_expr(cx)?,
        right.object().as_expr(cx)?,
    ]);
    cx.call_function(
        &Symbol::qualified("cas", "simplify"),
        Args::new(vec![cx.factory().expr(expr)?]),
    )
}
