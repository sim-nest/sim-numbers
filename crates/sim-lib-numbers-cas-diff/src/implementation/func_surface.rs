//! Expr-only helpers for symbolic function surfaces.

use sim_kernel::{Cx, Error, Expr, Result, Symbol, Value};
use sim_lib_numbers_cas::{CasExpr, expr_to_cas_expr};

const NO_SYMBOLIC_BODY: &str = "NotDifferentiable: function value does not expose a symbolic body";

pub(crate) fn func_surface_body(cx: &mut Cx, value: &Value) -> Result<(Expr, CasExpr)> {
    let expr = value.object().as_expr(cx)?;
    if let Expr::Extension { tag, payload } = &expr
        && tag == &Symbol::qualified("numbers", "Func")
    {
        return Err(Error::Eval(native_func_message(payload)));
    }
    let Expr::Call { operator, args } = expr else {
        return Err(Error::Eval(NO_SYMBOLIC_BODY.to_owned()));
    };
    let Expr::Symbol(operator) = operator.as_ref() else {
        return Err(Error::Eval(NO_SYMBOLIC_BODY.to_owned()));
    };
    if operator != &Symbol::new("fn") {
        return Err(Error::Eval(NO_SYMBOLIC_BODY.to_owned()));
    }
    let [vars_expr, body_expr] = args.as_slice() else {
        return Err(Error::Eval(
            "NotDifferentiable: function value had an invalid fn surface".to_owned(),
        ));
    };
    let body = expr_to_cas_expr(cx, body_expr)?.ok_or_else(|| {
        Error::Eval("NotDifferentiable: function body is not CAS-compatible".to_owned())
    })?;
    Ok((vars_expr.clone(), body))
}

fn native_func_message(payload: &Expr) -> String {
    if let Some(reason) = map_symbol(payload, "symbolic-loss-reason") {
        return format!("{NO_SYMBOLIC_BODY}: {reason}");
    }
    if let Some(status) = map_symbol(payload, "symbolic-status") {
        return format!("{NO_SYMBOLIC_BODY}: {status}");
    }
    NO_SYMBOLIC_BODY.to_owned()
}

fn map_symbol<'a>(expr: &'a Expr, key: &str) -> Option<&'a Symbol> {
    let Expr::Map(fields) = expr else {
        return None;
    };
    fields.iter().find_map(|(candidate, value)| {
        if candidate == &Expr::Symbol(Symbol::new(key))
            && let Expr::Symbol(symbol) = value
        {
            return Some(symbol);
        }
        None
    })
}
