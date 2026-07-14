//! Expr-only helpers for symbolic function surfaces.

use sim_kernel::{Cx, Error, Expr, Result, Symbol, Value};
use sim_lib_numbers_cas::{CasExpr, expr_to_cas_expr};

const NO_SYMBOLIC_BODY: &str = "NotDifferentiable: function value does not expose a symbolic body";

pub(crate) fn func_surface_body(cx: &mut Cx, value: &Value) -> Result<(Expr, CasExpr)> {
    let expr = value.object().as_expr(cx)?;
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
