//! The `integrate-sym` callable: the function object that parses a value into a
//! `CasExpr`, integrates it, and returns the symbolic result.

use std::any::Any;

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, Object, Result, Symbol,
    Value,
};
use sim_lib_numbers_cas::{cas_expr_to_surface_expr, cas_expr_to_value, value_to_cas_expr};
use sim_lib_numbers_core::domains;

use super::integrate::{integrate_cas, integrate_sym_symbol};

#[derive(Clone)]
pub(crate) struct IntegrateSymFunction;

impl Object for IntegrateSymFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", integrate_sym_symbol()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for IntegrateSymFunction {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Function"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(integrate_sym_symbol()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for IntegrateSymFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [expr_value, var] = values.as_slice() else {
            return Err(Error::Eval(format!(
                "{} expects exactly two arguments",
                integrate_sym_symbol()
            )));
        };
        let var = extract_symbolish(cx, var)?.ok_or_else(|| {
            Error::Eval(format!(
                "{} expects a quoted symbol or symbol as its second argument",
                integrate_sym_symbol()
            ))
        })?;
        if let Some(number) = cx.number_value_ref(expr_value.clone())?
            && number.domain == domains::func()
        {
            return integrate_func_value(cx, expr_value.clone(), &var);
        }
        let expr = value_to_cas_expr(cx, expr_value.clone())?;
        let integral = integrate_cas(cx, &expr, &var)?;
        cas_expr_to_value(cx, integral)
    }
}

use sim_lib_numbers_cas::extract_symbolish;

fn integrate_func_value(cx: &mut Cx, value: Value, var: &Symbol) -> Result<Value> {
    let expr = value.object().as_expr(cx)?;
    let Expr::Call { operator, args } = expr else {
        return Err(Error::Eval(
            "function value does not expose a symbolic body".to_owned(),
        ));
    };
    let Expr::Symbol(operator) = operator.as_ref() else {
        return Err(Error::Eval(
            "function value does not expose a symbolic body".to_owned(),
        ));
    };
    if *operator != Symbol::new("fn") {
        return Err(Error::Eval(
            "function value does not expose a symbolic body".to_owned(),
        ));
    }
    let [vars_expr, body_expr] = args.as_slice() else {
        return Err(Error::Eval(
            "function value had an invalid fn surface".to_owned(),
        ));
    };
    let body = value_to_cas_expr(cx, cx.factory().expr(body_expr.clone())?)?;
    let integral = integrate_cas(cx, &body, var)?;
    let integral_expr = cas_expr_to_surface_expr(cx, &integral)?;
    cx.eval_expr(Expr::Call {
        operator: Box::new(Expr::Symbol(Symbol::new("fn"))),
        args: vec![vars_expr.clone(), integral_expr],
    })
}
