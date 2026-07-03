//! The `as-f64` callable: the function object that truncates a
//! continued-fraction value to an `f64` under a work bound, plus the builtin
//! continued-fraction constructor.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, NumberLiteral, Object,
    Result, Symbol, Value,
};
use sim_lib_numbers_core::domains;

use super::{
    domain::builtin_symbol,
    value::{ContinuedFraction, ExoticReal},
};

/// The qualified symbol of the `as-f64` function this domain installs, which
/// truncates a continued-fraction value to an `f64` under a work bound.
pub fn as_f64_symbol() -> Symbol {
    domains::domain("as-f64")
}

pub fn plain_as_f64_symbol() -> Symbol {
    Symbol::new("as-f64")
}

#[derive(Clone)]
pub(crate) struct AsF64Function {
    pub(crate) symbol: Symbol,
}

impl Object for AsF64Function {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for AsF64Function {
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
        Ok(Expr::Symbol(self.symbol.clone()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for AsF64Function {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [value, max_work] = values.as_slice() else {
            return Err(Error::Eval(format!(
                "{} expects exactly two arguments",
                self.symbol
            )));
        };
        let Some(cf) = value.object().downcast_ref::<ContinuedFraction>() else {
            return Err(Error::Eval(format!(
                "as-f64 expected a continued fraction number value, found {}",
                value.object().display(cx)?
            )));
        };
        let max_work = require_usize(cx, max_work.clone(), "as-f64 max_work")?;
        let (approx, _) = cf.as_f64(max_work);
        cx.factory()
            .number_literal(domains::f64(), canonical_f64(approx))
    }
}

pub(crate) fn builtin_cf(
    name: &'static str,
    head: Vec<i128>,
    tail: Option<super::value::CfTail>,
) -> Arc<ContinuedFraction> {
    Arc::new(ContinuedFraction::builtin(
        builtin_symbol(name),
        name,
        head,
        tail,
    ))
}

fn canonical_f64(value: f64) -> String {
    let rendered = value.to_string();
    if rendered == "-0" {
        "0".to_owned()
    } else {
        rendered
    }
}

fn require_usize(cx: &mut Cx, value: Value, context: &str) -> Result<usize> {
    let Some(number) = cx.number_value_ref(value.clone())? else {
        return Err(Error::Eval(format!(
            "{context} expected a non-negative integer, found {}",
            value.object().display(cx)?
        )));
    };
    let Some(NumberLiteral { canonical, .. }) = number.literal else {
        return Err(Error::Eval(format!(
            "{context} must be a literal non-negative integer"
        )));
    };
    canonical
        .parse::<usize>()
        .map_err(|_| Error::Eval(format!("{context} must be a literal non-negative integer")))
}
