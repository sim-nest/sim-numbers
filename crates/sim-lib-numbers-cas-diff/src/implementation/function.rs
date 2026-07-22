//! The `CasDiffLib` and the `diff` callable: the library wiring that registers
//! symbolic differentiation and the `integrate-sym` function with the runtime.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, Cx, DefaultFactory, Dependency, Error, Export, Expr,
    Factory, Lib, LibManifest, LibTarget, Linker, Object, Result, Symbol, Value, Version,
};
use sim_lib_numbers_cas::{
    cas_expr_to_surface_expr, cas_expr_to_value, extract_symbolish, value_to_cas_expr,
};
use sim_lib_numbers_core::domains;

use super::diff::{diff_cas, diff_symbol};
use super::func_surface::func_surface_body;
use super::integrate::integrate_sym_symbol;
use super::integrate_function::IntegrateSymFunction;

/// The CAS differentiation and integration library.
///
/// Loading this [`Lib`] registers the `diff` and `integrate-sym` functions over
/// `numbers/cas` expressions. It requires the `numbers/cas` domain to be loaded
/// first.
pub struct CasDiffLib;

impl CasDiffLib {
    /// Construct the CAS differentiation library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CasDiffLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for CasDiffLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: domains::cas_diff(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![
                Export::Function {
                    symbol: diff_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: integrate_sym_symbol(),
                    function_id: None,
                },
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.function_value(
            diff_symbol(),
            DefaultFactory
                .opaque(Arc::new(DiffFunction))
                .expect("diff function should be boxable"),
        )?;
        linker.function_value(
            integrate_sym_symbol(),
            DefaultFactory
                .opaque(Arc::new(IntegrateSymFunction))
                .expect("integrate-sym function should be boxable"),
        )?;
        Ok(())
    }
}

#[derive(Clone)]
struct DiffFunction;

impl Object for DiffFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", diff_symbol()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for DiffFunction {
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
        Ok(Expr::Symbol(diff_symbol()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for DiffFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [expr_value, var] = values.as_slice() else {
            return Err(Error::Eval(format!(
                "{} expects exactly two arguments",
                diff_symbol()
            )));
        };
        let var = extract_symbolish(cx, var)?.ok_or_else(|| {
            Error::Eval(format!(
                "{} expects a quoted symbol or symbol as its second argument",
                diff_symbol()
            ))
        })?;
        if let Some(number) = cx.number_value_ref(expr_value.clone())?
            && number.domain == domains::func()
        {
            return diff_func_value(cx, expr_value.clone(), &var);
        }
        let expr = value_to_cas_expr(cx, expr_value.clone())?;
        let derivative = diff_cas(cx, &expr, &var)?;
        cas_expr_to_value(cx, derivative)
    }
}

fn diff_func_value(cx: &mut Cx, value: Value, var: &Symbol) -> Result<Value> {
    let (vars_expr, body) = func_surface_body(cx, &value)?;
    let derivative = diff_cas(cx, &body, var)?;
    let derivative_expr = cas_expr_to_surface_expr(cx, &derivative)?;
    cx.eval_expr(Expr::Call {
        operator: Box::new(Expr::Symbol(Symbol::new("fn"))),
        args: vec![vars_expr, derivative_expr],
    })
}
