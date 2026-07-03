//! The `CasEvalLib` and the `eval-cas` callable: the library wiring that
//! registers CAS evaluation with the runtime.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, Cx, DefaultFactory, Dependency, Error, Export, Expr,
    Factory, Lib, LibManifest, LibTarget, Linker, Object, Result, Symbol, Value, Version,
};
use sim_lib_numbers_cas::value_to_cas_expr;
use sim_lib_numbers_core::domains;

use super::eval::{eval_cas, eval_cas_symbol};

/// The CAS evaluation library.
///
/// Loading this [`Lib`] registers the `eval-cas` function, which evaluates a
/// `numbers/cas` expression against an environment. It requires the
/// `numbers/cas` domain to be loaded first.
pub struct CasEvalLib;

impl CasEvalLib {
    /// Construct the CAS evaluation library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CasEvalLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for CasEvalLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: domains::cas_eval(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Function {
                symbol: eval_cas_symbol(),
                function_id: None,
            }],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.function_value(
            eval_cas_symbol(),
            DefaultFactory
                .opaque(Arc::new(EvalCasFunction))
                .expect("eval-cas function should be boxable"),
        )?;
        Ok(())
    }
}

#[derive(Clone)]
struct EvalCasFunction;

impl Object for EvalCasFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", eval_cas_symbol()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for EvalCasFunction {
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
        Ok(Expr::Symbol(eval_cas_symbol()))
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for EvalCasFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [expr] = values.as_slice() else {
            return Err(Error::Eval(format!(
                "{} expects exactly one argument",
                eval_cas_symbol()
            )));
        };
        let env = cx.env().clone();
        let expr = value_to_cas_expr(cx, expr.clone())?;
        eval_cas(cx, &expr, &env)
    }
}
