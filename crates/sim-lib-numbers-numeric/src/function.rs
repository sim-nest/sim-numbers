//! The numeric domain library and its operation symbols (`numeric-diff`,
//! `integrate`, `integrate-adapt`, `ode-solve`, `numeric/compose`, and
//! `numeric/run-composed`) registered into the runtime.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, Cx, DefaultFactory, Dependency, Export, Expr, Factory,
    Lib, LibManifest, LibTarget, Linker, Object, RawArgs, Result, Symbol, Value, Version,
};
use sim_lib_numbers_core::domains;

use super::runtime;

/// Returns the symbol bound to the `numeric-diff` operation.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_numeric::{
///     integrate_adapt_symbol, integrate_symbol, numeric_diff_symbol, ode_solve_symbol,
/// };
///
/// assert_eq!(numeric_diff_symbol().to_string(), "numeric-diff");
/// assert_eq!(integrate_symbol().to_string(), "integrate");
/// assert_eq!(integrate_adapt_symbol().to_string(), "integrate-adapt");
/// assert_eq!(ode_solve_symbol().to_string(), "ode-solve");
/// ```
pub fn numeric_diff_symbol() -> Symbol {
    Symbol::new("numeric-diff")
}

/// Returns the symbol bound to the fixed-rule `integrate` operation.
pub fn integrate_symbol() -> Symbol {
    Symbol::new("integrate")
}

/// Returns the symbol bound to the adaptive `integrate-adapt` operation.
pub fn integrate_adapt_symbol() -> Symbol {
    Symbol::new("integrate-adapt")
}

/// Returns the symbol bound to the `ode-solve` operation.
pub fn ode_solve_symbol() -> Symbol {
    Symbol::new("ode-solve")
}

/// Returns the symbol bound to the `numeric/compose` operation.
pub fn numeric_compose_symbol() -> Symbol {
    Symbol::qualified("numeric", "compose")
}

/// Returns the symbol bound to the `numeric/run-composed` operation.
pub fn numeric_run_composed_symbol() -> Symbol {
    Symbol::qualified("numeric", "run-composed")
}

fn function_symbols() -> [Symbol; 6] {
    [
        numeric_diff_symbol(),
        integrate_symbol(),
        integrate_adapt_symbol(),
        ode_solve_symbol(),
        numeric_compose_symbol(),
        numeric_run_composed_symbol(),
    ]
}

#[derive(Clone)]
struct NumericFunction {
    symbol: Symbol,
}

impl Object for NumericFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for NumericFunction {
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

impl Callable for NumericFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        if self.symbol == numeric_diff_symbol() {
            runtime::call_numeric_diff(cx, args)
        } else if self.symbol == integrate_symbol() {
            runtime::call_integrate(cx, args)
        } else if self.symbol == integrate_adapt_symbol() {
            runtime::call_integrate_adapt(cx, args)
        } else if self.symbol == numeric_compose_symbol() {
            runtime::call_numeric_compose(cx, args)
        } else if self.symbol == numeric_run_composed_symbol() {
            runtime::call_numeric_run_composed(cx, args)
        } else {
            runtime::call_ode_solve(cx, args)
        }
    }

    fn call_exprs(&self, cx: &mut Cx, args: RawArgs) -> Result<Value> {
        let args = args.into_exprs();
        if self.symbol == numeric_diff_symbol() {
            runtime::call_numeric_diff_exprs(cx, args)
        } else if self.symbol == integrate_symbol() {
            runtime::call_integrate_exprs(cx, args)
        } else if self.symbol == integrate_adapt_symbol() {
            runtime::call_integrate_adapt_exprs(cx, args)
        } else if self.symbol == numeric_compose_symbol() {
            runtime::call_numeric_compose_exprs(cx, args)
        } else if self.symbol == numeric_run_composed_symbol() {
            runtime::call_numeric_run_composed_exprs(cx, args)
        } else {
            runtime::call_ode_solve_exprs(cx, args)
        }
    }
}

/// Library that installs the numeric evaluation surface: the `numeric-diff`,
/// `integrate`, `integrate-adapt`, `ode-solve`, `numeric/compose`, and
/// `numeric/run-composed` callables that dispatch to the registered numeric
/// backends.
pub struct NumericNumbersLib;

impl NumericNumbersLib {
    /// Creates a new `NumericNumbersLib` ready to be loaded into a runtime.
    pub fn new() -> Self {
        Self
    }
}

impl Default for NumericNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for NumericNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: domains::numeric(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: function_symbols()
                .into_iter()
                .map(|symbol| Export::Function {
                    symbol,
                    function_id: None,
                })
                .collect(),
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        for symbol in function_symbols() {
            linker.function_value(
                symbol.clone(),
                DefaultFactory.opaque(Arc::new(NumericFunction { symbol }))?,
            )?;
        }
        Ok(())
    }
}
