//! Loadable callables for Burg estimation and DFT interpolation.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, Expr, Factory, Linker, Object, RawArgs, Result,
    Symbol, Value,
};

use crate::runtime_spectral::{execute_exprs, execute_values};

/// Symbol of the Lisp-facing Burg operation (`signal/burg`).
pub fn signal_burg_symbol() -> Symbol {
    Symbol::qualified("signal", "burg")
}

/// Symbol of the Lisp-facing DFT interpolation operation (`signal/dft-interpolate`).
pub fn signal_dft_interpolate_symbol() -> Symbol {
    Symbol::qualified("signal", "dft-interpolate")
}

#[derive(Clone, Copy)]
pub(crate) enum SpectralOperation {
    Burg,
    DftInterpolate,
}

impl SpectralOperation {
    pub(crate) fn symbol(self) -> Symbol {
        match self {
            Self::Burg => signal_burg_symbol(),
            Self::DftInterpolate => signal_dft_interpolate_symbol(),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Burg => "signal/burg",
            Self::DftInterpolate => "signal/dft-interpolate",
        }
    }
}

#[derive(Clone)]
struct SpectralFunction {
    operation: SpectralOperation,
}

impl Object for SpectralFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", self.operation.symbol()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for SpectralFunction {
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
        Ok(Expr::Symbol(self.operation.symbol()))
    }

    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for SpectralFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        execute_values(cx, self.operation, args.into_vec())
    }

    fn call_exprs(&self, cx: &mut Cx, args: RawArgs) -> Result<Value> {
        execute_exprs(cx, self.operation, args.into_exprs())
    }
}

pub(crate) fn spectral_symbols() -> [Symbol; 2] {
    [signal_burg_symbol(), signal_dft_interpolate_symbol()]
}

pub(crate) fn load_spectral_operations(linker: &mut Linker<'_>) -> Result<()> {
    for operation in [SpectralOperation::Burg, SpectralOperation::DftInterpolate] {
        linker.function_value(
            operation.symbol(),
            DefaultFactory.opaque(Arc::new(SpectralFunction { operation }))?,
        )?;
    }
    Ok(())
}

/// Calls `signal/burg` with evaluated samples and keyword pairs.
pub fn call_signal_burg(cx: &mut Cx, args: Args) -> Result<Value> {
    execute_values(cx, SpectralOperation::Burg, args.into_vec())
}

/// Calls `signal/dft-interpolate` with evaluated bins and keyword pairs.
pub fn call_signal_dft_interpolate(cx: &mut Cx, args: Args) -> Result<Value> {
    execute_values(cx, SpectralOperation::DftInterpolate, args.into_vec())
}

pub(crate) fn argument_error(operation: SpectralOperation) -> sim_kernel::Error {
    sim_kernel::Error::Eval(format!(
        "{} expects one input followed by keyword pairs",
        operation.name()
    ))
}
