//! Loadable callables for the real signal-operation runtime.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, Linker, Object, RawArgs,
    Result, Symbol, Value,
};

use crate::runtime_convolution::{execute_exprs, execute_values};

/// Symbol of the Lisp-facing convolution operation (`signal/convolve`).
pub fn signal_convolve_symbol() -> Symbol {
    Symbol::qualified("signal", "convolve")
}

/// Symbol of the Lisp-facing cross-correlation operation (`signal/correlate`).
pub fn signal_correlate_symbol() -> Symbol {
    Symbol::qualified("signal", "correlate")
}

/// Symbol of the Lisp-facing guarded deconvolution operation (`signal/deconvolve`).
pub fn signal_deconvolve_symbol() -> Symbol {
    Symbol::qualified("signal", "deconvolve")
}

#[derive(Clone, Copy)]
pub(crate) enum Operation {
    Convolve,
    Correlate,
    Deconvolve,
}

impl Operation {
    fn symbol(self) -> Symbol {
        match self {
            Self::Convolve => signal_convolve_symbol(),
            Self::Correlate => signal_correlate_symbol(),
            Self::Deconvolve => signal_deconvolve_symbol(),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Convolve => "signal/convolve",
            Self::Correlate => "signal/correlate",
            Self::Deconvolve => "signal/deconvolve",
        }
    }
}

#[derive(Clone)]
struct SignalOperationFunction {
    operation: Operation,
}

impl Object for SignalOperationFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", self.operation.symbol()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for SignalOperationFunction {
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

impl Callable for SignalOperationFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        execute_values(cx, self.operation, args.into_vec())
    }

    fn call_exprs(&self, cx: &mut Cx, args: RawArgs) -> Result<Value> {
        execute_exprs(cx, self.operation, args.into_exprs())
    }
}

pub(crate) fn operation_symbols() -> [Symbol; 3] {
    [
        signal_convolve_symbol(),
        signal_correlate_symbol(),
        signal_deconvolve_symbol(),
    ]
}

pub(crate) fn load_operations(linker: &mut Linker<'_>) -> Result<()> {
    for operation in [
        Operation::Convolve,
        Operation::Correlate,
        Operation::Deconvolve,
    ] {
        linker.function_value(
            operation.symbol(),
            DefaultFactory.opaque(Arc::new(SignalOperationFunction { operation }))?,
        )?;
    }
    Ok(())
}

/// Calls `signal/convolve` with evaluated signal, kernel, and keyword pairs.
pub fn call_signal_convolve(cx: &mut Cx, args: Args) -> Result<Value> {
    execute_values(cx, Operation::Convolve, args.into_vec())
}

/// Calls `signal/correlate` with evaluated signal, kernel, and keyword pairs.
pub fn call_signal_correlate(cx: &mut Cx, args: Args) -> Result<Value> {
    execute_values(cx, Operation::Correlate, args.into_vec())
}

/// Calls `signal/deconvolve` with evaluated observation, kernel, and keyword pairs.
pub fn call_signal_deconvolve(cx: &mut Cx, args: Args) -> Result<Value> {
    execute_values(cx, Operation::Deconvolve, args.into_vec())
}

pub(crate) fn argument_error(operation: Operation) -> Error {
    Error::Eval(format!(
        "{} expects signal, kernel, and optional keyword pairs",
        operation.name()
    ))
}
