#![forbid(unsafe_code)]

//! Implementation of the `Func` number domain, wiring its domain registration,
//! function operations, and function value type together.

mod domain;
mod function;
mod value;

pub use domain::{FuncNumbersLib, func_class_symbol, func_domain_symbol};
pub use function::{call_symbol, fn_symbol, grad_symbol};
pub use value::{Func, FuncMetadata, NativeFn, SymbolicStatus};
