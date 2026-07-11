#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! Evaluation of `numbers/cas` symbolic expressions: the `eval-cas` function
//! that walks a `CasExpr` against an environment, in either numeric or
//! symbolic mode, and the surface `Expr`/`CasExpr` bridge it uses.
//!
//! # Examples
//!
//! Evaluate an expression symbolically against an empty environment: an unbound
//! variable survives as a symbolic CAS value rather than erroring, and lowers
//! back to its surface form:
//!
//! ```
//! use std::sync::Arc;
//! use sim_kernel::{Cx, DefaultFactory, Env, Expr, NoopEvalPolicy, Symbol};
//! use sim_lib_numbers_cas::CasNumbersLib;
//! use sim_lib_numbers_cas_eval::{eval_cas_symbolic, expr_to_cas};
//!
//! let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
//! cx.load_lib(&CasNumbersLib::new()).unwrap();
//!
//! let tree = expr_to_cas(&mut cx, &Expr::Symbol(Symbol::new("x"))).unwrap();
//! let env = Env::default();
//! let value = eval_cas_symbolic(&mut cx, &tree, &env).unwrap();
//! assert_eq!(value.object().as_expr(&mut cx).unwrap(), Expr::Symbol(Symbol::new("x")));
//! ```

mod implementation;

pub use implementation::{
    CasEvalLib, cas_to_expr, eval_cas, eval_cas_symbol, eval_cas_symbolic, expr_to_cas,
};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
