#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/cas` domain: the core computer-algebra layer. Defines the
//! symbolic expression tree ([`CasExpr`]), its value citizen, the conversions
//! to and from surface `Expr`/`Value`, and the `cas/var` and `cas/simplify`
//! functions other CAS crates build on.
//!
//! # Examples
//!
//! Build a symbolic tree directly and simplify it. Adding the variable `x` to a
//! literal `0` folds the zero term away, and the simplified result lowers back
//! to a surface `Expr`:
//!
//! ```
//! use std::sync::Arc;
//! use sim_kernel::{Cx, DefaultFactory, Expr, Factory, NoopEvalPolicy, Symbol};
//! use sim_lib_numbers_cas::{
//!     cas_expr_to_surface_expr, simplify_expr, CasExpr, CasNumbersLib,
//! };
//!
//! let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
//! cx.load_lib(&CasNumbersLib::new()).unwrap();
//!
//! let zero = cx
//!     .factory()
//!     .number_literal(Symbol::qualified("numbers", "i64"), "0".to_owned())
//!     .unwrap();
//! let zero_leaf = CasExpr::num(&mut cx, zero).unwrap();
//! let tree = CasExpr::Op(
//!     Symbol::qualified("math", "add"),
//!     vec![CasExpr::Var(Symbol::new("x")), zero_leaf],
//! );
//! let simplified = simplify_expr(&mut cx, tree).unwrap();
//! let surface = cas_expr_to_surface_expr(&mut cx, &simplified).unwrap();
//! assert_eq!(surface, Expr::Symbol(Symbol::new("x")));
//! ```

mod implementation;

pub use implementation::{
    CasExpr, CasNumbersLib, canonical_eq, cas_domain_symbol, cas_expr_to_surface_expr,
    cas_expr_to_value, cas_simplify_symbol, cas_value_class_symbol, cas_var_symbol,
    expr_to_cas_expr, extract_symbolish, free_vars, literal_number, simplify_expr,
    value_to_cas_expr,
};

#[cfg(test)]
mod tests;

/// Cookbook recipes for this domain, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
