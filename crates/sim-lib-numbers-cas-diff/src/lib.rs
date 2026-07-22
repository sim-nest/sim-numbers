#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! Symbolic differentiation and integration over the `numbers/cas` expression
//! tree: the `diff` and `integrate-sym` functions plus an extensible registry
//! of per-operator differentiation rules.
//!
//! # Examples
//!
//! Teach the differentiator a new operator through the extensible rule
//! registry, then look the rule up. Rules map an operator's arguments and the
//! variable of differentiation to a derivative
//! [`CasExpr`](sim_lib_numbers_cas::CasExpr):
//!
//! ```
//! use sim_kernel::Symbol;
//! use sim_lib_numbers_cas::CasExpr;
//! use sim_lib_numbers_cas_diff::{global_diff_registry, register_diff_rule};
//!
//! // d/dx ln(x) = 1/x, expressed structurally as (/ 1 x).
//! register_diff_rule(
//!     Symbol::new("ln-doctest"),
//!     Box::new(|args: &[CasExpr], _var: &Symbol| {
//!         let [arg] = args else { return None };
//!         Some(CasExpr::Op(
//!             Symbol::qualified("math", "div"),
//!             vec![CasExpr::Var(Symbol::new("1")), arg.clone()],
//!         ))
//!     }),
//! )
//! .unwrap();
//!
//! let registry = global_diff_registry().read().unwrap();
//! let derivative = registry.apply(
//!     &Symbol::new("ln-doctest"),
//!     &[CasExpr::Var(Symbol::new("x"))],
//!     &Symbol::new("x"),
//! );
//! assert!(matches!(derivative, Some(CasExpr::Op(_, _))));
//! ```

mod implementation;

pub use implementation::{
    CasDiffLib, CasDiffRegistry, DiffRule, diff_cas, diff_symbol, global_diff_registry,
    integrate_cas, integrate_sym_symbol, override_diff_rule, register_diff_rule,
};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
