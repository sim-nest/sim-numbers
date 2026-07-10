#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! Function number domain: callable function values built over CAS or native
//! bodies, with `fn`, `call`, and `grad` operations for the `Func` domain.

mod implementation;

pub use implementation::{
    Func, FuncMetadata, FuncNumbersLib, NativeFn, call_symbol, fn_symbol, func_class_symbol,
    func_domain_symbol, grad_symbol,
};

#[cfg(test)]
mod tests;

/// Cookbook recipes for this domain, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
