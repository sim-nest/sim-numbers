#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! Exotic number domains, currently the lazy continued-fraction domain
//! (`numbers/cf`): infinite-precision reals carried as continued-fraction
//! coefficient streams, their builtin constants, and the `as-f64` truncation
//! function.

mod implementation;

pub use implementation::{
    ContinuedFraction, ExoticNumbersLib, ExoticReal, as_f64_symbol, builtin_symbol, number_domain,
};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
