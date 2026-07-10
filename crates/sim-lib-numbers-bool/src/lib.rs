#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/bool` domain: boolean literals and values as the base of the
//! number-promotion lattice, with edges widening into the integer and float
//! domains.

mod implementation;
mod literal;

pub use implementation::{BoolNumberDomain, BoolNumbersLib, number_domain};

/// Cookbook recipes for this domain, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
