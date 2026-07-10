#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/bigint` domain: arbitrary-precision integer literals and
//! values, their exact arithmetic, and promotion into `rational`.

mod implementation;
mod literal;
mod ops;

pub use implementation::{BigIntNumberDomain, BigIntNumbersLib, number_domain};

/// Cookbook recipes for this domain, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
