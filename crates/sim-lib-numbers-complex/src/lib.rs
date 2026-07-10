#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/complex` domain: complex literals and values, their arithmetic,
//! and the promotion edges from `i64`, `f64`, and `rational` into the complex
//! sink of the scalar lattice.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for the complex domain, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
