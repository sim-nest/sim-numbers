#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/f32` domain: single-precision floating-point literals and
//! values, their scalar arithmetic, and promotion into `f64`.

mod implementation;
mod literal;
mod ops;

pub use implementation::{F32NumberDomain, F32NumbersLib, number_domain};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
