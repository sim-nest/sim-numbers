#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/i64` domain: 64-bit signed-integer literals and values, their
//! scalar arithmetic, and promotion edges into `f64` and `rational`.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for this domain, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
