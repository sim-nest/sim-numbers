#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The fixed-width integer domains (`numbers/i8` .. `numbers/i128`,
//! `numbers/u8` .. `numbers/u128`): their literals, values, and the widening
//! promotion edges through the signed and unsigned integer lattice.

mod implementation;
mod literal;

pub use implementation::FixedNumbersLib;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
