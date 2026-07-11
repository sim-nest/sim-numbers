#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Automatic differentiation primitives: forward-mode dual numbers, a
//! reverse-mode evaluation tape, and the `Scalarish` numeric trait they share.

mod implementation;

pub use implementation::{Dual, Scalarish, Tape, TapeNode, Var};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
