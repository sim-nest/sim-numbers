#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Fifth-order Radau IIA integration for stiff ODEs and index-1 DAEs.

mod solver;
pub use solver::*;

/// Cookbook recipes embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
