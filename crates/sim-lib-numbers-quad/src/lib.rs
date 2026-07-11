#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Quadrature and finite-difference backends for the numeric domain: fixed and
//! adaptive integration rules plus finite-difference differentiators, packaged
//! as a registered numeric plugin library.

mod implementation;

pub use implementation::QuadNumbersLib;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
