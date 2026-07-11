#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! Runge-Kutta ODE integrators for the numeric domain: fixed-step and adaptive
//! solver backends registered as numeric `ode-solve` plugins.

mod implementation;

pub use implementation::RkNumbersLib;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
