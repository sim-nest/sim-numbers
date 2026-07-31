#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Probability, descriptive statistics, finite Markov transition estimation,
//! and fairness metric helpers for number-domain data.
//!
//! Descriptive statistics and disparate-impact helpers also expose Claim
//! surfaces. The Claim values carry their subject, predicate, and evidence table
//! as inspectable runtime data, so callers can browse both the computed metric
//! and the inputs used to justify it. [`fit_markov`] keeps the finite vocabulary,
//! exact counts, additive smoothing, held-out likelihood, deterministic
//! serialization, and corpus provenance inspectable instead of hiding learned
//! weights.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
