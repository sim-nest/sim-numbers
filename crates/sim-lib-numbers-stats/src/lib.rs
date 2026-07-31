#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Probability, descriptive statistics, deterministic clustering, streaming
//! quantiles, finite Markov and hidden-state sequence inference, and fairness
//! helpers for number-domain data.
//!
//! Descriptive statistics and disparate-impact helpers also expose Claim
//! surfaces. The Claim values carry their subject, predicate, and evidence table
//! as inspectable runtime data, so callers can browse both the computed metric
//! and the inputs used to justify it. [`fit_markov`] keeps the finite vocabulary,
//! exact counts, additive smoothing, held-out likelihood, deterministic
//! serialization, and corpus provenance inspectable instead of hiding learned
//! weights. [`QuantileSketch`] makes rank error and retained memory explicit;
//! [`forward_backward`], [`viterbi`], and [`fit_hmm`] keep normalization,
//! convergence, bounded work, numerical repair, and termination evidence.
//! [`fit_kmeans`] and [`fit_gmm`] add seeded initialization, bounded convergence,
//! regularized covariance, singular-component policy, and model-selection
//! evidence without taking ownership of sequence alignment.

mod implementation;

pub use implementation::*;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
