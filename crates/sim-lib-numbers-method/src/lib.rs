#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Validated, bounded numerical method plans and common evidence.
//!
//! This crate composes existing domain reports without replacing them. It owns
//! only the vocabulary needed to ask the same boundedness, termination, error,
//! precision, and execution-identity questions across numerical domains.

#[cfg(feature = "adapters")]
mod adapters;
mod datum;
mod model;
mod runtime;

#[cfg(feature = "adapters")]
pub use adapters::{DenseSolveAdapter, SignalEstimatorAdapter, StatsKMeansAdapter};
pub use datum::{CanonicalDatum, DatumError};
pub use model::{
    CriterionId, ErrorMeasure, ExecutionIdentity, MethodError, MethodEvidence, MethodId,
    MethodPlan, PrecisionId, RefusalId, Termination, ToleranceSet, WorkLimit, WorkReceipt,
};
pub use runtime::{MethodNumbersLib, evidence_schema_symbol};

/// Cookbook recipes embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
