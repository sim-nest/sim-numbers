#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic one-dimensional Fourier, cosine, and sine transforms.
//!
//! The policy types make transform conventions part of a reusable plan rather
//! than ambient assumptions. Transform buffers use the canonical complex and
//! f64 tensor specializations owned by the number stack.

mod error;
mod fft;
mod plan;
mod reference;
mod runtime;
mod transform;

pub use error::SignalError;
pub use plan::{
    DctType, Direction, DstType, LengthPolicy, Normalization, PaddingPolicy, PlacementPolicy,
    SignConvention, SignalBuffer, SignalView, SignalViewMut, SpectrumPacking, Stride,
    TransformKind, TransformPlan,
};
pub use reference::{reference_dct, reference_dft, reference_dst};
pub use runtime::{SignalNumbersLib, call_signal_transform, signal_transform_symbol};
pub use transform::{transform, transform_in_place};

/// Cookbook recipes for deterministic signal transforms, embedded at build
/// time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod algorithm_tests;
#[cfg(test)]
mod conformance;
#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod runtime_tests;
