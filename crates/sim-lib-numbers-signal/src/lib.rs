#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic one- and multidimensional Fourier, cosine, and sine
//! transforms.
//!
//! The policy types make transform conventions part of a reusable plan rather
//! than ambient assumptions. Transform buffers use canonical complex and f64
//! tensors; external plans operate through caller-supplied Table or Dir blocks
//! under an explicit scratch ceiling.

mod block_io;
mod blocked;
mod error;
mod fft;
mod io_plan;
mod multidimensional;
mod plan;
mod reference;
mod runtime;
mod tensor_view;
mod transform;

pub use blocked::{
    BlockedTensor, read_blocked_tensor, transform_nd_blocked, write_blocked_tensor,
    write_complex_f64_block, write_f64_block,
};
pub use error::SignalError;
pub use io_plan::{TransformPrecision, TransformReport, TransformResources, transform_plan_digest};
pub use multidimensional::{TensorTransform, transform_nd};
pub use plan::{
    DctType, Direction, DstType, LengthPolicy, Normalization, PaddingPolicy, PlacementPolicy,
    SignConvention, SignalBuffer, SignalView, SignalViewMut, SpectrumPacking, Stride,
    TransformKind, TransformPlan,
};
pub use reference::{reference_dct, reference_dft, reference_dst};
pub use runtime::{SignalNumbersLib, call_signal_transform, signal_transform_symbol};
pub use tensor_view::TensorView;
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
mod multidimensional_tests;
#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod runtime_tests;
