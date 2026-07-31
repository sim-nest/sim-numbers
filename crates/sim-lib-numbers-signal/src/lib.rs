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
mod transform;

pub use error::SignalError;
pub use plan::{
    DctType, Direction, DstType, LengthPolicy, Normalization, PaddingPolicy, PlacementPolicy,
    SignConvention, SignalBuffer, SignalView, SignalViewMut, SpectrumPacking, Stride,
    TransformKind, TransformPlan,
};
pub use reference::{reference_dct, reference_dft, reference_dst};
pub use transform::{transform, transform_in_place};

#[cfg(test)]
mod algorithm_tests;
#[cfg(test)]
mod plan_tests;
