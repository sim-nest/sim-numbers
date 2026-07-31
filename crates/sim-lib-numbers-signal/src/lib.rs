#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic transforms, convolution, classical spectral estimation, and
//! guarded deconvolution over canonical SIM tensors.
//!
//! The policy types make transform conventions part of a reusable plan rather
//! than ambient assumptions. Transform buffers use canonical complex and f64
//! tensors; external plans operate through caller-supplied Table or Dir blocks
//! under an explicit scratch ceiling. Convolution and correlation plans make
//! geometry, boundaries, normalization, lag order, cost selection, retained
//! spans, and block latency visible. Deconvolution requires a regularizer and
//! returns singular-bin and residual evidence instead of dividing blindly.

mod block_io;
mod blocked;
mod blocked_convolution;
mod convolution;
mod convolution_plan;
mod correlation;
mod deconvolution;
mod error;
mod fft;
mod io_plan;
mod lomb;
mod multidimensional;
mod multitaper;
mod periodogram;
mod plan;
mod reference;
mod runtime;
mod runtime_convolution;
mod runtime_convolution_callable;
mod runtime_convolution_render;
mod runtime_convolution_value;
mod spectrum_core;
mod spectrum_types;
mod tensor_view;
mod transform;
mod window;

pub use blocked::{
    BlockedTensor, read_blocked_tensor, transform_nd_blocked, write_blocked_tensor,
    write_complex_f64_block, write_f64_block,
};
pub use blocked_convolution::{
    BlockBoundaryReport, BlockConvolutionMethod, BlockConvolutionPlan, BlockConvolutionReport,
    BlockConvolutionResult, convolve_blocked,
};
pub use convolution::{ConvolutionReport, ConvolutionResult, convolve};
pub use convolution_plan::{
    BoundaryPolicy, ConvolutionAlgorithm, ConvolutionCostPlan, ConvolutionMode,
    ConvolutionNormalization, ConvolutionPlan, LinearOutput,
};
pub use correlation::{
    CorrelationNormalization, CorrelationPlan, CorrelationResult, LagOrder, correlate,
};
pub use deconvolution::{
    DeconvolutionMode, DeconvolutionPlan, DeconvolutionReport, DeconvolutionResult, Regularization,
    deconvolve,
};
pub use error::SignalError;
pub use io_plan::{TransformPrecision, TransformReport, TransformResources, transform_plan_digest};
pub use lomb::lomb_scargle;
pub use multidimensional::{TensorTransform, transform_nd};
pub use multitaper::multitaper;
pub use periodogram::{cross_spectrum, periodogram, welch};
pub use plan::{
    DctType, Direction, DstType, LengthPolicy, Normalization, PaddingPolicy, PlacementPolicy,
    SignConvention, SignalBuffer, SignalView, SignalViewMut, SpectrumPacking, Stride,
    TransformKind, TransformPlan,
};
pub use reference::{reference_dct, reference_dft, reference_dst};
pub use runtime::{SignalNumbersLib, call_signal_transform, signal_transform_symbol};
pub use runtime_convolution_callable::{
    call_signal_convolve, call_signal_correlate, call_signal_deconvolve, signal_convolve_symbol,
    signal_correlate_symbol, signal_deconvolve_symbol,
};
pub use spectrum_types::{
    CrossSpectrumEstimate, EstimatorEvidence, EstimatorKind, EstimatorLimits, FrequencyGridPolicy,
    LombScarglePlan, MultitaperPlan, PeriodogramPlan, SpectrumEstimate, SpectrumScaling,
    SpectrumScalingKind, SpectrumSide, WelchPlan,
};
pub use tensor_view::TensorView;
pub use transform::{transform, transform_in_place};
pub use window::{
    Window, WindowFunction, WindowMetrics, WindowNormalization, WindowSampling, WindowSpec,
};

/// Cookbook recipes for deterministic signal transforms, embedded at build
/// time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod algorithm_tests;
#[cfg(test)]
mod conformance;
#[cfg(test)]
mod convolution_tests;
#[cfg(test)]
mod multidimensional_tests;
#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod runtime_tests;
#[cfg(test)]
mod spectral_tests;
