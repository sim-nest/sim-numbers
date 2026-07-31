#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic transforms, convolution, autoregression, spectral estimation,
//! interpolation, and guarded deconvolution over canonical SIM tensors.
//!
//! The policy types make transform conventions part of a reusable plan rather
//! than ambient assumptions. Transform buffers use canonical complex and f64
//! tensors; external plans operate through caller-supplied Table or Dir blocks
//! under an explicit scratch ceiling. Convolution and correlation plans make
//! geometry, boundaries, normalization, lag order, cost selection, retained
//! spans, and block latency visible. Deconvolution requires a regularizer and
//! returns singular-bin and residual evidence instead of dividing blindly.
//! Burg models retain stability, effective-order, residual, and selection
//! evidence; periodic DFT and analytic-signal helpers share the same explicit
//! sign and normalization conventions.

mod analytic;
mod autoregressive;
mod block_io;
mod blocked;
mod blocked_convolution;
mod convolution;
mod convolution_plan;
mod correlation;
mod deconvolution;
mod error;
mod fft;
mod interpolate;
mod io_plan;
mod lomb;
mod mem;
mod multidimensional;
mod multitaper;
mod periodogram;
mod plan;
mod prediction;
mod reference;
mod runtime;
mod runtime_convolution;
mod runtime_convolution_callable;
mod runtime_convolution_render;
mod runtime_convolution_value;
mod runtime_spectral;
mod runtime_spectral_callable;
mod runtime_spectral_value;
mod spectrum_core;
mod spectrum_types;
mod tensor_view;
mod transform;
mod window;

pub use analytic::{
    AnalyticSignal, AnalyticSignalPlan, AnalyticSignalReport, EnvelopeFollowerPlan,
    InstantaneousFrequency, analytic_envelope, analytic_signal, envelope_follow, hilbert_transform,
    instantaneous_frequency, unwrap_phase,
};
pub use autoregressive::{
    ArModel, ArOrderCriterion, ArOrderScore, BurgEvidence, BurgPlan, BurgStability,
    BurgTermination, burg,
};
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
pub use interpolate::{
    DftIntegral, DftInterpolation, DftSeriesPlan, DftSeriesReport, EndpointConvention,
    NyquistConvention, Periodicity, dft_bin, dft_bin_real, dft_integrate, dft_interpolate,
};
pub use io_plan::{TransformPrecision, TransformReport, TransformResources, transform_plan_digest};
pub use lomb::lomb_scargle;
pub use mem::{MemSpectrumPlan, mem_spectrum};
pub use multidimensional::{TensorTransform, transform_nd};
pub use multitaper::multitaper;
pub use periodogram::{cross_spectrum, periodogram, welch};
pub use plan::{
    DctType, Direction, DstType, LengthPolicy, Normalization, PaddingPolicy, PlacementPolicy,
    SignConvention, SignalBuffer, SignalView, SignalViewMut, SpectrumPacking, Stride,
    TransformKind, TransformPlan,
};
pub use prediction::{
    PredictionDirection, PredictionPlan, PredictionResult, predict_backward, predict_forward,
};
pub use reference::{reference_dct, reference_dft, reference_dst};
pub use runtime::{SignalNumbersLib, call_signal_transform, signal_transform_symbol};
pub use runtime_convolution_callable::{
    call_signal_convolve, call_signal_correlate, call_signal_deconvolve, signal_convolve_symbol,
    signal_correlate_symbol, signal_deconvolve_symbol,
};
pub use runtime_spectral_callable::{
    call_signal_burg, call_signal_dft_interpolate, signal_burg_symbol,
    signal_dft_interpolate_symbol,
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
mod analytic_tests;
#[cfg(test)]
mod autoregressive_tests;
#[cfg(test)]
mod conformance;
#[cfg(test)]
mod convolution_tests;
#[cfg(test)]
mod interpolation_tests;
#[cfg(test)]
mod multidimensional_tests;
#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod runtime_tests;
#[cfg(test)]
mod spectral_tests;
