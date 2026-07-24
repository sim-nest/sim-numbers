#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! f32 tensor specialization: a contiguous `f32` tensor element type and its
//! `SpecTensor` backend for the f32 tensor domain.
//!
//! [`F32Tensor`] is a typed view over canonical `Tensor` storage with native
//! single-precision element-wise math, and [`F32TensorLib`] registers it as the
//! `f32` element-type backend for the base tensor domain.
//!
//! # Examples
//!
//! ```
//! use sim_lib_numbers_tensor::SpecTensor;
//! use sim_lib_numbers_tensor_f32::F32Tensor;
//!
//! let tensor = F32Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
//! let shifted = tensor.add_scalar(0.5);
//! let roundtrip = F32Tensor::from_uniform(&shifted.to_uniform()).unwrap();
//! assert_eq!(roundtrip, shifted);
//! ```

mod f32_tensor;

pub use f32_tensor::{F32Tensor, F32TensorLib, tensor_lib_symbol, tensor_spec_symbol};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
