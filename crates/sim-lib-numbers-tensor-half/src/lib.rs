#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Half-precision tensor specialization: contiguous `f16` and `bf16` element
//! types with CPU arithmetic widened to `f32`.
//!
//! [`F16Tensor`] and [`Bf16Tensor`] are typed views over canonical `Tensor`
//! storage. [`HalfTensorLib`] registers both descriptor values for the base
//! tensor domain.
//!
//! # Examples
//!
//! ```
//! use half::f16;
//! use sim_lib_numbers_tensor::{SpecTensor, domains};
//! use sim_lib_numbers_tensor_half::F16Tensor;
//!
//! let tensor = F16Tensor::new(vec![2], vec![f16::from_f32(1.0), f16::from_f32(2.0)]).unwrap();
//! let widened = tensor.add_f32_scalar(0.5);
//! assert_eq!(widened.dtype(), &domains::f32());
//! assert_eq!(F16Tensor::from_uniform(&tensor.to_uniform()).unwrap().as_slice(), tensor.as_slice());
//! ```

mod descriptor;
mod half_tensor;

pub use descriptor::{
    HalfTensorLib, bf16_tensor_spec_symbol, f16_tensor_spec_symbol, tensor_lib_symbol,
};
pub use half_tensor::{Bf16Tensor, F16Tensor};

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
