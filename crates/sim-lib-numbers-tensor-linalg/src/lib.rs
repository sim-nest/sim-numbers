#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! Reductions, linear-algebra operations, f32/f64 transcendentals, and
//! constructors over the tensor domain, registered as a tensor library.
//!
//! The single export is [`TensorLinalgLib`], a stateless [`Lib`] that, when
//! loaded, registers tensor operation callables over the base tensor domain.
//!
//! [`Lib`]: sim_kernel::Lib
//!
//! # Examples
//!
//! The library is identified by the `numbers/tensor-linalg` domain and exports
//! one function per operation it provides:
//!
//! ```
//! use sim_kernel::Lib;
//! use sim_lib_numbers_tensor_linalg::TensorLinalgLib;
//! use sim_lib_numbers_tensor::domains;
//!
//! let manifest = TensorLinalgLib::new().manifest();
//! assert_eq!(manifest.id, domains::tensor_linalg());
//! assert_eq!(manifest.exports.len(), 18);
//! ```

mod implementation;

pub use implementation::TensorLinalgLib;

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
