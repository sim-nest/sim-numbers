#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! Tensor broadcasting specialization: element-wise binary and unary tensor
//! operations with NumPy-style shape broadcasting and promotion rules.
//!
//! The single export is [`TensorBroadcastLib`], a stateless [`Lib`] that, when
//! loaded, installs the `math` element-wise operators and the scalar-to-tensor
//! promotion rule over the `numbers/tensor` domain.
//!
//! [`Lib`]: sim_kernel::Lib
//!
//! # Examples
//!
//! The library is registered against the tensor number domain and adds no
//! exports of its own (it layers behavior onto operators already declared by
//! the base tensor domain):
//!
//! ```
//! use sim_kernel::Lib;
//! use sim_lib_numbers_tensor_bcast::TensorBroadcastLib;
//! use sim_lib_numbers_tensor::domains;
//!
//! let manifest = TensorBroadcastLib::new().manifest();
//! assert_eq!(manifest.id, domains::tensor_bcast());
//! assert!(manifest.exports.is_empty());
//! ```

mod implementation;

pub use implementation::TensorBroadcastLib;

#[cfg(test)]
mod tests;
