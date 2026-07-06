#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! Bit-tensor specialization: a packed-word boolean tensor element type and its
//! `SpecTensor` backend, with bitwise operations over the tensor domain.
//!
//! [`BitTensor`] is the storage type (booleans packed into `u64` words) with
//! element-wise [`bit_and`](BitTensor::bit_and), [`bit_or`](BitTensor::bit_or),
//! and [`bit_xor`](BitTensor::bit_xor). [`BitTensorLib`] registers it as the
//! `bool` element-type backend for the base tensor domain.
//!
//! # Examples
//!
//! Pack booleans, combine two tensors bit-for-bit, and unpack the result:
//!
//! ```
//! use sim_lib_numbers_tensor_bit::BitTensor;
//!
//! let left = BitTensor::from_bools(vec![4], &[true, false, true, true]).unwrap();
//! let right = BitTensor::from_bools(vec![4], &[true, true, false, true]).unwrap();
//! let masked = left.bit_and(&right).unwrap();
//! assert_eq!(masked.to_bools(), vec![true, false, false, true]);
//! ```
//!
//! Shape mismatches fail closed rather than truncate:
//!
//! ```
//! use sim_lib_numbers_tensor_bit::BitTensor;
//!
//! let a = BitTensor::from_bools(vec![2], &[true, false]).unwrap();
//! let b = BitTensor::from_bools(vec![3], &[true, false, true]).unwrap();
//! assert!(a.bit_or(&b).is_none());
//! ```

mod bit_tensor;

pub use bit_tensor::{BitTensor, BitTensorLib, tensor_lib_symbol, tensor_spec_symbol};
