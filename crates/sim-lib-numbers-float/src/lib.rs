#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/f32` domain: single-precision floating-point literals and
//! values, their scalar arithmetic, and promotion into `f64`.

mod implementation;
mod literal;
mod ops;

pub use implementation::{F32NumberDomain, F32NumbersLib, number_domain};

#[cfg(test)]
mod tests;
