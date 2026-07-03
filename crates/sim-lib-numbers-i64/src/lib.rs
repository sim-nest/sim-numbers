#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/i64` domain: 64-bit signed-integer literals and values, their
//! scalar arithmetic, and promotion edges into `f64` and `rational`.

mod implementation;

pub use implementation::*;
