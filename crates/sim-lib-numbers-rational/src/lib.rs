#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/rational` domain: exact rational literals and values built over
//! bigint numerator/denominator pairs, their reduced arithmetic, and promotion
//! edges to and from the integer and `f64` domains.

mod implementation;

pub use implementation::*;
