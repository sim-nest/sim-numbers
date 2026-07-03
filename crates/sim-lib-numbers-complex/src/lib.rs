#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/complex` domain: complex literals and values, their arithmetic,
//! and the promotion edges from `i64`, `f64`, and `rational` into the complex
//! sink of the scalar lattice.

mod implementation;

pub use implementation::*;

#[cfg(test)]
mod tests;
