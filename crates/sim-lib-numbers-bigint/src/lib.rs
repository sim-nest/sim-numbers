#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/bigint` domain: arbitrary-precision integer literals and
//! values, their exact arithmetic, and promotion into `rational`.

mod implementation;
mod literal;
mod ops;

pub use implementation::{BigIntNumberDomain, BigIntNumbersLib, number_domain};

#[cfg(test)]
mod tests;
