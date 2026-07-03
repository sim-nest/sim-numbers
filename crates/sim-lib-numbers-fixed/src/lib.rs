#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The fixed-width integer domains (`numbers/i8` .. `numbers/i128`,
//! `numbers/u8` .. `numbers/u128`): their literals, values, and the widening
//! promotion edges through the signed and unsigned integer lattice.

mod implementation;
mod literal;

pub use implementation::FixedNumbersLib;

#[cfg(test)]
mod tests;
