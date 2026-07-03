#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Quadrature and finite-difference backends for the numeric domain: fixed and
//! adaptive integration rules plus finite-difference differentiators, packaged
//! as a registered numeric plugin library.

mod implementation;

pub use implementation::QuadNumbersLib;

#[cfg(test)]
mod tests;
