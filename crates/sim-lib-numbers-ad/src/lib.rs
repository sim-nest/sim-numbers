#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Automatic differentiation primitives: forward-mode dual numbers, a
//! reverse-mode evaluation tape, and the `Scalarish` numeric trait they share.

mod implementation;

pub use implementation::{Dual, Scalarish, Tape, TapeNode, Var};

#[cfg(test)]
mod tests;
