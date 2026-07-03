#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! Cross-domain arithmetic: the `math/add`, `math/sub`, `math/mul`, `math/div`,
//! and reduction entry points that coerce mixed-domain operands through the
//! promotion lattice and route symbolic inputs to the CAS when it is loaded.

mod implementation;

pub use implementation::*;

#[cfg(test)]
mod tests;
