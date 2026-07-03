//! Implementation of the quadrature library: differentiator backends,
//! quadrature rules, and shared value-arithmetic support.

#[path = "diff.rs"]
mod diff;
#[path = "quad.rs"]
mod quad;
#[path = "support.rs"]
mod support;

pub use quad::QuadNumbersLib;
