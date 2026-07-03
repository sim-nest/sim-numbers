#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! Runge-Kutta ODE integrators for the numeric domain: fixed-step and adaptive
//! solver backends registered as numeric `ode-solve` plugins.

mod implementation;

pub use implementation::RkNumbersLib;

#[cfg(test)]
mod tests;
