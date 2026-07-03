//! Implementation of the Runge-Kutta library: the ODE-solver backends and their
//! shared value-arithmetic support.

#[path = "solver.rs"]
mod solver;
#[path = "support.rs"]
mod support;

pub use solver::RkNumbersLib;
