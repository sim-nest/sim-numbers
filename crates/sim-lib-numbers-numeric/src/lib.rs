#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! Numeric evaluation surface: the `numeric` domain exposes `numeric-diff`,
//! `integrate`, `ode-solve`, and composed pipelines over a registry of pluggable
//! differentiator, quadrature, and ODE-solver backends.
//!
//! A composed pipeline pairs a [`sim_lib_numbers_func::Func`] with a numeric
//! domain such as quadrature or ODE solving, a method such as `simpson` or
//! `rk4`, and a state kind. The resulting [`ComposedPipeline`] is a first-class
//! runtime value that can be inspected and run through `numeric/run-composed`.

mod implementation;

pub use implementation::{
    ComposedPipeline, DiffOpts, Differentiator, NumericKind, NumericNumbersLib, NumericPlugin,
    OdeOpts, OdeProblem, OdeSolver, PipelineKind, QuadOpts, Quadrature, StateKind,
    global_numeric_registry, integrate_adapt_symbol, integrate_symbol, numeric_compose_symbol,
    numeric_diff_symbol, numeric_run_composed_symbol, ode_solve_symbol, register_differentiator,
    register_ode_solver, register_quadrature,
};

#[cfg(test)]
mod tests;
