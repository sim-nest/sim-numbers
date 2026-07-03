//! Implementation of the numeric domain: its library and operations, option
//! parsing, runtime dispatch, plugin registry, and backend traits.

#[path = "function.rs"]
mod function;
#[path = "options.rs"]
mod options;
#[path = "pipeline.rs"]
mod pipeline;
#[path = "pipeline_run.rs"]
mod pipeline_run;
#[path = "registry.rs"]
mod registry;
#[path = "runtime.rs"]
mod runtime;
#[path = "traits.rs"]
mod traits;

pub use function::{
    NumericNumbersLib, integrate_adapt_symbol, integrate_symbol, numeric_compose_symbol,
    numeric_diff_symbol, numeric_run_composed_symbol, ode_solve_symbol,
};
pub use pipeline::{ComposedPipeline, PipelineKind, StateKind};
pub use registry::{
    global_numeric_registry, register_differentiator, register_ode_solver, register_quadrature,
};
pub use traits::{
    DiffOpts, Differentiator, NumericKind, NumericPlugin, OdeOpts, OdeProblem, OdeSolver, QuadOpts,
    Quadrature,
};
