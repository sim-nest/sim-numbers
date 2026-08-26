//! Bounded optimization with explicit plans and honest convergence evidence.

use core::fmt;
use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_tensor_decomp::{
    SingularCutoff, SvdPlan, VectorForm, least_squares as svd_least_squares, numerical_rank,
    svd_f64,
};

/// Cookbook recipes embedded for runtime discovery.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

/// Loadable schema surface for optimization policy and evidence discovery.
#[derive(Default)]
pub struct OptimizeLib;
impl OptimizeLib {
    pub fn new() -> Self {
        Self
    }
}
impl Lib for OptimizeLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "optimize"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: optimize_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(optimize_schema_symbol(),cx.factory().string("bounded-brent projected-bfgs levenberg-marquardt trust-region-reflective active-set-linear-least-squares rank covariance termination work assignment-adapter".to_owned())?)
    }
}
/// Runtime inspection symbol.
pub fn optimize_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/optimize", "schema")
}

mod least_squares;
mod minimize;
mod model;
mod scalar;

pub use least_squares::*;
pub use minimize::*;
pub use model::*;
pub use scalar::*;

#[cfg(feature = "assignment")]
pub mod assignment {
    use sim_lib_discrete_graph::{
        Assignment, AssignmentPolicy, CostMatrix, GraphError, min_cost_assignment,
    };
    pub fn assign(
        costs: Vec<Vec<f64>>,
        policy: AssignmentPolicy<f64>,
    ) -> Result<Assignment<f64>, GraphError> {
        let matrix = CostMatrix::try_from(costs)?;
        min_cost_assignment(&matrix, policy)
    }
}

#[cfg(test)]
mod tests;
