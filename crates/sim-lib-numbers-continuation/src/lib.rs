#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Bounded pseudo-arclength continuation with explicit numerical evidence.

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_method::ExecutionIdentity;
use sim_lib_numbers_root::{
    JacobianSource, RootTermination, VectorPlan, VectorRoot, damped_newton,
};
use sim_lib_numbers_tensor_decomp::{
    SingularCutoff, SvdPlan, VectorForm, null_space, numerical_rank, svd_f64,
};

mod continuation;

pub use continuation::*;

/// Cookbook recipes embedded for runtime discovery.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

/// Loadable runtime discovery surface.
#[derive(Default)]
pub struct ContinuationLib;
impl ContinuationLib {
    /// Creates the runtime library.
    pub fn new() -> Self {
        Self
    }
}
impl Lib for ContinuationLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "continuation"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: continuation_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(continuation_schema_symbol(), cx.factory().string("pseudo-arclength residual-manifold seeds derivative orientation step-plan bordered-newton folds rank-loss domain-exit closed-loop evidence".to_owned())?)
    }
}
#[cfg(test)]
mod tests;
