#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Bounded scalar and vector root finding with reviewable evidence.
//!
//! A [`RootBracket`] is deliberately not interchangeable with a
//! [`RootEstimate`]. Only the former proves a sign-changing enclosure.

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_method::ExecutionIdentity;
use sim_lib_numbers_tensor_decomp::{
    SingularCutoff, SvdPlan, VectorForm, least_squares, numerical_rank, svd_f64,
};
use std::{error::Error, fmt};

mod bisection;
mod brent;
mod model;
mod newton;
mod secant;
mod vector;

pub use bisection::*;
pub use brent::*;
pub use model::*;
pub use newton::*;
pub use secant::*;
pub use vector::*;

/// Cookbook recipes embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
/// Loadable root-finding schema surface.
#[derive(Default)]
pub struct RootLib;
impl RootLib {
    /// Constructs the stateless library.
    pub fn new() -> Self {
        Self
    }
}
impl Lib for RootLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "root"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: root_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(root_schema_symbol(),cx.factory().string("bisection brent-dekker safeguarded-newton secant vector-newton broyden bracket estimate residual rank work execution".to_owned())?)
    }
}
/// Runtime inspection symbol.
pub fn root_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/root", "schema")
}

impl fmt::Display for RootTermination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for RootTermination {}

#[cfg(test)]
mod tests;
