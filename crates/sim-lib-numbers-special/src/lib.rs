#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Evidence-carrying real error, gamma, beta, and complete elliptic functions.

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use std::{error::Error, f64::consts::PI, fmt};

const EPS: f64 = 2.0e-15;
const MAX_ITERATIONS: usize = 512;

mod functions;

pub use functions::*;

/// Loadable manifest surface advertising the special-function runtime contract.
#[derive(Default)]
pub struct SpecialNumbersLib;
impl SpecialNumbersLib {
    /// Constructs the stateless library.
    pub fn new() -> Self {
        Self
    }
}
impl Lib for SpecialNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "special"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: special_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(special_schema_symbol(),cx.factory().string("erf erfc inverse-erf log-gamma gamma-p gamma-q beta-i elliptic-k elliptic-e accuracy-evidence".to_owned())?)
    }
}
/// Runtime inspection symbol for the installed function and evidence families.
pub fn special_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/special", "schema")
}

/// Embedded cookbook recipes.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
