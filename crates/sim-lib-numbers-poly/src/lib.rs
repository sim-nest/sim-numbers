#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Dense coefficient algebra, deliberately separate from symbolic CAS trees.

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_complex::ComplexValue;
use sim_lib_numbers_tensor_decomp::{SchurPlan, SvdPlan, VectorForm, real_schur_f64, svd_f64};
use std::{error::Error, fmt};

mod polynomial;

pub use polynomial::*;

/// Loadable coefficient-algebra surface.
#[derive(Default)]
pub struct PolyLib;
impl Lib for PolyLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "poly"),
            version: Version(env!("CARGO_PKG_VERSION").into()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![],
            capabilities: vec![],
            exports: vec![Export::Value {
                symbol: poly_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(poly_schema_symbol(),cx.factory().string("coefficient polynomial laurent-integer-exponent puiseux-rational-exponent horner compensated-horner exact-gcd div-rem roots backward-error pade rank-evidence".into())?)
    }
}
/// Runtime schema symbol.
pub fn poly_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/poly", "schema")
}
/// Embedded recipes.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
