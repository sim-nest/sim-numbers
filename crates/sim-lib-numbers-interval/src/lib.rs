#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Sealed interval certification: evidence may be inspected, never imported as authority.

use sim_kernel::{
    AbiVersion, Datum, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use std::{error::Error, fmt};

mod analysis;
mod interval;

pub use analysis::*;
pub use interval::*;

/// Loadable inspection surface; it does not export a certificate constructor.
#[derive(Default)]
pub struct IntervalLib;
impl Lib for IntervalLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "interval"),
            version: Version(env!("CARGO_PKG_VERSION").into()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![],
            capabilities: vec![],
            exports: vec![Export::Value {
                symbol: interval_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(interval_schema_symbol(),cx.factory().string("sealed estimate certified rational directed-binary64 refusal interval-newton krawczyk threshold inspection-only".into())?)
    }
}
/// Runtime schema symbol.
pub fn interval_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/interval", "schema")
}
/// Embedded recipes.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
