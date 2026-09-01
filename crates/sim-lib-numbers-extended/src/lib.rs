#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Normalized double-double arithmetic and the `numbers/extended` runtime
//! domain. A value is an unevaluated sum of two binary64 components with
//! `|lo| <= 0.5 ulp(hi)` for finite nonzero results.
//!
//! Addition and multiplication are built from Knuth/Dekker error-free
//! transforms (`two_sum`, `quick_two_sum`, and an FMA `two_prod`). Under
//! round-to-nearest binary64 arithmetic, their residuals are exact. Division
//! and square root use two correction steps. The elementary functions perform
//! explicit argument reduction followed by double-double Taylor evaluation.
//! Checked specimens require errors below 32 double-double ulps on their stated
//! compact domains; no correct-rounding claim is made.

use core::{cmp::Ordering, fmt, ops};
use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Export, Factory, Lib, LibManifest, LibTarget, Linker, NumberDomain,
    NumberLiteral, Object, PromotionRule, Result, Symbol, Value, Version,
};
use sim_lib_numbers_core::{RealScalar, domains};

mod extended;

pub use extended::*;

/// Recipes embedded for runtime discovery.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

/// Loadable runtime library registering the domain and its deliberate f64 and rational edges.
pub struct ExtendedNumbersLib;
impl ExtendedNumbersLib {
    /// Constructs the stateless installer.
    pub fn new() -> Self {
        Self
    }
}
impl Default for ExtendedNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}
impl Lib for ExtendedNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: number_domain(),
            version: Version(env!("CARGO_PKG_VERSION").into()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![],
            capabilities: vec![],
            exports: vec![Export::NumberDomain {
                symbol: number_domain(),
                number_domain_id: None,
            }],
        }
    }
    fn load(&self, _: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(ExtendedNumberDomain))
                .expect("box domain"),
        )?;
        for r in promotion_rules() {
            linker.promotion_rule(r);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
