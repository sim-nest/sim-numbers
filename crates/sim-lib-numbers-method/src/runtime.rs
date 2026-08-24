use sim_kernel::{
    AbiVersion, Dependency, Export, Lib, LibManifest, LibTarget, Linker, Result, Symbol, Version,
};

/// Symbol of the loadable common-evidence inspection schema.
pub fn evidence_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/method", "evidence-schema")
}

/// Loadable numerical-method inspection surface.
///
/// The exported schema names the fields guaranteed by [`crate::MethodEvidence`]
/// and its canonical [`sim_kernel::Datum`] projection. Algorithms and domain
/// adapters stay ordinary Rust; loading this library only advertises the stable
/// inspection contract to runtime callers.
#[derive(Default)]
pub struct MethodNumbersLib;

impl MethodNumbersLib {
    /// Constructs the stateless inspection library.
    pub fn new() -> Self {
        Self
    }
}

impl Lib for MethodNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "method"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: evidence_schema_symbol(),
            }],
        }
    }

    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.value(
            evidence_schema_symbol(),
            cx.factory().string(
                "method termination work requested achieved precision execution".to_owned(),
            )?,
        )
    }
}
