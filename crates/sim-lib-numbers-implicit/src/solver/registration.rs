//! Loadable implicit-solver library registration.

use super::*;

/// Loadable runtime library advertising the `radau-iia` DAE backend.
pub struct ImplicitNumbersLib;
impl ImplicitNumbersLib {
    /// Creates the stateless library.
    pub fn new() -> Self {
        Self
    }
}
impl Default for ImplicitNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}
impl Lib for ImplicitNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "implicit"),
            version: Version(env!("CARGO_PKG_VERSION").into()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: vec![],
            exports: vec![Export::Value {
                symbol: numeric_plugin_descriptor_symbol("numbers/implicit", "radau-iia"),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(
            numeric_plugin_descriptor_symbol("numbers/implicit", "radau-iia"),
            cx.factory().table(vec![
                (
                    Symbol::new("kind"),
                    cx.factory().string("numeric-plugin".into())?,
                ),
                (
                    Symbol::new("method"),
                    cx.factory().symbol(Symbol::new("radau-iia"))?,
                ),
                (
                    Symbol::new("plugin-kind"),
                    cx.factory().string("dae".into())?,
                ),
                (Symbol::new("adaptive"), cx.factory().bool(true)?),
                (Symbol::new("dense-path"), cx.factory().bool(true)?),
                (Symbol::new("events"), cx.factory().bool(true)?),
                (Symbol::new("jacobian"), cx.factory().bool(true)?),
                (Symbol::new("mass-matrix"), cx.factory().bool(true)?),
                (Symbol::new("dae-residual"), cx.factory().bool(true)?),
                (
                    Symbol::new("provider"),
                    cx.factory()
                        .symbol(Symbol::qualified("numbers", "implicit"))?,
                ),
            ])?,
        )?;
        Ok(())
    }
}
