//! Numeric plugin registration and loadable Runge-Kutta library surface.

use super::*;

/// Registered numeric plugin library that installs this crate's Runge-Kutta
/// ODE-solver backends.
///
/// Loading this [`Lib`] registers the fixed-step solvers (forward Euler,
/// backward Euler, midpoint, classic RK4) and the adaptive Runge-Kutta-Fehlberg
/// (RKF45) integrator as `ode-solve` plugins on the numeric surface, and
/// installs the plugin descriptor values that advertise each backend to the
/// registry.
///
/// # Examples
///
/// ```
/// use sim_kernel::Lib;
/// use sim_lib_numbers_rk::RkNumbersLib;
///
/// let lib = RkNumbersLib::new();
/// let manifest = lib.manifest();
/// // One descriptor export per registered solver (4 fixed-step plus RKF45).
/// assert_eq!(manifest.exports.len(), 6);
/// ```
pub struct RkNumbersLib;

impl RkNumbersLib {
    /// Creates the Runge-Kutta library. The value is stateless; all solver
    /// backends are installed when it is loaded into a [`Cx`].
    pub fn new() -> Self {
        Self
    }
}

impl Default for RkNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for RkNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: domains::rk(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: descriptor_exports(),
        }
    }

    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        register_plugins_once()?;
        install_descriptors(cx, linker)?;
        Ok(())
    }
}

static PLUGINS_REGISTERED: OnceLock<std::result::Result<(), String>> = OnceLock::new();

fn register_plugins_once() -> Result<()> {
    match PLUGINS_REGISTERED.get_or_init(|| register_plugins().map_err(|err| err.to_string())) {
        Ok(()) => Ok(()),
        Err(message) => Err(Error::Eval(message.clone())),
    }
}

fn register_plugins() -> Result<()> {
    for plugin in solvers() {
        register_ode_solver(plugin)?;
    }
    Ok(())
}

fn descriptor_exports() -> Vec<Export> {
    descriptor_specs()
        .into_iter()
        .map(|(name, _adaptive)| Export::Value {
            symbol: numeric_plugin_descriptor_symbol("numbers/rk", name),
        })
        .collect()
}

fn install_descriptors(cx: &sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
    for (name, adaptive) in descriptor_specs() {
        linker.value(
            numeric_plugin_descriptor_symbol("numbers/rk", name),
            cx.factory().table(vec![
                (
                    Symbol::new("kind"),
                    cx.factory().string("numeric-plugin".to_owned())?,
                ),
                (
                    Symbol::new("method"),
                    cx.factory().symbol(Symbol::new(name))?,
                ),
                (
                    Symbol::new("plugin-kind"),
                    cx.factory().string("ode".to_owned())?,
                ),
                (Symbol::new("adaptive"), cx.factory().bool(adaptive)?),
                (Symbol::new("fixed"), cx.factory().bool(!adaptive)?),
                (Symbol::new("scalar-state"), cx.factory().bool(true)?),
                (Symbol::new("tensor-state"), cx.factory().bool(true)?),
                (Symbol::new("dense-path"), cx.factory().bool(true)?),
                (Symbol::new("events"), cx.factory().bool(true)?),
                (Symbol::new("jacobian"), cx.factory().bool(false)?),
                (Symbol::new("mass-matrix"), cx.factory().bool(false)?),
                (Symbol::new("dae-residual"), cx.factory().bool(false)?),
                (Symbol::new("provider"), cx.factory().symbol(domains::rk())?),
                (
                    Symbol::new("api"),
                    cx.factory().string("numeric-plugin-v2".to_owned())?,
                ),
            ])?,
        )?;
    }
    Ok(())
}

fn descriptor_specs() -> Vec<(&'static str, bool)> {
    vec![
        ("forward-euler", false),
        ("backward-euler", false),
        ("midpoint", false),
        ("rk4", false),
        ("rkf45", true),
        ("dop853", true),
    ]
}

#[derive(Clone, Copy)]
pub(super) enum Method {
    ForwardEuler,
    BackwardEuler,
    Midpoint,
    Rk4,
    Rkf45,
    Dop853,
}

fn solvers() -> Vec<Arc<dyn OdeSolver>> {
    vec![
        Arc::new(RkPlugin::new(
            "forward-euler",
            NumericKind::OdeFixed,
            Method::ForwardEuler,
        )),
        Arc::new(RkPlugin::new(
            "backward-euler",
            NumericKind::OdeFixed,
            Method::BackwardEuler,
        )),
        Arc::new(RkPlugin::new(
            "midpoint",
            NumericKind::OdeFixed,
            Method::Midpoint,
        )),
        Arc::new(RkPlugin::new("rk4", NumericKind::OdeFixed, Method::Rk4)),
        Arc::new(RkPlugin::new(
            "rkf45",
            NumericKind::OdeAdaptive,
            Method::Rkf45,
        )),
        Arc::new(RkPlugin::new(
            "dop853",
            NumericKind::OdeAdaptive,
            Method::Dop853,
        )),
    ]
}
