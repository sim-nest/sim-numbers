//! The global numeric registry holding registered differentiator, quadrature,
//! and ODE-solver plugins keyed by name and kind.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock, RwLock},
};

use sim_kernel::{Error, Result, Symbol};

use super::traits::{Differentiator, NumericKind, OdeSolver, Quadrature};

#[derive(Default)]
pub struct NumericRegistry {
    differentiators: BTreeMap<Symbol, Arc<dyn Differentiator>>,
    quadrature_fixed: BTreeMap<Symbol, Arc<dyn Quadrature>>,
    quadrature_adaptive: BTreeMap<Symbol, Arc<dyn Quadrature>>,
    ode_fixed: BTreeMap<Symbol, Arc<dyn OdeSolver>>,
    ode_adaptive: BTreeMap<Symbol, Arc<dyn OdeSolver>>,
}

impl NumericRegistry {
    pub fn register_differentiator(&mut self, plugin: Arc<dyn Differentiator>) -> Result<()> {
        let name = plugin.name();
        insert_plugin(
            &mut self.differentiators,
            NumericKind::Differentiator,
            name,
            plugin,
        )
    }

    pub fn register_quadrature(&mut self, plugin: Arc<dyn Quadrature>) -> Result<()> {
        let name = plugin.name();
        match plugin.kind() {
            NumericKind::QuadratureFixed => insert_plugin(
                &mut self.quadrature_fixed,
                NumericKind::QuadratureFixed,
                name,
                plugin,
            ),
            NumericKind::QuadratureAdaptive => insert_plugin(
                &mut self.quadrature_adaptive,
                NumericKind::QuadratureAdaptive,
                name,
                plugin,
            ),
            kind => Err(Error::Eval(format!(
                "numeric plugin :{name} has invalid quadrature kind {kind:?}"
            ))),
        }
    }

    pub fn register_ode_solver(&mut self, plugin: Arc<dyn OdeSolver>) -> Result<()> {
        let name = plugin.name();
        match plugin.kind() {
            NumericKind::OdeFixed => {
                insert_plugin(&mut self.ode_fixed, NumericKind::OdeFixed, name, plugin)
            }
            NumericKind::OdeAdaptive => insert_plugin(
                &mut self.ode_adaptive,
                NumericKind::OdeAdaptive,
                name,
                plugin,
            ),
            kind => Err(Error::Eval(format!(
                "numeric plugin :{name} has invalid ODE kind {kind:?}"
            ))),
        }
    }

    pub fn differentiator(&self, method: &Symbol) -> Option<Arc<dyn Differentiator>> {
        self.differentiators.get(method).cloned()
    }

    pub fn quadrature_fixed(&self, method: &Symbol) -> Option<Arc<dyn Quadrature>> {
        self.quadrature_fixed.get(method).cloned()
    }

    pub fn quadrature_adaptive(&self, method: &Symbol) -> Option<Arc<dyn Quadrature>> {
        self.quadrature_adaptive.get(method).cloned()
    }

    pub fn ode_fixed(&self, method: &Symbol) -> Option<Arc<dyn OdeSolver>> {
        self.ode_fixed.get(method).cloned()
    }

    pub fn ode_adaptive(&self, method: &Symbol) -> Option<Arc<dyn OdeSolver>> {
        self.ode_adaptive.get(method).cloned()
    }
}

fn insert_plugin<T: ?Sized>(
    plugins: &mut BTreeMap<Symbol, Arc<T>>,
    kind: NumericKind,
    name: Symbol,
    plugin: Arc<T>,
) -> Result<()> {
    if plugins.contains_key(&name) {
        return Err(Error::Eval(format!(
            "numeric {kind:?} plugin :{name} is already registered"
        )));
    }
    plugins.insert(name, plugin);
    Ok(())
}

static GLOBAL_NUMERIC_REGISTRY: OnceLock<RwLock<NumericRegistry>> = OnceLock::new();

/// Returns the process-global numeric registry, creating it on first access.
pub fn global_numeric_registry() -> &'static RwLock<NumericRegistry> {
    GLOBAL_NUMERIC_REGISTRY.get_or_init(|| RwLock::new(NumericRegistry::default()))
}

/// Registers a differentiator backend in the global numeric registry.
///
/// # Errors
///
/// Returns an error if the global registry lock is poisoned.
pub fn register_differentiator(plugin: Arc<dyn Differentiator>) -> Result<()> {
    global_numeric_registry()
        .write()
        .map_err(|_| Error::PoisonedLock("numeric registry"))?
        .register_differentiator(plugin)
}

/// Registers a quadrature backend in the global numeric registry.
///
/// The plugin is routed to the fixed or adaptive slot according to its
/// [`NumericKind`](crate::NumericKind).
///
/// # Errors
///
/// Returns an error if the global registry lock is poisoned.
pub fn register_quadrature(plugin: Arc<dyn Quadrature>) -> Result<()> {
    global_numeric_registry()
        .write()
        .map_err(|_| Error::PoisonedLock("numeric registry"))?
        .register_quadrature(plugin)
}

/// Registers an ODE-solver backend in the global numeric registry.
///
/// The plugin is routed to the fixed or adaptive slot according to its
/// [`NumericKind`](crate::NumericKind).
///
/// # Errors
///
/// Returns an error if the global registry lock is poisoned.
pub fn register_ode_solver(plugin: Arc<dyn OdeSolver>) -> Result<()> {
    global_numeric_registry()
        .write()
        .map_err(|_| Error::PoisonedLock("numeric registry"))?
        .register_ode_solver(plugin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::implementation::traits::{
        DiffOpts, NumericCallable, NumericPlugin, OdeOpts, OdeProblem, QuadOpts,
    };
    use sim_kernel::{Cx, Value};
    use sim_lib_numbers_func::Func;

    struct TestPlugin {
        name: Symbol,
        kind: NumericKind,
    }

    impl TestPlugin {
        fn new(name: &str, kind: NumericKind) -> Self {
            Self {
                name: Symbol::new(name),
                kind,
            }
        }
    }

    impl NumericPlugin for TestPlugin {
        fn name(&self) -> Symbol {
            self.name.clone()
        }

        fn kind(&self) -> NumericKind {
            self.kind
        }
    }

    impl Differentiator for TestPlugin {
        fn diff_at(
            &self,
            _cx: &mut Cx,
            _f: &Func,
            _var: &Symbol,
            point: &Value,
            _opt: DiffOpts,
        ) -> Result<Value> {
            Ok(point.clone())
        }
    }

    impl Quadrature for TestPlugin {
        fn integrate(
            &self,
            _cx: &mut Cx,
            _f: &NumericCallable,
            _var: &Symbol,
            lo: &Value,
            _hi: &Value,
            _opt: QuadOpts,
        ) -> Result<Value> {
            Ok(lo.clone())
        }
    }

    impl OdeSolver for TestPlugin {
        fn solve(
            &self,
            _cx: &mut Cx,
            _problem: OdeProblem<'_>,
            _opt: OdeOpts,
        ) -> Result<Vec<(Value, Value)>> {
            Ok(Vec::new())
        }
    }

    fn differentiator(name: &str) -> Arc<dyn Differentiator> {
        Arc::new(TestPlugin::new(name, NumericKind::Differentiator))
    }

    fn quadrature(name: &str, kind: NumericKind) -> Arc<dyn Quadrature> {
        Arc::new(TestPlugin::new(name, kind))
    }

    fn ode_solver(name: &str, kind: NumericKind) -> Arc<dyn OdeSolver> {
        Arc::new(TestPlugin::new(name, kind))
    }

    #[test]
    fn duplicate_differentiator_registration_is_rejected() {
        let mut registry = NumericRegistry::default();
        registry
            .register_differentiator(differentiator("central-5"))
            .unwrap();

        let err = registry
            .register_differentiator(differentiator("central-5"))
            .unwrap_err();

        assert!(err.to_string().contains("already registered"), "{err}");
    }

    #[test]
    fn duplicate_quadrature_registration_is_rejected_per_kind() {
        let mut registry = NumericRegistry::default();
        registry
            .register_quadrature(quadrature("romberg", NumericKind::QuadratureFixed))
            .unwrap();

        let err = registry
            .register_quadrature(quadrature("romberg", NumericKind::QuadratureFixed))
            .unwrap_err();

        assert!(err.to_string().contains("already registered"), "{err}");
    }

    #[test]
    fn fixed_and_adaptive_quadrature_can_share_method_name() {
        let mut registry = NumericRegistry::default();
        let name = Symbol::new("romberg");

        registry
            .register_quadrature(quadrature("romberg", NumericKind::QuadratureFixed))
            .unwrap();
        registry
            .register_quadrature(quadrature("romberg", NumericKind::QuadratureAdaptive))
            .unwrap();

        assert!(registry.quadrature_fixed(&name).is_some());
        assert!(registry.quadrature_adaptive(&name).is_some());
    }

    #[test]
    fn duplicate_ode_registration_is_rejected_per_kind() {
        let mut registry = NumericRegistry::default();
        registry
            .register_ode_solver(ode_solver("rk4", NumericKind::OdeFixed))
            .unwrap();

        let err = registry
            .register_ode_solver(ode_solver("rk4", NumericKind::OdeFixed))
            .unwrap_err();

        assert!(err.to_string().contains("already registered"), "{err}");
    }
}
