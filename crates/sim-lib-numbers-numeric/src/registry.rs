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
    pub fn register_differentiator(&mut self, plugin: Arc<dyn Differentiator>) {
        self.differentiators.insert(plugin.name(), plugin);
    }

    pub fn register_quadrature(&mut self, plugin: Arc<dyn Quadrature>) {
        match plugin.kind() {
            NumericKind::QuadratureFixed => {
                self.quadrature_fixed.insert(plugin.name(), plugin);
            }
            NumericKind::QuadratureAdaptive => {
                self.quadrature_adaptive.insert(plugin.name(), plugin);
            }
            _ => {}
        }
    }

    pub fn register_ode_solver(&mut self, plugin: Arc<dyn OdeSolver>) {
        match plugin.kind() {
            NumericKind::OdeFixed => {
                self.ode_fixed.insert(plugin.name(), plugin);
            }
            NumericKind::OdeAdaptive => {
                self.ode_adaptive.insert(plugin.name(), plugin);
            }
            _ => {}
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
        .register_differentiator(plugin);
    Ok(())
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
        .register_quadrature(plugin);
    Ok(())
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
        .register_ode_solver(plugin);
    Ok(())
}
