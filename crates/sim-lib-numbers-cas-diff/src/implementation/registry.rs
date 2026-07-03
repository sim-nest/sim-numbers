//! The extensible differentiation-rule registry: a process-global map from an
//! operator symbol to a custom `diff` rule, letting other libraries teach the
//! differentiator new functions.

use std::{
    collections::BTreeMap,
    sync::{OnceLock, RwLock},
};

use sim_kernel::Symbol;
use sim_lib_numbers_cas::CasExpr;

/// A custom differentiation rule: maps an operator's arguments and the variable
/// of differentiation to a derivative tree, or `None` to decline.
pub type DiffRule = Box<dyn Fn(&[CasExpr], &Symbol) -> Option<CasExpr> + Send + Sync>;

/// A registry of per-operator differentiation rules.
///
/// # Examples
///
/// ```
/// use sim_kernel::Symbol;
/// use sim_lib_numbers_cas::CasExpr;
/// use sim_lib_numbers_cas_diff::CasDiffRegistry;
///
/// let mut registry = CasDiffRegistry::new();
/// registry.register_rule(
///     Symbol::new("id"),
///     Box::new(|args: &[CasExpr], _var: &Symbol| args.first().cloned()),
/// );
///
/// let out = registry.apply(
///     &Symbol::new("id"),
///     &[CasExpr::Var(Symbol::new("x"))],
///     &Symbol::new("x"),
/// );
/// assert!(matches!(out, Some(CasExpr::Var(_))));
/// // An unregistered operator yields `None`.
/// assert!(registry.apply(&Symbol::new("nope"), &[], &Symbol::new("x")).is_none());
/// ```
#[derive(Default)]
pub struct CasDiffRegistry {
    /// The registered rules, keyed by operator symbol.
    pub rules: BTreeMap<Symbol, DiffRule>,
}

impl CasDiffRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `rule` for `symbol`, returning any rule it replaced.
    pub fn register_rule(&mut self, symbol: Symbol, rule: DiffRule) -> Option<DiffRule> {
        self.rules.insert(symbol, rule)
    }

    /// Apply the rule registered for `symbol`, if any, to `args` and `var`.
    pub fn apply(&self, symbol: &Symbol, args: &[CasExpr], var: &Symbol) -> Option<CasExpr> {
        self.rules.get(symbol).and_then(|rule| rule(args, var))
    }
}

static REGISTRY: OnceLock<RwLock<CasDiffRegistry>> = OnceLock::new();

/// Access the process-global differentiation-rule registry.
pub fn global_diff_registry() -> &'static RwLock<CasDiffRegistry> {
    REGISTRY.get_or_init(|| RwLock::new(CasDiffRegistry::new()))
}

/// Register `rule` for `symbol` in the global registry, returning any rule it
/// replaced.
pub fn register_diff_rule(symbol: Symbol, rule: DiffRule) -> Option<DiffRule> {
    global_diff_registry()
        .write()
        .expect("CAS diff registry should not be poisoned")
        .register_rule(symbol, rule)
}

pub(crate) fn apply_registered_rule(
    symbol: &Symbol,
    args: &[CasExpr],
    var: &Symbol,
) -> Option<CasExpr> {
    global_diff_registry()
        .read()
        .expect("CAS diff registry should not be poisoned")
        .apply(symbol, args, var)
}
