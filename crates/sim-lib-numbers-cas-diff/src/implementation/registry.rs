//! The extensible differentiation-rule registry: a process-global map from an
//! operator symbol to a custom `diff` rule, letting other libraries teach the
//! differentiator new functions.

use std::{
    collections::BTreeMap,
    sync::{OnceLock, RwLock},
};

use sim_kernel::{Error, Result, Symbol};
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
/// )
/// .unwrap();
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

    /// Register `rule` for `symbol`.
    ///
    /// Returns an error when a rule is already registered for the same symbol.
    /// Use [`Self::override_rule`] when replacement is intentional.
    pub fn register_rule(&mut self, symbol: Symbol, rule: DiffRule) -> Result<()> {
        if self.rules.contains_key(&symbol) {
            return Err(Error::Eval(format!(
                "CAS diff rule for operator {symbol} is already registered"
            )));
        }
        self.rules.insert(symbol, rule);
        Ok(())
    }

    /// Replace the rule for `symbol`, returning any rule that was already there.
    pub fn override_rule(&mut self, symbol: Symbol, rule: DiffRule) -> Option<DiffRule> {
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

/// Register `rule` for `symbol` in the global registry.
///
/// # Errors
///
/// Returns an error if the global registry lock is poisoned or a rule already
/// exists for `symbol`.
pub fn register_diff_rule(symbol: Symbol, rule: DiffRule) -> Result<()> {
    global_diff_registry()
        .write()
        .map_err(|_| Error::PoisonedLock("CAS diff registry"))?
        .register_rule(symbol, rule)
}

/// Replace the global rule for `symbol`, returning any rule that was already
/// registered.
pub fn override_diff_rule(symbol: Symbol, rule: DiffRule) -> Option<DiffRule> {
    global_diff_registry()
        .write()
        .expect("CAS diff registry should not be poisoned")
        .override_rule(symbol, rule)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_rule() -> DiffRule {
        Box::new(|args: &[CasExpr], _var: &Symbol| args.first().cloned())
    }

    fn constant_rule(symbol: &'static str) -> DiffRule {
        Box::new(move |_args: &[CasExpr], _var: &Symbol| Some(CasExpr::Var(Symbol::new(symbol))))
    }

    #[test]
    fn duplicate_rule_registration_fails_closed() {
        let mut registry = CasDiffRegistry::new();
        registry
            .register_rule(Symbol::new("custom"), identity_rule())
            .unwrap();

        let err = registry
            .register_rule(Symbol::new("custom"), identity_rule())
            .unwrap_err();

        assert!(err.to_string().contains("already registered"), "{err}");
    }

    #[test]
    fn explicit_override_replaces_existing_rule() {
        let mut registry = CasDiffRegistry::new();
        let symbol = Symbol::new("custom");
        registry
            .register_rule(symbol.clone(), constant_rule("before"))
            .unwrap();

        let replaced = registry.override_rule(symbol.clone(), constant_rule("after"));
        let out = registry.apply(&symbol, &[], &Symbol::new("x"));

        assert!(replaced.is_some());
        assert!(matches!(out, Some(CasExpr::Var(value)) if value == Symbol::new("after")));
    }
}
