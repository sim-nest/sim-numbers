#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The number-literal codec surface: helpers that build the `numeric-plugin-v1`
//! descriptor symbols and table values advertising a numeric codec or method
//! provider to the runtime.

use sim_kernel::{Factory, Result, Symbol, Value};

/// Build the descriptor symbol `group/name` advertising a numeric codec or
/// method provider under the `numeric-plugin-v1` API.
///
/// # Examples
///
/// ```
/// use sim_kernel::Symbol;
/// use sim_lib_numbers_codec::numeric_plugin_descriptor_symbol;
///
/// assert_eq!(
///     numeric_plugin_descriptor_symbol("numbers/quad", "simpson"),
///     Symbol::qualified("numbers/quad", "simpson"),
/// );
/// ```
pub fn numeric_plugin_descriptor_symbol(group: &str, name: &str) -> Symbol {
    Symbol::qualified(group, name)
}

/// Build the `numeric-plugin-v1` descriptor table for `method`, recording its
/// plugin `kind`, whether it is `adaptive`, and the `provider` domain that
/// supplies it.
pub fn numeric_plugin_descriptor_value(
    factory: &dyn Factory,
    method: Symbol,
    kind: &str,
    adaptive: bool,
    provider: Symbol,
) -> Result<Value> {
    factory.table(vec![
        (
            Symbol::new("kind"),
            factory.string("numeric-plugin".to_owned())?,
        ),
        (Symbol::new("method"), factory.symbol(method)?),
        (Symbol::new("plugin-kind"), factory.string(kind.to_owned())?),
        (Symbol::new("adaptive"), factory.bool(adaptive)?),
        (Symbol::new("provider"), factory.symbol(provider)?),
        (
            Symbol::new("api"),
            factory.string("numeric-plugin-v1".to_owned())?,
        ),
    ])
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sim_kernel::{DefaultFactory, Expr, NoopEvalPolicy};
    use sim_lib_numbers_core::domains;

    use super::{numeric_plugin_descriptor_symbol, numeric_plugin_descriptor_value};

    #[test]
    fn descriptor_symbol_uses_namespaced_group() {
        assert_eq!(
            numeric_plugin_descriptor_symbol("numbers/quad", "simpson"),
            sim_kernel::Symbol::qualified("numbers/quad", "simpson")
        );
    }

    #[test]
    fn descriptor_value_has_stable_table_surface() {
        let factory = DefaultFactory;
        let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
        let value = numeric_plugin_descriptor_value(
            &factory,
            sim_kernel::Symbol::new("rkf45"),
            "ode",
            true,
            domains::rk(),
        )
        .unwrap();
        let Expr::Map(entries) = value.object().as_expr(&mut cx).unwrap() else {
            panic!("descriptor should encode as a map");
        };
        assert!(entries.iter().any(|(key, value)| {
            *key == Expr::Symbol(sim_kernel::Symbol::new("method"))
                && *value == Expr::Symbol(sim_kernel::Symbol::new("rkf45"))
        }));
    }
}
