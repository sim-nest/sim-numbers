//! Internal layout of the CAS differentiation library: the `diff` engine, the
//! `integrate` engine and its function object, the `function` `Lib` wiring, and
//! the extensible differentiation-rule `registry`.

mod diff;
mod func_surface;
mod function;
mod integrate;
mod integrate_function;
mod registry;

pub use diff::{diff_cas, diff_symbol};
pub use function::CasDiffLib;
pub use integrate::{integrate_cas, integrate_sym_symbol};
pub use registry::{
    CasDiffRegistry, DiffRule, global_diff_registry, override_diff_rule, register_diff_rule,
};
