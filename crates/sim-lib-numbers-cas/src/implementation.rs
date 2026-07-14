//! Internal layout of the `numbers/cas` domain: the symbolic value class
//! (`citizen`), the domain `Lib` (`domain`), the CAS functions (`function`),
//! the literal shape/class (`literal`), the simplifier (`simplify`), and the
//! `CasExpr` tree and conversions (`value`).

mod citizen;
mod domain;
mod function;
mod literal;
mod simplify;
mod value;

pub use citizen::cas_value_class_symbol;
pub use domain::{CasNumbersLib, cas_domain_symbol};
pub use function::{cas_simplify_symbol, cas_var_symbol, extract_symbolish};
pub use simplify::{expr_to_cas_expr, literal_number, simplify_expr, value_to_cas_expr};
pub use value::{CasExpr, cas_expr_to_surface_expr, cas_expr_to_value, free_vars};
