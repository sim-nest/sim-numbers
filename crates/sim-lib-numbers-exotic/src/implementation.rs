//! Internal layout of the `numbers/cf` continued-fraction domain: the domain
//! `Lib` and builtin constants (`domain`), the `as-f64` function (`function`),
//! the literal shape (`literal`), and the `ContinuedFraction` value (`value`).

mod domain;
mod function;
mod literal;
mod value;

pub use domain::{ExoticNumbersLib, builtin_symbol, number_domain};
pub use function::as_f64_symbol;
pub use value::{ContinuedFraction, ExoticReal};
