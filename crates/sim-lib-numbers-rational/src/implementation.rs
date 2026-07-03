#![forbid(unsafe_code)]

//! Internal layout of the `numbers/rational` domain: the domain `Lib`
//! (`domain`), integer-component helpers (`integer`), the literal shape
//! (`literal`), the arithmetic rules and promotions (`ops`), and the `Rational`
//! value (`value`).

mod domain;
mod integer;
mod literal;
mod ops;
#[cfg(test)]
mod tests;
mod value;

pub use domain::{
    RationalNumberDomain, RationalNumbersLib, add_symbol, div_symbol, f64_domain,
    literal_class_symbol, literal_instance_shape_symbol, mul_symbol, number_domain, pow_symbol,
    rational_value_class_symbol, sub_symbol,
};
pub use ops::{f64_decimal_to_rational, parse_rational_parts};
pub use value::Rational;
