#![forbid(unsafe_code)]

//! Internal layout of the `numbers/rational` domain: the domain `Lib`
//! (`domain`), integer-component helpers (`integer`), the literal shape
//! (`literal`), the arithmetic rules and promotions (`ops`), decimal and
//! rational-text parsing (`parse`), and the `Rational` value (`value`).

mod domain;
mod integer;
mod literal;
mod ops;
mod parse;
#[cfg(test)]
mod tests;
mod value;

pub use domain::{
    RationalNumberDomain, RationalNumbersLib, add_symbol, div_symbol, f64_domain,
    literal_class_symbol, literal_instance_shape_symbol, mul_symbol, number_domain, pow_symbol,
    rational_value_class_symbol, sub_symbol,
};
pub use parse::{
    f64_decimal_to_rational, f64_decimal_to_rational_checked, parse_rational_parts,
    parse_rational_parts_checked,
};
pub use value::Rational;
