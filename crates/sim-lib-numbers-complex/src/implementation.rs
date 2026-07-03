#![forbid(unsafe_code)]

//! Internal layout of the `numbers/complex` domain: the domain `Lib` and its
//! symbols (`literal`), the arithmetic rules and promotions (`ops`), the
//! number-value shape (`surface`), and the `ComplexValue` object (`value`).

mod literal;
mod ops;
mod surface;
mod value;

pub use literal::{
    ComplexNumberDomain, ComplexNumbersLib, add_symbol, div_symbol, f64_domain, i64_domain,
    literal_class_symbol, literal_instance_shape_symbol, mul_symbol, neg_symbol, number_domain,
    product_symbol, rational_domain, sub_symbol, sum_symbol,
};
pub use ops::{canonical_complex, canonical_f64, parse_complex_literal, parse_rational_as_f64};
pub use value::{ComplexValue, complex_value, complex_value_class_symbol};
