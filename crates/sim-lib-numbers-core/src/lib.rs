#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Shared substrate for scalar number domains: the number-value shape and
//! browse table ([`value_shape`]), the shared number-literal shape and class
//! ([`literal`]), and the scalar-domain spec, literal matcher, and op-loop
//! installer ([`scalar`]). The canonical number-domain symbol registry and
//! promotion-lattice documentation live in [`domains`].

pub mod domains;
pub mod literal;
pub mod scalar;
pub mod value_shape;

pub use literal::{
    NumberLiteralClass, NumberLiteralShape, class_surface_or_symbol, shape_surface_or_symbol,
};
pub use scalar::{
    DomainLiteralMatcher, ScalarBinaryOp, ScalarDomainSpec, ScalarLiteralMatcher, ScalarOps,
    ScalarReductionOp, ScalarUnaryOp, install_scalar_ops, number_domain_class_stub,
};
pub use value_shape::{
    DomainNumberValueShape, NumberDomainTableSpec, assert_value_shape_symbol, number_domain_table,
    value_shape_symbol,
};

#[cfg(test)]
mod tests;
