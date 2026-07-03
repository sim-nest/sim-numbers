#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! The n-dimensional tensor number domain: the uniform `Tensor` value, its
//! domain registration and constructors (`tensor`, `vec`, `mat`, ...), and the
//! `SpecTensor` interface that specialized element-type backends plug into.

mod implementation;
mod spec;

pub use implementation::{
    Tensor, TensorNumbersLib, build_scalar_tensor_value, build_tensor_value,
    flatten_tensor_scalar_cells, number_domain, tensor_dtype, tensor_value_class_symbol,
    tensor_value_ref,
};
pub use sim_lib_numbers_core::domains;
pub use spec::{
    SpecTensor, SpecTensorDescriptor, checked_element_count, element_count,
    number_literal_for_tensor_cell, parse_complex_literal_cell, parse_f64_literal_cell,
    parse_i64_literal_cell, parse_rational_literal_cell, spec_tensor_descriptor_value,
    spec_tensor_symbol,
};

#[cfg(test)]
mod tests;
