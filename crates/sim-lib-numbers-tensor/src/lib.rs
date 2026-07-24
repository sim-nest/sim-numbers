#![forbid(unsafe_code)]
#![allow(deprecated)]
#![deny(missing_docs)]

//! The n-dimensional tensor number domain: the uniform `Tensor` value, its
//! domain registration and constructors (`tensor`, `vec`, `mat`, ...), and the
//! `SpecTensor` interface that specialized element-type backends plug into.
//! Tensor shape and dtype stay canonical while the open [`TensorStorage`]
//! contract supplies boxed host or loadable resident storage. Observation
//! through [`Tensor::cell`], [`Tensor::cells`], and [`Tensor::materialize`] is
//! checked so encoding and projection report resident readback failures.

mod implementation;
mod spec;

pub use implementation::{
    BoxedTensorStorage, CpuTensorExecutor, SubmissionEvidence, Tensor, TensorCell, TensorExecError,
    TensorExecution, TensorExecutor, TensorExecutorCard, TensorLocation, TensorMeta,
    TensorNumbersLib, TensorOp, TensorRequest, TensorSite, TensorStorage, TypedTensorStorage,
    active_tensor_executor, add_op_symbol, build_scalar_tensor_value, build_tensor_value,
    cast_op_symbol, cast_symbol, cast_tensor, cast_tensor_value, div_op_symbol,
    execute_tensor_binary_op, execute_tensor_request, execute_tensor_unary_op,
    flatten_tensor_scalar_cells, index_op_symbol, map_op_symbol, mat_op_symbol, mul_op_symbol,
    neg_op_symbol, number_domain, pow_op_symbol, rem_op_symbol, reshape_op_symbol,
    scalar_op_symbol, slice_op_symbol, sub_op_symbol, tensor_dtype, tensor_execute_capability,
    tensor_executor_symbol, tensor_op_symbol, tensor_site_symbol, tensor_value_class_symbol,
    tensor_value_ref, vec_op_symbol,
};
pub use sim_lib_numbers_core::domains;
pub use spec::{
    MAX_TENSOR_CELLS, SpecTensor, SpecTensorDescriptor, bounded_element_count,
    checked_element_count, element_count, number_literal_for_tensor_cell, parse_bf16_literal_cell,
    parse_complex_literal_cell, parse_f16_literal_cell, parse_f32_literal_cell,
    parse_f64_literal_cell, parse_i64_literal_cell, parse_rational_literal_cell,
    spec_tensor_descriptor_value, spec_tensor_symbol,
};

/// Cookbook recipes for this domain, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
