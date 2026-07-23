//! Implementation of the tensor domain: its value class citizen, domain
//! registration, constructor operations, and the `Tensor` value type.

mod citizen;
mod dimension;
mod domain;
mod function;
mod storage;
mod validation;
mod value;

pub use citizen::tensor_value_class_symbol;
pub use domain::{TensorNumbersLib, number_domain};
pub use storage::{BoxedTensorStorage, TensorLocation, TensorStorage};
pub use value::{
    Tensor, build_scalar_tensor_value, build_tensor_value, flatten_tensor_scalar_cells,
    tensor_dtype, tensor_value_ref,
};
