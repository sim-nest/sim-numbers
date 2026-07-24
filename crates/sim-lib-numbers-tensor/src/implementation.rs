//! Implementation of the tensor domain: its value class citizen, domain
//! registration, constructor operations, and the `Tensor` value type.

mod cast;
mod citizen;
mod dimension;
mod domain;
mod execution;
mod function;
mod storage;
mod tensor_site;
mod validation;
mod value;

pub use cast::{cast_symbol, cast_tensor, cast_tensor_value};
pub use citizen::tensor_value_class_symbol;
pub use domain::{TensorNumbersLib, number_domain};
pub use execution::{
    CpuTensorExecutor, SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorMeta, TensorOp, TensorRequest, cast_op_symbol, index_op_symbol,
    map_op_symbol, mat_op_symbol, reshape_op_symbol, scalar_op_symbol, slice_op_symbol,
    tensor_execute_capability, tensor_executor_symbol, tensor_op_symbol, tensor_site_symbol,
    vec_op_symbol,
};
pub use storage::{
    BoxedTensorStorage, TensorCell, TensorLocation, TensorStorage, TypedTensorStorage,
};
pub use tensor_site::TensorSite;
pub use value::{
    Tensor, build_scalar_tensor_value, build_tensor_value, flatten_tensor_scalar_cells,
    tensor_dtype, tensor_value_ref,
};
