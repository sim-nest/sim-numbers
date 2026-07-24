//! Implementation of the tensor domain: its value class citizen, domain
//! registration, constructor operations, and the `Tensor` value type.

mod cast;
mod citizen;
mod dimension;
mod domain;
mod elementwise;
mod execution;
mod execution_math_support;
mod execution_ops;
mod function;
mod storage;
mod tensor_site;
mod validation;
mod value;

pub use cast::{cast_symbol, cast_tensor, cast_tensor_value};
pub use citizen::tensor_value_class_symbol;
pub use domain::{TensorNumbersLib, number_domain};
pub use elementwise::{
    add_op_symbol, div_op_symbol, execute_tensor_binary_op, execute_tensor_unary_op, mul_op_symbol,
    neg_op_symbol, pow_op_symbol, rem_op_symbol, sub_op_symbol,
};
pub use execution::{
    CpuTensorExecutor, SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorMeta, TensorOp, TensorRequest, active_tensor_executor,
    cast_op_symbol, execute_tensor_request, index_op_symbol, map_op_symbol, mat_op_symbol,
    reshape_op_symbol, scalar_op_symbol, slice_op_symbol, tensor_execute_capability,
    tensor_executor_symbol, tensor_op_symbol, tensor_site_symbol, vec_op_symbol,
};
pub use execution_ops::{
    cos_op_symbol, dot_op_symbol, execute_tensor_dot, execute_tensor_matmul, execute_tensor_norm,
    execute_tensor_reduction, execute_tensor_transcendental, execute_tensor_transpose,
    exp_op_symbol, matmul_exec_op_symbol, max_op_symbol, min_op_symbol, norm_op_symbol,
    sin_op_symbol, sqrt_op_symbol, sum_op_symbol, tensor_executor_math_op_symbols,
    transpose_exec_op_symbol,
};
pub use storage::{
    BoxedTensorStorage, TensorCell, TensorLocation, TensorStorage, TypedTensorStorage,
};
pub use tensor_site::TensorSite;
pub use value::{
    Tensor, build_scalar_tensor_value, build_tensor_value, flatten_tensor_scalar_cells,
    tensor_dtype, tensor_value_ref,
};
