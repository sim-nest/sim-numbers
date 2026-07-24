//! Executor-routed tensor reductions, linear algebra, and f32 scalar functions.
//!
//! These operations use checked [`TensorRequest`](super::execution::TensorRequest)
//! values so host, GPU, or remote providers see the same operation symbols as the
//! CPU fallback. Reductions in this module reduce the whole tensor to one rank-0
//! scalar. Empty sum and norm return zero; empty min and max fail closed. Floating
//! tensors follow IEEE 754 propagation for NaN, infinity, and signed zero. Half
//! tensors are widened to `f32` for reductions and transcendentals because SIM's
//! scalar half domains are storage formats rather than arithmetic domains.

use sim_kernel::{Cx, Error, Result, Symbol};

use super::{
    execution::{TensorExecError, TensorMeta, TensorOp, TensorRequest, execute_tensor_request},
    execution_math_support::{
        ProductSpec, float_output_dtype, matches_tensor_transcendental, matmul_output_shape,
        norm_value, reduce_min_max, reduce_sum, reduction_output_dtype, reduction_pair_dtype,
        scalar_tensor, sum_products, tensor_from_cells, transcendental_cell,
    },
    value::Tensor,
};

/// Open operation symbol for reducing all tensor cells with addition.
pub fn sum_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/sum")
}

/// Open operation symbol for reducing all tensor cells to the minimum value.
pub fn min_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/min")
}

/// Open operation symbol for reducing all tensor cells to the maximum value.
pub fn max_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/max")
}

/// Open operation symbol for Euclidean norm over all tensor cells.
pub fn norm_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/norm")
}

/// Open operation symbol for matrix transpose.
pub fn transpose_exec_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/transpose")
}

/// Open operation symbol for vector dot product.
pub fn dot_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/dot")
}

/// Open operation symbol for matrix multiplication.
pub fn matmul_exec_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/matmul")
}

/// Open operation symbol for element-wise square root.
pub fn sqrt_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/sqrt")
}

/// Open operation symbol for element-wise exponential.
pub fn exp_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/exp")
}

/// Open operation symbol for element-wise sine.
pub fn sin_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/sin")
}

/// Open operation symbol for element-wise cosine.
pub fn cos_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/cos")
}

/// Returns the operation symbols accepted by executor math providers.
pub fn tensor_executor_math_op_symbols() -> Vec<Symbol> {
    vec![
        sum_op_symbol(),
        min_op_symbol(),
        max_op_symbol(),
        norm_op_symbol(),
        transpose_exec_op_symbol(),
        dot_op_symbol(),
        matmul_exec_op_symbol(),
        sqrt_op_symbol(),
        exp_op_symbol(),
        sin_op_symbol(),
        cos_op_symbol(),
    ]
}

pub(crate) fn is_tensor_executor_math_op(symbol: &Symbol) -> bool {
    tensor_executor_math_op_symbols()
        .iter()
        .any(|candidate| candidate == symbol)
}

/// Runs a whole-tensor sum, min, or max through the active executor.
pub fn execute_tensor_reduction(cx: &mut Cx, operator: Symbol, tensor: &Tensor) -> Result<Tensor> {
    let output_dtype = reduction_output_dtype(tensor);
    let op = TensorOp::without_attributes(cx, operator)?;
    execute_tensor_request(
        cx,
        TensorRequest::new(
            op,
            vec![tensor.clone()],
            TensorMeta::new(Vec::new(), output_dtype),
        ),
    )
}

/// Runs a Euclidean whole-tensor norm through the active executor.
pub fn execute_tensor_norm(cx: &mut Cx, tensor: &Tensor) -> Result<Tensor> {
    let op = TensorOp::without_attributes(cx, norm_op_symbol())?;
    execute_tensor_request(
        cx,
        TensorRequest::new(
            op,
            vec![tensor.clone()],
            TensorMeta::new(Vec::new(), float_output_dtype(tensor)),
        ),
    )
}

/// Runs a rank-2 transpose through the active executor.
pub fn execute_tensor_transpose(cx: &mut Cx, tensor: &Tensor) -> Result<Tensor> {
    let [rows, cols] = tensor.shape() else {
        return Err(Error::Eval("transpose expects a rank-2 tensor".to_owned()));
    };
    let op = TensorOp::without_attributes(cx, transpose_exec_op_symbol())?;
    execute_tensor_request(
        cx,
        TensorRequest::new(
            op,
            vec![tensor.clone()],
            TensorMeta::new(vec![*cols, *rows], tensor.dtype().clone()),
        ),
    )
}

/// Runs a vector dot product through the active executor.
pub fn execute_tensor_dot(cx: &mut Cx, left: &Tensor, right: &Tensor) -> Result<Tensor> {
    if left.shape().len() != 1 || right.shape().len() != 1 {
        return Err(Error::Eval("dot expects two rank-1 tensors".to_owned()));
    }
    if left.shape() != right.shape() {
        return Err(Error::Eval(
            "dot expects vectors with matching lengths".to_owned(),
        ));
    }
    let op = TensorOp::without_attributes(cx, dot_op_symbol())?;
    execute_tensor_request(
        cx,
        TensorRequest::new(
            op,
            vec![left.clone(), right.clone()],
            TensorMeta::new(Vec::new(), reduction_pair_dtype(left, right)),
        ),
    )
}

/// Runs vector or matrix multiplication through the active executor.
pub fn execute_tensor_matmul(cx: &mut Cx, left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let shape = matmul_output_shape(left.shape(), right.shape())?;
    let op = TensorOp::without_attributes(cx, matmul_exec_op_symbol())?;
    execute_tensor_request(
        cx,
        TensorRequest::new(
            op,
            vec![left.clone(), right.clone()],
            TensorMeta::new(shape, reduction_pair_dtype(left, right)),
        ),
    )
}

/// Runs an f32/f64/half element-wise transcendental through the active executor.
pub fn execute_tensor_transcendental(
    cx: &mut Cx,
    operator: Symbol,
    tensor: &Tensor,
) -> Result<Tensor> {
    if !matches_tensor_transcendental(&operator) {
        return Err(Error::Eval(format!(
            "unsupported tensor transcendental operation {operator}"
        )));
    }
    let op = TensorOp::without_attributes(cx, operator)?;
    execute_tensor_request(
        cx,
        TensorRequest::new(
            op,
            vec![tensor.clone()],
            TensorMeta::new(tensor.shape().to_vec(), float_output_dtype(tensor)),
        ),
    )
}

pub(crate) fn execute_tensor_math_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let operation = &request.operation.symbol;
    if *operation == sum_op_symbol()
        || *operation == min_op_symbol()
        || *operation == max_op_symbol()
    {
        execute_reduction_request(cx, request)
    } else if *operation == norm_op_symbol() {
        execute_norm_request(cx, request)
    } else if *operation == transpose_exec_op_symbol() {
        execute_transpose_request(cx, request)
    } else if *operation == dot_op_symbol() {
        execute_dot_request(cx, request)
    } else if *operation == matmul_exec_op_symbol() {
        execute_matmul_request(cx, request)
    } else if matches_tensor_transcendental(operation) {
        execute_transcendental_request(cx, request)
    } else {
        Err(TensorExecError::unsupported(
            operation.clone(),
            "unknown tensor math operation",
        ))
    }
}

fn execute_reduction_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "tensor reduction expects exactly one tensor input",
        ));
    };
    let value = if request.operation.symbol == sum_op_symbol() {
        reduce_sum(cx, tensor)?
    } else if request.operation.symbol == min_op_symbol() {
        reduce_min_max(cx, tensor, false)?
    } else {
        reduce_min_max(cx, tensor, true)?
    };
    scalar_tensor(cx, request.output.dtype().clone(), value)
}

fn execute_norm_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "tensor norm expects exactly one tensor input",
        ));
    };
    let value = norm_value(cx, tensor)?;
    scalar_tensor(cx, request.output.dtype().clone(), value)
}

fn execute_transpose_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "transpose expects exactly one tensor input",
        ));
    };
    let [rows, cols] = tensor.shape() else {
        return Err(TensorExecError::invalid("transpose expects rank-2 input"));
    };
    let mut out = Vec::with_capacity(tensor.len());
    for col in 0..*cols {
        for row in 0..*rows {
            out.push(
                tensor
                    .cell(row * cols + col)
                    .map_err(TensorExecError::from)?,
            );
        }
    }
    tensor_from_cells(
        cx,
        request.output.shape().to_vec(),
        request.output.dtype().clone(),
        out,
    )
}

fn execute_dot_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let [left, right] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "dot expects exactly two tensor inputs",
        ));
    };
    if left.shape().len() != 1 || right.shape().len() != 1 || left.shape() != right.shape() {
        return Err(TensorExecError::invalid(
            "dot expects matching rank-1 tensor inputs",
        ));
    }
    let value = sum_products(
        cx,
        ProductSpec {
            left,
            right,
            left_start: 0,
            right_start: 0,
            count: left.shape()[0],
            left_stride: 1,
            right_stride: 1,
        },
    )?;
    scalar_tensor(cx, request.output.dtype().clone(), value)
}

fn execute_matmul_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let [left, right] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "matmul expects exactly two tensor inputs",
        ));
    };
    let out_shape =
        matmul_output_shape(left.shape(), right.shape()).map_err(TensorExecError::from)?;
    if out_shape != request.output.shape() {
        return Err(TensorExecError::shape(format!(
            "matmul output shape {:?} did not match {:?}",
            out_shape,
            request.output.shape()
        )));
    }
    match (left.shape(), right.shape()) {
        ([n], [m]) if n == m => {
            let value = sum_products(
                cx,
                ProductSpec {
                    left,
                    right,
                    left_start: 0,
                    right_start: 0,
                    count: *n,
                    left_stride: 1,
                    right_stride: 1,
                },
            )?;
            scalar_tensor(cx, request.output.dtype().clone(), value)
        }
        ([rows, inner_left], [inner_right, cols]) if inner_left == inner_right => {
            let mut out = Vec::with_capacity(rows * cols);
            for row in 0..*rows {
                for col in 0..*cols {
                    out.push(sum_products(
                        cx,
                        ProductSpec {
                            left,
                            right,
                            left_start: row * inner_left,
                            right_start: col,
                            count: *inner_left,
                            left_stride: 1,
                            right_stride: *cols,
                        },
                    )?);
                }
            }
            tensor_from_cells(cx, out_shape, request.output.dtype().clone(), out)
        }
        ([rows, inner_left], [inner_right]) if inner_left == inner_right => {
            let mut out = Vec::with_capacity(*rows);
            for row in 0..*rows {
                out.push(sum_products(
                    cx,
                    ProductSpec {
                        left,
                        right,
                        left_start: row * inner_left,
                        right_start: 0,
                        count: *inner_left,
                        left_stride: 1,
                        right_stride: 1,
                    },
                )?);
            }
            tensor_from_cells(cx, out_shape, request.output.dtype().clone(), out)
        }
        ([inner_left], [inner_right, cols]) if inner_left == inner_right => {
            let mut out = Vec::with_capacity(*cols);
            for col in 0..*cols {
                out.push(sum_products(
                    cx,
                    ProductSpec {
                        left,
                        right,
                        left_start: 0,
                        right_start: col,
                        count: *inner_left,
                        left_stride: 1,
                        right_stride: *cols,
                    },
                )?);
            }
            tensor_from_cells(cx, out_shape, request.output.dtype().clone(), out)
        }
        _ => Err(TensorExecError::invalid(
            "matmul supports rank-1 and rank-2 tensors with matching inner dimensions",
        )),
    }
}

fn execute_transcendental_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "transcendental expects exactly one tensor input",
        ));
    };
    let cells = tensor.cells().map_err(TensorExecError::from)?;
    let mut out = Vec::with_capacity(cells.len());
    for cell in cells.iter() {
        out.push(transcendental_cell(
            cx,
            cell,
            &request.operation.symbol,
            request.output.dtype(),
        )?);
    }
    tensor_from_cells(
        cx,
        request.output.shape().to_vec(),
        request.output.dtype().clone(),
        out,
    )
}
