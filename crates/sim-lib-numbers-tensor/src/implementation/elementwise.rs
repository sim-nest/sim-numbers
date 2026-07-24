//! Element-wise tensor operation request construction and CPU semantics.

use sim_kernel::{Cx, Error, Result, Symbol, Value};

use crate::spec::bounded_element_count;

use super::{
    execution::{TensorExecError, TensorMeta, TensorOp, TensorRequest, execute_tensor_request},
    value::Tensor,
};

/// Open operation symbol for element-wise tensor addition.
pub fn add_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/add")
}

/// Open operation symbol for element-wise tensor subtraction.
pub fn sub_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/sub")
}

/// Open operation symbol for element-wise tensor multiplication.
pub fn mul_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/mul")
}

/// Open operation symbol for element-wise tensor division.
pub fn div_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/div")
}

/// Open operation symbol for element-wise tensor remainder.
pub fn rem_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/rem")
}

/// Open operation symbol for element-wise tensor exponentiation.
pub fn pow_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/pow")
}

/// Open operation symbol for element-wise tensor negation.
pub fn neg_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/neg")
}

pub(crate) fn tensor_elementwise_op_symbols() -> Vec<Symbol> {
    vec![
        add_op_symbol(),
        sub_op_symbol(),
        mul_op_symbol(),
        div_op_symbol(),
        rem_op_symbol(),
        pow_op_symbol(),
        neg_op_symbol(),
    ]
}

/// Runs a binary tensor operation through the active executor, or CPU when no
/// executor is bound in the environment.
pub fn execute_tensor_binary_op(
    cx: &mut Cx,
    operator: Symbol,
    left: &Tensor,
    right: &Tensor,
) -> Result<Tensor> {
    let op_symbol = binary_tensor_op_symbol(&operator)
        .ok_or_else(|| Error::Eval(format!("unsupported tensor binary operator {operator}")))?;
    let output = binary_output_meta(cx, &operator, left, right)?;
    let op = TensorOp::without_attributes(cx, op_symbol)?;
    execute_tensor_request(
        cx,
        TensorRequest::new(op, vec![left.clone(), right.clone()], output),
    )
}

/// Runs a unary tensor operation through the active executor, or CPU when no
/// executor is bound in the environment.
pub fn execute_tensor_unary_op(cx: &mut Cx, operator: Symbol, tensor: &Tensor) -> Result<Tensor> {
    let op_symbol = unary_tensor_op_symbol(&operator)
        .ok_or_else(|| Error::Eval(format!("unsupported tensor unary operator {operator}")))?;
    let output = TensorMeta::new(tensor.shape().to_vec(), tensor.dtype().clone());
    let op = TensorOp::without_attributes(cx, op_symbol)?;
    execute_tensor_request(cx, TensorRequest::new(op, vec![tensor.clone()], output))
}

pub(crate) fn is_elementwise_binary_op(symbol: &Symbol) -> bool {
    binary_math_operator(symbol).is_some()
}

pub(crate) fn is_elementwise_unary_op(symbol: &Symbol) -> bool {
    unary_math_operator(symbol).is_some()
}

pub(crate) fn execute_elementwise_binary_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let operator = binary_math_operator(&request.operation.symbol).ok_or_else(|| {
        TensorExecError::unsupported(
            request.operation.symbol.clone(),
            "unknown element-wise binary operation",
        )
    })?;
    let [left, right] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "element-wise binary operation expects exactly two tensor inputs",
        ));
    };
    let shape = broadcast_shape(left.shape(), right.shape()).map_err(TensorExecError::from)?;
    if shape != request.output.shape() {
        return Err(TensorExecError::shape(format!(
            "element-wise output shape {:?} did not match {:?}",
            shape,
            request.output.shape()
        )));
    }
    bounded_element_count(&shape).map_err(TensorExecError::from)?;
    let mut cells = Vec::with_capacity(bounded_element_count(&shape).unwrap_or(0));
    for coord in Tensor::coordinates(&shape) {
        let left_cell = select_cell(left, &coord, &shape).map_err(TensorExecError::from)?;
        let right_cell = select_cell(right, &coord, &shape).map_err(TensorExecError::from)?;
        cells.push(
            cx.apply_value_number_binary_op(&operator, left_cell, right_cell)
                .map_err(TensorExecError::from)?,
        );
    }
    Tensor::new_checked(cx, shape, request.output.dtype().clone(), cells)
        .map_err(TensorExecError::from)
}

pub(crate) fn execute_elementwise_unary_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let operator = unary_math_operator(&request.operation.symbol).ok_or_else(|| {
        TensorExecError::unsupported(
            request.operation.symbol.clone(),
            "unknown element-wise unary operation",
        )
    })?;
    let [tensor] = request.inputs.as_ref() else {
        return Err(TensorExecError::invalid(
            "element-wise unary operation expects exactly one tensor input",
        ));
    };
    if tensor.shape() != request.output.shape() {
        return Err(TensorExecError::shape(format!(
            "element-wise output shape {:?} did not match {:?}",
            tensor.shape(),
            request.output.shape()
        )));
    }
    let source = tensor.cells().map_err(TensorExecError::from)?;
    let mut cells = Vec::with_capacity(source.len());
    for cell in source.iter().cloned() {
        cells.push(
            cx.apply_value_number_unary_op(&operator, cell)
                .map_err(TensorExecError::from)?,
        );
    }
    Tensor::new_checked(
        cx,
        tensor.shape().to_vec(),
        request.output.dtype().clone(),
        cells,
    )
    .map_err(TensorExecError::from)
}

fn binary_output_meta(
    cx: &mut Cx,
    operator: &Symbol,
    left: &Tensor,
    right: &Tensor,
) -> Result<TensorMeta> {
    let shape = broadcast_shape(left.shape(), right.shape())?;
    let len = bounded_element_count(&shape)?;
    let dtype = if len == 0 {
        left.dtype().clone()
    } else {
        let coord = vec![0; shape.len()];
        let left_cell = select_cell(left, &coord, &shape)?;
        let right_cell = select_cell(right, &coord, &shape)?;
        let sample = cx.apply_value_number_binary_op(operator, left_cell, right_cell)?;
        scalar_domain(cx, &sample)?
    };
    Ok(TensorMeta::new(shape, dtype))
}

fn scalar_domain(cx: &mut Cx, value: &Value) -> Result<Symbol> {
    let Some(number) = cx.number_value_ref(value.clone())? else {
        return Err(Error::Eval(
            "element-wise tensor operation produced a non-number cell".to_owned(),
        ));
    };
    Ok(number.domain)
}

fn binary_tensor_op_symbol(operator: &Symbol) -> Option<Symbol> {
    if *operator == Symbol::qualified("math", "add") {
        Some(add_op_symbol())
    } else if *operator == Symbol::qualified("math", "sub") {
        Some(sub_op_symbol())
    } else if *operator == Symbol::qualified("math", "mul") {
        Some(mul_op_symbol())
    } else if *operator == Symbol::qualified("math", "div") {
        Some(div_op_symbol())
    } else if *operator == Symbol::qualified("math", "rem") {
        Some(rem_op_symbol())
    } else if *operator == Symbol::qualified("math", "pow") {
        Some(pow_op_symbol())
    } else {
        None
    }
}

fn unary_tensor_op_symbol(operator: &Symbol) -> Option<Symbol> {
    (*operator == Symbol::qualified("math", "neg")).then(neg_op_symbol)
}

fn binary_math_operator(op_symbol: &Symbol) -> Option<Symbol> {
    if *op_symbol == add_op_symbol() {
        Some(Symbol::qualified("math", "add"))
    } else if *op_symbol == sub_op_symbol() {
        Some(Symbol::qualified("math", "sub"))
    } else if *op_symbol == mul_op_symbol() {
        Some(Symbol::qualified("math", "mul"))
    } else if *op_symbol == div_op_symbol() {
        Some(Symbol::qualified("math", "div"))
    } else if *op_symbol == rem_op_symbol() {
        Some(Symbol::qualified("math", "rem"))
    } else if *op_symbol == pow_op_symbol() {
        Some(Symbol::qualified("math", "pow"))
    } else {
        None
    }
}

fn unary_math_operator(op_symbol: &Symbol) -> Option<Symbol> {
    (*op_symbol == neg_op_symbol()).then(|| Symbol::qualified("math", "neg"))
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut out = Vec::with_capacity(rank);
    for axis in 0..rank {
        let left_dim = *left
            .get(left.len().wrapping_sub(rank - axis))
            .unwrap_or(&1usize);
        let right_dim = *right
            .get(right.len().wrapping_sub(rank - axis))
            .unwrap_or(&1usize);
        if left_dim == right_dim {
            out.push(left_dim);
        } else if left_dim == 1 {
            out.push(right_dim);
        } else if right_dim == 1 {
            out.push(left_dim);
        } else {
            return Err(Error::Eval(format!(
                "cannot broadcast tensor shapes {left:?} and {right:?}"
            )));
        }
    }
    Ok(out)
}

fn select_cell(tensor: &Tensor, coord: &[usize], result_shape: &[usize]) -> Result<Value> {
    let shape = tensor.shape();
    let rank_gap = result_shape.len().saturating_sub(shape.len());
    let mut local = Vec::with_capacity(shape.len());
    for (axis, dim) in shape.iter().enumerate() {
        let result_axis = axis + rank_gap;
        let coord_value = coord
            .get(result_axis)
            .copied()
            .ok_or_else(|| Error::Eval("tensor broadcast axis mismatch".to_owned()))?;
        local.push(if *dim == 1 { 0 } else { coord_value });
    }
    let flat = Tensor::flat_offset(shape, &local)?;
    tensor.cell(flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_extent_broadcast_does_not_expand_to_one() {
        assert_eq!(broadcast_shape(&[0], &[1]).unwrap(), vec![0]);
        assert_eq!(broadcast_shape(&[1, 0], &[3, 1]).unwrap(), vec![3, 0]);
    }

    #[test]
    fn incompatible_shapes_fail_closed() {
        let err = broadcast_shape(&[2, 3], &[2]).unwrap_err();
        assert!(err.to_string().contains("cannot broadcast"));
    }

    #[test]
    fn tensor_op_symbols_map_to_scalar_symbols() {
        assert_eq!(
            binary_math_operator(&add_op_symbol()),
            Some(Symbol::qualified("math", "add"))
        );
        assert_eq!(
            unary_math_operator(&neg_op_symbol()),
            Some(Symbol::qualified("math", "neg"))
        );
    }
}
