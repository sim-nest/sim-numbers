//! Shared CPU helpers for executor-routed tensor math.

use std::sync::Arc;

use half::{bf16, f16};
use sim_kernel::{Cx, Error, NumberLiteral, Result, Symbol, Value};
use sim_lib_numbers_core::domains;

use super::{
    execution::TensorExecError,
    value::{Tensor, build_tensor_value},
};
use crate::number_literal_for_tensor_cell;

pub(crate) fn reduce_sum(
    cx: &mut Cx,
    tensor: &Tensor,
) -> std::result::Result<Value, TensorExecError> {
    if tensor.is_empty() {
        return zero_value(cx, &reduction_output_dtype(tensor));
    }
    let source = tensor.cells().map_err(TensorExecError::from)?;
    let mut cells = source.iter().cloned();
    let mut acc = cells
        .next()
        .ok_or_else(|| TensorExecError::invalid("tensor cells disappeared during sum"))?;
    for cell in cells {
        acc = cx
            .apply_value_number_binary_op(&Symbol::qualified("math", "add"), acc, cell)
            .map_err(TensorExecError::from)?;
    }
    Ok(acc)
}

pub(crate) fn reduce_min_max(
    cx: &mut Cx,
    tensor: &Tensor,
    max: bool,
) -> std::result::Result<Value, TensorExecError> {
    let cells = tensor.cells().map_err(TensorExecError::from)?;
    if cells.is_empty() {
        return Err(TensorExecError::invalid("min/max over an empty tensor"));
    }
    let mut best = numeric_cell(cx, &cells[0])?;
    for cell in &cells[1..] {
        let next = numeric_cell(cx, cell)?;
        best = if best.value.is_nan() || next.value.is_nan() {
            NumericCell {
                value: f64::NAN,
                domain: best.domain,
            }
        } else if (max && next.value > best.value) || (!max && next.value < best.value) {
            next
        } else {
            best
        };
    }
    numeric_value(cx, &best.domain, best.value)
}

pub(crate) fn norm_value(
    cx: &mut Cx,
    tensor: &Tensor,
) -> std::result::Result<Value, TensorExecError> {
    if tensor.is_empty() {
        return zero_value(cx, &float_output_dtype(tensor));
    }
    let mut acc = 0.0_f64;
    for cell in tensor.cells().map_err(TensorExecError::from)?.iter() {
        let value = numeric_cell(cx, cell)?.value;
        acc += value * value;
    }
    numeric_value(cx, &float_output_dtype(tensor), acc.sqrt())
}

pub(crate) struct ProductSpec<'a> {
    pub(crate) left: &'a Tensor,
    pub(crate) right: &'a Tensor,
    pub(crate) left_start: usize,
    pub(crate) right_start: usize,
    pub(crate) count: usize,
    pub(crate) left_stride: usize,
    pub(crate) right_stride: usize,
}

pub(crate) fn sum_products(
    cx: &mut Cx,
    spec: ProductSpec<'_>,
) -> std::result::Result<Value, TensorExecError> {
    let mut acc = zero_value(cx, &reduction_pair_dtype(spec.left, spec.right))?;
    for inner in 0..spec.count {
        let left_cell = spec
            .left
            .cell(spec.left_start + inner * spec.left_stride)
            .map_err(TensorExecError::from)?;
        let right_cell = spec
            .right
            .cell(spec.right_start + inner * spec.right_stride)
            .map_err(TensorExecError::from)?;
        let product = cx
            .apply_value_number_binary_op(&Symbol::qualified("math", "mul"), left_cell, right_cell)
            .map_err(TensorExecError::from)?;
        acc = cx
            .apply_value_number_binary_op(&Symbol::qualified("math", "add"), acc, product)
            .map_err(TensorExecError::from)?;
    }
    Ok(acc)
}

pub(crate) fn matmul_output_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    match (left, right) {
        ([n], [m]) if n == m => Ok(Vec::new()),
        ([rows, inner_left], [inner_right, cols]) if inner_left == inner_right => {
            Ok(vec![*rows, *cols])
        }
        ([rows, inner_left], [inner_right]) if inner_left == inner_right => Ok(vec![*rows]),
        ([inner_left], [inner_right, cols]) if inner_left == inner_right => Ok(vec![*cols]),
        _ => Err(Error::Eval(
            "matmul supports rank-1 and rank-2 tensors with matching inner dimensions".to_owned(),
        )),
    }
}

pub(crate) fn scalar_tensor(
    cx: &mut Cx,
    dtype: Symbol,
    value: Value,
) -> std::result::Result<Tensor, TensorExecError> {
    tensor_from_cells(cx, Vec::new(), dtype, vec![value])
}

pub(crate) fn tensor_from_cells(
    cx: &mut Cx,
    shape: Vec<usize>,
    dtype: Symbol,
    cells: Vec<Value>,
) -> std::result::Result<Tensor, TensorExecError> {
    build_tensor_value(cx, shape, Some(dtype), cells)
        .map_err(TensorExecError::from)?
        .object()
        .downcast_ref::<Tensor>()
        .cloned()
        .ok_or_else(|| TensorExecError::invalid("tensor executor produced a non-tensor value"))
}

pub(crate) fn reduction_output_dtype(tensor: &Tensor) -> Symbol {
    if tensor.dtype() == &domains::f16() || tensor.dtype() == &domains::bf16() {
        domains::f32()
    } else {
        tensor.dtype().clone()
    }
}

pub(crate) fn reduction_pair_dtype(left: &Tensor, right: &Tensor) -> Symbol {
    if left.dtype() == right.dtype() {
        reduction_output_dtype(left)
    } else if left.dtype() == &domains::f64() || right.dtype() == &domains::f64() {
        domains::f64()
    } else {
        domains::f32()
    }
}

pub(crate) fn float_output_dtype(tensor: &Tensor) -> Symbol {
    if tensor.dtype() == &domains::f64() {
        domains::f64()
    } else {
        domains::f32()
    }
}

pub(crate) fn matches_tensor_transcendental(symbol: &Symbol) -> bool {
    *symbol == super::execution_ops::sqrt_op_symbol()
        || *symbol == super::execution_ops::exp_op_symbol()
        || *symbol == super::execution_ops::sin_op_symbol()
        || *symbol == super::execution_ops::cos_op_symbol()
}

pub(crate) fn transcendental_cell(
    cx: &mut Cx,
    value: &Value,
    operator: &Symbol,
    output_dtype: &Symbol,
) -> std::result::Result<Value, TensorExecError> {
    let input = numeric_cell(cx, value)?.value;
    let output = if *operator == super::execution_ops::sqrt_op_symbol() {
        input.sqrt()
    } else if *operator == super::execution_ops::exp_op_symbol() {
        input.exp()
    } else if *operator == super::execution_ops::sin_op_symbol() {
        input.sin()
    } else {
        input.cos()
    };
    numeric_value(cx, output_dtype, output)
}

fn zero_value(cx: &mut Cx, dtype: &Symbol) -> std::result::Result<Value, TensorExecError> {
    let text = if dtype == &domains::f16() || dtype == &domains::bf16() {
        "0.0"
    } else {
        "0"
    };
    cx.factory()
        .number_literal(dtype.clone(), text.to_owned())
        .map_err(TensorExecError::from)
}

#[derive(Clone)]
struct NumericCell {
    domain: Symbol,
    value: f64,
}

fn numeric_cell(cx: &mut Cx, value: &Value) -> std::result::Result<NumericCell, TensorExecError> {
    let literal = number_literal_for_tensor_cell(value)
        .or_else(|| cx.number_value_ref(value.clone()).ok().flatten()?.literal)
        .ok_or_else(|| TensorExecError::invalid("tensor math expects numeric tensor cells"))?;
    let domain = literal.domain.clone();
    let value = parse_numeric_literal(literal)?;
    Ok(NumericCell { domain, value })
}

fn parse_numeric_literal(literal: NumberLiteral) -> std::result::Result<f64, TensorExecError> {
    if literal.domain == domains::f64() {
        literal.canonical.parse::<f64>().map_err(parse_error)
    } else if literal.domain == domains::f32() {
        Ok(f64::from(
            literal.canonical.parse::<f32>().map_err(parse_error)?,
        ))
    } else if literal.domain == domains::f16() {
        Ok(f64::from(
            f16::from_f32(literal.canonical.parse::<f32>().map_err(parse_error)?).to_f32(),
        ))
    } else if literal.domain == domains::bf16() {
        Ok(f64::from(
            bf16::from_f32(literal.canonical.parse::<f32>().map_err(parse_error)?).to_f32(),
        ))
    } else if literal.domain == domains::i64() {
        literal
            .canonical
            .parse::<i64>()
            .map(|value| value as f64)
            .map_err(parse_error)
    } else if literal.domain == domains::rational() {
        parse_rational(&literal.canonical)
    } else {
        literal.canonical.parse::<f64>().map_err(parse_error)
    }
}

fn parse_rational(text: &str) -> std::result::Result<f64, TensorExecError> {
    let Some((numerator, denominator)) = text.split_once('/') else {
        return text.parse::<f64>().map_err(parse_error);
    };
    let numerator = numerator.trim().parse::<f64>().map_err(parse_error)?;
    let denominator = denominator.trim().parse::<f64>().map_err(parse_error)?;
    Ok(numerator / denominator)
}

fn numeric_value(
    cx: &mut Cx,
    domain: &Symbol,
    value: f64,
) -> std::result::Result<Value, TensorExecError> {
    let text = if domain == &domains::f64() {
        value.to_string()
    } else if domain == &domains::f32() || domain == &domains::f16() || domain == &domains::bf16() {
        (value as f32).to_string()
    } else if domain == &domains::i64() {
        (value as i64).to_string()
    } else {
        value.to_string()
    };
    cx.factory()
        .number_literal(domain.clone(), text)
        .map_err(TensorExecError::from)
}

fn parse_error(error: impl std::fmt::Display) -> TensorExecError {
    TensorExecError::invalid(Arc::<str>::from(format!(
        "tensor math could not parse numeric cell: {error}"
    )))
}
