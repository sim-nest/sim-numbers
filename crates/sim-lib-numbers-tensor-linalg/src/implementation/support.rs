//! Shared helpers for the linear-algebra ops: tensor shape/rank validation,
//! argument extraction, and element-wise value arithmetic.

use sim_kernel::{DefaultFactory, Error, Expr, Factory, Result, Symbol, Value, force_list_to_vec};
use sim_lib_numbers_core::domains;
use sim_lib_numbers_tensor::{Tensor, tensor_value_ref};

pub fn expect_tensor(value: &Value) -> Result<&Tensor> {
    tensor_value_ref(value).ok_or_else(|| Error::Eval("expected a tensor value".to_owned()))
}

pub fn expect_vector(value: &Value) -> Result<&Tensor> {
    let tensor = expect_tensor(value)?;
    if tensor.shape.len() != 1 {
        return Err(Error::Eval("expected a rank-1 tensor".to_owned()));
    }
    Ok(tensor)
}

pub fn expect_matrix(value: &Value) -> Result<&Tensor> {
    let tensor = expect_tensor(value)?;
    if tensor.shape.len() != 2 {
        return Err(Error::Eval("expected a rank-2 tensor".to_owned()));
    }
    Ok(tensor)
}

pub fn add(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "add"), left, right)
}

pub fn sub(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "sub"), left, right)
}

pub fn mul(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "mul"), left, right)
}

pub fn div(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "div"), left, right)
}

pub fn pow(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "pow"), left, right)
}

pub fn neg(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    cx.apply_value_number_unary_op(&Symbol::qualified("math", "neg"), value)
}

pub fn i64_number(value: i64) -> Result<Value> {
    DefaultFactory.number_literal(domains::i64(), value.to_string())
}

pub fn extract_optional_symbol(cx: &mut sim_kernel::Cx, value: &Value) -> Result<Option<Symbol>> {
    match value.object().as_expr(cx)? {
        Expr::Nil => Ok(None),
        Expr::Symbol(symbol) => Ok(Some(symbol)),
        Expr::Quote { expr, .. } => match *expr {
            Expr::Symbol(symbol) => Ok(Some(symbol)),
            _ => Err(Error::Eval("expected quoted dtype symbol".to_owned())),
        },
        _ => Err(Error::Eval("expected dtype symbol or nil".to_owned())),
    }
}

pub fn extract_shape(cx: &mut sim_kernel::Cx, value: &Value) -> Result<Vec<usize>> {
    if let Some(list) = value.object().as_list() {
        return force_list_to_vec(cx, list, "tensor shape")?
            .iter()
            .map(|item| extract_usize(item, "shape dimension"))
            .collect();
    }
    match value.object().as_expr(cx)? {
        Expr::Vector(items) => items
            .into_iter()
            .map(|expr| match expr {
                Expr::String(text) => text
                    .parse::<usize>()
                    .map_err(|_| Error::Eval("shape dimensions must be usize strings".to_owned())),
                Expr::Number(number) => number
                    .canonical
                    .parse::<usize>()
                    .map_err(|_| Error::Eval("shape dimensions must be usize numbers".to_owned())),
                _ => Err(Error::Eval("shape dimensions must be numeric".to_owned())),
            })
            .collect(),
        _ => Err(Error::Eval("shape must be a list or vector".to_owned())),
    }
}

pub fn extract_usize(value: &Value, context: &str) -> Result<usize> {
    let mut cx = sim_kernel::Cx::new(
        std::sync::Arc::new(sim_kernel::NoopEvalPolicy),
        std::sync::Arc::new(DefaultFactory),
    );
    match value.object().as_expr(&mut cx)? {
        Expr::Number(number) => number
            .canonical
            .parse::<usize>()
            .map_err(|_| Error::Eval(format!("{context} must be a non-negative integer"))),
        Expr::String(text) => text
            .parse::<usize>()
            .map_err(|_| Error::Eval(format!("{context} must be a non-negative integer"))),
        _ => Err(Error::Eval(format!(
            "{context} must be a numeric literal or decimal string"
        ))),
    }
}

pub use sim_lib_numbers_tensor::bounded_element_count;
