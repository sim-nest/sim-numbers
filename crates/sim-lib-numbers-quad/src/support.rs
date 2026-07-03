//! Shared value-arithmetic helpers used by the quadrature and differentiation
//! backends to evaluate functions and combine number-domain values.

use std::sync::Arc;

use sim_kernel::{Args, Cx, Error, Result, Symbol, Value};
use sim_lib_numbers_core::domains;
use sim_lib_numbers_func::Func;

pub fn f64_value(cx: &mut Cx, value: f64) -> Result<Value> {
    cx.factory()
        .number_literal(domains::f64(), value.to_string())
}

pub fn value_to_f64(cx: &mut Cx, value: &Value, context: &str) -> Result<f64> {
    value
        .object()
        .display(cx)?
        .parse::<f64>()
        .map_err(|_| Error::Eval(format!("{context} expected an f64-compatible value")))
}

pub fn add(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "add"), left, right)
}

pub fn sub(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "sub"), left, right)
}

pub fn mul(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "mul"), left, right)
}

pub fn div(cx: &mut Cx, left: Value, right: Value) -> Result<Value> {
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "div"), left, right)
}

pub fn scale(cx: &mut Cx, value: Value, scalar: f64) -> Result<Value> {
    let scalar = f64_value(cx, scalar)?;
    mul(cx, value, scalar)
}

pub fn add_scaled(cx: &mut Cx, acc: Value, value: Value, scalar: f64) -> Result<Value> {
    let scaled = scale(cx, value, scalar)?;
    add(cx, acc, scaled)
}

pub fn zero_like(cx: &mut Cx, seed: Value) -> Result<Value> {
    scale(cx, seed, 0.0)
}

pub fn call_unary_func(cx: &mut Cx, func: &Func, point: Value) -> Result<Value> {
    let value = cx.factory().opaque(Arc::new(func.clone()))?;
    cx.call_value(value, Args::new(vec![point]))
}

pub fn abs_error(cx: &mut Cx, left: Value, right: Value) -> Result<f64> {
    let diff = sub(cx, left, right)?;
    Ok(value_to_f64(cx, &diff, "numeric error")?.abs())
}
