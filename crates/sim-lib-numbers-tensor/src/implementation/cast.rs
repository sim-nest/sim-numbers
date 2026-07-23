//! Explicit tensor dtype conversion with checked narrowing and deterministic
//! rounding.

use std::sync::Arc;

use half::{bf16, f16};
use sim_kernel::{Cx, Error, Expr, QuoteMode, Result, Symbol, Value};

use crate::{
    Tensor, TypedTensorStorage, domains, number_literal_for_tensor_cell, tensor_value_ref,
};

/// Returns the qualified function symbol for explicit tensor dtype conversion.
pub fn cast_symbol() -> Symbol {
    Symbol::qualified("tensor", "cast")
}

/// Casts a tensor into `target_dtype`.
///
/// Supported targets are `numbers/i64`, `numbers/f32`, `numbers/f64`,
/// `numbers/f16`, and `numbers/bf16`. Floating-to-integer casts round ties to
/// even and reject NaN, infinity, and out-of-range results. Floating narrowing
/// preserves NaN, infinity, and signed zero, and rejects finite overflow into an
/// infinite result.
pub fn cast_tensor(tensor: &Tensor, target_dtype: Symbol) -> Result<Tensor> {
    if tensor.dtype() == &target_dtype {
        return Ok(tensor.clone());
    }
    let cells = tensor
        .cells()?
        .iter()
        .enumerate()
        .map(|(index, value)| CastCell::from_value(value, index))
        .collect::<Result<Vec<_>>>()?;
    let shape = tensor.shape().to_vec();
    if target_dtype == domains::i64() {
        let out = cells
            .iter()
            .enumerate()
            .map(|(index, cell)| cell.to_i64(index))
            .collect::<Result<Vec<_>>>()?;
        return Tensor::from_storage(
            shape,
            target_dtype,
            Arc::new(TypedTensorStorage::<i64>::new(out)),
        );
    }
    if target_dtype == domains::f32() {
        let out = cells
            .iter()
            .enumerate()
            .map(|(index, cell)| cell.to_f32(index, domains::f32()))
            .collect::<Result<Vec<_>>>()?;
        return Tensor::from_storage(
            shape,
            target_dtype,
            Arc::new(TypedTensorStorage::<f32>::new(out)),
        );
    }
    if target_dtype == domains::f64() {
        let out = cells
            .iter()
            .copied()
            .map(CastCell::to_f64)
            .collect::<Vec<_>>();
        return Tensor::from_storage(
            shape,
            target_dtype,
            Arc::new(TypedTensorStorage::<f64>::new(out)),
        );
    }
    if target_dtype == domains::f16() {
        let out = cells
            .iter()
            .enumerate()
            .map(|(index, cell)| cell.to_f16(index))
            .collect::<Result<Vec<_>>>()?;
        return Tensor::from_storage(
            shape,
            target_dtype,
            Arc::new(TypedTensorStorage::<f16>::new(out)),
        );
    }
    if target_dtype == domains::bf16() {
        let out = cells
            .iter()
            .enumerate()
            .map(|(index, cell)| cell.to_bf16(index))
            .collect::<Result<Vec<_>>>()?;
        return Tensor::from_storage(
            shape,
            target_dtype,
            Arc::new(TypedTensorStorage::<bf16>::new(out)),
        );
    }
    Err(Error::Eval(format!(
        "tensor/cast does not support target dtype {target_dtype}"
    )))
}

/// Casts a tensor value into `target_dtype` and returns it as a runtime value.
pub fn cast_tensor_value(cx: &mut Cx, value: Value, target_dtype: Symbol) -> Result<Value> {
    let tensor = tensor_value_ref(&value)
        .ok_or_else(|| Error::Eval("tensor/cast expects a tensor value".to_owned()))?;
    if tensor.dtype() == &target_dtype {
        return Ok(value);
    }
    cx.factory()
        .opaque(Arc::new(cast_tensor(tensor, target_dtype)?))
}

pub(crate) fn cast_function_impl(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let [tensor_value, dtype_value] = values.as_slice() else {
        return Err(Error::Eval(
            "tensor/cast expects exactly two arguments: tensor dtype".to_owned(),
        ));
    };
    let dtype = extract_dtype_symbol(cx, dtype_value)?;
    cast_tensor_value(cx, tensor_value.clone(), dtype)
}

#[derive(Clone, Copy)]
enum CastCell {
    I64(i64),
    F32(f32),
    F64(f64),
    F16(f16),
    Bf16(bf16),
}

impl CastCell {
    fn from_value(value: &Value, index: usize) -> Result<Self> {
        let literal = number_literal_for_tensor_cell(value).ok_or_else(|| {
            Error::Eval(format!(
                "tensor/cast source cell {index} does not expose a numeric literal"
            ))
        })?;
        if literal.domain == domains::i64() {
            return Ok(Self::I64(parse_literal(
                &literal.canonical,
                index,
                &literal.domain,
            )?));
        }
        if literal.domain == domains::f32() {
            return Ok(Self::F32(parse_literal(
                &literal.canonical,
                index,
                &literal.domain,
            )?));
        }
        if literal.domain == domains::f64() {
            return Ok(Self::F64(parse_literal(
                &literal.canonical,
                index,
                &literal.domain,
            )?));
        }
        if literal.domain == domains::f16() {
            let value = parse_literal::<f32>(&literal.canonical, index, &literal.domain)?;
            return Ok(Self::F16(f16::from_f32(value)));
        }
        if literal.domain == domains::bf16() {
            let value = parse_literal::<f32>(&literal.canonical, index, &literal.domain)?;
            return Ok(Self::Bf16(bf16::from_f32(value)));
        }
        Err(Error::Eval(format!(
            "tensor/cast does not support source dtype {} at cell {index}",
            literal.domain
        )))
    }

    fn to_f64(self) -> f64 {
        match self {
            Self::I64(value) => value as f64,
            Self::F32(value) => f64::from(value),
            Self::F64(value) => value,
            Self::F16(value) => f64::from(value.to_f32()),
            Self::Bf16(value) => f64::from(value.to_f32()),
        }
    }

    fn to_f32(self, index: usize, target: Symbol) -> Result<f32> {
        match self {
            Self::I64(value) => Ok(value as f32),
            Self::F32(value) => Ok(value),
            Self::F64(value) => {
                let narrowed = value as f32;
                reject_finite_overflow(value, narrowed.is_infinite(), index, &target)?;
                Ok(narrowed)
            }
            Self::F16(value) => Ok(value.to_f32()),
            Self::Bf16(value) => Ok(value.to_f32()),
        }
    }

    fn to_i64(self, index: usize) -> Result<i64> {
        let value = self.to_f64();
        if !value.is_finite() {
            return Err(Error::Eval(format!(
                "tensor/cast cannot cast non-finite cell {index} to {}",
                domains::i64()
            )));
        }
        let rounded = value.round_ties_even();
        const I64_MIN_INCLUSIVE: f64 = -9_223_372_036_854_775_808.0;
        const I64_MAX_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
        if !(I64_MIN_INCLUSIVE..I64_MAX_EXCLUSIVE).contains(&rounded) {
            return Err(Error::Eval(format!(
                "tensor/cast cell {index} overflows {}",
                domains::i64()
            )));
        }
        Ok(rounded as i64)
    }

    fn to_f16(self, index: usize) -> Result<f16> {
        let source = self.to_f64();
        let narrowed = f16::from_f32(self.to_f32(index, domains::f16())?);
        reject_finite_overflow(source, narrowed.is_infinite(), index, &domains::f16())?;
        Ok(narrowed)
    }

    fn to_bf16(self, index: usize) -> Result<bf16> {
        let source = self.to_f64();
        let narrowed = bf16::from_f32(self.to_f32(index, domains::bf16())?);
        reject_finite_overflow(source, narrowed.is_infinite(), index, &domains::bf16())?;
        Ok(narrowed)
    }
}

fn reject_finite_overflow(
    source: f64,
    narrowed_is_infinite: bool,
    index: usize,
    target: &Symbol,
) -> Result<()> {
    if source.is_finite() && narrowed_is_infinite {
        return Err(Error::Eval(format!(
            "tensor/cast cell {index} overflows {target}"
        )));
    }
    Ok(())
}

fn parse_literal<T: std::str::FromStr>(canonical: &str, index: usize, domain: &Symbol) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    canonical.parse::<T>().map_err(|err| {
        Error::Eval(format!(
            "tensor/cast cell {index} in {domain} has invalid canonical literal {canonical:?}: {err}"
        ))
    })
}

fn extract_dtype_symbol(cx: &mut Cx, value: &Value) -> Result<Symbol> {
    match value.object().as_expr(cx)? {
        Expr::Symbol(symbol) => Ok(symbol),
        Expr::Quote {
            mode: QuoteMode::Quote,
            expr,
        } => match *expr {
            Expr::Symbol(symbol) => Ok(symbol),
            _ => Err(Error::Eval(
                "tensor/cast expected a symbol dtype".to_owned(),
            )),
        },
        _ => Err(Error::Eval(
            "tensor/cast expected a symbol dtype".to_owned(),
        )),
    }
}
