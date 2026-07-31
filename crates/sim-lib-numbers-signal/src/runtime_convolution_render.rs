//! Runtime values for signal-operation results and policy evidence.

use sim_kernel::{Cx, Result, Symbol, Value};

use crate::{ConvolutionAlgorithm, CorrelationNormalization, LagOrder, Regularization};

pub(crate) fn real_values(cx: &mut Cx, values: &[f64]) -> Result<Value> {
    let values = values
        .iter()
        .map(|value| f64_value(cx, *value))
        .collect::<Result<Vec<_>>>()?;
    cx.factory().list(values)
}

pub(crate) fn usize_value(cx: &mut Cx, value: usize) -> Result<Value> {
    number_value(cx, value.to_string())
}

pub(crate) fn signed_value(cx: &mut Cx, value: isize) -> Result<Value> {
    number_value(cx, value.to_string())
}

pub(crate) fn symbol_value(cx: &mut Cx, value: &str) -> Result<Value> {
    cx.factory().symbol(Symbol::new(value))
}

pub(crate) fn algorithm_name(algorithm: ConvolutionAlgorithm) -> &'static str {
    match algorithm {
        ConvolutionAlgorithm::Auto => "auto",
        ConvolutionAlgorithm::Direct => "direct",
        ConvolutionAlgorithm::Fft => "fft",
    }
}

pub(crate) fn correlation_normalization_name(
    normalization: CorrelationNormalization,
) -> &'static str {
    match normalization {
        CorrelationNormalization::None => "none",
        CorrelationNormalization::Biased => "biased",
        CorrelationNormalization::Unbiased => "unbiased",
        CorrelationNormalization::Coefficient => "coefficient",
    }
}

pub(crate) fn lag_order_name(order: LagOrder) -> &'static str {
    match order {
        LagOrder::Ascending => "ascending",
        LagOrder::Descending => "descending",
    }
}

pub(crate) fn regularization_name(regularization: Regularization) -> &'static str {
    match regularization {
        Regularization::Tikhonov { .. } => "tikhonov",
        Regularization::Truncated => "truncated",
    }
}

pub(crate) fn f64_value(cx: &mut Cx, value: f64) -> Result<Value> {
    number_value(
        cx,
        if value == 0.0 {
            "0".into()
        } else {
            value.to_string()
        },
    )
}

fn number_value(cx: &mut Cx, canonical: String) -> Result<Value> {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "f64"), canonical)
}
