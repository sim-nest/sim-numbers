//! Numeric runtime-value parsing for real signal operations.

use sim_kernel::{Cx, Error, NumberLiteral, Result, Value, force_list_to_vec};

use crate::runtime_convolution_callable::Operation;

pub(crate) fn real_list(
    cx: &mut Cx,
    operation: Operation,
    value: &Value,
    name: &str,
) -> Result<Vec<f64>> {
    let list = value.object().as_list().ok_or(Error::TypeMismatch {
        expected: "list",
        found: "non-list",
    })?;
    force_list_to_vec(cx, list, &format!("{} {name}", operation.name()))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            number_value(cx, value, &format!("{name}[{index}]"))?
                .canonical
                .parse::<f64>()
                .map_err(|_| {
                    Error::Eval(format!("{} {name}[{index}] must be f64", operation.name()))
                })
        })
        .collect()
}

pub(crate) fn number_value(cx: &mut Cx, value: &Value, name: &str) -> Result<NumberLiteral> {
    value
        .object()
        .as_number_value()
        .ok_or(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        })?
        .number_literal(cx)?
        .ok_or_else(|| Error::Eval(format!("{name} has no numeric literal")))
}
