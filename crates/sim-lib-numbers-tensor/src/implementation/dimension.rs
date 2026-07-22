//! Dimension parsing shared by tensor callables and the tensor read constructor.

use sim_kernel::{Cx, Error, Expr, Result, Value, force_list_to_vec};

pub(super) fn extract_dims(cx: &mut Cx, value: &Value, context: &str) -> Result<Vec<usize>> {
    let list = value
        .object()
        .as_list()
        .ok_or_else(|| Error::Eval(format!("{context} must be a list or vector of dimensions")))?;
    let values = force_list_to_vec(cx, list, context)?;
    values
        .iter()
        .map(|entry| extract_usize(cx, entry, context))
        .collect()
}

pub(super) fn extract_usize(cx: &mut Cx, value: &Value, context: &str) -> Result<usize> {
    match value.object().as_expr(cx)? {
        Expr::Number(number) => number
            .canonical
            .parse::<usize>()
            .map_err(|_| Error::Eval(format!("{context} expects non-negative integer dimensions"))),
        Expr::String(text) => text
            .parse::<usize>()
            .map_err(|_| Error::Eval(format!("{context} expects non-negative integer dimensions"))),
        _ => Err(Error::Eval(format!(
            "{context} expects non-negative integer dimensions"
        ))),
    }
}
