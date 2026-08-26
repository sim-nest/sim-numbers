//! Canonical concatenation, padding, and differencing operations.

use super::*;

pub(super) fn concat(
    cx: &mut Cx,
    r: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    if r.inputs.is_empty() {
        return Err(TensorExecError::invalid(
            "concat/stack requires at least one input",
        ));
    }
    let mut stack = r.operation.symbol == stack_op_symbol();
    let column = r.operation.symbol == column_stack_op_symbol();
    let axis = if column {
        if r.inputs[0].shape().len() == 1 {
            stack = true;
            1
        } else {
            1
        }
    } else {
        match attrs(r)? {
            CanonicalAttrs::Axis { axis } => *axis,
            _ => {
                return Err(TensorExecError::invalid(
                    "concat/stack requires explicit Axis attributes",
                ));
            }
        }
    };
    let base = &r.inputs[0];
    if r.inputs.iter().any(|t| t.dtype() != base.dtype()) {
        return Err(TensorExecError::invalid(
            "concat/stack refuses dtype mismatch",
        ));
    }
    if stack {
        if axis > base.shape().len() || r.inputs.iter().any(|t| t.shape() != base.shape()) {
            return Err(TensorExecError::invalid("stack axis or shape mismatch"));
        }
        let values = r
            .inputs
            .iter()
            .map(|t| cells(cx, t))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let inner: usize = base.shape()[axis..].iter().product();
        let outer: usize = base.shape()[..axis].iter().product();
        let mut all = Vec::new();
        for o in 0..outer {
            for value in &values {
                all.extend_from_slice(&value[o * inner..(o + 1) * inner]);
            }
        }
        return output(cx, r, all);
    }
    if axis >= base.shape().len()
        || r.inputs.iter().any(|t| {
            t.shape().len() != base.shape().len()
                || t.shape()
                    .iter()
                    .enumerate()
                    .any(|(i, n)| i != axis && *n != base.shape()[i])
        })
    {
        return Err(TensorExecError::invalid(
            "concat axis, rank, or shape mismatch",
        ));
    }
    let inner: usize = base.shape()[axis + 1..].iter().product();
    let outer: usize = base.shape()[..axis].iter().product();
    let vals = r
        .inputs
        .iter()
        .map(|t| cells(cx, t))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut out = Vec::new();
    for o in 0..outer {
        for (t, v) in r.inputs.iter().zip(&vals) {
            let width = t.shape()[axis] * inner;
            out.extend_from_slice(&v[o * width..(o + 1) * width])
        }
    }
    output(cx, r, out)
}
pub(super) fn pad(cx: &mut Cx, r: &TensorRequest) -> std::result::Result<Tensor, TensorExecError> {
    let t = unary(r)?;
    let CanonicalAttrs::Pad { widths, mode } = attrs(r)? else {
        return Err(TensorExecError::invalid(
            "pad requires explicit Pad attributes",
        ));
    };
    if widths.len() != t.shape().len() {
        return Err(TensorExecError::invalid(
            "pad widths must match tensor rank",
        ));
    }
    let PadMode::Constant(fill) = mode;
    let src = cells(cx, t)?;
    let out_shape = r.output.shape();
    let out_strides = strides(out_shape);
    let in_strides = strides(t.shape());
    let mut out = vec![*fill; out_shape.iter().product()];
    for (flat, v) in src.into_iter().enumerate() {
        let mut rem = flat;
        let mut dst = 0;
        for axis in 0..t.shape().len() {
            let coord = rem / in_strides[axis];
            rem %= in_strides[axis];
            dst += (coord + widths[axis].0) * out_strides[axis]
        }
        out[dst] = v
    }
    output(cx, r, out)
}
pub(super) fn diff(cx: &mut Cx, r: &TensorRequest) -> std::result::Result<Tensor, TensorExecError> {
    let t = unary(r)?;
    let CanonicalAttrs::Diff { axis, periods } = *attrs(r)? else {
        return Err(TensorExecError::invalid("diff requires Diff attributes"));
    };
    if axis >= t.shape().len() || periods > t.shape()[axis] {
        return Err(TensorExecError::invalid("diff axis or periods invalid"));
    }
    let mut shape = t.shape().to_vec();
    let mut v = cells(cx, t)?;
    for _ in 0..periods {
        let inner: usize = shape[axis + 1..].iter().product();
        let outer: usize = shape[..axis].iter().product();
        let width = shape[axis];
        let mut next = Vec::new();
        for o in 0..outer {
            for i in 0..width - 1 {
                for j in 0..inner {
                    next.push(
                        v[o * width * inner + (i + 1) * inner + j]
                            - v[o * width * inner + i * inner + j],
                    )
                }
            }
        }
        v = next;
        shape[axis] -= 1
    }
    output(cx, r, v)
}
