//! Canonical, executor-routed tensor vocabulary.
//!
//! Every non-tensor parameter is carried by [`CanonicalAttrs`] in the operation
//! descriptor.  This keeps endpoint, axis, tolerance, padding, and empty-input
//! policy visible to providers instead of inheriting ambient defaults.

use std::{any::Any, sync::Arc};

use sim_kernel::{Cx, Error, Expr, Object, Result, Symbol};

use super::{
    execution::{TensorExecError, TensorMeta, TensorOp, TensorRequest, execute_tensor_request},
    execution_math_support::{numeric_f64, numeric_value, tensor_from_cells},
    value::Tensor,
};

/// Explicit padding behavior. Only constant padding is presently canonical.
#[derive(Clone, Debug, PartialEq)]
pub enum PadMode {
    /// Fill every padded cell with the supplied scalar.
    Constant(f64),
}

/// Explicit parameters and edge policy carried by canonical operation requests.
#[derive(Clone, Debug, PartialEq)]
pub enum CanonicalAttrs {
    /// Finite arithmetic progression and endpoint policy.
    Range {
        /// First value.
        start: f64,
        /// Endpoint bound.
        stop: f64,
        /// Nonzero increment.
        step: f64,
        /// Whether an exactly reached endpoint is included.
        inclusive: bool,
    },
    /// Counted linear, logarithmic, or geometric space.
    Space {
        /// First value or exponent.
        start: f64,
        /// Last value or exponent.
        stop: f64,
        /// Requested output count; zero is valid.
        count: usize,
        /// Whether the last value is the exact endpoint.
        endpoint: bool,
        /// Optional logarithm base.
        base: Option<f64>,
        /// Whether interpolation is geometric.
        geometric: bool,
    },
    /// Identity-grid dimensions and diagonal offset.
    Eye {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
        /// Signed diagonal offset.
        diagonal: isize,
    },
    /// Scalar repeated by `full`.
    Full {
        /// Repeated value.
        value: f64,
    },
    /// Explicit row-major axis.
    Axis {
        /// Zero-based axis.
        axis: usize,
    },
    /// Per-axis padding widths and mode.
    Pad {
        /// `(before, after)` widths for every axis.
        widths: Arc<[(usize, usize)]>,
        /// Explicit fill behavior.
        mode: PadMode,
    },
    /// Inclusive clipping interval.
    Clip {
        /// Lower bound.
        minimum: f64,
        /// Upper bound.
        maximum: f64,
    },
    /// Repeated finite difference parameters.
    Diff {
        /// Axis along which differences are taken.
        axis: usize,
        /// Number of difference passes.
        periods: usize,
    },
    /// Separate closeness tolerances and NaN policy.
    Close {
        /// Relative tolerance.
        relative: f64,
        /// Absolute tolerance.
        absolute: f64,
        /// Whether paired NaNs compare close.
        equal_nan: bool,
    },
    /// Operation has no scalar parameters.
    None,
}

impl Object for CanonicalAttrs {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<tensor-op-attributes {self:?}>"))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl sim_kernel::ObjectCompat for CanonicalAttrs {
    fn class(&self, cx: &mut Cx) -> Result<sim_kernel::ClassRef> {
        sim_lib_numbers_core::number_domain_class_stub(cx)
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Err(Error::Eval(
            "tensor operation attributes are request descriptors, not expressions".into(),
        ))
    }
}

macro_rules! symbols { ($($name:ident => $text:literal),+ $(,)?) => {$(
    #[doc = concat!("Canonical `", $text, "` operation symbol.")]
    pub fn $name() -> Symbol { Symbol::qualified("tensor", concat!("op/", $text)) }
)+}; }
symbols! {
    arange_op_symbol=>"arange", linspace_op_symbol=>"linspace", logspace_op_symbol=>"logspace",
    geomspace_op_symbol=>"geomspace", eye_op_symbol=>"eye", diag_op_symbol=>"diag",
    full_op_symbol=>"full", outer_op_symbol=>"outer", concat_op_symbol=>"concat",
    stack_op_symbol=>"stack", column_stack_op_symbol=>"column-stack", pad_op_symbol=>"pad",
    argmax_op_symbol=>"argmax", argmin_op_symbol=>"argmin", where_op_symbol=>"where",
    nonzero_op_symbol=>"nonzero", unique_op_symbol=>"unique", clip_op_symbol=>"clip",
    diff_op_symbol=>"diff", cumsum_op_symbol=>"cumsum", maximum_op_symbol=>"maximum",
    minimum_op_symbol=>"minimum", sign_op_symbol=>"sign", signbit_op_symbol=>"signbit",
    isfinite_op_symbol=>"isfinite", isclose_op_symbol=>"isclose", allclose_op_symbol=>"allclose"
}

/// All canonical operation symbols advertised by a capable provider.
pub fn canonical_tensor_op_symbols() -> Vec<Symbol> {
    vec![
        arange_op_symbol(),
        linspace_op_symbol(),
        logspace_op_symbol(),
        geomspace_op_symbol(),
        eye_op_symbol(),
        diag_op_symbol(),
        full_op_symbol(),
        outer_op_symbol(),
        concat_op_symbol(),
        stack_op_symbol(),
        column_stack_op_symbol(),
        pad_op_symbol(),
        argmax_op_symbol(),
        argmin_op_symbol(),
        where_op_symbol(),
        nonzero_op_symbol(),
        unique_op_symbol(),
        clip_op_symbol(),
        diff_op_symbol(),
        cumsum_op_symbol(),
        maximum_op_symbol(),
        minimum_op_symbol(),
        sign_op_symbol(),
        signbit_op_symbol(),
        isfinite_op_symbol(),
        isclose_op_symbol(),
        allclose_op_symbol(),
    ]
}
pub(crate) fn is_canonical_tensor_op(symbol: &Symbol) -> bool {
    canonical_tensor_op_symbols().contains(symbol)
}

/// Submits a canonical operation. Providers that decline are handled by the
/// existing executor fallback policy.
pub fn execute_canonical_tensor_op(
    cx: &mut Cx,
    symbol: Symbol,
    inputs: Vec<Tensor>,
    output: TensorMeta,
    attrs: CanonicalAttrs,
) -> Result<Tensor> {
    let attributes = cx.factory().opaque(Arc::new(attrs))?;
    execute_tensor_request(
        cx,
        TensorRequest::new(TensorOp::new(symbol, attributes), inputs, output),
    )
}

fn attrs(request: &TensorRequest) -> std::result::Result<&CanonicalAttrs, TensorExecError> {
    request
        .operation
        .attributes
        .object()
        .downcast_ref::<CanonicalAttrs>()
        .ok_or_else(|| {
            TensorExecError::invalid("canonical tensor operation requires explicit CanonicalAttrs")
        })
}
fn cells(cx: &mut Cx, tensor: &Tensor) -> std::result::Result<Vec<f64>, TensorExecError> {
    tensor
        .cells()
        .map_err(TensorExecError::from)?
        .iter()
        .map(|v| numeric_f64(cx, v))
        .collect()
}
fn output(
    cx: &mut Cx,
    request: &TensorRequest,
    values: Vec<f64>,
) -> std::result::Result<Tensor, TensorExecError> {
    let vals = values
        .into_iter()
        .map(|v| numeric_value(cx, request.output.dtype(), v))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    tensor_from_cells(
        cx,
        request.output.shape().to_vec(),
        request.output.dtype().clone(),
        vals,
    )
}
fn unary(request: &TensorRequest) -> std::result::Result<&Tensor, TensorExecError> {
    request
        .inputs
        .first()
        .filter(|_| request.inputs.len() == 1)
        .ok_or_else(|| TensorExecError::invalid("operation expects one tensor input"))
}
fn pair(request: &TensorRequest) -> std::result::Result<(&Tensor, &Tensor), TensorExecError> {
    match request.inputs.as_ref() {
        [a, b] => Ok((a, b)),
        _ => Err(TensorExecError::invalid(
            "operation expects two tensor inputs",
        )),
    }
}
fn strides(shape: &[usize]) -> Vec<usize> {
    (0..shape.len())
        .map(|i| shape[i + 1..].iter().product())
        .collect()
}

pub(crate) fn execute_canonical_request(
    cx: &mut Cx,
    request: &TensorRequest,
) -> std::result::Result<Tensor, TensorExecError> {
    let op = &request.operation.symbol;
    if *op == arange_op_symbol() {
        let CanonicalAttrs::Range {
            start,
            stop,
            step,
            inclusive,
        } = *attrs(request)?
        else {
            return Err(TensorExecError::invalid("arange requires Range attributes"));
        };
        if !start.is_finite() || !stop.is_finite() || !step.is_finite() || step == 0.0 {
            return Err(TensorExecError::invalid(
                "arange requires finite bounds and a nonzero finite step",
            ));
        }
        let mut out = Vec::new();
        let mut v = start;
        let forward = step > 0.0;
        while if forward {
            v < stop || (inclusive && v <= stop)
        } else {
            v > stop || (inclusive && v >= stop)
        } {
            out.push(v);
            v += step;
            if out.len() > request.output.shape().iter().product() {
                return Err(TensorExecError::invalid("arange output count overflow"));
            }
        }
        return output(cx, request, out);
    }
    if [
        linspace_op_symbol(),
        logspace_op_symbol(),
        geomspace_op_symbol(),
    ]
    .contains(op)
    {
        let CanonicalAttrs::Space {
            start,
            stop,
            count,
            endpoint,
            base,
            geometric,
        } = *attrs(request)?
        else {
            return Err(TensorExecError::invalid(
                "space operation requires Space attributes",
            ));
        };
        if !start.is_finite()
            || !stop.is_finite()
            || base.is_some_and(|b| !b.is_finite() || b <= 0.0)
        {
            return Err(TensorExecError::invalid(
                "space operation requires finite inputs and a positive finite base",
            ));
        }
        if count == 0 {
            return output(cx, request, Vec::new());
        }
        let denom = if endpoint && count > 1 {
            count - 1
        } else {
            count
        };
        let vals = (0..count)
            .map(|i| {
                let t = if denom == 0 {
                    0.0
                } else {
                    i as f64 / denom as f64
                };
                let v = if geometric {
                    if start == 0.0 || stop == 0.0 || start.signum() != stop.signum() {
                        f64::NAN
                    } else {
                        start.signum()
                            * (start.abs().ln() + t * (stop.abs().ln() - start.abs().ln())).exp()
                    }
                } else {
                    start + t * (stop - start)
                };
                base.map_or(v, |b| b.powf(v))
            })
            .collect();
        return output(cx, request, vals);
    }
    if *op == full_op_symbol() {
        let CanonicalAttrs::Full { value } = *attrs(request)? else {
            return Err(TensorExecError::invalid("full requires Full attributes"));
        };
        return output(
            cx,
            request,
            vec![value; request.output.shape().iter().product()],
        );
    }
    if *op == eye_op_symbol() {
        let CanonicalAttrs::Eye {
            rows,
            cols,
            diagonal,
        } = *attrs(request)?
        else {
            return Err(TensorExecError::invalid("eye requires Eye attributes"));
        };
        let mut v = vec![
            0.0;
            rows.checked_mul(cols)
                .ok_or_else(|| TensorExecError::invalid("eye shape overflow"))?
        ];
        for r in 0..rows {
            let c = r as isize + diagonal;
            if c >= 0 && (c as usize) < cols {
                v[r * cols + c as usize] = 1.0
            }
        }
        return output(cx, request, v);
    }
    if *op == outer_op_symbol() {
        let (a, b) = pair(request)?;
        let av = cells(cx, a)?;
        let bv = cells(cx, b)?;
        return output(
            cx,
            request,
            av.iter()
                .flat_map(|x| bv.iter().map(move |y| x * y))
                .collect(),
        );
    }
    if *op == diag_op_symbol() {
        let t = unary(request)?;
        let v = cells(cx, t)?;
        if t.shape().len() == 1 {
            let n = v.len();
            let mut o = vec![0.0; n * n];
            for i in 0..n {
                o[i * n + i] = v[i]
            }
            return output(cx, request, o);
        }
        if let [r, c] = t.shape() {
            return output(
                cx,
                request,
                (0..(*r).min(*c)).map(|i| v[i * c + i]).collect(),
            );
        }
        return Err(TensorExecError::invalid("diag expects rank one or two"));
    }
    if *op == concat_op_symbol() || *op == stack_op_symbol() || *op == column_stack_op_symbol() {
        return concat(cx, request);
    }
    if *op == pad_op_symbol() {
        return pad(cx, request);
    }
    if *op == argmax_op_symbol() || *op == argmin_op_symbol() {
        let v = cells(cx, unary(request)?)?;
        if v.is_empty() {
            return Err(TensorExecError::invalid("argmin/argmax reject empty input"));
        }
        let max = *op == argmax_op_symbol();
        let mut best = 0;
        for i in 1..v.len() {
            if v[best].is_nan()
                || (!v[i].is_nan() && ((max && v[i] > v[best]) || (!max && v[i] < v[best])))
            {
                best = i
            }
        }
        return output(cx, request, vec![best as f64]);
    }
    if *op == nonzero_op_symbol() {
        let v = cells(cx, unary(request)?)?;
        return output(
            cx,
            request,
            v.iter()
                .enumerate()
                .filter(|(_, x)| **x != 0.0)
                .map(|(i, _)| i as f64)
                .collect(),
        );
    }
    if *op == unique_op_symbol() {
        let mut out = Vec::new();
        for v in cells(cx, unary(request)?)? {
            if !out.iter().any(|x: &f64| x.to_bits() == v.to_bits()) {
                out.push(v)
            }
        }
        return output(cx, request, out);
    }
    if *op == clip_op_symbol() {
        let CanonicalAttrs::Clip { minimum, maximum } = *attrs(request)? else {
            return Err(TensorExecError::invalid("clip requires Clip attributes"));
        };
        if minimum > maximum {
            return Err(TensorExecError::invalid("clip minimum exceeds maximum"));
        }
        let values = cells(cx, unary(request)?)?
            .into_iter()
            .map(|v| v.max(minimum).min(maximum))
            .collect();
        return output(cx, request, values);
    }
    if *op == diff_op_symbol() {
        return diff(cx, request);
    }
    if *op == cumsum_op_symbol() {
        let input = cells(cx, unary(request)?)?;
        let values = super::reduction::cumsum_f64(&input, super::reduction::SumMode::Naive);
        return output(cx, request, values);
    }
    if [
        maximum_op_symbol(),
        minimum_op_symbol(),
        isclose_op_symbol(),
    ]
    .contains(op)
    {
        let (a, b) = pair(request)?;
        if a.shape() != b.shape() || a.dtype() != b.dtype() {
            return Err(TensorExecError::invalid(
                "elementwise canonical operations require identical shape and dtype",
            ));
        }
        let av = cells(cx, a)?;
        let bv = cells(cx, b)?;
        if *op == isclose_op_symbol() {
            let CanonicalAttrs::Close {
                relative,
                absolute,
                equal_nan,
            } = *attrs(request)?
            else {
                return Err(TensorExecError::invalid(
                    "isclose requires Close attributes",
                ));
            };
            valid_tolerances(relative, absolute)?;
            return output(
                cx,
                request,
                av.into_iter()
                    .zip(bv)
                    .map(|(a, b)| {
                        ((a == b)
                            || (equal_nan && a.is_nan() && b.is_nan())
                            || ((a - b).abs() <= absolute + relative * b.abs()))
                            as u8 as f64
                    })
                    .collect(),
            );
        }
        return output(
            cx,
            request,
            av.into_iter()
                .zip(bv)
                .map(|(a, b)| {
                    if *op == maximum_op_symbol() {
                        a.max(b)
                    } else {
                        a.min(b)
                    }
                })
                .collect(),
        );
    }
    if [sign_op_symbol(), signbit_op_symbol(), isfinite_op_symbol()].contains(op) {
        let vals = cells(cx, unary(request)?)?
            .into_iter()
            .map(|v| {
                if *op == sign_op_symbol() {
                    v.signum()
                } else if *op == signbit_op_symbol() {
                    v.is_sign_negative() as u8 as f64
                } else {
                    v.is_finite() as u8 as f64
                }
            })
            .collect();
        return output(cx, request, vals);
    }
    if *op == allclose_op_symbol() {
        let (a, b) = pair(request)?;
        if a.shape() != b.shape() || a.dtype() != b.dtype() {
            return Err(TensorExecError::invalid(
                "allclose requires identical shape and dtype",
            ));
        }
        let CanonicalAttrs::Close {
            relative,
            absolute,
            equal_nan,
        } = *attrs(request)?
        else {
            return Err(TensorExecError::invalid(
                "allclose requires Close attributes",
            ));
        };
        valid_tolerances(relative, absolute)?;
        let yes = cells(cx, a)?.into_iter().zip(cells(cx, b)?).all(|(a, b)| {
            (a == b)
                || (equal_nan && a.is_nan() && b.is_nan())
                || (a - b).abs() <= absolute + relative * b.abs()
        });
        return output(cx, request, vec![yes as u8 as f64]);
    }
    if *op == where_op_symbol() {
        if let [condition, yes, no] = request.inputs.as_ref() {
            if condition.shape() != yes.shape()
                || yes.shape() != no.shape()
                || yes.dtype() != no.dtype()
            {
                return Err(TensorExecError::invalid(
                    "where requires identical shapes and matching branch dtypes",
                ));
            }
            let c = cells(cx, condition)?;
            let y = cells(cx, yes)?;
            let n = cells(cx, no)?;
            return output(
                cx,
                request,
                c.into_iter()
                    .enumerate()
                    .map(|(i, v)| if v != 0.0 { y[i] } else { n[i] })
                    .collect(),
            );
        }
        return Err(TensorExecError::invalid(
            "where expects condition, yes, and no tensors",
        ));
    }
    Err(TensorExecError::unsupported(
        op.clone(),
        "unknown canonical tensor operation",
    ))
}

fn valid_tolerances(r: f64, a: f64) -> std::result::Result<(), TensorExecError> {
    if r < 0.0 || a < 0.0 || !r.is_finite() || !a.is_finite() {
        Err(TensorExecError::invalid(
            "closeness tolerances must be finite and non-negative",
        ))
    } else {
        Ok(())
    }
}
mod array;

use array::*;
