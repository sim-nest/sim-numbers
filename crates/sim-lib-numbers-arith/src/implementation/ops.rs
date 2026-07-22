//! Runtime implementations for arithmetic operator functions.

use sim_kernel::{Cx, Error, PreparedArgs, Result, Symbol, Value};
use sim_shape::Bindings;

use super::cas_route::{coerce_arith_argument, route_via_cas, should_route_via_cas};
use super::{BinaryOpKind, ReductionOpKind, cmp_symbol, neg_symbol};

pub(crate) fn add_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Add)
}

pub(crate) fn sub_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Sub)
}

pub(crate) fn mul_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Mul)
}

pub(crate) fn div_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Div)
}

pub(crate) fn rem_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Rem)
}

pub(crate) fn pow_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Pow)
}

pub(crate) fn cmp_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [left, right] = prepared.values() else {
        return Err(Error::Eval("cmp expects exactly two arguments".to_owned()));
    };
    let left = coerce_arith_argument(cx, left.clone())?;
    let right = coerce_arith_argument(cx, right.clone())?;
    require_number_arg(cx, left.clone(), cmp_symbol(), 0)?;
    require_number_arg(cx, right.clone(), cmp_symbol(), 1)?;
    cx.apply_value_number_binary_op(&cmp_symbol(), left, right)
}

pub(crate) fn neg_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let operand = prepared
        .get(0)
        .cloned()
        .ok_or_else(|| Error::Eval("neg expects exactly one argument".to_owned()))?;
    require_number_arg(cx, operand, neg_symbol(), 0)?;
    cx.apply_value_number_unary_op(&neg_symbol(), prepared.get(0).cloned().unwrap())
}

pub(crate) fn sum_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    reduction_impl(cx, prepared, ReductionOpKind::Sum)
}

pub(crate) fn product_impl(
    cx: &mut Cx,
    prepared: &PreparedArgs,
    _bindings: Bindings,
) -> Result<Value> {
    reduction_impl(cx, prepared, ReductionOpKind::Product)
}

fn fold_binary_impl(cx: &mut Cx, prepared: &PreparedArgs, kind: BinaryOpKind) -> Result<Value> {
    let values = prepared
        .values()
        .iter()
        .map(|value| coerce_arith_argument(cx, value.clone()))
        .collect::<Result<Vec<_>>>()?;
    if should_route_via_cas(cx, &values)?
        && let Some(value) = route_via_cas(cx, kind.symbol(), &values)?
    {
        return Ok(value);
    }
    let [first, second, rest @ ..] = values.as_slice() else {
        return Err(Error::Eval(format!(
            "{} expects at least two arguments",
            kind.display_name()
        )));
    };
    require_number_arg(cx, first.clone(), kind.symbol(), 0)?;
    require_number_arg(cx, second.clone(), kind.symbol(), 1)?;
    let mut acc = cx.apply_value_number_binary_op(&kind.symbol(), first.clone(), second.clone())?;
    for (offset, value) in rest.iter().enumerate() {
        require_number_arg(cx, value.clone(), kind.symbol(), offset + 2)?;
        acc = cx.apply_value_number_binary_op(&kind.symbol(), acc, value.clone())?;
    }
    Ok(acc)
}

fn reduction_impl(cx: &mut Cx, prepared: &PreparedArgs, kind: ReductionOpKind) -> Result<Value> {
    let values = prepared
        .values()
        .iter()
        .map(|value| coerce_arith_argument(cx, value.clone()))
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        return Err(Error::Eval(format!(
            "{} expects at least one argument",
            kind.display_name()
        )));
    }
    let operands = number_args(cx, &values, kind.symbol())?;
    cx.apply_value_number_reduction_op(&kind.symbol(), operands)
}

fn require_number_arg(cx: &mut Cx, value: Value, function: Symbol, index: usize) -> Result<()> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        });
    };
    if cx
        .registry()
        .number_domain_by_symbol(&number.domain)
        .is_none()
    {
        return Err(Error::Eval(format!(
            "{} arg {} uses unloaded number domain {}",
            function, index, number.domain
        )));
    }
    Ok(())
}

fn number_args(cx: &mut Cx, values: &[Value], function: Symbol) -> Result<Vec<Value>> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            require_number_arg(cx, value.clone(), function.clone(), index)?;
            Ok(value.clone())
        })
        .collect()
}
