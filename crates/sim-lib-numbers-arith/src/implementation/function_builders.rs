//! FunctionObject construction for arithmetic operators and aliases.

use std::sync::Arc;

use sim_kernel::{Demand, LoadCx, Symbol};
use sim_shape::{AnyShape, CaptureShape, FunctionCase, FunctionObject, ListShape};

use super::ops::{
    add_impl, cmp_impl, div_impl, mul_impl, neg_impl, pow_impl, product_impl, rem_impl, sub_impl,
    sum_impl,
};
use super::{BinaryOpKind, ReductionOpKind, UnaryOpKind, cmp_symbol};

pub(super) fn build_binary_function(
    cx: &mut LoadCx,
    kind: BinaryOpKind,
) -> (Symbol, FunctionObject) {
    build_binary_named_function(cx, kind.symbol(), kind)
}

fn build_binary_named_function(
    cx: &mut LoadCx,
    symbol: Symbol,
    kind: BinaryOpKind,
) -> (Symbol, FunctionObject) {
    let case = FunctionCase {
        id: cx.fresh_case_id(),
        name: Symbol::qualified(symbol.to_string(), "numbers"),
        args: Arc::new(ListShape::with_rest(
            vec![captured_number("left"), captured_number("right")],
            captured_number("rest"),
        )),
        result: Some(number_shape()),
        demand: vec![Demand::Value, Demand::Value],
        priority: 10,
        implementation: match kind {
            BinaryOpKind::Add => add_impl,
            BinaryOpKind::Sub => sub_impl,
            BinaryOpKind::Mul => mul_impl,
            BinaryOpKind::Div => div_impl,
            BinaryOpKind::Rem => rem_impl,
            BinaryOpKind::Pow => pow_impl,
        },
    };
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, vec![case]),
    )
}

pub(super) fn build_cmp_function(cx: &mut LoadCx) -> (Symbol, FunctionObject) {
    let symbol = cmp_symbol();
    let case = FunctionCase {
        id: cx.fresh_case_id(),
        name: Symbol::qualified(symbol.to_string(), "numbers"),
        args: Arc::new(ListShape::new(vec![
            captured_number("left"),
            captured_number("right"),
        ])),
        result: Some(number_shape()),
        demand: vec![Demand::Value, Demand::Value],
        priority: 10,
        implementation: cmp_impl,
    };
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, vec![case]),
    )
}

pub(super) fn build_unary_function(cx: &mut LoadCx, kind: UnaryOpKind) -> (Symbol, FunctionObject) {
    build_unary_named_function(cx, kind.symbol(), kind)
}

fn build_unary_named_function(
    cx: &mut LoadCx,
    symbol: Symbol,
    _kind: UnaryOpKind,
) -> (Symbol, FunctionObject) {
    let case = FunctionCase {
        id: cx.fresh_case_id(),
        name: Symbol::qualified(symbol.to_string(), "numbers"),
        args: Arc::new(ListShape::new(vec![captured_number("value")])),
        result: Some(number_shape()),
        demand: vec![Demand::Value],
        priority: 10,
        implementation: neg_impl,
    };
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, vec![case]),
    )
}

#[derive(Clone, Copy)]
pub(super) enum AliasCase {
    Binary(BinaryOpKind),
    Unary(UnaryOpKind),
}

pub(super) fn build_alias_function(
    cx: &mut LoadCx,
    symbol: Symbol,
    cases: &[AliasCase],
) -> (Symbol, FunctionObject) {
    let mut built = Vec::with_capacity(cases.len());
    for case in cases {
        built.push(match case {
            AliasCase::Binary(kind) => build_binary_named_function(cx, symbol.clone(), *kind).1,
            AliasCase::Unary(kind) => build_unary_named_function(cx, symbol.clone(), *kind).1,
        });
    }
    let cases = built
        .into_iter()
        .flat_map(|function| function.cases)
        .collect();
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, cases),
    )
}

pub(super) fn build_reduction_function(
    cx: &mut LoadCx,
    kind: ReductionOpKind,
) -> (Symbol, FunctionObject) {
    let symbol = kind.symbol();
    let case = FunctionCase {
        id: cx.fresh_case_id(),
        name: Symbol::qualified(symbol.to_string(), "numbers"),
        args: Arc::new(ListShape::with_rest(
            vec![captured_number("first")],
            captured_number("rest"),
        )),
        result: Some(number_shape()),
        demand: vec![Demand::Value],
        priority: 10,
        implementation: match kind {
            ReductionOpKind::Sum => sum_impl,
            ReductionOpKind::Product => product_impl,
        },
    };
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, vec![case]),
    )
}

fn number_shape() -> Arc<AnyShape> {
    Arc::new(AnyShape)
}

fn captured_number(name: &str) -> Arc<CaptureShape> {
    Arc::new(CaptureShape::new(Symbol::new(name), number_shape()))
}
