use std::sync::Arc;

use sim_kernel::{DefaultFactory, Expr, NoopEvalPolicy, NumberLiteral};
use sim_lib_numbers_core::domains;

use crate::{BoolNumbersLib, number_domain};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&BoolNumbersLib::new()).unwrap();
    cx
}

#[test]
fn bool_literal_round_trips_through_expr() {
    let mut cx = cx();
    let value = cx
        .factory()
        .number_literal(number_domain(), "true".to_owned())
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: number_domain(),
            canonical: "true".to_owned(),
        })
    );

    let value = cx.factory().bool(true).unwrap();
    let number = cx.number_value_ref(value).unwrap().unwrap();
    assert_eq!(number.domain, number_domain());
    assert_eq!(number.literal.unwrap().canonical, "true");
}

#[test]
fn bool_arithmetic_uses_boolean_rules() {
    let mut cx = cx();
    let left = cx.factory().bool(true).unwrap();
    let right = cx.factory().bool(false).unwrap();
    let value = cx
        .apply_value_number_binary_op(&sim_kernel::Symbol::qualified("math", "add"), left, right)
        .unwrap();
    assert_eq!(value.object().as_expr(&mut cx).unwrap(), Expr::Bool(true));
}

#[test]
fn bool_registers_value_promotions() {
    let cx = cx();
    let edges = cx.registry().value_promotion_rules();
    assert!(
        edges
            .iter()
            .any(|rule| rule.from_domain == domains::bool() && rule.to_domain == domains::i64())
    );
    assert!(
        edges
            .iter()
            .any(|rule| rule.from_domain == domains::bool() && rule.to_domain == domains::f64())
    );
}
