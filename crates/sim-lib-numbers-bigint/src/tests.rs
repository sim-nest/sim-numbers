use std::sync::Arc;

use sim_kernel::{DefaultFactory, Expr, NoopEvalPolicy, NumberLiteral, Symbol};
use sim_lib_numbers_core::domains;

use crate::{BigIntNumbersLib, number_domain};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&BigIntNumbersLib::new()).unwrap();
    cx
}

#[test]
fn bigint_literal_round_trips_through_expr() {
    let mut cx = cx();
    let value = cx
        .factory()
        .number_literal(
            number_domain(),
            "1267650600228229401496703205376".to_owned(),
        )
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: number_domain(),
            canonical: "1267650600228229401496703205376".to_owned(),
        })
    );
}

#[test]
fn bigint_arithmetic_adds_large_values() {
    let mut cx = cx();
    let left = cx
        .factory()
        .number_literal(
            number_domain(),
            "1267650600228229401496703205376".to_owned(),
        )
        .unwrap();
    let right = cx
        .factory()
        .number_literal(number_domain(), "24".to_owned())
        .unwrap();
    let value = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "add"), left, right)
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: number_domain(),
            canonical: "1267650600228229401496703205400".to_owned(),
        })
    );
}

#[test]
fn bigint_registers_value_promotions() {
    let cx = cx();
    let edges = cx.registry().value_promotion_rules();
    assert!(
        edges
            .iter()
            .any(|rule| rule.from_domain == domains::i64() && rule.to_domain == domains::bigint())
    );
    assert!(
        edges
            .iter()
            .any(|rule| rule.from_domain == domains::bigint()
                && rule.to_domain == domains::rational())
    );
}
