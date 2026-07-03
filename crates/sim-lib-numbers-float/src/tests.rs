use std::sync::Arc;

use sim_kernel::{DefaultFactory, Expr, NoopEvalPolicy, NumberLiteral, Symbol};
use sim_lib_numbers_core::domains;

use crate::{F32NumbersLib, number_domain};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&F32NumbersLib::new()).unwrap();
    cx
}

#[test]
fn f32_literal_round_trips_through_expr() {
    let mut cx = cx();
    let value = cx
        .factory()
        .number_literal(number_domain(), "1.5".to_owned())
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: number_domain(),
            canonical: "1.5".to_owned(),
        })
    );
}

#[test]
fn f32_arithmetic_adds_values() {
    let mut cx = cx();
    let left = cx
        .factory()
        .number_literal(number_domain(), "1.25".to_owned())
        .unwrap();
    let right = cx
        .factory()
        .number_literal(number_domain(), "2.5".to_owned())
        .unwrap();
    let value = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "add"), left, right)
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: number_domain(),
            canonical: "3.75".to_owned(),
        })
    );
}

#[test]
fn f32_registers_value_promotion_to_f64() {
    let cx = cx();
    assert!(
        cx.registry()
            .value_promotion_rules()
            .iter()
            .any(|rule| rule.from_domain == domains::f32() && rule.to_domain == domains::f64())
    );
}
