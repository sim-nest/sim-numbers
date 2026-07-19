use std::sync::Arc;

use sim_kernel::{DefaultFactory, Error, Expr, NoopEvalPolicy, NumberLiteral, Symbol, Value};
use sim_lib_numbers_core::{MagnitudeLimit, domains};

use crate::{BigIntNumbersLib, number_domain};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&BigIntNumbersLib::new()).unwrap();
    cx
}

fn bigint_value(cx: &mut sim_kernel::Cx, text: impl Into<String>) -> Value {
    cx.factory()
        .number_literal(number_domain(), text.into())
        .unwrap()
}

fn assert_eval_error_contains(err: Error, needles: &[&str]) {
    let Error::Eval(message) = err else {
        panic!("expected Eval error, got {err:?}");
    };
    for needle in needles {
        assert!(
            message.contains(needle),
            "expected error {message:?} to contain {needle:?}"
        );
    }
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
fn bigint_pow_keeps_normal_exact_result() {
    let mut cx = cx();
    let left = bigint_value(&mut cx, "2");
    let right = bigint_value(&mut cx, "64");
    let value = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "pow"), left, right)
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: number_domain(),
            canonical: "18446744073709551616".to_owned(),
        })
    );
}

#[test]
fn bigint_pow_rejects_huge_result_before_allocating() {
    let mut cx = cx();
    let left = bigint_value(&mut cx, "2");
    let right = bigint_value(&mut cx, "1000000");
    let err = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "pow"), left, right)
        .unwrap_err();
    assert_eval_error_contains(err, &["pow result", "too large"]);
}

#[test]
fn bigint_pow_allows_large_exponent_when_result_is_small() {
    let mut cx = cx();
    let left = bigint_value(&mut cx, "-1");
    let right = bigint_value(&mut cx, "1000001");
    let value = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "pow"), left, right)
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: number_domain(),
            canonical: "-1".to_owned(),
        })
    );
}

#[test]
fn bigint_mul_rejects_results_over_magnitude_limit() {
    let mut cx = cx();
    let factor_digits = MagnitudeLimit::default_arbitrary_precision().max_decimal_digits() / 2 + 2;
    let factor = "9".repeat(factor_digits);
    let left = bigint_value(&mut cx, factor.clone());
    let right = bigint_value(&mut cx, factor);
    let err = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "mul"), left, right)
        .unwrap_err();
    assert_eval_error_contains(err, &["mul result", "too large"]);
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
