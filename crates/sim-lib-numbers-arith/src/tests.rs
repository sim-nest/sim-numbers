use std::sync::Arc;

use sim_kernel::{Args, DefaultFactory, Expr, NoopEvalPolicy, NumberLiteral, Symbol};

use crate::NumbersArithmeticLib;

#[test]
fn arithmetic_lib_registers_aliases() {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&NumbersArithmeticLib::new()).unwrap();
    for symbol in [Symbol::new("+"), Symbol::new("-"), Symbol::new("*")] {
        assert!(cx.registry().function_by_symbol(&symbol).is_some());
    }
}

#[test]
fn arithmetic_function_dispatches_to_registered_number_ops() {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&NumbersArithmeticLib::new()).unwrap();
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    let value = cx
        .call_function(
            &Symbol::new("+"),
            Args::new(vec![
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "i64"), "2".to_owned())
                    .unwrap(),
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "i64"), "3".to_owned())
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "5".to_owned(),
        })
    );
}

#[test]
fn cmp_function_dispatches_through_number_promotion() {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&NumbersArithmeticLib::new()).unwrap();
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    let value = cx
        .call_function(
            &Symbol::qualified("math", "cmp"),
            Args::new(vec![
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "i64"), "2".to_owned())
                    .unwrap(),
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "f64"), "2.0".to_owned())
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "0".to_owned(),
        })
    );
}

#[test]
fn cmp_function_orders_bigint_values() {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&NumbersArithmeticLib::new()).unwrap();
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_bigint::BigIntNumbersLib::new())
        .unwrap();
    let value = cx
        .call_function(
            &Symbol::qualified("math", "cmp"),
            Args::new(vec![
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "bigint"), "5".to_owned())
                    .unwrap(),
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "bigint"), "3".to_owned())
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "1".to_owned(),
        })
    );
}
