use std::sync::Arc;

use sim_kernel::{Args, DefaultFactory, EagerPolicy, Expr, NumberLiteral, QuoteMode, Symbol};
use sim_lib_numbers_cas::CasExpr;

use crate::{CasDiffLib, diff_cas, diff_symbol, integrate_cas};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas::CasNumbersLib::new())
        .unwrap();
    cx.load_lib(&CasDiffLib::new()).unwrap();
    cx
}

fn quoted(name: &str) -> Expr {
    Expr::Quote {
        mode: QuoteMode::Quote,
        expr: Box::new(Expr::Symbol(Symbol::new(name))),
    }
}

#[test]
fn polynomial_derivative_simplifies() {
    let mut cx = cx();
    let polynomial = cx
        .eval_expr(Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("+"))),
            args: vec![
                Expr::Call {
                    operator: Box::new(Expr::Symbol(Symbol::new("*"))),
                    args: vec![
                        Expr::Number(NumberLiteral {
                            domain: Symbol::qualified("numbers", "i64"),
                            canonical: "2".to_owned(),
                        }),
                        quoted("x"),
                    ],
                },
                Expr::Call {
                    operator: Box::new(Expr::Symbol(Symbol::new("^"))),
                    args: vec![
                        quoted("x"),
                        Expr::Number(NumberLiteral {
                            domain: Symbol::qualified("numbers", "i64"),
                            canonical: "2".to_owned(),
                        }),
                    ],
                },
            ],
        })
        .unwrap();
    let var = cx.factory().expr(quoted("x")).unwrap();
    let value = cx
        .call_function(&diff_symbol(), Args::new(vec![polynomial, var]))
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::List(vec![
            Expr::Symbol(Symbol::new("+")),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "2".to_owned(),
            }),
            Expr::List(vec![
                Expr::Symbol(Symbol::new("*")),
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "2".to_owned(),
                }),
                Expr::Symbol(Symbol::new("x")),
            ]),
        ])
    );
}

#[test]
fn sin_derivative_is_cos() {
    let mut cx = cx();
    let expr = CasExpr::Op(Symbol::new("sin"), vec![CasExpr::Var(Symbol::new("x"))]);
    let derivative = diff_cas(&mut cx, &expr, &Symbol::new("x")).unwrap();
    assert_eq!(
        sim_lib_numbers_cas::cas_expr_to_value(&mut cx, derivative)
            .unwrap()
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::List(vec![
            Expr::Symbol(Symbol::new("cos")),
            Expr::Symbol(Symbol::new("x")),
        ])
    );
}

#[test]
fn product_derivative_with_respect_to_x_is_y() {
    let mut cx = cx();
    let expr = CasExpr::Op(
        Symbol::qualified("math", "mul"),
        vec![
            CasExpr::Var(Symbol::new("x")),
            CasExpr::Var(Symbol::new("y")),
        ],
    );
    let derivative = diff_cas(&mut cx, &expr, &Symbol::new("x")).unwrap();
    assert_eq!(
        sim_lib_numbers_cas::cas_expr_to_value(&mut cx, derivative)
            .unwrap()
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::Symbol(Symbol::new("y"))
    );
}

#[test]
fn integrate_power_with_max_exponent_errors_instead_of_overflowing() {
    let mut cx = cx();
    let exponent = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "i64"), i64::MAX.to_string())
        .unwrap();
    let expr = CasExpr::Op(
        Symbol::qualified("math", "pow"),
        vec![
            CasExpr::Var(Symbol::new("x")),
            CasExpr::num(&mut cx, exponent).unwrap(),
        ],
    );
    let err = integrate_cas(&mut cx, &expr, &Symbol::new("x")).unwrap_err();
    assert!(
        err.to_string().contains("overflow"),
        "expected an overflow diagnostic, got: {err}"
    );
}

#[test]
fn unknown_function_stays_as_diff_form() {
    let mut cx = cx();
    let expr = CasExpr::Op(
        Symbol::new("my-f"),
        vec![
            CasExpr::Var(Symbol::new("x")),
            CasExpr::Var(Symbol::new("y")),
        ],
    );
    let derivative = diff_cas(&mut cx, &expr, &Symbol::new("x")).unwrap();
    assert_eq!(
        sim_lib_numbers_cas::cas_expr_to_value(&mut cx, derivative)
            .unwrap()
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::List(vec![
            Expr::Symbol(Symbol::new("diff")),
            Expr::List(vec![
                Expr::Symbol(Symbol::new("my-f")),
                Expr::Symbol(Symbol::new("x")),
                Expr::Symbol(Symbol::new("y")),
            ]),
            Expr::Symbol(Symbol::new("x")),
        ])
    );
}
