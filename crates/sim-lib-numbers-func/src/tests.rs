use std::sync::Arc;

use sim_kernel::{
    Args, Cx, DefaultFactory, EagerPolicy, Error, Expr, NumberLiteral, QuoteMode, Symbol,
};

use crate::{Func, FuncNumbersLib};

fn test_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas::CasNumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas_diff::CasDiffLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas_eval::CasEvalLib::new())
        .unwrap();
    cx.load_lib(&FuncNumbersLib::new()).unwrap();
    cx
}

fn number(text: &str) -> Expr {
    Expr::Number(NumberLiteral {
        domain: Symbol::qualified("numbers", "i64"),
        canonical: text.to_owned(),
    })
}

#[test]
fn fn_builder_and_call_surface_work() {
    let mut cx = test_cx();
    let expr = Expr::Call {
        operator: Box::new(Expr::Symbol(Symbol::new("call"))),
        args: vec![
            Expr::Call {
                operator: Box::new(Expr::Symbol(Symbol::new("fn"))),
                args: vec![
                    Expr::List(vec![
                        Expr::Symbol(Symbol::new("x")),
                        Expr::Symbol(Symbol::new("y")),
                    ]),
                    Expr::Call {
                        operator: Box::new(Expr::Symbol(Symbol::new("+"))),
                        args: vec![
                            Expr::Symbol(Symbol::new("x")),
                            Expr::Symbol(Symbol::new("y")),
                        ],
                    },
                ],
            },
            number("2"),
            number("3"),
        ],
    };
    let value = cx.eval_expr(expr).unwrap();
    assert_eq!(value.object().as_expr(&mut cx).unwrap(), number("5"));
}

#[test]
fn function_plus_constant_remains_callable() {
    let mut cx = test_cx();
    let value = cx
        .eval_expr(Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("+"))),
            args: vec![
                Expr::Call {
                    operator: Box::new(Expr::Symbol(Symbol::new("fn"))),
                    args: vec![
                        Expr::List(vec![Expr::Symbol(Symbol::new("x"))]),
                        Expr::Symbol(Symbol::new("x")),
                    ],
                },
                number("1"),
            ],
        })
        .unwrap();
    let out = cx
        .call_value(
            value,
            Args::new(vec![
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "i64"), "4".to_owned())
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(out.object().as_expr(&mut cx).unwrap(), number("5"));
}

#[test]
fn native_only_functions_report_not_differentiable() {
    let mut cx = test_cx();
    let native = cx
        .factory()
        .opaque(Arc::new(Func::native(
            vec![Symbol::new("x")],
            Arc::new(|cx, args| {
                let [value] = args else {
                    return Err(Error::Eval("expected one arg".to_owned()));
                };
                cx.apply_value_number_binary_op(
                    &Symbol::qualified("math", "add"),
                    value.clone(),
                    cx.factory()
                        .number_literal(Symbol::qualified("numbers", "i64"), "1".to_owned())?,
                )
            }),
        )))
        .unwrap();
    let err = cx
        .call_function(
            &Symbol::new("diff"),
            Args::new(vec![
                native,
                cx.factory()
                    .expr(Expr::Quote {
                        mode: QuoteMode::Quote,
                        expr: Box::new(Expr::Symbol(Symbol::new("x"))),
                    })
                    .unwrap(),
            ]),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Eval(message) if message.contains("NotDifferentiable")));
}
