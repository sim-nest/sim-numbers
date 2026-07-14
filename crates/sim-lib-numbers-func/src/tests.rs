use std::sync::Arc;

use sim_kernel::{
    Args, Callable, Cx, DefaultFactory, EagerPolicy, Error, Expr, NumberLiteral, QuoteMode, Symbol,
};

use crate::{Func, FuncMetadata, FuncNumbersLib};

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

fn number_value(cx: &mut Cx, text: &str) -> sim_kernel::Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "i64"), text.to_owned())
        .unwrap()
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

#[test]
fn native_wrong_arity_is_error_not_panic() {
    let mut cx = test_cx();
    let native = cx
        .factory()
        .opaque(Arc::new(Func::native(
            vec![Symbol::new("x")],
            Arc::new(|_cx, args| Ok(args[0].clone())),
        )))
        .unwrap();

    let err = cx.call_value(native, Args::new(Vec::new())).unwrap_err();

    assert!(
        matches!(err, Error::Eval(message) if message == "function expected 1 arguments but received 0")
    );
}

#[test]
fn browse_args_shape_reports_func_arity() {
    let mut cx = test_cx();
    let func = Func::native(
        vec![Symbol::new("x"), Symbol::new("y")],
        Arc::new(|_cx, args| Ok(args[0].clone())),
    );
    let shape = func
        .browse_args_shape(&mut cx)
        .unwrap()
        .expect("Func should report an argument shape");
    let shape = shape.object().as_shape().expect("shape protocol");
    let one = number_value(&mut cx, "1");
    let two = number_value(&mut cx, "2");
    let three = number_value(&mut cx, "3");

    let exact = cx.factory().list(vec![one.clone(), two.clone()]).unwrap();
    let too_short = cx.factory().list(vec![one.clone()]).unwrap();
    let too_long = cx
        .factory()
        .list(vec![one.clone(), two.clone(), three])
        .unwrap();

    assert!(shape.check_value(&mut cx, exact).unwrap().accepted);
    assert!(!shape.check_value(&mut cx, too_short).unwrap().accepted);
    assert!(!shape.check_value(&mut cx, too_long).unwrap().accepted);

    let func_value = cx.factory().opaque(Arc::new(func)).unwrap();
    let out = cx
        .call_value(func_value, Args::new(vec![one, two]))
        .unwrap();
    assert_eq!(out.object().as_expr(&mut cx).unwrap(), number("1"));
}

#[test]
fn native_with_preserves_metadata() {
    let mut cx = test_cx();
    let payload = cx.factory().string("payload".to_owned()).unwrap();
    let metadata = FuncMetadata {
        source: Some(Symbol::qualified("test", "native")),
        differentiator_hint: Some(Symbol::qualified("test", "diff")),
        payload: Some(payload),
    };

    let func = Func::native_with(
        vec![Symbol::new("x")],
        Arc::new(|_cx, args| Ok(args[0].clone())),
        metadata,
    );

    assert_eq!(
        func.metadata.source,
        Some(Symbol::qualified("test", "native"))
    );
    assert_eq!(
        func.metadata.differentiator_hint,
        Some(Symbol::qualified("test", "diff"))
    );
    let payload = func.metadata.payload.as_ref().expect("payload");
    assert_eq!(
        payload.object().as_expr(&mut cx).unwrap(),
        Expr::String("payload".to_owned())
    );
}

#[test]
fn body_mismatch_unrepresentable() {
    let native = Func::native(
        vec![Symbol::new("x")],
        Arc::new(|_cx, args| Ok(args[0].clone())),
    );
    assert!(native.is_native());
    assert!(native.body_cas().is_none());

    let symbolic = Func::symbolic(
        vec![Symbol::new("x")],
        sim_lib_numbers_cas::CasExpr::Var(Symbol::new("x")),
    );
    assert!(!symbolic.is_native());
    assert!(symbolic.body_cas().is_some());
}
