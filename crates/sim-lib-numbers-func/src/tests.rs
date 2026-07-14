use std::sync::Arc;

use sim_kernel::{
    Args, Callable, Cx, DefaultFactory, EagerPolicy, Error, Expr, LengthResult, NumberLiteral,
    QuoteMode, Symbol,
};
use sim_lib_numbers_cas::{CasExpr, canonical_eq, expr_to_cas_expr};
use sim_lib_numbers_cas_diff::{diff_symbol, integrate_sym_symbol};

use crate::{Func, FuncMetadata, FuncNumbersLib, SymbolicStatus, fn_symbol, grad_symbol};

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

fn quoted(name: &str) -> Expr {
    Expr::Quote {
        mode: QuoteMode::Quote,
        expr: Box::new(Expr::Symbol(Symbol::new(name))),
    }
}

fn number_value(cx: &mut Cx, text: &str) -> sim_kernel::Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "i64"), text.to_owned())
        .unwrap()
}

fn zero_func_value(cx: &mut Cx) -> sim_kernel::Value {
    let zero = number_value(cx, "0");
    let zero = CasExpr::num(cx, zero).unwrap();
    cx.factory()
        .opaque(Arc::new(Func::symbolic(Vec::new(), zero)))
        .unwrap()
}

fn cas_number(cx: &mut Cx, text: &str) -> CasExpr {
    let value = number_value(cx, text);
    CasExpr::num(cx, value).unwrap()
}

fn symbolic_func_value(cx: &mut Cx, vars: Vec<Symbol>, body: CasExpr) -> sim_kernel::Value {
    cx.factory()
        .opaque(Arc::new(Func::symbolic(vars, body)))
        .unwrap()
}

fn native_increment_func_value(cx: &mut Cx) -> sim_kernel::Value {
    cx.factory()
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
        .unwrap()
}

fn returned_func_body(cx: &mut Cx, value: &sim_kernel::Value) -> CasExpr {
    let Expr::Call { operator, args } = value.object().as_expr(cx).unwrap() else {
        panic!("expected returned function surface");
    };
    assert_eq!(operator.as_ref(), &Expr::Symbol(fn_symbol()));
    let [_, body_expr] = args.as_slice() else {
        panic!("expected fn surface to carry vars and a body");
    };
    expr_to_cas_expr(cx, body_expr)
        .unwrap()
        .expect("returned function body should be CAS-compatible")
}

fn promote_to_func_via_add(cx: &mut Cx, value: sim_kernel::Value) -> sim_kernel::Value {
    let zero = zero_func_value(cx);
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "add"), value, zero)
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
fn symbolic_plus_symbolic_stays_symbolic() {
    let mut cx = test_cx();
    let x = Symbol::new("x");
    let left = symbolic_func_value(&mut cx, vec![x.clone()], CasExpr::Var(x.clone()));
    let right = symbolic_func_value(&mut cx, vec![x.clone()], CasExpr::Var(x.clone()));

    let sum = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "add"), left, right)
        .unwrap();
    let func = sum.object().downcast_ref::<Func>().unwrap();

    assert!(func.body_cas().is_some());
    assert_eq!(func.symbolic_status(), SymbolicStatus::Available);
}

#[test]
fn symbolic_plus_native_reports_mixed_native_loss() {
    let mut cx = test_cx();
    let x = Symbol::new("x");
    let symbolic = symbolic_func_value(&mut cx, vec![x.clone()], CasExpr::Var(x));
    let native = native_increment_func_value(&mut cx);

    let mixed = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "add"), symbolic, native)
        .unwrap();
    let func = mixed.object().downcast_ref::<Func>().unwrap();

    assert!(func.body_cas().is_none());
    assert_eq!(func.symbolic_status(), SymbolicStatus::mixed_native());
    let err = cx
        .call_function(
            &diff_symbol(),
            Args::new(vec![mixed, cx.factory().expr(quoted("x")).unwrap()]),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Eval(message) if message.contains("numbers/func/mixed-native")));
}

#[test]
fn mixed_native_eval_unchanged() {
    let mut cx = test_cx();
    let x = Symbol::new("x");
    let symbolic = symbolic_func_value(&mut cx, vec![x.clone()], CasExpr::Var(x));
    let native = native_increment_func_value(&mut cx);

    let mixed = cx
        .apply_value_number_binary_op(&Symbol::qualified("math", "add"), symbolic, native)
        .unwrap();
    let arg = number_value(&mut cx, "4");
    let out = cx.call_value(mixed, Args::new(vec![arg])).unwrap();

    assert_eq!(out.object().as_expr(&mut cx).unwrap(), number("9"));
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
    assert!(matches!(err, Error::Eval(message) if message.contains("numbers/func/native-only")));
}

#[test]
fn diff_func_with_sin_body_survives_surface_bridge() {
    let mut cx = test_cx();
    let x = Symbol::new("x");
    let func = symbolic_func_value(
        &mut cx,
        vec![x.clone()],
        CasExpr::Op(Symbol::new("sin"), vec![CasExpr::Var(x.clone())]),
    );
    let var = cx.factory().expr(quoted("x")).unwrap();

    let derivative = cx
        .call_function(&diff_symbol(), Args::new(vec![func, var]))
        .unwrap();
    let body = returned_func_body(&mut cx, &derivative);
    let expected = CasExpr::Op(Symbol::new("cos"), vec![CasExpr::Var(x)]);

    assert!(canonical_eq(&mut cx, &body, &expected).unwrap());
}

#[test]
fn integrate_func_with_pow_body_survives_surface_bridge() {
    let mut cx = test_cx();
    let x = Symbol::new("x");
    let exponent = cas_number(&mut cx, "2");
    let func = symbolic_func_value(
        &mut cx,
        vec![x.clone()],
        CasExpr::Op(
            Symbol::qualified("math", "pow"),
            vec![CasExpr::Var(x.clone()), exponent],
        ),
    );
    let var = cx.factory().expr(quoted("x")).unwrap();

    let integral = cx
        .call_function(&integrate_sym_symbol(), Args::new(vec![func, var]))
        .unwrap();
    let body = returned_func_body(&mut cx, &integral);
    let coefficient = cx
        .factory()
        .number_literal(
            Symbol::qualified("numbers", "f64"),
            "0.3333333333333333".to_owned(),
        )
        .unwrap();
    let cubed_exponent = cas_number(&mut cx, "3");
    let expected = CasExpr::Op(
        Symbol::qualified("math", "mul"),
        vec![
            CasExpr::num(&mut cx, coefficient).unwrap(),
            CasExpr::Op(
                Symbol::qualified("math", "pow"),
                vec![CasExpr::Var(x), cubed_exponent],
            ),
        ],
    );

    assert!(canonical_eq(&mut cx, &body, &expected).unwrap());
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
    assert_eq!(func.symbolic_status(), SymbolicStatus::ProvidedByHint);
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

    let symbolic = Func::symbolic(vec![Symbol::new("x")], CasExpr::Var(Symbol::new("x")));
    assert!(!symbolic.is_native());
    assert!(symbolic.body_cas().is_some());
}

#[test]
fn promote_symbolic_cas_lifts_free_var() {
    let mut cx = test_cx();
    let x = Symbol::new("x");
    let cas = sim_lib_numbers_cas::cas_expr_to_value(&mut cx, CasExpr::Var(x.clone())).unwrap();

    let promoted = promote_to_func_via_add(&mut cx, cas);
    let func = promoted.object().downcast_ref::<Func>().unwrap();
    assert_eq!(func.vars, vec![x]);

    let gradient = cx
        .call_function(&grad_symbol(), Args::new(vec![promoted]))
        .unwrap();
    let gradient = gradient.object().as_list().expect("grad returns a list");
    assert_eq!(gradient.len(&mut cx).unwrap(), LengthResult::Known(1));
}

#[test]
fn promote_concrete_number_is_nullary() {
    let mut cx = test_cx();
    let three = number_value(&mut cx, "3");
    let promoted = promote_to_func_via_add(&mut cx, three);
    let func = promoted.object().downcast_ref::<Func>().unwrap();
    assert!(func.vars.is_empty());

    let out = cx.call_value(promoted, Args::new(Vec::new())).unwrap();
    assert_eq!(out.object().as_expr(&mut cx).unwrap(), number("3"));
}
