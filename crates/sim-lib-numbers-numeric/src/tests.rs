use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, EagerPolicy, Error, Expr, Factory, NumberLiteral,
    Object, QuoteMode, Symbol, Value,
};
use sim_lib_numbers_cas::CasExpr;
use sim_lib_numbers_func::{Func, FuncMetadata};

use crate::{
    ComposedPipeline, DiffOpts, Differentiator, NumericKind, NumericNumbersLib, NumericPlugin,
    PipelineKind, StateKind, numeric_compose_symbol, register_differentiator,
};

fn test_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas::CasNumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas_diff::CasDiffLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas_eval::CasEvalLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_func::FuncNumbersLib::new())
        .unwrap();
    cx.load_lib(&NumericNumbersLib::new()).unwrap();
    cx
}

fn f64_number(text: &str) -> Expr {
    Expr::Number(NumberLiteral {
        domain: Symbol::qualified("numbers", "f64"),
        canonical: text.to_owned(),
    })
}

fn quoted(name: &str) -> Expr {
    Expr::Quote {
        mode: QuoteMode::Quote,
        expr: Box::new(Expr::Symbol(Symbol::new(name))),
    }
}

fn f64_value(cx: &mut Cx, canonical: &str) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "f64"), canonical.to_owned())
        .unwrap()
}

fn value_to_f64(cx: &mut Cx, value: &Value) -> f64 {
    value.object().display(cx).unwrap().parse::<f64>().unwrap()
}

#[derive(Clone)]
struct PlainUnary {
    name: &'static str,
    f: fn(f64) -> f64,
}

impl Object for PlainUnary {
    fn display(&self, _cx: &mut Cx) -> sim_kernel::Result<String> {
        Ok(format!("#<plain-callable {}>", self.name))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for PlainUnary {
    fn class(&self, _cx: &mut Cx) -> sim_kernel::Result<ClassRef> {
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }

    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for PlainUnary {
    fn call(&self, cx: &mut Cx, args: Args) -> sim_kernel::Result<Value> {
        let values = args.into_vec();
        let [x] = values.as_slice() else {
            return Err(Error::Eval("plain unary expected one arg".to_owned()));
        };
        let x = value_to_f64(cx, x);
        Ok(f64_value(cx, (self.f)(x).to_string().as_str()))
    }
}

fn plain_unary(cx: &mut Cx, name: &'static str, f: fn(f64) -> f64) -> Value {
    cx.factory()
        .opaque(Arc::new(PlainUnary { name, f }))
        .unwrap()
}

struct ExactTestDifferentiator {
    name: Symbol,
    sentinel: &'static str,
}

impl NumericPlugin for ExactTestDifferentiator {
    fn name(&self) -> Symbol {
        self.name.clone()
    }

    fn kind(&self) -> NumericKind {
        NumericKind::Differentiator
    }
}

impl Differentiator for ExactTestDifferentiator {
    fn diff_at(
        &self,
        cx: &mut Cx,
        _f: &Func,
        _var: &Symbol,
        _point: &Value,
        _opt: DiffOpts,
    ) -> sim_kernel::Result<Value> {
        Ok(f64_value(cx, self.sentinel))
    }
}

fn register_exact_test_differentiator(name: &str, sentinel: &'static str) -> Symbol {
    let name = Symbol::new(name);
    register_differentiator(Arc::new(ExactTestDifferentiator {
        name: name.clone(),
        sentinel,
    }))
    .unwrap();
    name
}

fn native_quadratic_func(cx: &mut Cx, metadata: FuncMetadata) -> Value {
    cx.factory()
        .opaque(Arc::new(Func::native_with(
            vec![Symbol::new("x")],
            Arc::new(|cx, args| {
                let [x] = args else {
                    return Err(Error::Eval("expected one arg".to_owned()));
                };
                let x2 = cx.apply_value_number_binary_op(
                    &Symbol::qualified("math", "mul"),
                    x.clone(),
                    x.clone(),
                )?;
                cx.apply_value_number_binary_op(&Symbol::qualified("math", "add"), x2, x.clone())
            }),
            metadata,
        )))
        .unwrap()
}

fn symbolic_quadratic_func(cx: &mut Cx, metadata: FuncMetadata) -> Value {
    let x = Symbol::new("x");
    let body = CasExpr::Op(
        Symbol::qualified("math", "add"),
        vec![
            CasExpr::Op(
                Symbol::qualified("math", "mul"),
                vec![CasExpr::Var(x.clone()), CasExpr::Var(x.clone())],
            ),
            CasExpr::Var(x.clone()),
        ],
    );
    cx.factory()
        .opaque(Arc::new(Func::symbolic_with(vec![x], body, metadata)))
        .unwrap()
}

fn numeric_diff_at_three(cx: &mut Cx, func: Value, method: Option<Symbol>) -> Value {
    let var = cx.factory().symbol(Symbol::new("x")).unwrap();
    let point = f64_value(cx, "3.0");
    let mut args = vec![func, var, point];
    if let Some(method) = method {
        let method = cx.factory().symbol(method).unwrap();
        let options = cx
            .factory()
            .table(vec![(Symbol::new(":method"), method)])
            .unwrap();
        args.push(options);
    }
    cx.call_function(&Symbol::new("numeric-diff"), Args::new(args))
        .unwrap()
}

fn native_identity_func(cx: &mut Cx) -> sim_kernel::Value {
    cx.factory()
        .opaque(Arc::new(Func::native(
            vec![Symbol::new("x")],
            Arc::new(|_cx, args| {
                let [x] = args else {
                    return Err(Error::Eval("expected one arg".to_owned()));
                };
                Ok(x.clone())
            }),
        )))
        .unwrap()
}

#[test]
fn composed_pipeline_table_value_round_trips() {
    let mut cx = test_cx();
    let pipeline = ComposedPipeline::new(
        sim_kernel::Ref::Symbol(Symbol::new("test-func")),
        PipelineKind::OdeSolve,
        Symbol::new("rk4"),
        StateKind::F64,
    );

    let table = pipeline.table_value(cx.factory()).unwrap();
    let table_impl = table.object().as_table_impl().unwrap();
    assert_eq!(
        table_impl
            .get(&mut cx, Symbol::new("kind"))
            .unwrap()
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::String("composed-pipeline".to_owned())
    );
    assert_eq!(
        table_impl
            .get(&mut cx, Symbol::new("domain"))
            .unwrap()
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::Symbol(Symbol::new("ode-solve"))
    );
    assert_eq!(
        table_impl
            .get(&mut cx, Symbol::new("method"))
            .unwrap()
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::Symbol(Symbol::new("rk4"))
    );
    assert_eq!(
        table_impl
            .get(&mut cx, Symbol::new("state"))
            .unwrap()
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::Symbol(Symbol::new("f64"))
    );
    assert_eq!(
        table_impl
            .get(&mut cx, Symbol::new("func"))
            .unwrap()
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::Symbol(Symbol::new("test-func"))
    );
}

#[test]
fn numeric_compose_returns_composed_pipeline_value() {
    let mut cx = test_cx();
    let func = native_identity_func(&mut cx);
    let kind = cx.factory().symbol(Symbol::new(":ode-solve")).unwrap();
    let method = cx.factory().symbol(Symbol::new("rk4")).unwrap();
    let state = cx.factory().symbol(Symbol::new(":f64")).unwrap();

    let value = cx
        .call_function(
            &numeric_compose_symbol(),
            Args::new(vec![func, kind, method, state]),
        )
        .unwrap();
    let pipeline = value.object().downcast_ref::<ComposedPipeline>().unwrap();
    assert_eq!(pipeline.kind, PipelineKind::OdeSolve);
    assert_eq!(pipeline.method, Symbol::new("rk4"));
    assert_eq!(pipeline.state, StateKind::F64);
    assert!(matches!(pipeline.func_ref, sim_kernel::Ref::Handle(_)));
}

#[test]
fn unknown_method_errors_cleanly() {
    let mut cx = test_cx();
    let err = cx
        .eval_expr(Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("numeric-diff"))),
            args: vec![
                Expr::Call {
                    operator: Box::new(Expr::Symbol(Symbol::new("fn"))),
                    args: vec![
                        Expr::List(vec![Expr::Symbol(Symbol::new("x"))]),
                        Expr::Symbol(Symbol::new("x")),
                    ],
                },
                quoted("x"),
                f64_number("2.0"),
                Expr::Symbol(Symbol::new(":method")),
                quoted("no-such-method"),
            ],
        })
        .unwrap_err();
    assert!(matches!(err, Error::Eval(message) if message.contains("UnknownNumericMethod")));
}

#[test]
fn native_func_can_be_passed_to_numeric_diff() {
    let mut cx = test_cx();
    let func = cx
        .factory()
        .opaque(Arc::new(Func::native(
            vec![Symbol::new("x")],
            Arc::new(|cx, args| {
                let [x] = args else {
                    return Err(Error::Eval("expected one arg".to_owned()));
                };
                let x2 = cx.apply_value_number_binary_op(
                    &Symbol::qualified("math", "mul"),
                    x.clone(),
                    x.clone(),
                )?;
                cx.apply_value_number_binary_op(&Symbol::qualified("math", "add"), x2, x.clone())
            }),
        )))
        .unwrap();
    let out = cx
        .call_function(
            &Symbol::new("numeric-diff"),
            Args::new(vec![
                func,
                cx.factory().expr(quoted("x")).unwrap(),
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "f64"), "3.0".to_owned())
                    .unwrap(),
            ]),
        )
        .unwrap();
    let rendered = out
        .object()
        .display(&mut cx)
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!((rendered - 7.0).abs() < 1.0e-3);
}

#[test]
fn numeric_diff_accepts_plain_callable() {
    let mut cx = test_cx();
    let func = plain_unary(&mut cx, "plain-quadratic-auto", |x| x * x + x);

    let out = numeric_diff_at_three(&mut cx, func, None);

    assert!((value_to_f64(&mut cx, &out) - 7.0).abs() < 1.0e-3);
}

#[test]
fn auto_uses_hinted_differentiator_before_finite_difference() {
    let mut cx = test_cx();
    let method = register_exact_test_differentiator("test-exact-review15-02", "42.25");
    let func = native_quadratic_func(
        &mut cx,
        FuncMetadata {
            differentiator_hint: Some(method),
            ..FuncMetadata::default()
        },
    );

    let out = numeric_diff_at_three(&mut cx, func, None);

    assert!((value_to_f64(&mut cx, &out) - 42.25).abs() < f64::EPSILON);
    let diagnostics = cx.take_diagnostics();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("method=auto->test-exact-review15-02")
    }));
}

#[test]
fn explicit_registered_differentiator_still_routes_directly() {
    let mut cx = test_cx();
    let method = register_exact_test_differentiator("test-explicit-review15-02", "42.25");
    let func = native_quadratic_func(&mut cx, FuncMetadata::default());

    let out = numeric_diff_at_three(&mut cx, func, Some(method));

    assert!((value_to_f64(&mut cx, &out) - 42.25).abs() < f64::EPSILON);
    let diagnostics = cx.take_diagnostics();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic
            .message
            .contains("method=test-explicit-review15-02")
    }));
}

#[test]
fn auto_without_hint_still_finite_difference() {
    let mut cx = test_cx();
    let func = native_quadratic_func(&mut cx, FuncMetadata::default());

    let out = numeric_diff_at_three(&mut cx, func, None);

    assert!((value_to_f64(&mut cx, &out) - 7.0).abs() < 1.0e-3);
    let diagnostics = cx.take_diagnostics();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("method=auto steps=2"))
    );
}

#[test]
fn symbolic_body_still_wins_over_hint() {
    let mut cx = test_cx();
    let method = register_exact_test_differentiator("test-symbolic-loser-review15-02", "42.25");
    let func = symbolic_quadratic_func(
        &mut cx,
        FuncMetadata {
            differentiator_hint: Some(method),
            ..FuncMetadata::default()
        },
    );

    let out = numeric_diff_at_three(&mut cx, func, None);

    assert_eq!(out.object().display(&mut cx).unwrap(), "7");
    let diagnostics = cx.take_diagnostics();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("method=auto steps=1"))
    );
}

#[test]
fn symbolic_func_still_uses_exact_diff() {
    let mut cx = test_cx();
    let out = cx
        .eval_expr(Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("numeric-diff"))),
            args: vec![
                Expr::Call {
                    operator: Box::new(Expr::Symbol(Symbol::new("fn"))),
                    args: vec![
                        Expr::List(vec![Expr::Symbol(Symbol::new("x"))]),
                        Expr::Call {
                            operator: Box::new(Expr::Symbol(Symbol::new("+"))),
                            args: vec![
                                Expr::Call {
                                    operator: Box::new(Expr::Symbol(Symbol::new("*"))),
                                    args: vec![quoted("x"), quoted("x")],
                                },
                                quoted("x"),
                            ],
                        },
                    ],
                },
                quoted("x"),
                f64_number("3.0"),
            ],
        })
        .unwrap();
    assert_eq!(out.object().display(&mut cx).unwrap(), "7");
    let diagnostics = cx.take_diagnostics();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("method=auto"))
    );
}
