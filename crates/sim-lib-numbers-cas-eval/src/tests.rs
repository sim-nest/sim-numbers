use std::{
    any::Any,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use sim_kernel::{
    Args, DefaultFactory, EagerPolicy, Env, Expr, NumberLiteral, NumberValue, Object, ObjectCompat,
    QuoteMode, Result, Symbol, Value,
};
use sim_lib_numbers_cas::CasExpr;

use crate::{CasEvalLib, cas_to_expr, eval_cas, eval_cas_symbol, eval_cas_symbolic, expr_to_cas};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas::CasNumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_cas_diff::CasDiffLib::new())
        .unwrap();
    cx.load_lib(&CasEvalLib::new()).unwrap();
    cx
}

fn quoted(name: &str) -> Expr {
    Expr::Quote {
        mode: QuoteMode::Quote,
        expr: Box::new(Expr::Symbol(Symbol::new(name))),
    }
}

struct InspectErrorNumber {
    inspections: AtomicUsize,
}

impl InspectErrorNumber {
    fn new() -> Self {
        Self {
            inspections: AtomicUsize::new(0),
        }
    }
}

impl Object for InspectErrorNumber {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<inspect-error-number>".to_owned())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ObjectCompat for InspectErrorNumber {
    fn as_expr(&self, _cx: &mut sim_kernel::Cx) -> Result<Expr> {
        Ok(Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "7".to_owned(),
        }))
    }

    fn as_number_value(&self) -> Option<&dyn NumberValue> {
        Some(self)
    }
}

impl NumberValue for InspectErrorNumber {
    fn number_domain(&self, _cx: &mut sim_kernel::Cx) -> Result<Symbol> {
        if self.inspections.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(sim_kernel::Error::Eval(
                "test number inspection failed".to_owned(),
            ));
        }
        Ok(Symbol::qualified("numbers", "i64"))
    }

    fn number_literal(&self, _cx: &mut sim_kernel::Cx) -> Result<Option<NumberLiteral>> {
        Ok(Some(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "7".to_owned(),
        }))
    }
}

fn inspect_error_number(cx: &mut sim_kernel::Cx) -> Value {
    cx.factory()
        .opaque(Arc::new(InspectErrorNumber::new()))
        .unwrap()
}

#[test]
fn expr_round_trips_through_cas_conversion() {
    let mut cx = cx();
    let expr = Expr::List(vec![
        Expr::Symbol(Symbol::new("+")),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "1".to_owned(),
        }),
        Expr::Symbol(Symbol::new("x")),
    ]);
    let cas = expr_to_cas(&mut cx, &expr).unwrap();
    assert_eq!(cas_to_expr(&mut cx, &cas).unwrap(), expr);
}

#[test]
fn evals_symbolic_sum_against_env() {
    let mut cx = cx();
    let expr = expr_to_cas(
        &mut cx,
        &Expr::List(vec![
            Expr::Symbol(Symbol::new("+")),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "1".to_owned(),
            }),
            Expr::Symbol(Symbol::new("x")),
        ]),
    )
    .unwrap();
    let mut env = Env::default();
    env.define(
        Symbol::new("x"),
        cx.factory()
            .number_literal(Symbol::qualified("numbers", "i64"), "3".to_owned())
            .unwrap(),
    );
    let value = eval_cas(&mut cx, &expr, &env).unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "4".to_owned(),
        })
    );
}

#[test]
fn inspect_error_propagates_from_symbolic_eval() {
    let mut cx = cx();
    let expr = CasExpr::Op(
        Symbol::qualified("math", "add"),
        vec![
            CasExpr::Num(inspect_error_number(&mut cx)),
            CasExpr::Var(Symbol::new("x")),
        ],
    );

    let err = eval_cas_symbolic(&mut cx, &expr, &Env::default()).unwrap_err();

    assert!(err.to_string().contains("test number inspection failed"));
}

#[test]
fn derivative_evaluates_at_a_point() {
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
    let derivative = cx
        .call_function(&Symbol::new("diff"), Args::new(vec![polynomial, var]))
        .unwrap();
    let x = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "i64"), "3".to_owned())
        .unwrap();
    cx.env_mut().define(Symbol::new("x"), x);
    let value = cx
        .call_function(&eval_cas_symbol(), Args::new(vec![derivative]))
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "8".to_owned(),
        })
    );
}
