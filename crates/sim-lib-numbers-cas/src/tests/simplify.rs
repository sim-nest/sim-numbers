use std::{
    any::Any,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use sim_kernel::{NumberValue, Object, ObjectCompat};

use super::*;

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
fn cas_simplifier_folds_nested_constants() {
    let mut cx = cx();
    let inner = cx
        .eval_expr(Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("+"))),
            args: vec![
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "1".to_owned(),
                }),
                Expr::Quote {
                    mode: QuoteMode::Quote,
                    expr: Box::new(Expr::Symbol(Symbol::new("a"))),
                },
            ],
        })
        .unwrap();
    let value = cx
        .call_function(
            &Symbol::qualified("math", "add"),
            Args::new(vec![
                inner,
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "i64"), "2".to_owned())
                    .unwrap(),
            ]),
        )
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::List(vec![
            Expr::Symbol(Symbol::new("+")),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "3".to_owned(),
            }),
            Expr::Symbol(Symbol::new("a")),
        ])
    );
}

#[test]
fn cas_simplifier_absorbs_zero_product() {
    let mut cx = cx();
    let symbolic = cx
        .eval_expr(Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("*"))),
            args: vec![
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "0".to_owned(),
                }),
                Expr::Quote {
                    mode: QuoteMode::Quote,
                    expr: Box::new(Expr::Symbol(Symbol::new("x"))),
                },
            ],
        })
        .unwrap();
    let value = cx
        .call_function(&cas_simplify_symbol(), Args::new(vec![symbolic]))
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
fn inspect_error_propagates_from_simplify() {
    let mut cx = cx();
    let tree = CasExpr::Op(
        Symbol::qualified("math", "add"),
        vec![CasExpr::Num(inspect_error_number(&mut cx)), var("x")],
    );

    let err = simplify_expr(&mut cx, tree).unwrap_err();

    assert!(err.to_string().contains("test number inspection failed"));
}

#[test]
fn zero_pow_zero_not_one() {
    let mut cx = cx();
    let zero = cas_number(&mut cx, "0");
    let tree = CasExpr::Op(
        Symbol::qualified("math", "pow"),
        vec![zero.clone(), zero.clone()],
    );

    let simplified = simplify_expr(&mut cx, tree).unwrap();

    assert_eq!(
        cas_expr_to_surface_expr(&mut cx, &simplified).unwrap(),
        Expr::List(vec![
            Expr::Symbol(Symbol::new("^")),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "0".to_owned(),
            }),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "0".to_owned(),
            }),
        ])
    );
}

#[test]
fn constant_add_and_mul_fold() {
    let mut cx = cx();
    let add_args = vec![
        cas_number(&mut cx, "1"),
        cas_number(&mut cx, "2"),
        cas_number(&mut cx, "3"),
    ];
    let add = simplify_expr(
        &mut cx,
        CasExpr::Op(Symbol::qualified("math", "add"), add_args),
    )
    .unwrap();
    assert_eq!(
        cas_expr_to_surface_expr(&mut cx, &add).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "6".to_owned(),
        })
    );

    let mul_args = vec![
        cas_number(&mut cx, "2"),
        cas_number(&mut cx, "3"),
        cas_number(&mut cx, "4"),
    ];
    let mul = simplify_expr(
        &mut cx,
        CasExpr::Op(Symbol::qualified("math", "mul"), mul_args),
    )
    .unwrap();
    assert_eq!(
        cas_expr_to_surface_expr(&mut cx, &mul).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "24".to_owned(),
        })
    );
}

#[test]
fn cas_simplify_propagates_sort_key_error_instead_of_panicking() {
    use sim_kernel::{Cx, Error};

    // A number value whose surface-`Expr` lowering fails, so computing its CAS
    // sort key errors on the public simplify path.
    struct UnlowerableNumber;

    impl Object for UnlowerableNumber {
        fn display(&self, _cx: &mut Cx) -> Result<String> {
            Ok("#<unlowerable-number>".to_owned())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl ObjectCompat for UnlowerableNumber {
        fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
            Err(Error::Eval(
                "unlowerable number cannot lower to a surface expr".to_owned(),
            ))
        }

        fn as_number_value(&self) -> Option<&dyn NumberValue> {
            Some(self)
        }
    }

    impl NumberValue for UnlowerableNumber {
        fn number_domain(&self, _cx: &mut Cx) -> Result<Symbol> {
            Ok(Symbol::qualified("numbers", "i64"))
        }
    }

    let mut cx = cx();
    let cell = cx.factory().opaque(Arc::new(UnlowerableNumber)).unwrap();
    let cell = CasExpr::num(&mut cx, cell).unwrap();
    // A commutative op with two retained operands reaches the sort; the number's
    // key lowering must surface as an Err rather than a panic.
    let tree = CasExpr::Op(
        Symbol::qualified("math", "add"),
        vec![cell, CasExpr::Var(Symbol::new("x"))],
    );
    let result = simplify_expr(&mut cx, tree);
    assert!(result.is_err());
}

#[test]
fn free_vars_preserves_first_seen_order() {
    let tree = CasExpr::Op(
        Symbol::qualified("math", "add"),
        vec![
            CasExpr::Var(Symbol::new("y")),
            CasExpr::Op(
                Symbol::qualified("math", "mul"),
                vec![
                    CasExpr::Var(Symbol::new("x")),
                    CasExpr::Var(Symbol::new("y")),
                ],
            ),
        ],
    );

    assert_eq!(free_vars(&tree), vec![Symbol::new("y"), Symbol::new("x")]);
}

#[test]
fn num_rejects_non_number_value() {
    let mut cx = cx();
    let text = cx.factory().string("not-a-number".to_owned()).unwrap();
    assert!(CasExpr::num(&mut cx, text).is_err());

    let list = cx.factory().list(Vec::new()).unwrap();
    assert!(CasExpr::num(&mut cx, list).is_err());
}
