use std::sync::Arc;

use sim_codec::encode_value_with_codec;
use sim_codec_lisp::LispCodecLib;
use sim_kernel::{
    Args, DefaultFactory, EagerPolicy, EncodeOptions, Expr, NumberLiteral, QuoteMode, Symbol,
    read_construct_capability,
};

use crate::{
    CasExpr, CasNumbersLib, cas_simplify_symbol, cas_value_class_symbol, cas_var_symbol, free_vars,
    simplify_expr,
};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&CasNumbersLib::new()).unwrap();
    let lisp = LispCodecLib::new(cx.registry_mut().fresh_codec_id()).unwrap();
    cx.load_lib(&lisp).unwrap();
    cx
}

#[test]
fn cas_var_builds_symbolic_number() {
    let mut cx = cx();
    let value = cx
        .call_function(
            &cas_var_symbol(),
            Args::new(vec![
                cx.factory()
                    .expr(Expr::Quote {
                        mode: QuoteMode::Quote,
                        expr: Box::new(Expr::Symbol(Symbol::new("x"))),
                    })
                    .unwrap(),
            ]),
        )
        .unwrap();
    let encoded = encode_value_with_codec(
        &mut cx,
        &Symbol::qualified("codec", "lisp"),
        &value,
        EncodeOptions::default(),
    )
    .unwrap()
    .into_text()
    .unwrap();
    assert_eq!(encoded, "x");
}

#[test]
fn cas_add_prints_as_canonical_lisp() {
    let mut cx = cx();
    let expr = Expr::Call {
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
    };
    let value = cx.eval_expr(expr).unwrap();
    let encoded = encode_value_with_codec(
        &mut cx,
        &Symbol::qualified("codec", "lisp"),
        &value,
        EncodeOptions::default(),
    )
    .unwrap()
    .into_text()
    .unwrap();
    assert_eq!(encoded, "(+ 1 a)");
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
fn cas_simplify_propagates_sort_key_error_instead_of_panicking() {
    use std::any::Any;

    use sim_kernel::{Cx, Error, NumberValue, Object, ObjectCompat, Result};

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

#[test]
fn cas_citizen_read_constructor_round_trips_expression() {
    let mut cx = cx();
    let value = cx
        .eval_expr(Expr::Call {
            operator: Box::new(Expr::Symbol(Symbol::new("+"))),
            args: vec![
                Expr::Quote {
                    mode: QuoteMode::Quote,
                    expr: Box::new(Expr::Symbol(Symbol::new("x"))),
                },
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "1".to_owned(),
                }),
            ],
        })
        .unwrap();
    sim_citizen::check_value_fixture_with_wrong_version(
        &mut cx,
        value,
        Some(vec![
            Expr::Symbol(Symbol::new("v999")),
            Expr::Symbol(Symbol::new("x")),
        ]),
    )
    .unwrap();
}

#[test]
fn cas_ops_accept_citizen_values() {
    let mut cx = cx();
    cx.grant(read_construct_capability());
    let constructed = cx
        .read_construct(
            &cas_value_class_symbol(),
            vec![
                cx.factory().symbol(Symbol::new("v1")).unwrap(),
                cx.factory()
                    .expr(Expr::List(vec![
                        Expr::Symbol(Symbol::new("+")),
                        Expr::Symbol(Symbol::new("x")),
                        Expr::Number(NumberLiteral {
                            domain: Symbol::qualified("numbers", "i64"),
                            canonical: "1".to_owned(),
                        }),
                    ]))
                    .unwrap(),
            ],
        )
        .unwrap();
    let value = cx
        .call_function(
            &Symbol::qualified("math", "add"),
            Args::new(vec![
                constructed,
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
            Expr::Symbol(Symbol::new("x")),
        ])
    );
}
