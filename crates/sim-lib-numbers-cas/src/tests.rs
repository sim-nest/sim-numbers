use std::sync::Arc;

use sim_codec::{Input, decode_with_codec, encode_value_with_codec};
use sim_codec_lisp::{LispCodecLib, encode_object_lisp};
use sim_kernel::{
    Args, CapabilitySet, CodecId, DefaultFactory, EagerPolicy, EncodeOptions, EncodePosition, Expr,
    NumberLiteral, QuoteMode, ReadPolicy, Result, Symbol, TrustLevel, Value, WriteCx,
    read_construct_capability,
};

use crate::{
    CasExpr, CasNumbersLib, canonical_eq, cas_expr_to_surface_expr, cas_expr_to_value,
    cas_simplify_symbol, cas_value_class_symbol, cas_var_symbol, expr_to_cas_expr, free_vars,
    simplify_expr, value_to_cas_expr,
};

mod simplify;

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

fn var(name: &str) -> CasExpr {
    CasExpr::Var(Symbol::new(name))
}

fn number_value(cx: &mut sim_kernel::Cx, canonical: &str) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "i64"), canonical.to_owned())
        .unwrap()
}

fn cas_number(cx: &mut sim_kernel::Cx, canonical: &str) -> CasExpr {
    let value = number_value(cx, canonical);
    CasExpr::num(cx, value).unwrap()
}

fn read_construct_policy() -> ReadPolicy {
    ReadPolicy {
        trust: TrustLevel::TrustedSource,
        capabilities: CapabilitySet::new().grant(read_construct_capability()),
    }
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
fn roundtrip_cas_shaped_subset_semantic() {
    let mut cx = cx();
    let mut examples = Vec::new();

    examples.push(cas_number(&mut cx, "1"));
    examples.push(var("x"));
    examples.push(CasExpr::Op(
        Symbol::qualified("math", "add"),
        vec![var("x"), cas_number(&mut cx, "1")],
    ));
    examples.push(CasExpr::Op(
        Symbol::qualified("math", "neg"),
        vec![var("x")],
    ));
    examples.push(CasExpr::Op(
        Symbol::qualified("math", "rem"),
        vec![var("x"), cas_number(&mut cx, "2")],
    ));
    examples.push(CasExpr::Op(Symbol::new("sin"), vec![var("x")]));
    examples.push(CasExpr::Op(Symbol::new("cos"), vec![var("x")]));
    examples.push(CasExpr::Op(Symbol::new("ln"), vec![var("x")]));
    examples.push(CasExpr::Op(Symbol::new("exp"), vec![var("x")]));
    examples.push(CasExpr::Op(
        Symbol::new("diff"),
        vec![CasExpr::Op(Symbol::new("sin"), vec![var("x")]), var("x")],
    ));

    for expr in examples {
        let surface = cas_expr_to_surface_expr(&mut cx, &expr).unwrap();
        let parsed = expr_to_cas_expr(&mut cx, &surface).unwrap().unwrap();
        assert!(
            canonical_eq(&mut cx, &expr, &parsed).unwrap(),
            "round-trip changed {surface:?} into {parsed:?}"
        );
    }
}

#[test]
fn infix_prefix_postfix_enter_cas() {
    let mut cx = cx();
    let infix = Expr::Infix {
        operator: Symbol::new("+"),
        left: Box::new(Expr::Symbol(Symbol::new("a"))),
        right: Box::new(Expr::Symbol(Symbol::new("b"))),
    };
    let parsed = expr_to_cas_expr(&mut cx, &infix).unwrap().unwrap();
    assert!(
        canonical_eq(
            &mut cx,
            &parsed,
            &CasExpr::Op(Symbol::qualified("math", "add"), vec![var("a"), var("b")]),
        )
        .unwrap()
    );

    let prefix = Expr::Prefix {
        operator: Symbol::new("-"),
        arg: Box::new(Expr::Symbol(Symbol::new("x"))),
    };
    let parsed = expr_to_cas_expr(&mut cx, &prefix).unwrap().unwrap();
    assert!(
        canonical_eq(
            &mut cx,
            &parsed,
            &CasExpr::Op(Symbol::qualified("math", "neg"), vec![var("x")]),
        )
        .unwrap()
    );

    let postfix = Expr::Postfix {
        operator: Symbol::new("!"),
        arg: Box::new(Expr::Symbol(Symbol::new("n"))),
    };
    let parsed = expr_to_cas_expr(&mut cx, &postfix).unwrap().unwrap();
    assert!(
        canonical_eq(
            &mut cx,
            &parsed,
            &CasExpr::Op(Symbol::new("!"), vec![var("n")]),
        )
        .unwrap()
    );
}

#[test]
fn nested_non_cas_is_none_not_err() {
    let mut cx = cx();
    let non_cas = Expr::Call {
        operator: Box::new(Expr::Symbol(Symbol::new("f"))),
        args: vec![Expr::String("hi".to_owned())],
    };

    assert!(expr_to_cas_expr(&mut cx, &non_cas).unwrap().is_none());

    let value = cx.factory().expr(non_cas).unwrap();
    let error = value_to_cas_expr(&mut cx, value).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected a numeric or symbolic CAS input")
    );
}

#[test]
fn cas_read_construct_roundtrips_noncanonical_operator() {
    let mut cx = cx();
    cx.grant(read_construct_capability());
    let version = cx.factory().symbol(Symbol::new("v1")).unwrap();
    let expr_value = cx
        .factory()
        .expr(Expr::List(vec![
            Expr::Symbol(Symbol::new("sin")),
            Expr::Symbol(Symbol::new("x")),
        ]))
        .unwrap();
    let constructed = cx
        .read_construct(&cas_value_class_symbol(), vec![version, expr_value])
        .unwrap();
    let original = value_to_cas_expr(&mut cx, constructed.clone()).unwrap();
    assert!(
        canonical_eq(
            &mut cx,
            &original,
            &CasExpr::Op(Symbol::new("sin"), vec![var("x")]),
        )
        .unwrap()
    );

    let default_encoded = encode_value_with_codec(
        &mut cx,
        &Symbol::qualified("codec", "lisp"),
        &constructed,
        EncodeOptions::default(),
    )
    .unwrap()
    .into_text()
    .unwrap();
    assert_eq!(default_encoded, "(sin x)");

    let encoded_quote = {
        let mut write = WriteCx {
            cx: &mut cx,
            codec: CodecId(0),
            options: EncodeOptions {
                position: EncodePosition::Quote,
                ..Default::default()
            },
        };
        encode_object_lisp(&mut write, constructed.clone()).unwrap()
    };
    assert_eq!(encoded_quote, "#(numbers/Cas v1 (sin x))");

    let decoded = decode_with_codec(
        &mut cx,
        &Symbol::qualified("codec", "lisp"),
        Input::Text(encoded_quote.clone()),
        read_construct_policy(),
    )
    .unwrap();
    let decoded_expr = expr_to_cas_expr(&mut cx, &decoded).unwrap().unwrap();
    assert!(canonical_eq(&mut cx, &original, &decoded_expr).unwrap());

    let decoded_value = cas_expr_to_value(&mut cx, decoded_expr).unwrap();
    let reencoded = {
        let mut write = WriteCx {
            cx: &mut cx,
            codec: CodecId(0),
            options: EncodeOptions {
                position: EncodePosition::Quote,
                ..Default::default()
            },
        };
        encode_object_lisp(&mut write, decoded_value).unwrap()
    };
    assert_eq!(reencoded, encoded_quote);
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
