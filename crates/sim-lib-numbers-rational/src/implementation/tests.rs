use std::sync::Arc;

use num_bigint::BigInt;
use sim_codec::{Input, decode_with_codec, encode_value_with_codec};
use sim_codec_lisp::{LispCodecLib, encode_object_lisp};
use sim_kernel::{
    CapabilitySet, DefaultFactory, EagerPolicy, EncodeOptions, EncodePosition, Error, Expr,
    ReadPolicy, Symbol, TrustLevel, read_construct_capability,
};
use sim_lib_numbers_core::MagnitudeLimit;

use crate::{
    RationalNumbersLib, f64_decimal_to_rational_checked,
    implementation::{add_symbol, ops::rational_add_value_rule, value::make_rational},
};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_bigint::BigIntNumbersLib::new())
        .unwrap();
    cx.load_lib(&RationalNumbersLib::new()).unwrap();
    cx
}

fn assert_eval_error_contains(err: Error, needles: &[&str]) {
    let Error::Eval(message) = err else {
        panic!("expected Eval error, got {err:?}");
    };
    for needle in needles {
        assert!(
            message.contains(needle),
            "expected error {message:?} to contain {needle:?}"
        );
    }
}

#[test]
fn rational_addition_stays_exact() {
    let mut cx = cx();
    let value = cx
        .apply_value_number_binary_op(
            &add_symbol(),
            cx.factory()
                .number_literal(Symbol::qualified("numbers", "rational"), "1/2".to_owned())
                .unwrap(),
            cx.factory()
                .number_literal(Symbol::qualified("numbers", "rational"), "1/3".to_owned())
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(sim_kernel::NumberLiteral {
            domain: Symbol::qualified("numbers", "rational"),
            canonical: "5/6".to_owned(),
        })
    );
}

#[test]
fn mixed_bigint_and_i64_rational_values_add() {
    let mut cx = cx();
    let numerator = cx
        .factory()
        .number_literal(
            Symbol::qualified("numbers", "bigint"),
            "1267650600228229401496703205376".to_owned(),
        )
        .unwrap();
    let denominator = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "i64"), "3".to_owned())
        .unwrap();
    let rational = make_rational(&mut cx, numerator, denominator).unwrap();
    let one_third = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "rational"), "1/3".to_owned())
        .unwrap();
    let value = rational_add_value_rule(&mut cx, rational, one_third).unwrap();
    assert_eq!(
        value.object().as_expr(&mut cx).unwrap(),
        Expr::Number(sim_kernel::NumberLiteral {
            domain: Symbol::qualified("numbers", "rational"),
            canonical: "1267650600228229401496703205377/3".to_owned(),
        })
    );
}

#[test]
fn decimal_to_rational_accepts_ordinary_precision() {
    assert_eq!(
        f64_decimal_to_rational_checked("0.25").unwrap(),
        Some((BigInt::from(1), BigInt::from(4)))
    );
}

#[test]
fn decimal_to_rational_rejects_excessive_fractional_precision() {
    let digits = MagnitudeLimit::default_arbitrary_precision().max_decimal_digits() + 1;
    let literal = format!("0.{}", "1".repeat(digits));
    let err = f64_decimal_to_rational_checked(&literal).unwrap_err();
    assert_eval_error_contains(err, &["decimal literal too precise", "exceeds limit"]);
}

#[test]
fn noncompact_rationals_round_trip_as_read_constructs() {
    let mut cx = cx();
    let lisp_id = cx.registry_mut().fresh_codec_id();
    let lisp = LispCodecLib::new(lisp_id).unwrap();
    cx.load_lib(&lisp).unwrap();
    let numerator = cx
        .factory()
        .number_literal(
            Symbol::qualified("numbers", "bigint"),
            "1267650600228229401496703205376".to_owned(),
        )
        .unwrap();
    let denominator = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "i64"), "3".to_owned())
        .unwrap();
    let value = make_rational(&mut cx, numerator, denominator).unwrap();

    let compact_value = cx
        .factory()
        .number_literal(Symbol::qualified("numbers", "rational"), "1/2".to_owned())
        .unwrap();
    let compact = encode_value_with_codec(
        &mut cx,
        &Symbol::qualified("codec", "lisp"),
        &compact_value,
        EncodeOptions::default(),
    )
    .unwrap()
    .into_text()
    .unwrap();
    assert_eq!(compact, "1/2");

    cx.grant(read_construct_capability());
    let encoded = encode_object_lisp(
        &mut sim_kernel::WriteCx {
            cx: &mut cx,
            codec: lisp_id,
            options: EncodeOptions {
                position: EncodePosition::Quote,
                ..Default::default()
            },
        },
        value,
    )
    .unwrap();
    assert_eq!(
        encoded,
        "#(numbers/Rational 1267650600228229401496703205376 3)"
    );

    let decoded = decode_with_codec(
        &mut cx,
        &Symbol::qualified("codec", "lisp"),
        Input::Text(encoded),
        ReadPolicy {
            trust: TrustLevel::TrustedSource,
            capabilities: CapabilitySet::new().grant(read_construct_capability()),
        },
    )
    .unwrap();
    let Expr::Extension { tag, .. } = decoded else {
        panic!("expected decoded noncompact rational to stay structured");
    };
    assert_eq!(tag, Symbol::qualified("numbers", "Rational"));
}
