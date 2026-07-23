use sim_kernel::{Args, DefaultFactory, Factory};

use crate::{
    Tensor, build_tensor_value, cast_symbol, cast_tensor, domains, parse_f32_literal_cell,
    parse_i64_literal_cell, tensor_value_ref,
};

use super::{number, test_cx};

#[test]
fn tensor_cast_to_f32_rounds_and_preserves_signed_zero_and_specials() {
    let mut cx = test_cx();
    let tensor = build_tensor_value(
        &mut cx,
        vec![4],
        Some(domains::f64()),
        vec![
            number("f64", "16777217"),
            number("f64", "-0"),
            number("f64", "inf"),
            number("f64", "NaN"),
        ],
    )
    .unwrap();
    let tensor = tensor_value_ref(&tensor).unwrap();
    let cast = cast_tensor(tensor, domains::f32()).unwrap();
    assert_eq!(cast.dtype(), &domains::f32());
    let cells = cast.cells().unwrap();
    assert_eq!(parse_f32_literal_cell(&cells[0]).unwrap(), 16_777_216.0);
    assert!(
        parse_f32_literal_cell(&cells[1])
            .unwrap()
            .is_sign_negative()
    );
    assert!(parse_f32_literal_cell(&cells[2]).unwrap().is_infinite());
    assert!(parse_f32_literal_cell(&cells[3]).unwrap().is_nan());
}

#[test]
fn tensor_cast_to_i64_uses_ties_even_and_checks_specials() {
    let mut cx = test_cx();
    let tensor = build_tensor_value(
        &mut cx,
        vec![3],
        Some(domains::f64()),
        vec![
            number("f64", "2.5"),
            number("f64", "3.5"),
            number("f64", "-2.5"),
        ],
    )
    .unwrap();
    let cast = cast_tensor(tensor_value_ref(&tensor).unwrap(), domains::i64()).unwrap();
    let cells = cast.cells().unwrap();
    assert_eq!(
        cells
            .iter()
            .map(parse_i64_literal_cell)
            .collect::<Option<Vec<_>>>()
            .unwrap(),
        vec![2, 4, -2]
    );

    let non_finite = build_tensor_value(
        &mut cx,
        vec![1],
        Some(domains::f64()),
        vec![number("f64", "inf")],
    )
    .unwrap();
    let err = match cast_tensor(tensor_value_ref(&non_finite).unwrap(), domains::i64()) {
        Ok(_) => panic!("non-finite cast to i64 must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("non-finite"));
}

#[test]
fn tensor_cast_to_half_rejects_finite_overflow() {
    let mut cx = test_cx();
    let tensor = build_tensor_value(
        &mut cx,
        vec![3],
        Some(domains::f32()),
        vec![
            number("f32", "1.5"),
            number("f32", "-0"),
            number("f32", "inf"),
        ],
    )
    .unwrap();
    let half = cast_tensor(tensor_value_ref(&tensor).unwrap(), domains::f16()).unwrap();
    assert_eq!(half.dtype(), &domains::f16());
    let widened = cast_tensor(&half, domains::f32()).unwrap();
    let cells = widened.cells().unwrap();
    assert_eq!(parse_f32_literal_cell(&cells[0]).unwrap(), 1.5);
    assert!(
        parse_f32_literal_cell(&cells[1])
            .unwrap()
            .is_sign_negative()
    );
    assert!(parse_f32_literal_cell(&cells[2]).unwrap().is_infinite());

    let overflowing = build_tensor_value(
        &mut cx,
        vec![1],
        Some(domains::f32()),
        vec![number("f32", "70000")],
    )
    .unwrap();
    let err = match cast_tensor(tensor_value_ref(&overflowing).unwrap(), domains::f16()) {
        Ok(_) => panic!("finite f16 overflow must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("overflows"));
}

#[test]
fn tensor_cast_function_is_explicit_and_rejects_unsupported_domains() {
    let mut cx = test_cx();
    let tensor = build_tensor_value(
        &mut cx,
        vec![2],
        Some(domains::f64()),
        vec![number("f64", "1.25"), number("f64", "2.5")],
    )
    .unwrap();
    let cast = cx
        .call_function(
            &cast_symbol(),
            Args::new(vec![tensor, DefaultFactory.symbol(domains::f32()).unwrap()]),
        )
        .unwrap();
    let cast = tensor_value_ref(&cast).unwrap();
    assert_eq!(cast.dtype(), &domains::f32());
    let cells = cast.cells().unwrap();
    assert_eq!(parse_f32_literal_cell(&cells[0]).unwrap(), 1.25);

    let complex =
        Tensor::new_exact(vec![1], domains::complex(), vec![number("complex", "1+2i")]).unwrap();
    let err = match cast_tensor(&complex, domains::f32()) {
        Ok(_) => panic!("complex source cast to f32 must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("source dtype"));

    let err = match cast_tensor(cast, domains::bool()) {
        Ok(_) => panic!("unsupported target dtype must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("target dtype"));
}
