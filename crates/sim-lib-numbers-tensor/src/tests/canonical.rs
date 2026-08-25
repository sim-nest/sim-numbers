use sim_kernel::Symbol;

use crate::{
    CanonicalAttrs, CpuTensorExecutor, PadMode, TensorExecutor, TensorMeta, allclose_op_symbol,
    arange_op_symbol, build_tensor_value, execute_canonical_tensor_op, isclose_op_symbol,
    maximum_op_symbol, nonzero_op_symbol, pad_op_symbol, tensor_value_ref,
};

use super::{number, test_cx};

fn tensor(
    cx: &mut sim_kernel::Cx,
    shape: Vec<usize>,
    domain: &str,
    values: &[&str],
) -> crate::Tensor {
    let dtype = Symbol::qualified("numbers", domain);
    let value = build_tensor_value(
        cx,
        shape,
        Some(dtype.clone()),
        values.iter().map(|v| number(domain, v)).collect(),
    )
    .unwrap();
    tensor_value_ref(&value).unwrap().clone()
}
fn text(cx: &mut sim_kernel::Cx, t: &crate::Tensor) -> Vec<String> {
    t.cells()
        .unwrap()
        .iter()
        .map(|v| v.object().display(cx).unwrap())
        .collect()
}

#[test]
fn construction_selection_and_edge_policies_are_explicit() {
    let mut cx = test_cx();
    let range = execute_canonical_tensor_op(
        &mut cx,
        arange_op_symbol(),
        vec![],
        TensorMeta::new(vec![3], Symbol::qualified("numbers", "f64")),
        CanonicalAttrs::Range {
            start: 0.0,
            stop: 1.0,
            step: 0.5,
            inclusive: true,
        },
    )
    .unwrap();
    assert_eq!(text(&mut cx, &range), ["0", "0.5", "1"]);
    let nz = execute_canonical_tensor_op(
        &mut cx,
        nonzero_op_symbol(),
        vec![range],
        TensorMeta::new(vec![2], Symbol::qualified("numbers", "i64")),
        CanonicalAttrs::None,
    )
    .unwrap();
    assert_eq!(text(&mut cx, &nz), ["1", "2"]);
    let empty = execute_canonical_tensor_op(
        &mut cx,
        arange_op_symbol(),
        vec![],
        TensorMeta::new(vec![0], Symbol::qualified("numbers", "f32")),
        CanonicalAttrs::Range {
            start: 1.0,
            stop: 1.0,
            step: 1.0,
            inclusive: false,
        },
    )
    .unwrap();
    assert!(empty.is_empty());
}

#[test]
fn strict_shape_tolerance_nan_and_padding_rules_hold() {
    let mut cx = test_cx();
    let a = tensor(&mut cx, vec![2], "f64", &["NaN", "-0.0"]);
    let b = tensor(&mut cx, vec![2], "f64", &["NaN", "0.0"]);
    let close = execute_canonical_tensor_op(
        &mut cx,
        isclose_op_symbol(),
        vec![a.clone(), b.clone()],
        TensorMeta::new(vec![2], Symbol::qualified("numbers", "f64")),
        CanonicalAttrs::Close {
            relative: 0.0,
            absolute: 0.0,
            equal_nan: true,
        },
    )
    .unwrap();
    assert_eq!(text(&mut cx, &close), ["1", "1"]);
    let bad = execute_canonical_tensor_op(
        &mut cx,
        allclose_op_symbol(),
        vec![a.clone(), b],
        TensorMeta::new(vec![], Symbol::qualified("numbers", "f64")),
        CanonicalAttrs::Close {
            relative: -1.0,
            absolute: 0.0,
            equal_nan: false,
        },
    );
    assert!(bad.unwrap_err().to_string().contains("non-negative"));
    let finite = tensor(&mut cx, vec![2], "i64", &["1", "2"]);
    let padded = execute_canonical_tensor_op(
        &mut cx,
        pad_op_symbol(),
        vec![finite.clone()],
        TensorMeta::new(vec![4], Symbol::qualified("numbers", "i64")),
        CanonicalAttrs::Pad {
            widths: vec![(1, 1)].into(),
            mode: PadMode::Constant(9.0),
        },
    )
    .unwrap();
    assert_eq!(text(&mut cx, &padded), ["9", "1", "2", "9"]);
    let mismatch = tensor(&mut cx, vec![1], "i64", &["1"]);
    assert!(
        execute_canonical_tensor_op(
            &mut cx,
            maximum_op_symbol(),
            vec![finite, mismatch],
            TensorMeta::new(vec![2], Symbol::qualified("numbers", "i64")),
            CanonicalAttrs::None
        )
        .is_err()
    );
}

#[test]
fn cpu_provider_card_advertises_the_complete_vocabulary() {
    let card = CpuTensorExecutor::new().card();
    assert!(card.operations.contains(&arange_op_symbol()));
    assert!(card.operations.contains(&allclose_op_symbol()));
}

#[test]
fn explicit_sum_modes_expose_catastrophic_cancellation() {
    use crate::{SumMode, cumsum_f64, sum_f64};
    let v = [1.0e16, 1.0, -1.0e16];
    assert_eq!(sum_f64(&v, SumMode::Naive), 0.0);
    assert_eq!(sum_f64(&v, SumMode::Neumaier), 1.0);
    assert_eq!(cumsum_f64(&v, SumMode::Neumaier).last(), Some(&1.0));
    assert_eq!(sum_f64(&[1., 2., 3., 4.], SumMode::Pairwise), 10.0);
    // The exact rational sum is one; the compensated mode preserves it despite
    // the two terms sixteen orders of magnitude larger.
    assert_eq!(sum_f64(&[1.0e16, 1.0, -1.0e16], SumMode::Neumaier), 1.0);
}
