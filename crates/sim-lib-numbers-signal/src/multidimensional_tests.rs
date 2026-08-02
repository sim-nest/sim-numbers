// conformance: strided multidimensional and bounded Table-backed transforms.

use std::sync::Arc;

use sim_kernel::{AssocTable, Cx, DefaultFactory, EagerPolicy, Symbol};
use sim_lib_numbers_tensor::SpecTensor;

use crate::{
    PlacementPolicy, SignalBuffer, TensorView, TransformKind, TransformPlan, TransformPrecision,
    TransformResources, read_blocked_tensor, reference_dft, transform_nd, transform_nd_blocked,
    transform_plan_digest, write_blocked_tensor, write_complex_f64_block,
};

const TOLERANCE: f64 = 3.0e-10;

#[test]
fn separable_fft_over_transposed_strided_view_matches_axis_definitions() {
    let physical = [
        (1.0, 0.0),
        (2.0, -0.5),
        (3.0, 0.25),
        (4.0, 1.0),
        (5.0, -1.0),
        (6.0, 0.75),
    ];
    // Logical 3x2 transpose of the physical 2x3 row-major tensor.
    let view = TensorView::complex(&physical, vec![3, 2], vec![1, 3], 0).unwrap();
    assert_eq!(view.strides(), [1, 3]);

    let plan = TransformPlan::new(TransformKind::Fft, 1);
    let actual = transform_nd(view, &[0, 1], &plan).unwrap();
    let SignalBuffer::Complex(actual) = actual.output else {
        panic!("complex FFT must return a complex tensor");
    };
    assert_eq!(actual.shape(), [3, 2]);

    let logical = [
        physical[0],
        physical[3],
        physical[1],
        physical[4],
        physical[2],
        physical[5],
    ];
    let mut after_axis_zero = [(0.0, 0.0); 6];
    for column in 0..2 {
        let line = [logical[column], logical[2 + column], logical[4 + column]];
        let expected =
            reference_dft(&line, plan.direction, plan.sign).expect("direct axis transform");
        for row in 0..3 {
            after_axis_zero[row * 2 + column] = expected[row];
        }
    }
    let mut expected = [(0.0, 0.0); 6];
    for row in 0..3 {
        let line = [after_axis_zero[row * 2], after_axis_zero[row * 2 + 1]];
        let transformed =
            reference_dft(&line, plan.direction, plan.sign).expect("direct axis transform");
        expected[row * 2..row * 2 + 2].copy_from_slice(&transformed);
    }

    assert_complex_close(actual.as_slice(), &expected);
}

#[test]
fn tensor_view_rejects_overlapping_or_out_of_bounds_layouts() {
    let values = [0.0; 4];
    assert!(TensorView::real(&values, vec![2, 2], vec![1, 1], 0).is_err());
    assert!(TensorView::real(&values, vec![2, 2], vec![3, 1], 0).is_err());
}

#[test]
fn blocked_table_transform_matches_in_memory_with_bounded_scratch() {
    let shape = vec![4, 4];
    let values = (0..16)
        .map(|index| {
            let value = index as f64;
            ((value * 0.37).sin(), (value * 0.19).cos())
        })
        .collect::<Vec<_>>();
    let view = TensorView::complex(&values, shape, vec![4, 1], 0).unwrap();
    let resources = TransformResources {
        max_scratch_bytes: 4096,
        block_len: 3,
    };
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let store = AssocTable::new();
    let blocked = write_blocked_tensor(
        &mut cx,
        &store,
        Symbol::qualified("test", "blocked-fft"),
        &view,
        resources,
    )
    .unwrap();
    assert_eq!(blocked.block_len(), 3);
    assert_eq!(
        blocked,
        crate::BlockedTensor::new(
            Symbol::qualified("test", "blocked-fft"),
            vec![4, 4],
            TransformPrecision::ComplexF64,
            3,
        )
        .unwrap()
    );

    let mut blocked_plan = TransformPlan::new(TransformKind::Fft, 1);
    blocked_plan.placement = PlacementPolicy::InPlace;
    let report =
        transform_nd_blocked(&mut cx, &store, &blocked, &[0, 1], &blocked_plan, resources).unwrap();
    assert_eq!(report.passes, 2);
    assert!(report.io_blocks > 0);
    assert!(report.scratch_bytes <= resources.max_scratch_bytes);
    assert_eq!(report.precision, TransformPrecision::ComplexF64);
    assert_eq!(
        report.plan_digest,
        transform_plan_digest(
            blocked.shape(),
            &[0, 1],
            &blocked_plan,
            TransformPrecision::ComplexF64,
            Some(resources),
        )
    );

    let SignalBuffer::Complex(blocked_output) =
        read_blocked_tensor(&mut cx, &store, &blocked).unwrap()
    else {
        panic!("blocked FFT must remain complex");
    };
    let in_memory_plan = TransformPlan::new(TransformKind::Fft, 1);
    let in_memory = transform_nd(view, &[0, 1], &in_memory_plan).unwrap();
    let SignalBuffer::Complex(in_memory_output) = in_memory.output else {
        panic!("in-memory FFT must remain complex");
    };
    assert_eq!(blocked_output.shape(), [4, 4]);
    assert_complex_close(blocked_output.as_slice(), in_memory_output.as_slice());
}

#[test]
fn blocked_plan_rejects_insufficient_scratch_before_execution() {
    let values = vec![(1.0, 0.0); 16];
    let view = TensorView::complex(&values, vec![4, 4], vec![4, 1], 0).unwrap();
    let write_resources = TransformResources {
        max_scratch_bytes: 1024,
        block_len: 2,
    };
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let store = AssocTable::new();
    let blocked = write_blocked_tensor(
        &mut cx,
        &store,
        Symbol::qualified("test", "scratch-limit"),
        &view,
        write_resources,
    )
    .unwrap();
    let mut plan = TransformPlan::new(TransformKind::Fft, 1);
    plan.placement = PlacementPolicy::InPlace;
    let too_small = TransformResources {
        max_scratch_bytes: 64,
        block_len: 2,
    };
    assert!(transform_nd_blocked(&mut cx, &store, &blocked, &[0, 1], &plan, too_small).is_err());
}

#[test]
fn caller_can_seed_external_descriptor_one_block_at_a_time() {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    let store = AssocTable::new();
    let tensor = crate::BlockedTensor::new(
        Symbol::qualified("test", "incremental-blocks"),
        vec![2, 2],
        TransformPrecision::ComplexF64,
        2,
    )
    .unwrap();
    assert_eq!(tensor.block_count(), 2);
    write_complex_f64_block(&mut cx, &store, &tensor, 0, &[(1.0, -1.0), (2.0, -2.0)]).unwrap();
    write_complex_f64_block(&mut cx, &store, &tensor, 1, &[(3.0, -3.0), (4.0, -4.0)]).unwrap();
    let SignalBuffer::Complex(materialized) =
        read_blocked_tensor(&mut cx, &store, &tensor).unwrap()
    else {
        panic!("complex descriptor must materialize complex cells");
    };
    assert_eq!(
        materialized.as_slice(),
        &[(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)]
    );
}

fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.0 - expected.0).abs() <= TOLERANCE);
        assert!((actual.1 - expected.1).abs() <= TOLERANCE);
    }
}
