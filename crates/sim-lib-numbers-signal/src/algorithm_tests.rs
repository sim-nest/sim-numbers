use crate::{
    DctType, Direction, DstType, Normalization, PlacementPolicy, SignalBuffer, SignalView,
    SignalViewMut, SpectrumPacking, Stride, TransformKind, TransformPlan, reference_dct,
    reference_dft, reference_dst, transform, transform_in_place,
};
use sim_lib_numbers_tensor_cmplxf::ComplexFTensor;
use sim_lib_numbers_tensor_f64::F64Tensor;

const TOLERANCE: f64 = 2.0e-10;

fn assert_complex_close(left: &[(f64, f64)], right: &[(f64, f64)]) {
    assert_eq!(left.len(), right.len());
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left.0 - right.0).abs() <= TOLERANCE && (left.1 - right.1).abs() <= TOLERANCE,
            "complex value {index}: {left:?} != {right:?}"
        );
    }
}

fn assert_real_close(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= TOLERANCE,
            "real value {index}: {left} != {right}"
        );
    }
}

#[test]
fn radix2_mixed_radix_and_bluestein_agree_with_direct_dft() {
    for len in 1..=16 {
        let input = (0..len)
            .map(|index| {
                (
                    (index as f64 * 0.37).sin(),
                    (index as f64 * 0.19).cos() * 0.25,
                )
            })
            .collect::<Vec<_>>();
        let expected = reference_dft(
            &input,
            Direction::Forward,
            crate::SignConvention::NegativeForward,
        )
        .unwrap();
        let plan = TransformPlan::new(TransformKind::Fft, len);
        let SignalBuffer::Complex(actual) = transform(&plan, SignalView::Complex(&input)).unwrap()
        else {
            panic!("FFT must return complex output");
        };
        assert_complex_close(actual.as_slice(), &expected);
    }
}

#[test]
fn canonical_tensor_storage_is_the_public_input_and_output_boundary() {
    let input = ComplexFTensor::new(
        vec![4],
        vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    )
    .unwrap();
    let plan = TransformPlan::new(TransformKind::Fft, 4);
    let SignalBuffer::Complex(output) =
        transform(&plan, SignalView::from_complex_tensor(&input)).unwrap()
    else {
        panic!("complex FFT must return canonical complex tensor storage");
    };
    assert_eq!(output.shape(), &[4]);
    assert_eq!(output.dtype().as_qualified_str(), "numbers/complex");
    assert_complex_close(
        output.as_slice(),
        &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
    );

    let real = F64Tensor::new(vec![4], vec![1.0, 0.0, -1.0, 0.0]).unwrap();
    let mut real_plan = TransformPlan::new(TransformKind::Dct(DctType::II), 4);
    real_plan.normalization = Normalization::Orthonormal;
    let SignalBuffer::Real(output) =
        transform(&real_plan, SignalView::from_real_tensor(&real)).unwrap()
    else {
        panic!("DCT must return canonical f64 tensor storage");
    };
    assert_eq!(output.shape(), &[4]);
    assert_eq!(output.dtype().as_qualified_str(), "numbers/f64");
}

#[test]
fn strided_in_place_transform_updates_only_selected_cells() {
    let mut values = [
        (99.0, 99.0),
        (1.0, 0.0),
        (98.0, 98.0),
        (0.0, 0.0),
        (97.0, 97.0),
        (0.0, 0.0),
        (96.0, 96.0),
        (0.0, 0.0),
    ];
    let mut plan = TransformPlan::new(TransformKind::Fft, 4);
    plan.stride = Stride::new(1, 2).unwrap();
    plan.placement = PlacementPolicy::InPlace;
    transform_in_place(&plan, SignalViewMut::Complex(&mut values)).unwrap();
    assert_complex_close(
        &[values[1], values[3], values[5], values[7]],
        &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
    );
    assert_eq!(
        [values[0], values[2], values[4], values[6]],
        [(99.0, 99.0), (98.0, 98.0), (97.0, 97.0), (96.0, 96.0)]
    );
}

#[test]
fn in_place_transform_rejects_padding_that_cannot_fit() {
    let mut values = [(1.0, 0.0), (0.0, 0.0)];
    let mut plan = TransformPlan::new(TransformKind::Fft, 4);
    plan.length = crate::LengthPolicy::Pad;
    plan.padding = crate::PaddingPolicy::Zero;
    plan.placement = PlacementPolicy::InPlace;
    assert!(matches!(
        transform_in_place(&plan, SignalViewMut::Complex(&mut values)),
        Err(crate::SignalError::InvalidPolicy {
            policy: "length",
            ..
        })
    ));
}

#[test]
fn real_fft_full_and_half_packing_round_trip() {
    let input = [0.25, -1.0, 2.5, 0.5, -0.75, 3.0, 1.25];
    for packing in [SpectrumPacking::Full, SpectrumPacking::HermitianHalf] {
        let mut plan = TransformPlan::new(TransformKind::RealFft, input.len());
        plan.packing = packing;
        let SignalBuffer::Complex(spectrum) = transform(&plan, SignalView::Real(&input)).unwrap()
        else {
            panic!("real FFT must return complex output");
        };
        plan.direction = Direction::Inverse;
        let SignalBuffer::Real(actual) =
            transform(&plan, SignalView::Complex(spectrum.as_slice())).unwrap()
        else {
            panic!("inverse real FFT must return real output");
        };
        assert_real_close(actual.as_slice(), &input);
    }
}

#[test]
fn every_dct_and_dst_definition_round_trips() {
    let input = [0.25, -1.0, 2.5, 0.5, -0.75];
    for kind in [DctType::I, DctType::II, DctType::III, DctType::IV] {
        let forward =
            reference_dct(&input, kind, Direction::Forward, Normalization::Inverse).unwrap();
        let inverse =
            reference_dct(&forward, kind, Direction::Inverse, Normalization::Inverse).unwrap();
        assert_real_close(&inverse, &input);
    }
    for kind in [DstType::I, DstType::II, DstType::III, DstType::IV] {
        let forward =
            reference_dst(&input, kind, Direction::Forward, Normalization::Inverse).unwrap();
        let inverse =
            reference_dst(&forward, kind, Direction::Inverse, Normalization::Inverse).unwrap();
        assert_real_close(&inverse, &input);
    }
}

#[test]
fn orthonormal_cosine_and_sine_definitions_preserve_energy() {
    let input = [0.5, -1.25, 2.0, 0.75, -0.5, 1.0];
    let energy = input.iter().map(|value| value * value).sum::<f64>();
    for kind in [DctType::I, DctType::II, DctType::III, DctType::IV] {
        let output =
            reference_dct(&input, kind, Direction::Forward, Normalization::Orthonormal).unwrap();
        let output_energy = output.iter().map(|value| value * value).sum::<f64>();
        assert!((output_energy - energy).abs() <= TOLERANCE);
        let inverse = reference_dct(
            &output,
            kind,
            Direction::Inverse,
            Normalization::Orthonormal,
        )
        .unwrap();
        assert_real_close(&inverse, &input);
    }
    for kind in [DstType::I, DstType::II, DstType::III, DstType::IV] {
        let output =
            reference_dst(&input, kind, Direction::Forward, Normalization::Orthonormal).unwrap();
        let output_energy = output.iter().map(|value| value * value).sum::<f64>();
        assert!((output_energy - energy).abs() <= TOLERANCE);
        let inverse = reference_dst(
            &output,
            kind,
            Direction::Inverse,
            Normalization::Orthonormal,
        )
        .unwrap();
        assert_real_close(&inverse, &input);
    }
}
