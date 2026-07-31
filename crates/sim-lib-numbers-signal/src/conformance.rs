//! Conformance: definition-level signal fixtures shared by transform users.

use std::f64::consts::TAU;

use crate::{
    Direction, Normalization, SignalBuffer, SignalError, SignalView, SpectrumPacking,
    TransformKind, TransformPlan, reference_dft, transform,
};

const TOLERANCE: f64 = 3.0e-10;

#[test]
fn impulse_and_tone_land_in_definition_level_bins() {
    let impulse = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
    let plan = TransformPlan::new(TransformKind::Fft, impulse.len());
    let SignalBuffer::Complex(spectrum) = transform(&plan, SignalView::Complex(&impulse)).unwrap()
    else {
        panic!("FFT must return complex values");
    };
    assert_eq!(
        spectrum.as_slice(),
        &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    );

    let len = 16;
    let bin = 3;
    let tone = (0..len)
        .map(|sample| (TAU * bin as f64 * sample as f64 / len as f64).cos())
        .collect::<Vec<_>>();
    let mut plan = TransformPlan::new(TransformKind::RealFft, len);
    plan.packing = SpectrumPacking::HermitianHalf;
    let SignalBuffer::Complex(spectrum) = transform(&plan, SignalView::Real(&tone)).unwrap() else {
        panic!("real FFT must return complex values");
    };
    for (frequency, (real, imag)) in spectrum.as_slice().iter().copied().enumerate() {
        let magnitude = real.hypot(imag);
        if frequency == bin {
            assert!((magnitude - len as f64 / 2.0).abs() <= TOLERANCE);
        } else {
            assert!(magnitude <= TOLERANCE, "unexpected tone bin {frequency}");
        }
    }
}

#[test]
fn prime_length_fast_transform_agrees_with_direct_definition() {
    let input = (0..11)
        .map(|index| ((index as f64 * 0.31).sin(), (index as f64 * 0.47).cos()))
        .collect::<Vec<_>>();
    let expected = reference_dft(
        &input,
        Direction::Forward,
        crate::SignConvention::NegativeForward,
    )
    .unwrap();
    let plan = TransformPlan::new(TransformKind::Fft, input.len());
    let SignalBuffer::Complex(actual) = transform(&plan, SignalView::Complex(&input)).unwrap()
    else {
        panic!("FFT must return complex values");
    };
    for (actual, expected) in actual.as_slice().iter().zip(expected) {
        assert!((actual.0 - expected.0).abs() <= TOLERANCE);
        assert!((actual.1 - expected.1).abs() <= TOLERANCE);
    }
}

#[test]
fn orthonormal_fft_satisfies_parseval_and_round_trips() {
    let input = [
        (0.5, -0.25),
        (1.25, 0.75),
        (-2.0, 0.5),
        (0.125, -1.0),
        (0.75, 0.0),
    ];
    let mut plan = TransformPlan::new(TransformKind::Fft, input.len());
    plan.normalization = Normalization::Orthonormal;
    let SignalBuffer::Complex(spectrum) = transform(&plan, SignalView::Complex(&input)).unwrap()
    else {
        panic!("FFT must return complex values");
    };
    let input_energy = input
        .iter()
        .map(|(real, imag)| real * real + imag * imag)
        .sum::<f64>();
    let spectrum_energy = spectrum
        .as_slice()
        .iter()
        .map(|(real, imag)| real * real + imag * imag)
        .sum::<f64>();
    assert!((input_energy - spectrum_energy).abs() <= TOLERANCE);

    plan.direction = Direction::Inverse;
    let SignalBuffer::Complex(recovered) =
        transform(&plan, SignalView::Complex(spectrum.as_slice())).unwrap()
    else {
        panic!("inverse FFT must return complex values");
    };
    for (actual, expected) in recovered.as_slice().iter().zip(input) {
        assert!((actual.0 - expected.0).abs() <= TOLERANCE);
        assert!((actual.1 - expected.1).abs() <= TOLERANCE);
    }
}

#[test]
fn empty_and_non_finite_inputs_fail_closed() {
    let empty = TransformPlan::new(TransformKind::Fft, 0);
    assert!(matches!(
        transform(&empty, SignalView::Complex(&[])),
        Err(SignalError::InvalidLength { .. })
    ));

    let plan = TransformPlan::new(TransformKind::Fft, 2);
    assert_eq!(
        transform(&plan, SignalView::Complex(&[(1.0, 0.0), (f64::NAN, 0.0)])),
        Err(SignalError::NonFinite {
            index: 1,
            component: "real"
        })
    );
}
