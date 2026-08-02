// conformance: convolution, correlation, blocking, and guarded spectral inverse.

use crate::{
    BlockConvolutionMethod, BlockConvolutionPlan, BoundaryPolicy, ConvolutionAlgorithm,
    ConvolutionMode, ConvolutionNormalization, ConvolutionPlan, CorrelationNormalization,
    CorrelationPlan, DeconvolutionPlan, LagOrder, LinearOutput, Regularization, convolve,
    convolve_blocked, correlate, deconvolve,
};

const TOLERANCE: f64 = 2.0e-9;

#[test]
fn automatic_cost_plan_is_inspectable_and_selects_both_paths() {
    let plan = ConvolutionPlan::linear_full();
    let small = plan.inspect(4, 3).unwrap();
    assert_eq!(small.selected, ConvolutionAlgorithm::Direct);
    assert!(small.direct_cost_units < small.fft_cost_units);
    assert_eq!(small.fft_len, 8);

    let large = plan.inspect(256, 129).unwrap();
    assert_eq!(large.selected, ConvolutionAlgorithm::Fft);
    assert!(large.fft_cost_units < large.direct_cost_units);
    assert_eq!(large.fft_len, 512);
    assert!(large.fft_scratch_bytes > 0);
}

#[test]
fn direct_and_fft_convolution_agree_for_every_linear_span() {
    let signal = (0..33)
        .map(|index| (index as f64 * 0.37).sin() + index as f64 / 71.0)
        .collect::<Vec<_>>();
    let kernel = (0..9)
        .map(|index| (index as f64 * 0.23).cos() - 0.4)
        .collect::<Vec<_>>();
    for span in [LinearOutput::Full, LinearOutput::Same, LinearOutput::Valid] {
        let mut direct = ConvolutionPlan::linear_full();
        direct.mode = ConvolutionMode::Linear(span);
        direct.algorithm = ConvolutionAlgorithm::Direct;
        let mut fast = direct.clone();
        fast.algorithm = ConvolutionAlgorithm::Fft;
        let direct = convolve(&signal, &kernel, &direct).unwrap();
        let fast = convolve(&signal, &kernel, &fast).unwrap();
        assert_close(direct.samples.as_slice(), fast.samples.as_slice());
        assert_eq!(direct.report.retained_start, fast.report.retained_start);
        assert_eq!(direct.report.retained_len, fast.report.retained_len);
    }
}

#[test]
fn full_and_circular_convolution_are_commutative_and_identity_preserving() {
    let left = [0.5, -1.0, 2.0, 0.25, 3.0];
    let right = [1.5, 0.0, -0.5];
    let mut plan = ConvolutionPlan::linear_full();
    plan.algorithm = ConvolutionAlgorithm::Direct;
    let left_right = convolve(&left, &right, &plan).unwrap();
    let right_left = convolve(&right, &left, &plan).unwrap();
    assert_close(left_right.samples.as_slice(), right_left.samples.as_slice());
    assert_close(
        convolve(&left, &[1.0], &plan).unwrap().samples.as_slice(),
        &left,
    );

    let mut circular = ConvolutionPlan::circular(5);
    circular.algorithm = ConvolutionAlgorithm::Fft;
    let left_right = convolve(&left, &right, &circular).unwrap();
    let right_left = convolve(&right, &left, &circular).unwrap();
    assert_close(left_right.samples.as_slice(), right_left.samples.as_slice());
    assert_close(
        convolve(&left, &[1.0], &circular)
            .unwrap()
            .samples
            .as_slice(),
        &left,
    );
}

#[test]
fn overlap_add_and_save_match_direct_with_exact_span_reports() {
    let signal = (0..13)
        .map(|index| (index as f64 * 0.19).sin())
        .collect::<Vec<_>>();
    let kernel = [0.2, -0.4, 0.7, 0.1];
    for span in [LinearOutput::Full, LinearOutput::Same, LinearOutput::Valid] {
        let mut ordinary = ConvolutionPlan::linear_full();
        ordinary.mode = ConvolutionMode::Linear(span);
        ordinary.algorithm = ConvolutionAlgorithm::Direct;
        let expected = convolve(&signal, &kernel, &ordinary).unwrap();
        for method in [
            BlockConvolutionMethod::OverlapAdd,
            BlockConvolutionMethod::OverlapSave,
        ] {
            let blocked = convolve_blocked(
                &signal,
                &kernel,
                &BlockConvolutionPlan {
                    convolution: ordinary.clone(),
                    method,
                    fft_len: 8,
                },
            )
            .unwrap();
            assert_close(
                blocked.convolution.samples.as_slice(),
                expected.samples.as_slice(),
            );
            assert_eq!(blocked.blocked.input_span_per_block, 5);
            assert_eq!(blocked.blocked.retained_span_per_block, 5);
            assert_eq!(blocked.blocked.latency_samples, 5);
            assert_eq!(
                blocked.blocked.boundary.retained_start,
                expected.report.retained_start
            );
            assert_eq!(
                blocked.blocked.boundary.retained_len,
                expected.report.retained_len
            );
            if method == BlockConvolutionMethod::OverlapSave {
                assert_eq!(blocked.blocked.boundary.left_padding, 3);
                assert_eq!(blocked.blocked.boundary.discarded_prefix_per_block, 3);
            }
        }
    }
}

#[test]
fn correlation_has_reflected_pair_symmetry_and_typed_normalization() {
    let left = [1.0, -2.0, 0.5, 3.0];
    let right = [0.25, 2.0, -1.0];
    let mut plan = CorrelationPlan::linear_full();
    plan.algorithm = ConvolutionAlgorithm::Fft;
    let left_right = correlate(&left, &right, &plan).unwrap();
    let right_left = correlate(&right, &left, &plan).unwrap();
    for (lag, value) in left_right.lags.iter().zip(left_right.samples.as_slice()) {
        let opposite = right_left
            .lags
            .iter()
            .position(|candidate| candidate == &-*lag)
            .unwrap();
        assert!((value - right_left.samples.as_slice()[opposite]).abs() <= TOLERANCE);
    }

    plan.normalization = CorrelationNormalization::Unbiased;
    plan.lag_order = LagOrder::Descending;
    let normalized = correlate(&left, &right, &plan).unwrap();
    assert!(normalized.lags.windows(2).all(|pair| pair[0] > pair[1]));
    assert!(
        normalized
            .samples
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn guarded_deconvolution_recovers_regular_inputs_and_reports_singular_bins() {
    let signal = [0.5, -1.0, 2.0, 0.25, 1.5, -0.75];
    let kernel = [1.0, 0.25];
    let observation = convolve(&signal, &kernel, &ConvolutionPlan::linear_full()).unwrap();
    let recovered = deconvolve(
        observation.samples.as_slice(),
        &kernel,
        &DeconvolutionPlan::tikhonov(1.0e-12, 1.0e-10),
    )
    .unwrap();
    assert_close(recovered.samples.as_slice(), &signal);
    assert!(recovered.report.singular_bins.is_empty());
    assert!(recovered.report.residual_l2 <= 1.0e-9);

    let singular_kernel = [1.0, -1.0];
    let observation = convolve(&signal, &singular_kernel, &ConvolutionPlan::linear_full()).unwrap();
    let singular = deconvolve(
        observation.samples.as_slice(),
        &singular_kernel,
        &DeconvolutionPlan {
            mode: crate::DeconvolutionMode::LinearFull,
            regularization: Regularization::Tikhonov { lambda: 1.0e-8 },
            singular_threshold: 1.0e-12,
        },
    )
    .unwrap();
    assert!(singular.report.singular_bins.contains(&0));
    assert_eq!(singular.report.minimum_kernel_magnitude, 0.0);
    assert!(
        singular
            .samples
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(singular.report.maximum_inverse_gain.is_finite());
    assert!(singular.report.residual_l2.is_finite());
}

#[test]
fn invalid_boundaries_and_degenerate_normalizations_fail_closed() {
    let mut plan = ConvolutionPlan::linear_full();
    plan.boundary = BoundaryPolicy::Periodic;
    assert!(convolve(&[1.0], &[1.0], &plan).is_err());

    plan.boundary = BoundaryPolicy::ZeroPad;
    plan.normalization = ConvolutionNormalization::KernelSum;
    assert!(convolve(&[1.0, 2.0], &[1.0, -1.0], &plan).is_err());
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= TOLERANCE,
            "sample {index}: {actual} != {expected}"
        );
    }
}
