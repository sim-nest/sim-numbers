use sim_lib_numbers_signal::{
    BoundaryMode, DuplicateXPolicy, ExtrapolationPolicy, InterpolationMethod, InterpolationPlan,
    SavitzkyGolaySpec, ToeplitzPlan, Wavelet, WaveletPlan, apply_savitzky_golay, dwt, idwt,
    interpolate_samples, savitzky_golay, solve_toeplitz,
};

fn main() {
    let samples = [0.5, -1.0, 2.5, 4.0, -3.25, 1.5, 0.125, 7.0, -2.0];
    let wavelet_plan = WaveletPlan {
        wavelet: Wavelet::LeGall53,
        levels: 3,
        boundary: BoundaryMode::Symmetric,
    };
    let coefficients = dwt(&samples, &wavelet_plan).unwrap();
    let reconstructed = idwt(&coefficients, &wavelet_plan).unwrap();
    let roundtrip_error = samples
        .iter()
        .zip(reconstructed)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f64::max);

    let spacing = 0.25;
    let filter = savitzky_golay(SavitzkyGolaySpec {
        window_length: 7,
        polynomial_order: 3,
        derivative_order: 2,
        sample_spacing: spacing,
        ..SavitzkyGolaySpec::default()
    })
    .unwrap();
    let parabola = (-3..=3)
        .map(|index| {
            let x = f64::from(index) * spacing;
            3.0 * x * x + 2.0 * x + 1.0
        })
        .collect::<Vec<_>>();
    let derivative = apply_savitzky_golay(&parabola, &filter, BoundaryMode::Zero).unwrap();

    let toeplitz = solve_toeplitz(
        &[4.0, 1.0, 0.5],
        &[4.0, 2.0, -1.0],
        &[5.5, 6.0, 3.5],
        ToeplitzPlan::default(),
    )
    .unwrap();

    let interpolation = interpolate_samples(
        &[0.0, 1.0, 1.0, 2.0],
        &[0.0, 1.0, 3.0, 4.0],
        &[0.5, 1.0, 1.5],
        InterpolationPlan {
            method: InterpolationMethod::Monotone,
            duplicates: DuplicateXPolicy::Average,
            extrapolation: ExtrapolationPolicy::Reject,
        },
    )
    .unwrap();

    println!(
        "wavelet-levels={} roundtrip-max-error={roundtrip_error:.12}",
        coefficients.levels.len()
    );
    println!(
        "sg-second-derivative={:.6} toeplitz-residual={:.12}",
        derivative[3], toeplitz.diagnostics.residual_l2
    );
    println!(
        "interpolation={:?} duplicates={} extrapolated={}",
        interpolation.values,
        interpolation.report.duplicates_resolved,
        interpolation.report.extrapolated_points
    );
}
