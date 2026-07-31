//! Plan execution over strided views and canonical tensor buffers.

use sim_lib_numbers_tensor_cmplxf::ComplexFTensor;
use sim_lib_numbers_tensor_f64::F64Tensor;

use crate::{
    Direction, LengthPolicy, Normalization, PlacementPolicy, SignalBuffer, SignalError, SignalView,
    SignalViewMut, SpectrumPacking, TransformKind, TransformPlan,
    fft::{Complex, fft},
    reference::{reference_dct, reference_dft, reference_dst},
};

/// Executes an out-of-place transform into canonical tensor storage.
pub fn transform(plan: &TransformPlan, input: SignalView<'_>) -> Result<SignalBuffer, SignalError> {
    plan.validate()?;
    if plan.placement != PlacementPolicy::OutOfPlace {
        return Err(SignalError::InvalidPolicy {
            policy: "placement",
            reason: "transform requires OutOfPlace; use transform_in_place for InPlace",
        });
    }
    match plan.kind {
        TransformKind::Dft | TransformKind::Fft => {
            let values = expect_complex(input)?;
            let selected = select_complex(plan, values, plan.len)?;
            let mut output = if plan.kind == TransformKind::Dft {
                reference_dft(&selected, plan.direction, plan.sign)?
            } else {
                fft(
                    &selected
                        .iter()
                        .copied()
                        .map(Complex::from)
                        .collect::<Vec<_>>(),
                    plan.sign.angle_sign(plan.direction),
                )?
                .into_iter()
                .map(Into::into)
                .collect()
            };
            apply_fourier_normalization(&mut output, plan);
            complex_buffer(output)
        }
        TransformKind::RealFft => transform_real_fft(plan, input),
        TransformKind::Dct(kind) => {
            let selected = select_real(plan, expect_real(input)?, plan.len)?;
            real_buffer(reference_dct(
                &selected,
                kind,
                plan.direction,
                plan.normalization,
            )?)
        }
        TransformKind::Dst(kind) => {
            let selected = select_real(plan, expect_real(input)?, plan.len)?;
            real_buffer(reference_dst(
                &selected,
                kind,
                plan.direction,
                plan.normalization,
            )?)
        }
    }
}

/// Executes a same-representation transform into mutable caller storage.
///
/// Complex DFT/FFT and real DCT/DST plans are supported. Real FFT is rejected
/// because its real/complex representations and packed lengths differ.
pub fn transform_in_place(
    plan: &TransformPlan,
    input: SignalViewMut<'_>,
) -> Result<(), SignalError> {
    plan.validate()?;
    if plan.placement != PlacementPolicy::InPlace {
        return Err(SignalError::InvalidPolicy {
            policy: "placement",
            reason: "transform_in_place requires InPlace",
        });
    }
    if plan.kind == TransformKind::RealFft {
        return Err(SignalError::InvalidPolicy {
            policy: "placement",
            reason: "real FFT changes representation and may change packed length",
        });
    }
    let mut out_of_place = plan.clone();
    out_of_place.placement = PlacementPolicy::OutOfPlace;
    let stride = plan.stride;
    match input {
        SignalViewMut::Complex(values) => {
            let SignalBuffer::Complex(output) =
                transform(&out_of_place, SignalView::Complex(values))?
            else {
                return Err(SignalError::InputKind {
                    expected: "real",
                    actual: "complex",
                });
            };
            for (index, value) in output.as_slice().iter().copied().enumerate() {
                values[stride.physical_index(index)?] = value;
            }
        }
        SignalViewMut::Real(values) => {
            let SignalBuffer::Real(output) = transform(&out_of_place, SignalView::Real(values))?
            else {
                return Err(SignalError::InputKind {
                    expected: "complex",
                    actual: "real",
                });
            };
            for (index, value) in output.as_slice().iter().copied().enumerate() {
                values[stride.physical_index(index)?] = value;
            }
        }
    }
    Ok(())
}

fn transform_real_fft(
    plan: &TransformPlan,
    input: SignalView<'_>,
) -> Result<SignalBuffer, SignalError> {
    match plan.direction {
        Direction::Forward => {
            let selected = select_real(plan, expect_real(input)?, plan.len)?;
            let mut output = fft(
                &selected
                    .into_iter()
                    .map(|value| Complex::new(value, 0.0))
                    .collect::<Vec<_>>(),
                plan.sign.angle_sign(Direction::Forward),
            )?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
            apply_fourier_normalization(&mut output, plan);
            if plan.packing == SpectrumPacking::HermitianHalf {
                output.truncate(plan.len / 2 + 1);
            }
            complex_buffer(output)
        }
        Direction::Inverse => {
            let packed_len = match plan.packing {
                SpectrumPacking::Full => plan.len,
                SpectrumPacking::HermitianHalf => plan.len / 2 + 1,
            };
            let selected = select_complex(plan, expect_complex(input)?, packed_len)?;
            let spectrum = match plan.packing {
                SpectrumPacking::Full => selected,
                SpectrumPacking::HermitianHalf => unpack_hermitian(&selected, plan.len),
            };
            let mut output = fft(
                &spectrum.into_iter().map(Complex::from).collect::<Vec<_>>(),
                plan.sign.angle_sign(Direction::Inverse),
            )?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
            apply_fourier_normalization(&mut output, plan);
            let tolerance = 64.0 * f64::EPSILON * plan.len.max(1) as f64;
            let real = output
                .into_iter()
                .enumerate()
                .map(|(index, (real, imag))| {
                    if imag.abs() > tolerance * real.abs().max(1.0) {
                        Err(SignalError::InvalidPolicy {
                            policy: "packing",
                            reason: "inverse real FFT spectrum is not Hermitian",
                        })
                    } else if !real.is_finite() {
                        Err(SignalError::NonFinite {
                            index,
                            component: "value",
                        })
                    } else {
                        Ok(real)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            real_buffer(real)
        }
    }
}

fn unpack_hermitian(packed: &[(f64, f64)], len: usize) -> Vec<(f64, f64)> {
    let mut output = vec![(0.0, 0.0); len];
    output[..packed.len()].copy_from_slice(packed);
    for (frequency, value) in output.iter_mut().enumerate().skip(packed.len()) {
        let mirror = len - frequency;
        *value = (packed[mirror].0, -packed[mirror].1);
    }
    output
}

fn apply_fourier_normalization(output: &mut [(f64, f64)], plan: &TransformPlan) {
    let scale = match (plan.normalization, plan.direction) {
        (Normalization::None, _) => 1.0,
        (Normalization::Forward, Direction::Forward)
        | (Normalization::Inverse, Direction::Inverse) => 1.0 / plan.len as f64,
        (Normalization::Forward | Normalization::Inverse, _) => 1.0,
        (Normalization::Orthonormal, _) => 1.0 / (plan.len as f64).sqrt(),
    };
    for (real, imag) in output {
        *real *= scale;
        *imag *= scale;
    }
}

fn select_real(
    plan: &TransformPlan,
    values: &[f64],
    expected: usize,
) -> Result<Vec<f64>, SignalError> {
    let available = plan.stride.available(values.len());
    let selected_len = admitted_len(plan, expected, available)?;
    let mut selected = Vec::with_capacity(expected);
    for index in 0..selected_len {
        let value = values[plan.stride.physical_index(index)?];
        if !value.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "value",
            });
        }
        selected.push(value);
    }
    selected.resize(expected, 0.0);
    Ok(selected)
}

fn select_complex(
    plan: &TransformPlan,
    values: &[(f64, f64)],
    expected: usize,
) -> Result<Vec<(f64, f64)>, SignalError> {
    let available = plan.stride.available(values.len());
    let selected_len = admitted_len(plan, expected, available)?;
    let mut selected = Vec::with_capacity(expected);
    for index in 0..selected_len {
        let (real, imag) = values[plan.stride.physical_index(index)?];
        if !real.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "real",
            });
        }
        if !imag.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "imag",
            });
        }
        selected.push((real, imag));
    }
    selected.resize(expected, (0.0, 0.0));
    Ok(selected)
}

fn admitted_len(
    plan: &TransformPlan,
    expected: usize,
    available: usize,
) -> Result<usize, SignalError> {
    match plan.length {
        LengthPolicy::Exact if available == expected => Ok(expected),
        LengthPolicy::Exact => Err(SignalError::LengthMismatch {
            expected,
            actual: available,
        }),
        LengthPolicy::Pad if available <= expected => Ok(available),
        LengthPolicy::Pad => Err(SignalError::LengthMismatch {
            expected,
            actual: available,
        }),
        LengthPolicy::Truncate if available >= expected => Ok(expected),
        LengthPolicy::Truncate => Err(SignalError::LengthMismatch {
            expected,
            actual: available,
        }),
    }
}

fn expect_complex(input: SignalView<'_>) -> Result<&[(f64, f64)], SignalError> {
    match input {
        SignalView::Complex(values) => Ok(values),
        SignalView::Real(_) => Err(SignalError::InputKind {
            expected: "complex",
            actual: "real",
        }),
    }
}

fn expect_real(input: SignalView<'_>) -> Result<&[f64], SignalError> {
    match input {
        SignalView::Real(values) => Ok(values),
        SignalView::Complex(_) => Err(SignalError::InputKind {
            expected: "real",
            actual: "complex",
        }),
    }
}

fn complex_buffer(values: Vec<(f64, f64)>) -> Result<SignalBuffer, SignalError> {
    let len = values.len();
    for (index, (real, imag)) in values.iter().copied().enumerate() {
        if !real.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "real",
            });
        }
        if !imag.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "imag",
            });
        }
    }
    ComplexFTensor::new(vec![len], values)
        .map(SignalBuffer::Complex)
        .ok_or(SignalError::InvalidLength {
            len,
            reason: "complex tensor shape overflowed",
        })
}

fn real_buffer(values: Vec<f64>) -> Result<SignalBuffer, SignalError> {
    let len = values.len();
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "value",
            });
        }
    }
    F64Tensor::new(vec![len], values)
        .map(SignalBuffer::Real)
        .ok_or(SignalError::InvalidLength {
            len,
            reason: "real tensor shape overflowed",
        })
}
