//! Definition-level O(N^2) reference transforms.
//!
//! These routines intentionally remain direct definitions. They provide the
//! oracle for fast implementations and are not described as FFT algorithms.

use std::f64::consts::{PI, SQRT_2, TAU};

use crate::{
    DctType, Direction, DstType, Normalization, SignConvention, SignalError, fft::Complex,
};

/// Computes the direct, unnormalized complex DFT definition.
///
/// The selected [`SignConvention`] and [`Direction`] determine the exponential
/// sign. No scaling is applied.
pub fn reference_dft(
    input: &[(f64, f64)],
    direction: Direction,
    sign: SignConvention,
) -> Result<Vec<(f64, f64)>, SignalError> {
    if input.is_empty() {
        return Err(SignalError::InvalidLength {
            len: 0,
            reason: "DFT requires at least one value",
        });
    }
    validate_complex(input)?;
    let sign = sign.angle_sign(direction);
    let len = input.len() as f64;
    Ok((0..input.len())
        .map(|frequency| {
            input
                .iter()
                .copied()
                .enumerate()
                .fold(Complex::ZERO, |sum, (sample, value)| {
                    let angle = sign * TAU * frequency as f64 * sample as f64 / len;
                    sum + Complex::from(value) * Complex::cis(angle)
                })
                .into()
        })
        .collect())
}

/// Computes DCT-I, II, III, or IV directly from its cosine definition.
pub fn reference_dct(
    input: &[f64],
    kind: DctType,
    direction: Direction,
    normalization: Normalization,
) -> Result<Vec<f64>, SignalError> {
    validate_real(input)?;
    if kind == DctType::I && input.len() < 2 {
        return Err(SignalError::InvalidLength {
            len: input.len(),
            reason: "DCT-I requires at least two values",
        });
    }
    if normalization == Normalization::Orthonormal {
        return Ok(orthonormal_dct(input, kind, direction));
    }
    let (kernel, factor) = match (kind, direction) {
        (DctType::I, _) => (DctType::I, 2.0 * (input.len() - 1) as f64),
        (DctType::II, Direction::Forward) | (DctType::III, Direction::Inverse) => {
            (DctType::II, 2.0 * input.len() as f64)
        }
        (DctType::III, Direction::Forward) | (DctType::II, Direction::Inverse) => {
            (DctType::III, 2.0 * input.len() as f64)
        }
        (DctType::IV, _) => (DctType::IV, 2.0 * input.len() as f64),
    };
    let mut output = raw_dct(input, kernel);
    apply_pair_normalization(&mut output, factor, direction, normalization);
    Ok(output)
}

/// Computes DST-I, II, III, or IV directly from its sine definition.
pub fn reference_dst(
    input: &[f64],
    kind: DstType,
    direction: Direction,
    normalization: Normalization,
) -> Result<Vec<f64>, SignalError> {
    validate_real(input)?;
    if normalization == Normalization::Orthonormal {
        return Ok(orthonormal_dst(input, kind, direction));
    }
    let (kernel, factor) = match (kind, direction) {
        (DstType::I, _) => (DstType::I, 2.0 * (input.len() + 1) as f64),
        (DstType::II, Direction::Forward) | (DstType::III, Direction::Inverse) => {
            (DstType::II, 2.0 * input.len() as f64)
        }
        (DstType::III, Direction::Forward) | (DstType::II, Direction::Inverse) => {
            (DstType::III, 2.0 * input.len() as f64)
        }
        (DstType::IV, _) => (DstType::IV, 2.0 * input.len() as f64),
    };
    let mut output = raw_dst(input, kernel);
    apply_pair_normalization(&mut output, factor, direction, normalization);
    Ok(output)
}

fn raw_dct(input: &[f64], kind: DctType) -> Vec<f64> {
    let len = input.len();
    match kind {
        DctType::I => (0..len)
            .map(|frequency| {
                input[0]
                    + if frequency % 2 == 0 {
                        input[len - 1]
                    } else {
                        -input[len - 1]
                    }
                    + 2.0
                        * input[1..len - 1]
                            .iter()
                            .enumerate()
                            .map(|(offset, value)| {
                                let sample = offset + 1;
                                value
                                    * (PI * frequency as f64 * sample as f64 / (len - 1) as f64)
                                        .cos()
                            })
                            .sum::<f64>()
            })
            .collect(),
        DctType::II => (0..len)
            .map(|frequency| {
                2.0 * input
                    .iter()
                    .enumerate()
                    .map(|(sample, value)| {
                        value
                            * (PI * frequency as f64 * (2 * sample + 1) as f64 / (2 * len) as f64)
                                .cos()
                    })
                    .sum::<f64>()
            })
            .collect(),
        DctType::III => (0..len)
            .map(|frequency| {
                input[0]
                    + 2.0
                        * input[1..]
                            .iter()
                            .enumerate()
                            .map(|(offset, value)| {
                                let sample = offset + 1;
                                value
                                    * (PI * sample as f64 * (2 * frequency + 1) as f64
                                        / (2 * len) as f64)
                                        .cos()
                            })
                            .sum::<f64>()
            })
            .collect(),
        DctType::IV => (0..len)
            .map(|frequency| {
                2.0 * input
                    .iter()
                    .enumerate()
                    .map(|(sample, value)| {
                        value
                            * (PI * (2 * frequency + 1) as f64 * (2 * sample + 1) as f64
                                / (4 * len) as f64)
                                .cos()
                    })
                    .sum::<f64>()
            })
            .collect(),
    }
}

fn raw_dst(input: &[f64], kind: DstType) -> Vec<f64> {
    let len = input.len();
    match kind {
        DstType::I => (0..len)
            .map(|frequency| {
                2.0 * input
                    .iter()
                    .enumerate()
                    .map(|(sample, value)| {
                        value
                            * (PI * (frequency + 1) as f64 * (sample + 1) as f64 / (len + 1) as f64)
                                .sin()
                    })
                    .sum::<f64>()
            })
            .collect(),
        DstType::II => (0..len)
            .map(|frequency| {
                2.0 * input
                    .iter()
                    .enumerate()
                    .map(|(sample, value)| {
                        value
                            * (PI * (frequency + 1) as f64 * (2 * sample + 1) as f64
                                / (2 * len) as f64)
                                .sin()
                    })
                    .sum::<f64>()
            })
            .collect(),
        DstType::III => (0..len)
            .map(|frequency| {
                let endpoint = if frequency % 2 == 0 {
                    input[len - 1]
                } else {
                    -input[len - 1]
                };
                endpoint
                    + 2.0
                        * input[..len - 1]
                            .iter()
                            .enumerate()
                            .map(|(sample, value)| {
                                value
                                    * (PI * (sample + 1) as f64 * (2 * frequency + 1) as f64
                                        / (2 * len) as f64)
                                        .sin()
                            })
                            .sum::<f64>()
            })
            .collect(),
        DstType::IV => (0..len)
            .map(|frequency| {
                2.0 * input
                    .iter()
                    .enumerate()
                    .map(|(sample, value)| {
                        value
                            * (PI * (2 * frequency + 1) as f64 * (2 * sample + 1) as f64
                                / (4 * len) as f64)
                                .sin()
                    })
                    .sum::<f64>()
            })
            .collect(),
    }
}

fn orthonormal_dct(input: &[f64], kind: DctType, direction: Direction) -> Vec<f64> {
    let len = input.len();
    match kind {
        DctType::I => matrix_apply(input, |row, column| {
            let row_weight = endpoint_weight(row, len);
            let column_weight = endpoint_weight(column, len);
            (2.0 / (len - 1) as f64).sqrt()
                * row_weight
                * column_weight
                * (PI * row as f64 * column as f64 / (len - 1) as f64).cos()
        }),
        DctType::II | DctType::III => {
            let transpose = (kind == DctType::II) == (direction == Direction::Inverse);
            matrix_apply(input, |row, column| {
                let (frequency, sample) = if transpose {
                    (column, row)
                } else {
                    (row, column)
                };
                (2.0 / len as f64).sqrt()
                    * if frequency == 0 { 1.0 / SQRT_2 } else { 1.0 }
                    * (PI * frequency as f64 * (2 * sample + 1) as f64 / (2 * len) as f64).cos()
            })
        }
        DctType::IV => matrix_apply(input, |row, column| {
            (2.0 / len as f64).sqrt()
                * (PI * (2 * row + 1) as f64 * (2 * column + 1) as f64 / (4 * len) as f64).cos()
        }),
    }
}

fn orthonormal_dst(input: &[f64], kind: DstType, direction: Direction) -> Vec<f64> {
    let len = input.len();
    match kind {
        DstType::I => matrix_apply(input, |row, column| {
            (2.0 / (len + 1) as f64).sqrt()
                * (PI * (row + 1) as f64 * (column + 1) as f64 / (len + 1) as f64).sin()
        }),
        DstType::II | DstType::III => {
            let transpose = (kind == DstType::II) == (direction == Direction::Inverse);
            matrix_apply(input, |row, column| {
                let (frequency, sample) = if transpose {
                    (column, row)
                } else {
                    (row, column)
                };
                (2.0 / len as f64).sqrt()
                    * if frequency + 1 == len {
                        1.0 / SQRT_2
                    } else {
                        1.0
                    }
                    * (PI * (frequency + 1) as f64 * (2 * sample + 1) as f64 / (2 * len) as f64)
                        .sin()
            })
        }
        DstType::IV => matrix_apply(input, |row, column| {
            (2.0 / len as f64).sqrt()
                * (PI * (2 * row + 1) as f64 * (2 * column + 1) as f64 / (4 * len) as f64).sin()
        }),
    }
}

fn endpoint_weight(index: usize, len: usize) -> f64 {
    if index == 0 || index + 1 == len {
        1.0 / SQRT_2
    } else {
        1.0
    }
}

fn matrix_apply(input: &[f64], coefficient: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    (0..input.len())
        .map(|row| {
            input
                .iter()
                .enumerate()
                .map(|(column, value)| coefficient(row, column) * value)
                .sum()
        })
        .collect()
}

fn apply_pair_normalization(
    output: &mut [f64],
    factor: f64,
    direction: Direction,
    normalization: Normalization,
) {
    let scale = match (normalization, direction) {
        (Normalization::None, _) => 1.0,
        (Normalization::Forward, Direction::Forward)
        | (Normalization::Inverse, Direction::Inverse) => 1.0 / factor,
        (Normalization::Forward | Normalization::Inverse, _) => 1.0,
        (Normalization::Orthonormal, _) => unreachable!("handled by orthonormal definitions"),
    };
    output.iter_mut().for_each(|value| *value *= scale);
}

fn validate_real(input: &[f64]) -> Result<(), SignalError> {
    if input.is_empty() {
        return Err(SignalError::InvalidLength {
            len: 0,
            reason: "real transforms require at least one value",
        });
    }
    for (index, value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "value",
            });
        }
    }
    Ok(())
}

fn validate_complex(input: &[(f64, f64)]) -> Result<(), SignalError> {
    for (index, (real, imag)) in input.iter().copied().enumerate() {
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
    Ok(())
}
