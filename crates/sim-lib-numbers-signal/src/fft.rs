//! Fast complex transforms: radix-2, mixed-radix, and Bluestein paths.

use std::{
    f64::consts::{PI, TAU},
    ops::{Add, AddAssign, Mul, Sub},
};

use crate::SignalError;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Complex {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

impl Complex {
    pub(crate) const ZERO: Self = Self { re: 0.0, im: 0.0 };

    pub(crate) const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub(crate) fn cis(angle: f64) -> Self {
        Self::new(angle.cos(), angle.sin())
    }

    pub(crate) fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
}

impl From<(f64, f64)> for Complex {
    fn from((re, im): (f64, f64)) -> Self {
        Self::new(re, im)
    }
}

impl From<Complex> for (f64, f64) {
    fn from(value: Complex) -> Self {
        (value.re, value.im)
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

pub(crate) fn fft(input: &[Complex], angle_sign: f64) -> Result<Vec<Complex>, SignalError> {
    match input.len() {
        0 => Err(SignalError::InvalidLength {
            len: 0,
            reason: "FFT requires at least one value",
        }),
        1 => Ok(input.to_vec()),
        len if len.is_power_of_two() => {
            let mut output = input.to_vec();
            radix2_in_place(&mut output, angle_sign);
            Ok(output)
        }
        len if smallest_factor(len) < len => mixed_radix(input, angle_sign),
        _ => bluestein(input, angle_sign),
    }
}

fn smallest_factor(len: usize) -> usize {
    if len.is_multiple_of(2) {
        return 2;
    }
    let mut factor = 3;
    while factor <= len / factor {
        if len.is_multiple_of(factor) {
            return factor;
        }
        factor += 2;
    }
    len
}

fn mixed_radix(input: &[Complex], angle_sign: f64) -> Result<Vec<Complex>, SignalError> {
    let len = input.len();
    let radix = smallest_factor(len);
    let sub_len = len / radix;
    let mut sub_transforms = Vec::with_capacity(radix);
    for residue in 0..radix {
        let sequence = (0..sub_len)
            .map(|index| input[index * radix + residue])
            .collect::<Vec<_>>();
        sub_transforms.push(fft(&sequence, angle_sign)?);
    }

    let mut output = vec![Complex::ZERO; len];
    for (frequency, slot) in output.iter_mut().enumerate() {
        let mut sum = Complex::ZERO;
        for (residue, sub_transform) in sub_transforms.iter().enumerate() {
            let angle = angle_sign * TAU * (frequency as f64) * (residue as f64) / (len as f64);
            sum += sub_transform[frequency % sub_len] * Complex::cis(angle);
        }
        *slot = sum;
    }
    Ok(output)
}

fn bluestein(input: &[Complex], angle_sign: f64) -> Result<Vec<Complex>, SignalError> {
    let len = input.len();
    let convolution_len = len
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .and_then(usize::checked_next_power_of_two)
        .ok_or(SignalError::InvalidLength {
            len,
            reason: "Bluestein convolution length overflowed",
        })?;
    let mut a = vec![Complex::ZERO; convolution_len];
    let mut b = vec![Complex::ZERO; convolution_len];
    for index in 0..len {
        let phase = PI * (index as f64).powi(2) / len as f64;
        a[index] = input[index] * Complex::cis(angle_sign * phase);
        let chirp = Complex::cis(-angle_sign * phase);
        b[index] = chirp;
        if index != 0 {
            b[convolution_len - index] = chirp;
        }
    }
    radix2_in_place(&mut a, -1.0);
    radix2_in_place(&mut b, -1.0);
    for (left, right) in a.iter_mut().zip(b) {
        *left = *left * right;
    }
    radix2_in_place(&mut a, 1.0);
    let inverse_scale = 1.0 / convolution_len as f64;
    Ok((0..len)
        .map(|index| {
            let phase = PI * (index as f64).powi(2) / len as f64;
            a[index].scale(inverse_scale) * Complex::cis(angle_sign * phase)
        })
        .collect())
}

fn radix2_in_place(values: &mut [Complex], angle_sign: f64) {
    debug_assert!(values.len().is_power_of_two());
    let len = values.len();
    let mut target = 0;
    for source in 1..len {
        let mut bit = len >> 1;
        while target & bit != 0 {
            target ^= bit;
            bit >>= 1;
        }
        target ^= bit;
        if source < target {
            values.swap(source, target);
        }
    }

    let mut width = 2;
    while width <= len {
        let root = Complex::cis(angle_sign * TAU / width as f64);
        for start in (0..len).step_by(width) {
            let mut twiddle = Complex::new(1.0, 0.0);
            for offset in 0..width / 2 {
                let even = values[start + offset];
                let odd = values[start + offset + width / 2] * twiddle;
                values[start + offset] = even + odd;
                values[start + offset + width / 2] = even - odd;
                twiddle = twiddle * root;
            }
        }
        width *= 2;
    }
}
