//! Forward-mode dual numbers: a value paired with an `N`-slot gradient, with
//! arithmetic and elementary functions that propagate derivatives.

use std::{
    array,
    ops::{Add, Div, Mul, Neg, Sub},
};

use super::Scalarish;

/// A forward-mode dual number: a primal value paired with an `N`-slot gradient.
///
/// Each arithmetic and elementary operation propagates the derivative slots
/// alongside the value, so evaluating an expression on `Dual<N>` computes both
/// the result and its partial derivatives with respect to `N` seeded
/// directions in a single pass. `N` is the number of independent directions
/// (inputs) tracked; `Dual<0>` carries no gradient and reduces to plain `f64`
/// arithmetic on the value.
///
/// # Examples
///
/// Differentiate `f(x) = x * x + 3 * x` at `x = 2`, where `f'(x) = 2x + 3 = 7`:
///
/// ```
/// use sim_lib_numbers_ad::Dual;
///
/// let x = Dual::<1>::var(2.0, 0);
/// let y = x * x + Dual::<1>::cst(3.0) * x;
/// assert_eq!(y.v, 10.0);
/// assert_eq!(y.d, [7.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<const N: usize> {
    /// The primal value of the number.
    pub v: f64,
    /// The gradient: one partial derivative per tracked direction.
    pub d: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// Builds a constant: the given value with a zero gradient.
    pub fn cst(v: f64) -> Self {
        Self { v, d: [0.0; N] }
    }

    /// Builds an independent variable: the given value seeded with derivative
    /// `1.0` in gradient slot `slot` (and zero elsewhere).
    ///
    /// A `slot` outside `0..N` seeds no direction, yielding a constant.
    pub fn var(v: f64, slot: usize) -> Self {
        let mut d = [0.0; N];
        if let Some(seed) = d.get_mut(slot) {
            *seed = 1.0;
        }
        Self { v, d }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            v: self.v + rhs.v,
            d: array::from_fn(|index| self.d[index] + rhs.d[index]),
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            v: self.v - rhs.v,
            d: array::from_fn(|index| self.d[index] - rhs.d[index]),
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            v: self.v * rhs.v,
            d: array::from_fn(|index| self.d[index].mul_add(rhs.v, rhs.d[index] * self.v)),
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.v * rhs.v;
        Self {
            v: self.v / rhs.v,
            d: array::from_fn(|index| (self.d[index] * rhs.v - self.v * rhs.d[index]) / denom),
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            v: -self.v,
            d: array::from_fn(|index| -self.d[index]),
        }
    }
}

impl<const N: usize> Scalarish for Dual<N> {
    fn from_f64(x: f64) -> Self {
        Self::cst(x)
    }

    fn sin(self) -> Self {
        let cos_v = self.v.cos();
        Self {
            v: self.v.sin(),
            d: array::from_fn(|index| self.d[index] * cos_v),
        }
    }

    fn cos(self) -> Self {
        let sin_v = self.v.sin();
        Self {
            v: self.v.cos(),
            d: array::from_fn(|index| -self.d[index] * sin_v),
        }
    }

    fn exp(self) -> Self {
        let exp_v = self.v.exp();
        Self {
            v: exp_v,
            d: array::from_fn(|index| self.d[index] * exp_v),
        }
    }

    fn ln(self) -> Self {
        Self {
            v: self.v.ln(),
            d: array::from_fn(|index| self.d[index] / self.v),
        }
    }

    fn sqrt(self) -> Self {
        let sqrt_v = self.v.sqrt();
        Self {
            v: sqrt_v,
            d: array::from_fn(|index| self.d[index] / (2.0 * sqrt_v)),
        }
    }

    fn recip(self) -> Self {
        let denom = self.v * self.v;
        Self {
            v: self.v.recip(),
            d: array::from_fn(|index| -self.d[index] / denom),
        }
    }
}
