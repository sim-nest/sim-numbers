//! The `Scalarish` numeric trait: arithmetic plus elementary functions shared
//! by `f64` and the autodiff number types.

use std::ops::{Add, Div, Mul, Sub};

/// A minimal scalar numeric interface: the arithmetic operators plus a fixed
/// set of elementary functions, shared by `f64` and the autodiff number types.
///
/// Writing a numeric routine generic over `Scalarish` lets the same code run on
/// plain `f64` for the value and on [`Dual`](crate::Dual) to obtain forward-mode
/// derivatives, with no change to the body.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_ad::{Dual, Scalarish};
///
/// fn quadratic<S: Scalarish>(x: S) -> S {
///     x * x + S::from_f64(3.0) * x
/// }
///
/// // Evaluate on plain f64.
/// assert_eq!(quadratic(2.0), 10.0);
/// // Evaluate on a dual to also get the derivative at x = 2 (which is 7).
/// assert_eq!(quadratic(Dual::<1>::var(2.0, 0)).d, [7.0]);
/// ```
pub trait Scalarish:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    /// Lifts a plain `f64` into this scalar type (as a constant with zero
    /// derivative, for the differentiable types).
    fn from_f64(x: f64) -> Self;
    /// Returns the sine of `self`.
    fn sin(self) -> Self;
    /// Returns the cosine of `self`.
    fn cos(self) -> Self;
    /// Returns `e` raised to the power of `self`.
    fn exp(self) -> Self;
    /// Returns the natural logarithm of `self`.
    fn ln(self) -> Self;
    /// Returns the square root of `self`.
    fn sqrt(self) -> Self;
    /// Returns the reciprocal `1 / self`.
    fn recip(self) -> Self;
}

impl Scalarish for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn recip(self) -> Self {
        self.recip()
    }
}
