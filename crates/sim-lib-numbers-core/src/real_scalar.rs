//! Scalar bridge for dependency-light numerical algorithms.

/// A finite-real algorithm scalar, deliberately separate from SIM's runtime
/// number domains and promotion lattice.
///
/// Algorithms must reject non-finite inputs before calling arithmetic methods;
/// [`Self::from_f64`] is the canonical admission boundary.
pub trait RealScalar: Copy + PartialOrd + core::fmt::Debug + Send + Sync + 'static {
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Machine epsilon.
    const EPSILON: Self;
    /// Smallest positive normal value.
    const MIN_POSITIVE: Self;
    /// Largest finite value.
    const MAX: Self;

    /// Admits one finite canonical f64 value.
    fn from_f64(value: f64) -> Option<Self>;
    /// Converts to the canonical interchange scalar.
    fn to_f64(self) -> f64;
    /// Whether this value is finite.
    fn is_finite(self) -> bool;
    /// Magnitude.
    fn abs(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Fused multiply-add.
    fn mul_add(self, multiplier: Self, addend: Self) -> Self;
    /// Addition.
    fn add(self, rhs: Self) -> Self;
    /// Subtraction.
    fn sub(self, rhs: Self) -> Self;
    /// Multiplication.
    fn mul(self, rhs: Self) -> Self;
    /// Division.
    fn div(self, rhs: Self) -> Self;
}

impl RealScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f64::EPSILON;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX: Self = f64::MAX;

    fn from_f64(value: f64) -> Option<Self> {
        value.is_finite().then_some(value)
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn abs(self) -> Self {
        self.abs()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        self.mul_add(multiplier, addend)
    }
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
}

#[cfg(test)]
mod tests {
    use super::RealScalar;

    #[test]
    fn f64_bridge_rejects_non_finite_values() {
        assert_eq!(f64::from_f64(-0.0).unwrap().to_bits(), (-0.0f64).to_bits());
        assert!(f64::from_f64(f64::NAN).is_none());
        assert!(f64::from_f64(f64::INFINITY).is_none());
    }
}
