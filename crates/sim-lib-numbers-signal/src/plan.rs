//! Typed transform conventions and canonical signal buffers.

use std::num::NonZeroUsize;

use sim_lib_numbers_tensor_cmplxf::ComplexFTensor;
use sim_lib_numbers_tensor_f64::F64Tensor;

use crate::SignalError;

/// Direction in which a transform plan is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Map samples to transform coefficients.
    Forward,
    /// Map transform coefficients back to samples.
    Inverse,
}

/// Scaling convention applied by a forward/inverse transform pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// Apply no normalization in either direction.
    None,
    /// Normalize only the forward transform by its definition-level factor.
    Forward,
    /// Normalize only the inverse transform by its definition-level factor.
    Inverse,
    /// Use the orthonormal basis in both directions.
    Orthonormal,
}

/// Complex-exponential sign convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignConvention {
    /// Forward transforms use `exp(-i theta)` and inverse transforms use
    /// `exp(+i theta)`.
    NegativeForward,
    /// Forward transforms use `exp(+i theta)` and inverse transforms use
    /// `exp(-i theta)`.
    PositiveForward,
}

impl SignConvention {
    /// Returns the signed angle multiplier for `direction`.
    pub fn angle_sign(self, direction: Direction) -> f64 {
        match (self, direction) {
            (Self::NegativeForward, Direction::Forward)
            | (Self::PositiveForward, Direction::Inverse) => -1.0,
            (Self::PositiveForward, Direction::Forward)
            | (Self::NegativeForward, Direction::Inverse) => 1.0,
        }
    }
}

/// Packing of a real FFT spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumPacking {
    /// Store all `N` complex frequency bins.
    Full,
    /// Store bins `0..=N/2`; omitted bins are their Hermitian mirrors.
    HermitianHalf,
}

/// Relationship between the logical plan length and available input values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthPolicy {
    /// Require exactly the plan length.
    Exact,
    /// Admit a shorter input and extend it according to [`PaddingPolicy`].
    Pad,
    /// Require at least the plan length and ignore later values.
    Truncate,
}

/// Values used when [`LengthPolicy::Pad`] extends an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingPolicy {
    /// Do not synthesize values.
    Reject,
    /// Extend with real or complex zero values.
    Zero,
}

/// Logical selection of values from a physical one-dimensional buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stride {
    offset: usize,
    step: NonZeroUsize,
}

impl Stride {
    /// Contiguous selection starting at physical index zero.
    pub const fn contiguous() -> Self {
        Self {
            offset: 0,
            step: NonZeroUsize::MIN,
        }
    }

    /// Builds a selection from `offset` with a nonzero `step`.
    pub fn new(offset: usize, step: usize) -> Result<Self, SignalError> {
        let step = NonZeroUsize::new(step).ok_or(SignalError::ZeroStride)?;
        Ok(Self { offset, step })
    }

    /// First selected physical index.
    pub const fn offset(self) -> usize {
        self.offset
    }

    /// Distance between consecutive selected physical indices.
    pub const fn step(self) -> usize {
        self.step.get()
    }

    /// Returns the number of values reachable in a physical buffer of `len`.
    pub fn available(self, len: usize) -> usize {
        if self.offset >= len {
            0
        } else {
            1 + (len - 1 - self.offset) / self.step()
        }
    }

    /// Maps a logical index to a physical index with overflow checking.
    pub fn physical_index(self, logical: usize) -> Result<usize, SignalError> {
        logical
            .checked_mul(self.step())
            .and_then(|delta| self.offset.checked_add(delta))
            .ok_or(SignalError::StrideOverflow)
    }
}

impl Default for Stride {
    fn default() -> Self {
        Self::contiguous()
    }
}

/// Whether execution writes a distinct result or overwrites caller storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementPolicy {
    /// Return a new canonical tensor buffer.
    OutOfPlace,
    /// Overwrite a mutable caller slice of the same representation and length.
    InPlace,
}

/// One of the four standard discrete cosine transform definitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DctType {
    /// DCT-I.
    I,
    /// DCT-II.
    II,
    /// DCT-III.
    III,
    /// DCT-IV.
    IV,
}

/// One of the four standard discrete sine transform definitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DstType {
    /// DST-I.
    I,
    /// DST-II.
    II,
    /// DST-III.
    III,
    /// DST-IV.
    IV,
}

/// Mathematical transform selected by a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformKind {
    /// Direct O(N^2) complex discrete Fourier transform reference.
    Dft,
    /// Mixed-radix or Bluestein complex fast Fourier transform.
    Fft,
    /// FFT of real samples with explicit full or Hermitian-half packing.
    RealFft,
    /// Discrete cosine transform of the selected type.
    Dct(DctType),
    /// Discrete sine transform of the selected type.
    Dst(DstType),
}

/// Reusable, fully explicit transform plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformPlan {
    /// Transform definition.
    pub kind: TransformKind,
    /// Logical transform length.
    pub len: usize,
    /// Forward or inverse application.
    pub direction: Direction,
    /// Scaling convention.
    pub normalization: Normalization,
    /// Complex-exponential sign convention.
    pub sign: SignConvention,
    /// Real-spectrum packing.
    pub packing: SpectrumPacking,
    /// Length admission policy.
    pub length: LengthPolicy,
    /// Padding value policy.
    pub padding: PaddingPolicy,
    /// Input selection stride.
    pub stride: Stride,
    /// Output placement policy.
    pub placement: PlacementPolicy,
}

impl TransformPlan {
    /// Builds the conventional forward, inverse-normalized, out-of-place plan.
    pub fn new(kind: TransformKind, len: usize) -> Self {
        Self {
            kind,
            len,
            direction: Direction::Forward,
            normalization: Normalization::Inverse,
            sign: SignConvention::NegativeForward,
            packing: SpectrumPacking::Full,
            length: LengthPolicy::Exact,
            padding: PaddingPolicy::Reject,
            stride: Stride::contiguous(),
            placement: PlacementPolicy::OutOfPlace,
        }
    }

    /// Validates definition-level length and policy invariants.
    pub fn validate(&self) -> Result<(), SignalError> {
        if self.len == 0 {
            return Err(SignalError::InvalidLength {
                len: self.len,
                reason: "transforms require at least one value",
            });
        }
        if self.kind == TransformKind::Dct(DctType::I) && self.len < 2 {
            return Err(SignalError::InvalidLength {
                len: self.len,
                reason: "DCT-I requires at least two values",
            });
        }
        if self.packing != SpectrumPacking::Full && self.kind != TransformKind::RealFft {
            return Err(SignalError::InvalidPolicy {
                policy: "packing",
                reason: "Hermitian-half packing is defined only for real FFT",
            });
        }
        match (self.length, self.padding) {
            (LengthPolicy::Pad, PaddingPolicy::Zero)
            | (LengthPolicy::Exact | LengthPolicy::Truncate, PaddingPolicy::Reject) => Ok(()),
            (LengthPolicy::Pad, PaddingPolicy::Reject) => Err(SignalError::InvalidPolicy {
                policy: "padding",
                reason: "Pad length policy requires zero padding",
            }),
            (LengthPolicy::Exact | LengthPolicy::Truncate, PaddingPolicy::Zero) => {
                Err(SignalError::InvalidPolicy {
                    policy: "padding",
                    reason: "zero padding requires the Pad length policy",
                })
            }
        }
    }
}

/// Borrowed transform input.
#[derive(Clone, Copy, Debug)]
pub enum SignalView<'a> {
    /// Canonical complex cells as `(real, imag)` pairs.
    Complex(&'a [(f64, f64)]),
    /// Canonical real cells.
    Real(&'a [f64]),
}

impl<'a> SignalView<'a> {
    /// Borrows the native storage of a canonical complex tensor.
    pub fn from_complex_tensor(tensor: &'a ComplexFTensor) -> Self {
        Self::Complex(tensor.as_slice())
    }

    /// Borrows the native storage of a canonical f64 tensor.
    pub fn from_real_tensor(tensor: &'a F64Tensor) -> Self {
        Self::Real(tensor.as_slice())
    }

    /// Physical number of cells in the borrowed storage.
    pub fn physical_len(self) -> usize {
        match self {
            Self::Complex(values) => values.len(),
            Self::Real(values) => values.len(),
        }
    }
}

/// Mutable signal storage accepted by in-place transforms.
#[derive(Debug)]
pub enum SignalViewMut<'a> {
    /// Mutable complex `(real, imag)` cells.
    Complex(&'a mut [(f64, f64)]),
    /// Mutable real cells.
    Real(&'a mut [f64]),
}

/// Owned transform result in canonical tensor storage.
#[derive(Clone, Debug, PartialEq)]
pub enum SignalBuffer {
    /// Complex coefficient or sample tensor.
    Complex(ComplexFTensor),
    /// Real coefficient or sample tensor.
    Real(F64Tensor),
}

impl SignalBuffer {
    /// Logical one-dimensional result length.
    pub fn len(&self) -> usize {
        match self {
            Self::Complex(values) => values.as_slice().len(),
            Self::Real(values) => values.as_slice().len(),
        }
    }

    /// Whether the result has no cells.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
