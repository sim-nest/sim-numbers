//! Native f16 and bf16 tensor storage plus descriptor registration.

use std::{fmt, sync::Arc};

use half::{bf16, f16};
use sim_kernel::Symbol;
use sim_lib_numbers_tensor::{
    SpecTensor, Tensor, TypedTensorStorage, checked_element_count, domains,
    parse_bf16_literal_cell, parse_f16_literal_cell,
};

type F16Storage = TypedTensorStorage<f16>;
type Bf16Storage = TypedTensorStorage<bf16>;

/// A tensor view whose cells are IEEE 754 half-precision `f16` values.
#[derive(Clone)]
pub struct F16Tensor {
    pub(crate) tensor: Tensor,
}

impl F16Tensor {
    /// Builds an `f16` tensor from a shape and flat data buffer.
    ///
    /// Returns `None` when `data.len()` does not match the element count
    /// implied by `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<f16>) -> Option<Self> {
        let expected = checked_element_count(&shape).ok()?;
        if expected != data.len() {
            return None;
        }
        let storage = Arc::new(F16Storage::new(data));
        Tensor::from_storage(shape, domains::f16(), storage)
            .ok()
            .map(|tensor| Self { tensor })
    }

    /// Borrows the native row-major cell slice.
    pub fn as_slice(&self) -> &[f16] {
        self.storage().cell_slice()
    }

    /// Converts every half cell to a native `f32` tensor.
    pub fn to_f32_uniform(&self) -> Tensor {
        f32_tensor(
            self.shape().to_vec(),
            self.as_slice().iter().map(|value| value.to_f32()).collect(),
        )
    }

    /// Adds an `f32` scalar to every cell and returns an `f32` tensor.
    pub fn add_f32_scalar(&self, scalar: f32) -> Tensor {
        f32_tensor(
            self.shape().to_vec(),
            self.as_slice()
                .iter()
                .map(|value| value.to_f32() + scalar)
                .collect(),
        )
    }

    /// Sums the half cells after widening each cell to `f32`.
    pub fn sum_f32(&self) -> f32 {
        self.as_slice().iter().map(|value| value.to_f32()).sum()
    }

    fn storage(&self) -> &F16Storage {
        self.tensor
            .storage()
            .as_any()
            .downcast_ref::<F16Storage>()
            .expect("F16Tensor must hold f16 typed storage")
    }
}

impl fmt::Debug for F16Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F16Tensor")
            .field(&self.shape())
            .field(&self.as_slice())
            .finish()
    }
}

impl PartialEq for F16Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.as_slice() == other.as_slice()
    }
}

impl SpecTensor for F16Tensor {
    fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    fn dtype(&self) -> Symbol {
        domains::f16()
    }

    fn to_uniform(&self) -> Tensor {
        self.tensor.clone()
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        (tensor.dtype() == &domains::f16()).then_some(())?;
        if tensor.storage().as_any().is::<F16Storage>() {
            return Some(Self {
                tensor: tensor.clone(),
            });
        }
        Self::new(
            tensor.shape().to_vec(),
            tensor
                .cells()
                .ok()?
                .iter()
                .map(parse_f16_literal_cell)
                .collect::<Option<Vec<_>>>()?,
        )
    }
}

/// A tensor view whose cells are bfloat16 `bf16` values.
#[derive(Clone)]
pub struct Bf16Tensor {
    pub(crate) tensor: Tensor,
}

impl Bf16Tensor {
    /// Builds a `bf16` tensor from a shape and flat data buffer.
    ///
    /// Returns `None` when `data.len()` does not match the element count
    /// implied by `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<bf16>) -> Option<Self> {
        let expected = checked_element_count(&shape).ok()?;
        if expected != data.len() {
            return None;
        }
        let storage = Arc::new(Bf16Storage::new(data));
        Tensor::from_storage(shape, domains::bf16(), storage)
            .ok()
            .map(|tensor| Self { tensor })
    }

    /// Borrows the native row-major cell slice.
    pub fn as_slice(&self) -> &[bf16] {
        self.storage().cell_slice()
    }

    /// Converts every bfloat16 cell to a native `f32` tensor.
    pub fn to_f32_uniform(&self) -> Tensor {
        f32_tensor(
            self.shape().to_vec(),
            self.as_slice().iter().map(|value| value.to_f32()).collect(),
        )
    }

    /// Adds an `f32` scalar to every cell and returns an `f32` tensor.
    pub fn add_f32_scalar(&self, scalar: f32) -> Tensor {
        f32_tensor(
            self.shape().to_vec(),
            self.as_slice()
                .iter()
                .map(|value| value.to_f32() + scalar)
                .collect(),
        )
    }

    /// Sums the bfloat16 cells after widening each cell to `f32`.
    pub fn sum_f32(&self) -> f32 {
        self.as_slice().iter().map(|value| value.to_f32()).sum()
    }

    fn storage(&self) -> &Bf16Storage {
        self.tensor
            .storage()
            .as_any()
            .downcast_ref::<Bf16Storage>()
            .expect("Bf16Tensor must hold bf16 typed storage")
    }
}

impl fmt::Debug for Bf16Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Bf16Tensor")
            .field(&self.shape())
            .field(&self.as_slice())
            .finish()
    }
}

impl PartialEq for Bf16Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.as_slice() == other.as_slice()
    }
}

impl SpecTensor for Bf16Tensor {
    fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    fn dtype(&self) -> Symbol {
        domains::bf16()
    }

    fn to_uniform(&self) -> Tensor {
        self.tensor.clone()
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        (tensor.dtype() == &domains::bf16()).then_some(())?;
        if tensor.storage().as_any().is::<Bf16Storage>() {
            return Some(Self {
                tensor: tensor.clone(),
            });
        }
        Self::new(
            tensor.shape().to_vec(),
            tensor
                .cells()
                .ok()?
                .iter()
                .map(parse_bf16_literal_cell)
                .collect::<Option<Vec<_>>>()?,
        )
    }
}

pub(crate) fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::from_storage(
        shape,
        domains::f32(),
        Arc::new(TypedTensorStorage::<f32>::new(data)),
    )
    .expect("f32 widened tensor should preserve shape")
}
