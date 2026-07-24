#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! i64 tensor specialization: a contiguous `i64` tensor element type and its
//! `SpecTensor` backend, with overflow-checked operations that fall back to the
//! bigint domain.
//!
//! [`I64Tensor`] is a typed view over canonical `Tensor` storage; its checked
//! operations return an [`I64AddResult`] that widens to bigint on overflow.
//! [`I64TensorLib`] registers it as the `i64` element-type backend for the base
//! tensor domain.
//!
//! # Examples
//!
//! In-range additions stay on the fast specialized path:
//!
//! ```
//! use sim_lib_numbers_tensor_i64::{I64Tensor, I64AddResult};
//!
//! let tensor = I64Tensor::new(vec![3], vec![1, 2, 3]).unwrap();
//! assert!(matches!(
//!     tensor.checked_add_scalar(10),
//!     I64AddResult::Specialized(_)
//! ));
//! ```
//!
//! Overflow widens the whole result into a bigint uniform tensor:
//!
//! ```
//! use sim_lib_numbers_tensor_i64::{I64Tensor, I64AddResult};
//!
//! let tensor = I64Tensor::new(vec![2], vec![i64::MAX, 3]).unwrap();
//! assert!(matches!(
//!     tensor.checked_add_scalar(1),
//!     I64AddResult::Uniform(_)
//! ));
//! ```

use std::{fmt, sync::Arc};

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Factory, Lib, LibManifest, LibTarget, Linker,
    Result, Symbol, Version,
};
use sim_lib_numbers_tensor::{
    SpecTensor, SpecTensorDescriptor, Tensor, TypedTensorStorage, checked_element_count, domains,
    parse_i64_literal_cell, spec_tensor_descriptor_value, spec_tensor_symbol,
};

type I64Storage = TypedTensorStorage<i64>;

/// A tensor view whose cells are native `i64` values in canonical storage.
///
/// Storage is a flat `i64` slice in row-major order over the tensor
/// [`shape`](Self::shape). Native integer math runs directly on the buffer;
/// operations that would overflow widen into the bigint domain instead of
/// wrapping.
#[derive(Clone)]
pub struct I64Tensor {
    tensor: Tensor,
}

/// Outcome of an overflow-checked `i64` tensor operation.
#[derive(Clone)]
pub enum I64AddResult {
    /// Every cell stayed within `i64`; the native [`I64Tensor`] is returned.
    Specialized(I64Tensor),
    /// A cell overflowed `i64`; the result widened to a bigint uniform tensor.
    Uniform(Tensor),
}

impl I64Tensor {
    /// Builds an `i64` tensor from a shape and flat data buffer.
    ///
    /// Returns `None` when `data.len()` does not match the element count
    /// implied by `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<i64>) -> Option<Self> {
        let expected = checked_element_count(&shape).ok()?;
        if expected != data.len() {
            return None;
        }
        let storage = Arc::new(I64Storage::new(data));
        Tensor::from_storage(shape, domains::i64(), storage)
            .ok()
            .map(|tensor| Self { tensor })
    }

    /// Borrows the native row-major cell slice.
    pub fn as_slice(&self) -> &[i64] {
        self.storage().cell_slice()
    }

    /// Adds a scalar to every cell, widening to bigint on overflow.
    pub fn checked_add_scalar(&self, scalar: i64) -> I64AddResult {
        let mut out = Vec::with_capacity(self.as_slice().len());
        for value in self.as_slice() {
            match value.checked_add(scalar) {
                Some(sum) => out.push(sum),
                None => {
                    let tensor = Tensor::new_exact(
                        self.shape().to_vec(),
                        domains::bigint(),
                        self.as_slice()
                            .iter()
                            .map(|cell| {
                                DefaultFactory
                                    .number_literal(
                                        domains::bigint(),
                                        (i128::from(*cell) + i128::from(scalar)).to_string(),
                                    )
                                    .unwrap()
                            })
                            .collect(),
                    )
                    .expect("bigint overflow fallback should build a valid uniform tensor");
                    return I64AddResult::Uniform(tensor);
                }
            }
        }
        I64AddResult::Specialized(Self::new(self.shape().to_vec(), out).unwrap())
    }

    fn storage(&self) -> &I64Storage {
        self.tensor
            .storage()
            .as_any()
            .downcast_ref::<I64Storage>()
            .expect("I64Tensor must hold i64 typed storage")
    }
}

impl fmt::Debug for I64Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("I64Tensor")
            .field(&self.shape())
            .field(&self.as_slice())
            .finish()
    }
}

impl PartialEq for I64Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.as_slice() == other.as_slice()
    }
}

impl Eq for I64Tensor {}

impl SpecTensor for I64Tensor {
    fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    fn dtype(&self) -> Symbol {
        domains::i64()
    }

    fn to_uniform(&self) -> Tensor {
        self.tensor.clone()
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        (tensor.dtype() == &domains::i64()).then_some(())?;
        if tensor.storage().as_any().is::<I64Storage>() {
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
                .map(parse_i64_literal_cell)
                .collect::<Option<Vec<_>>>()?,
        )
    }
}

/// Registered library that installs the native-`i64` tensor backend.
///
/// Loading this [`Lib`] registers a [`SpecTensor`] descriptor binding the `i64`
/// element type to the [`I64Tensor`] storage, so the base tensor domain can
/// construct and round-trip `i64` tensors against the contiguous buffer.
pub struct I64TensorLib;

impl I64TensorLib {
    /// Creates the `i64`-tensor library. The value is stateless; the
    /// spec-tensor descriptor is installed when it is loaded into a
    /// [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for I64TensorLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for I64TensorLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: tensor_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: tensor_spec_symbol(),
            }],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.value(
            tensor_spec_symbol(),
            spec_tensor_descriptor_value(
                &DefaultFactory,
                SpecTensorDescriptor {
                    symbol: tensor_spec_symbol(),
                    dtype: domains::i64(),
                    implementation: "I64Tensor",
                    storage: "canonical Tensor storage over i64 cells",
                },
            )?,
        )
    }
}

/// The manifest id symbol for this library (`numbers/tensor-i64`).
pub fn tensor_lib_symbol() -> Symbol {
    domains::domain("tensor-i64")
}

/// The symbol under which the `i64`-tensor [`SpecTensor`] descriptor is
/// exported.
pub fn tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("i64")
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
