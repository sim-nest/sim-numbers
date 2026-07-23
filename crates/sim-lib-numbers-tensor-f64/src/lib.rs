#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! f64 tensor specialization: a contiguous `f64` tensor element type and its
//! `SpecTensor` backend for the f64 tensor domain.
//!
//! [`F64Tensor`] is a typed view over canonical `Tensor` storage with native
//! element-wise math, and [`F64TensorLib`] registers it as the `f64`
//! element-type backend for the base tensor domain.
//!
//! # Examples
//!
//! Native element-wise math runs on the buffer and round-trips through the
//! uniform tensor representation:
//!
//! ```
//! use sim_lib_numbers_tensor_f64::F64Tensor;
//! use sim_lib_numbers_tensor::SpecTensor;
//!
//! let tensor = F64Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
//! let shifted = tensor.add_scalar(10.0);
//! let roundtrip = F64Tensor::from_uniform(&shifted.to_uniform()).unwrap();
//! assert_eq!(roundtrip, shifted);
//! ```
//!
//! A mismatched element count fails closed:
//!
//! ```
//! use sim_lib_numbers_tensor_f64::F64Tensor;
//!
//! assert!(F64Tensor::new(vec![2, 2], vec![0.0]).is_none());
//! ```

use std::{fmt, sync::Arc, time::Instant};

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Factory, Lib, LibManifest, LibTarget, Linker,
    Result, Symbol, Version,
};
use sim_lib_numbers_tensor::{
    SpecTensor, SpecTensorDescriptor, Tensor, TypedTensorStorage, checked_element_count, domains,
    parse_f64_literal_cell, spec_tensor_descriptor_value, spec_tensor_symbol,
};

type F64Storage = TypedTensorStorage<f64>;

/// A tensor view whose cells are native `f64` values in canonical storage.
///
/// Storage is a flat `f64` slice in row-major order over the tensor
/// [`shape`](Self::shape). Working in native `f64` lets element-wise math run
/// directly on the typed storage behind the uniform [`Tensor`].
#[derive(Clone)]
pub struct F64Tensor {
    tensor: Tensor,
}

impl F64Tensor {
    /// Builds an `f64` tensor from a shape and flat data buffer.
    ///
    /// Returns `None` when `data.len()` does not match the element count
    /// implied by `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Option<Self> {
        let expected = checked_element_count(&shape).ok()?;
        if expected != data.len() {
            return None;
        }
        let storage = Arc::new(F64Storage::new(data));
        Tensor::from_storage(shape, domains::f64(), storage)
            .ok()
            .map(|tensor| Self { tensor })
    }

    /// Borrows the native row-major cell slice.
    pub fn as_slice(&self) -> &[f64] {
        self.storage().cell_slice()
    }

    /// Adds a scalar to every cell, operating directly on the native buffer.
    pub fn add_scalar(&self, scalar: f64) -> Self {
        Self::new(
            self.shape().to_vec(),
            self.as_slice().iter().map(|value| value + scalar).collect(),
        )
        .expect("f64 add should preserve tensor shape")
    }

    /// Adds a scalar through uniform cell observation as the reference path.
    pub fn add_uniform_scalar_slow(&self, scalar: f64) -> Tensor {
        let uniform = self.to_uniform();
        Tensor::new_exact(
            uniform.shape().to_vec(),
            uniform.dtype().clone(),
            uniform
                .cells()
                .expect("uniform f64 tensor storage should be observable")
                .iter()
                .map(|value| {
                    let literal = parse_f64_literal_cell(value).unwrap();
                    DefaultFactory
                        .number_literal(domains::f64(), (literal + scalar).to_string())
                        .unwrap()
                })
                .collect(),
        )
        .expect("uniform f64 tensor conversion should stay valid")
    }

    /// Times both add-scalar paths once and returns `(fast_ns, slow_ns)`.
    ///
    /// A coarse smoke measurement comparing the native
    /// [`add_scalar`](Self::add_scalar) against
    /// [`add_uniform_scalar_slow`](Self::add_uniform_scalar_slow); each
    /// duration is clamped to at least one nanosecond.
    pub fn smoke_speed_ratio(&self, scalar: f64) -> (u128, u128) {
        let fast_start = Instant::now();
        let _ = self.add_scalar(scalar);
        let fast = fast_start.elapsed().as_nanos();

        let slow_start = Instant::now();
        let _ = self.add_uniform_scalar_slow(scalar);
        let slow = slow_start.elapsed().as_nanos();
        (fast.max(1), slow.max(1))
    }

    fn storage(&self) -> &F64Storage {
        self.tensor
            .storage()
            .as_any()
            .downcast_ref::<F64Storage>()
            .expect("F64Tensor must hold f64 typed storage")
    }
}

impl fmt::Debug for F64Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F64Tensor")
            .field(&self.shape())
            .field(&self.as_slice())
            .finish()
    }
}

impl PartialEq for F64Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.as_slice() == other.as_slice()
    }
}

impl SpecTensor for F64Tensor {
    fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    fn dtype(&self) -> Symbol {
        domains::f64()
    }

    fn to_uniform(&self) -> Tensor {
        self.tensor.clone()
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        (tensor.dtype() == &domains::f64()).then_some(())?;
        if tensor.storage().as_any().is::<F64Storage>() {
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
                .map(parse_f64_literal_cell)
                .collect::<Option<Vec<_>>>()?,
        )
    }
}

/// Registered library that installs the native-`f64` tensor backend.
///
/// Loading this [`Lib`] registers a [`SpecTensor`] descriptor binding the `f64`
/// element type to the [`F64Tensor`] storage, so the base tensor domain can
/// construct and round-trip `f64` tensors against the contiguous buffer.
pub struct F64TensorLib;

impl F64TensorLib {
    /// Creates the `f64`-tensor library. The value is stateless; the
    /// spec-tensor descriptor is installed when it is loaded into a
    /// [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for F64TensorLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for F64TensorLib {
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
                    dtype: domains::f64(),
                    implementation: "F64Tensor",
                    storage: "canonical Tensor storage over f64 cells",
                },
            )?,
        )
    }
}

/// The manifest id symbol for this library (`numbers/tensor-f64`).
pub fn tensor_lib_symbol() -> Symbol {
    domains::domain("tensor-f64")
}

/// The symbol under which the `f64`-tensor [`SpecTensor`] descriptor is
/// exported.
pub fn tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("f64")
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
