//! Native f32 tensor storage and descriptor registration.

use std::{fmt, sync::Arc};

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Lib, LibManifest, LibTarget, Linker, Result,
    Symbol, Version,
};
use sim_lib_numbers_tensor::{
    SpecTensor, SpecTensorDescriptor, Tensor, TypedTensorStorage, checked_element_count, domains,
    parse_f32_literal_cell, spec_tensor_descriptor_value, spec_tensor_symbol,
};

type F32Storage = TypedTensorStorage<f32>;

/// A tensor view whose cells are native `f32` values in canonical storage.
///
/// Storage is a flat `f32` slice in row-major order over the tensor
/// [`shape`](Self::shape). Working in native `f32` lets single-precision
/// element-wise math run directly on the typed storage behind the uniform
/// [`Tensor`].
#[derive(Clone)]
pub struct F32Tensor {
    pub(crate) tensor: Tensor,
}

impl F32Tensor {
    /// Builds an `f32` tensor from a shape and flat data buffer.
    ///
    /// Returns `None` when `data.len()` does not match the element count
    /// implied by `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Option<Self> {
        let expected = checked_element_count(&shape).ok()?;
        if expected != data.len() {
            return None;
        }
        let storage = Arc::new(F32Storage::new(data));
        Tensor::from_storage(shape, domains::f32(), storage)
            .ok()
            .map(|tensor| Self { tensor })
    }

    /// Borrows the native row-major cell slice.
    pub fn as_slice(&self) -> &[f32] {
        self.storage().cell_slice()
    }

    /// Adds a scalar to every cell, operating directly on the native buffer.
    pub fn add_scalar(&self, scalar: f32) -> Self {
        Self::new(
            self.shape().to_vec(),
            self.as_slice().iter().map(|value| value + scalar).collect(),
        )
        .expect("f32 add should preserve tensor shape")
    }

    fn storage(&self) -> &F32Storage {
        self.tensor
            .storage()
            .as_any()
            .downcast_ref::<F32Storage>()
            .expect("F32Tensor must hold f32 typed storage")
    }
}

impl fmt::Debug for F32Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F32Tensor")
            .field(&self.shape())
            .field(&self.as_slice())
            .finish()
    }
}

impl PartialEq for F32Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.as_slice() == other.as_slice()
    }
}

impl SpecTensor for F32Tensor {
    fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    fn dtype(&self) -> Symbol {
        domains::f32()
    }

    fn to_uniform(&self) -> Tensor {
        self.tensor.clone()
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        (tensor.dtype() == &domains::f32()).then_some(())?;
        if tensor.storage().as_any().is::<F32Storage>() {
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
                .map(parse_f32_literal_cell)
                .collect::<Option<Vec<_>>>()?,
        )
    }
}

/// Registered library that installs the native-`f32` tensor backend.
///
/// Loading this [`Lib`] registers a [`SpecTensor`] descriptor binding the `f32`
/// element type to the [`F32Tensor`] storage, so the base tensor domain can
/// construct and round-trip `f32` tensors against the contiguous buffer.
pub struct F32TensorLib;

impl F32TensorLib {
    /// Creates the `f32`-tensor library. The value is stateless; the
    /// spec-tensor descriptor is installed when it is loaded into a
    /// [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for F32TensorLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for F32TensorLib {
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
                    dtype: domains::f32(),
                    implementation: "F32Tensor",
                    storage: "canonical Tensor storage over f32 cells",
                },
            )?,
        )
    }
}

/// The manifest id symbol for this library (`numbers/tensor-f32`).
pub fn tensor_lib_symbol() -> Symbol {
    domains::domain("tensor-f32")
}

/// The symbol under which the `f32`-tensor [`SpecTensor`] descriptor is
/// exported.
pub fn tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("f32")
}
