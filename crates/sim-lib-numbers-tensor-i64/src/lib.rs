#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! i64 tensor specialization: a contiguous `i64` tensor element type and its
//! `SpecTensor` backend, with overflow-checked operations that fall back to the
//! bigint domain.
//!
//! [`I64Tensor`] is the storage type (a flat `i64` buffer); its checked
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

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Factory, Lib, LibManifest, LibTarget, Linker,
    Result, Symbol, Version,
};
use sim_lib_numbers_tensor::{
    SpecTensor, SpecTensorDescriptor, Tensor, domains, element_count, parse_i64_literal_cell,
    spec_tensor_descriptor_value, spec_tensor_symbol,
};

/// A tensor whose cells are native `i64` values in a contiguous buffer.
///
/// Storage is a flat `Vec<i64>` in row-major order over the tensor
/// [`shape`](Self::shape). Native integer math runs directly on the buffer;
/// operations that would overflow widen into the bigint domain instead of
/// wrapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I64Tensor {
    shape: Vec<usize>,
    data: Vec<i64>,
}

/// Outcome of an overflow-checked `i64` tensor operation.
///
/// The fast specialized path is kept when every cell stays in range; on
/// overflow the whole result widens to a boxed uniform tensor in the bigint
/// domain.
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
        (element_count(&shape) == data.len()).then_some(Self { shape, data })
    }

    /// Adds a scalar to every cell, widening to bigint on overflow.
    ///
    /// Returns [`I64AddResult::Specialized`] when every cell fits in `i64`, or
    /// [`I64AddResult::Uniform`] (a bigint uniform tensor) as soon as any cell
    /// would overflow.
    pub fn checked_add_scalar(&self, scalar: i64) -> I64AddResult {
        let mut out = Vec::with_capacity(self.data.len());
        for value in &self.data {
            match value.checked_add(scalar) {
                Some(sum) => out.push(sum),
                None => {
                    let tensor = Tensor::new_exact(
                        self.shape.clone(),
                        domains::bigint(),
                        self.data
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
        I64AddResult::Specialized(Self {
            shape: self.shape.clone(),
            data: out,
        })
    }
}

impl SpecTensor for I64Tensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> Symbol {
        domains::i64()
    }

    fn to_uniform(&self) -> Tensor {
        Tensor::new_exact(
            self.shape.clone(),
            self.dtype(),
            self.data
                .iter()
                .map(|value| {
                    DefaultFactory
                        .number_literal(domains::i64(), value.to_string())
                        .unwrap()
                })
                .collect(),
        )
        .expect("i64 tensor storage should convert to a valid uniform tensor")
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        Some(Self {
            shape: tensor.shape().to_vec(),
            data: tensor
                .data()
                .iter()
                .map(parse_i64_literal_cell)
                .collect::<Option<Vec<_>>>()?,
        })
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
                    storage: "Vec<i64>",
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
mod tests {
    use sim_kernel::Lib;
    use sim_lib_numbers_tensor::domains;

    use super::{I64AddResult, I64Tensor, I64TensorLib, tensor_spec_symbol};

    #[test]
    fn overflow_widens_to_bigint_uniform_tensor() {
        let tensor = I64Tensor::new(vec![2], vec![i64::MAX, 3]).unwrap();
        let out = tensor.checked_add_scalar(1);
        match out {
            I64AddResult::Uniform(tensor) => {
                assert_eq!(tensor.dtype(), &domains::bigint());
                assert_eq!(tensor.shape(), &[2]);
            }
            I64AddResult::Specialized(_) => panic!("expected bigint widening"),
        }
    }

    #[test]
    fn lib_exports_spec_tensor_descriptor() {
        assert_eq!(
            I64TensorLib::new().manifest().exports[0].symbol(),
            &tensor_spec_symbol()
        );
    }
}
