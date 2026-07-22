#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Rational-i64 tensor specialization: a normalized `(numerator, denominator)`
//! i64-pair tensor element type and its `SpecTensor` backend for the rational
//! tensor domain.
//!
//! [`Rat64Tensor`] is the storage type (a flat vector of reduced `(num, den)`
//! pairs) and [`Rat64TensorLib`] registers it as the `rational` element-type
//! backend for the base tensor domain.
//!
//! # Examples
//!
//! Cells are reduced to lowest terms with the sign on the numerator, and the
//! tensor round-trips through the uniform representation:
//!
//! ```
//! use sim_lib_numbers_tensor_rat64::Rat64Tensor;
//! use sim_lib_numbers_tensor::SpecTensor;
//!
//! let tensor = Rat64Tensor::new(vec![2], vec![(2, 4), (-6, -8)]).unwrap();
//! let roundtrip = Rat64Tensor::from_uniform(&tensor.to_uniform()).unwrap();
//! assert_eq!(roundtrip.to_uniform().data().len(), 2);
//! ```
//!
//! A mismatched element count fails closed:
//!
//! ```
//! use sim_lib_numbers_tensor_rat64::Rat64Tensor;
//!
//! assert!(Rat64Tensor::new(vec![2, 2], vec![(1, 2)]).is_none());
//! ```

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Factory, Lib, LibManifest, LibTarget, Linker,
    Result, Symbol, Version,
};
use sim_lib_numbers_tensor::{
    SpecTensor, SpecTensorDescriptor, Tensor, checked_element_count, domains,
    parse_rational_literal_cell, spec_tensor_descriptor_value, spec_tensor_symbol,
};

/// A rational tensor whose cells are `(numerator, denominator)` `i64` pairs.
///
/// Storage is a flat `Vec<(i64, i64)>` in row-major order over the tensor
/// [`shape`](Self::shape). Every cell is normalized on construction: the
/// fraction is reduced by its greatest common divisor and the sign is carried
/// on the numerator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rat64Tensor {
    shape: Vec<usize>,
    data: Vec<(i64, i64)>,
}

impl Rat64Tensor {
    /// Builds a rational tensor from a shape and flat `(num, den)` data,
    /// normalizing each cell.
    ///
    /// Returns `None` when `data.len()` does not match the element count
    /// implied by `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<(i64, i64)>) -> Option<Self> {
        let expected = checked_element_count(&shape).ok()?;
        (expected == data.len()).then(|| Self {
            shape,
            data: data.into_iter().map(normalize).collect(),
        })
    }
}

impl SpecTensor for Rat64Tensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> Symbol {
        domains::rational()
    }

    fn to_uniform(&self) -> Tensor {
        Tensor::new_exact(
            self.shape.clone(),
            self.dtype(),
            self.data
                .iter()
                .map(|(num, den)| {
                    DefaultFactory
                        .number_literal(domains::rational(), format!("{num}/{den}"))
                        .unwrap()
                })
                .collect(),
        )
        .expect("rational tensor storage should convert to a valid uniform tensor")
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        Some(Self {
            shape: tensor.shape().to_vec(),
            data: tensor
                .data()
                .iter()
                .map(parse_rational_literal_cell)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .map(normalize)
                .collect(),
        })
    }
}

/// Registered library that installs the rational-`i64` tensor backend.
///
/// Loading this [`Lib`] registers a [`SpecTensor`] descriptor binding the
/// `rational` element type to the [`Rat64Tensor`] storage, so the base tensor
/// domain can construct and round-trip rational tensors.
pub struct Rat64TensorLib;

impl Rat64TensorLib {
    /// Creates the rational-tensor library. The value is stateless; the
    /// spec-tensor descriptor is installed when it is loaded into a
    /// [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for Rat64TensorLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for Rat64TensorLib {
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
                    dtype: domains::rational(),
                    implementation: "Rat64Tensor",
                    storage: "Vec<(i64,i64)>",
                },
            )?,
        )
    }
}

/// The manifest id symbol for this library (`numbers/tensor-rat64`).
pub fn tensor_lib_symbol() -> Symbol {
    domains::domain("tensor-rat64")
}

/// The symbol under which the rational-tensor [`SpecTensor`] descriptor is
/// exported.
pub fn tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("rat64")
}

fn normalize((mut num, mut den): (i64, i64)) -> (i64, i64) {
    if den < 0 {
        num = -num;
        den = -den;
    }
    let gcd = gcd(num.unsigned_abs(), den.unsigned_abs()) as i64;
    (num / gcd, den / gcd)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left.max(1)
}

/// Cookbook recipes for this domain, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests {
    use sim_kernel::Lib;
    use sim_lib_numbers_tensor::SpecTensor;

    use super::{Rat64Tensor, Rat64TensorLib, tensor_spec_symbol};

    #[test]
    fn rationals_are_normalized() {
        let tensor = Rat64Tensor::new(vec![2], vec![(2, 4), (-6, -8)]).unwrap();
        assert_eq!(tensor.to_uniform().data().len(), 2);
        let roundtrip = Rat64Tensor::from_uniform(&tensor.to_uniform()).unwrap();
        assert_eq!(roundtrip.data, vec![(1, 2), (3, 4)]);
    }

    #[test]
    fn constructor_rejects_overflowing_shape() {
        assert!(Rat64Tensor::new(vec![usize::MAX, 2], Vec::new()).is_none());
    }

    #[test]
    fn lib_exports_spec_tensor_descriptor() {
        assert_eq!(
            Rat64TensorLib::new().manifest().exports[0].symbol(),
            &tensor_spec_symbol()
        );
    }
}
