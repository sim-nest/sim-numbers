#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Complex-float tensor specialization: a `(real, imag)` f64-pair tensor element
//! type and its `SpecTensor` backend for the complex tensor domain.
//!
//! [`ComplexFTensor`] is the storage type (a flat vector of `(real, imag)`
//! pairs) and [`ComplexFTensorLib`] registers it as the `complex` element-type
//! backend for the base tensor domain.
//!
//! # Examples
//!
//! Build a complex tensor and confirm it round-trips through the uniform tensor
//! representation:
//!
//! ```
//! use sim_lib_numbers_tensor_cmplxf::ComplexFTensor;
//! use sim_lib_numbers_tensor::SpecTensor;
//!
//! let tensor = ComplexFTensor::new(vec![2], vec![(1.0, 2.0), (3.5, -4.0)]).unwrap();
//! let roundtrip = ComplexFTensor::from_uniform(&tensor.to_uniform()).unwrap();
//! assert_eq!(roundtrip, tensor);
//! ```
//!
//! A mismatched element count fails closed:
//!
//! ```
//! use sim_lib_numbers_tensor_cmplxf::ComplexFTensor;
//!
//! assert!(ComplexFTensor::new(vec![2, 2], vec![(0.0, 0.0)]).is_none());
//! ```

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Factory, Lib, LibManifest, LibTarget, Linker,
    Result, Symbol, Version,
};
use sim_lib_numbers_tensor::{
    SpecTensor, SpecTensorDescriptor, Tensor, domains, element_count, parse_complex_literal_cell,
    spec_tensor_descriptor_value, spec_tensor_symbol,
};

/// A complex tensor whose cells are `(real, imag)` `f64` pairs.
///
/// Storage is a flat `Vec<(f64, f64)>` in row-major order over the tensor
/// [`shape`](Self::shape). Conversion to and from the uniform tensor encodes
/// each cell as a `real+imagi` complex number literal.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexFTensor {
    shape: Vec<usize>,
    data: Vec<(f64, f64)>,
}

impl ComplexFTensor {
    /// Builds a complex tensor from a shape and flat `(real, imag)` data.
    ///
    /// Returns `None` when `data.len()` does not match the element count
    /// implied by `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<(f64, f64)>) -> Option<Self> {
        (element_count(&shape) == data.len()).then_some(Self { shape, data })
    }
}

impl SpecTensor for ComplexFTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> Symbol {
        domains::complex()
    }

    fn to_uniform(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            dtype: self.dtype(),
            data: self
                .data
                .iter()
                .map(|(real, imag)| {
                    DefaultFactory
                        .number_literal(domains::complex(), format!("{real}{imag:+}i"))
                        .unwrap()
                })
                .collect(),
        }
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        Some(Self {
            shape: tensor.shape.clone(),
            data: tensor
                .data
                .iter()
                .map(parse_complex_literal_cell)
                .collect::<Option<Vec<_>>>()?,
        })
    }
}

/// Registered library that installs the complex-float tensor backend.
///
/// Loading this [`Lib`] registers a [`SpecTensor`] descriptor binding the
/// `complex` element type to the [`ComplexFTensor`] storage, so the base tensor
/// domain can construct and round-trip complex tensors.
pub struct ComplexFTensorLib;

impl ComplexFTensorLib {
    /// Creates the complex-tensor library. The value is stateless; the
    /// spec-tensor descriptor is installed when it is loaded into a
    /// [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for ComplexFTensorLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for ComplexFTensorLib {
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
                    dtype: domains::complex(),
                    implementation: "ComplexFTensor",
                    storage: "Vec<(f64,f64)>",
                },
            )?,
        )
    }
}

/// The manifest id symbol for this library (`numbers/tensor-cmplxf`).
pub fn tensor_lib_symbol() -> Symbol {
    domains::domain("tensor-cmplxf")
}

/// The symbol under which the complex-tensor [`SpecTensor`] descriptor is
/// exported.
pub fn tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("cmplxf")
}

#[cfg(test)]
mod tests {
    use sim_kernel::Lib;

    use super::{ComplexFTensor, ComplexFTensorLib, SpecTensor, tensor_spec_symbol};

    #[test]
    fn complex_roundtrip_preserves_cells() {
        let tensor = ComplexFTensor::new(vec![2], vec![(1.0, 2.0), (3.5, -4.0)]).unwrap();
        let roundtrip = ComplexFTensor::from_uniform(&tensor.to_uniform()).unwrap();
        assert_eq!(roundtrip, tensor);
    }

    #[test]
    fn lib_exports_spec_tensor_descriptor() {
        assert_eq!(
            ComplexFTensorLib::new().manifest().exports[0].symbol(),
            &tensor_spec_symbol()
        );
    }
}
