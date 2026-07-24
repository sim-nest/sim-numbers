//! Descriptor registration for the half tensor storage backends.

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Lib, LibManifest, LibTarget, Linker, Result,
    Symbol, Version,
};
use sim_lib_numbers_tensor::{
    SpecTensorDescriptor, domains, spec_tensor_descriptor_value, spec_tensor_symbol,
};

/// Registered library that installs `f16` and `bf16` tensor backends.
pub struct HalfTensorLib;

impl HalfTensorLib {
    /// Creates the half-tensor library. The value is stateless; the descriptor
    /// values are installed when it is loaded into a [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for HalfTensorLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for HalfTensorLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: tensor_lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![
                Export::Value {
                    symbol: f16_tensor_spec_symbol(),
                },
                Export::Value {
                    symbol: bf16_tensor_spec_symbol(),
                },
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.value(
            f16_tensor_spec_symbol(),
            spec_tensor_descriptor_value(
                &DefaultFactory,
                SpecTensorDescriptor {
                    symbol: f16_tensor_spec_symbol(),
                    dtype: domains::f16(),
                    implementation: "F16Tensor",
                    storage: "canonical Tensor storage over f16 cells",
                },
            )?,
        )?;
        linker.value(
            bf16_tensor_spec_symbol(),
            spec_tensor_descriptor_value(
                &DefaultFactory,
                SpecTensorDescriptor {
                    symbol: bf16_tensor_spec_symbol(),
                    dtype: domains::bf16(),
                    implementation: "Bf16Tensor",
                    storage: "canonical Tensor storage over bf16 cells",
                },
            )?,
        )
    }
}

/// The manifest id symbol for this library (`numbers/tensor-half`).
pub fn tensor_lib_symbol() -> Symbol {
    domains::domain("tensor-half")
}

/// The symbol under which the `f16` tensor [`SpecTensor`](sim_lib_numbers_tensor::SpecTensor)
/// descriptor is exported.
pub fn f16_tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("f16")
}

/// The symbol under which the `bf16` tensor [`SpecTensor`](sim_lib_numbers_tensor::SpecTensor)
/// descriptor is exported.
pub fn bf16_tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("bf16")
}
