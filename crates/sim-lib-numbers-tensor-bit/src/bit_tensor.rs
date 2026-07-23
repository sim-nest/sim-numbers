//! Bit-packed boolean tensor storage, its `SpecTensor` backend, and the
//! library that registers it as the `bool` element-type backend.

use std::{any::Any, fmt, sync::Arc};

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Factory, Lib, LibManifest, LibTarget, Linker,
    Result, Symbol, Value, Version,
};
use sim_lib_numbers_tensor::{
    SpecTensor, SpecTensorDescriptor, Tensor, TensorLocation, TensorStorage, checked_element_count,
    domains, spec_tensor_descriptor_value, spec_tensor_symbol,
};

/// A boolean tensor stored as bit-packed `u64` words.
///
/// Each element occupies a single bit, so an `n`-element tensor is held in
/// `ceil(n / 64)` words. The logical [`shape`](Self::shape) and element count
/// drive layout; bitwise operations work directly on the packed words.
#[derive(Clone)]
pub struct BitTensor {
    tensor: Tensor,
}

struct BitTensorStorage {
    len: usize,
    words: Arc<[u64]>,
}

impl BitTensor {
    /// Packs a slice of booleans into a bit tensor of the given shape.
    ///
    /// Returns `None` when `bits.len()` does not match the element count
    /// implied by `shape`.
    pub fn from_bools(shape: Vec<usize>, bits: &[bool]) -> Option<Self> {
        let len = checked_element_count(&shape).ok()?;
        if len != bits.len() {
            return None;
        }
        let mut words = vec![0u64; len.div_ceil(64)];
        for (index, bit) in bits.iter().enumerate() {
            if *bit {
                words[index / 64] |= 1u64 << (index % 64);
            }
        }
        let storage = Arc::new(BitTensorStorage {
            len,
            words: words.into(),
        });
        Tensor::from_storage(shape, domains::bool(), storage)
            .ok()
            .map(|tensor| Self { tensor })
    }

    /// Unpacks the tensor back into one boolean per element, in flat order.
    pub fn to_bools(&self) -> Vec<bool> {
        self.storage().to_bools()
    }

    /// Element-wise bitwise OR with another bit tensor of the same shape.
    ///
    /// Returns `None` when the shapes differ.
    pub fn bit_or(&self, other: &Self) -> Option<Self> {
        map_words(self, other, |left, right| left | right)
    }

    /// Element-wise bitwise XOR with another bit tensor of the same shape.
    ///
    /// Returns `None` when the shapes differ.
    pub fn bit_xor(&self, other: &Self) -> Option<Self> {
        map_words(self, other, |left, right| left ^ right)
    }

    /// Element-wise bitwise AND with another bit tensor of the same shape.
    ///
    /// Returns `None` when the shapes differ.
    pub fn bit_and(&self, other: &Self) -> Option<Self> {
        map_words(self, other, |left, right| left & right)
    }

    fn storage(&self) -> &BitTensorStorage {
        self.tensor
            .storage()
            .as_any()
            .downcast_ref::<BitTensorStorage>()
            .expect("BitTensor must hold bit-packed storage")
    }
}

impl BitTensorStorage {
    fn to_bools(&self) -> Vec<bool> {
        (0..self.len)
            .map(|index| ((self.words[index / 64] >> (index % 64)) & 1) == 1)
            .collect()
    }
}

impl TensorStorage for BitTensorStorage {
    fn dtype(&self) -> &Symbol {
        static DTYPE: std::sync::OnceLock<Symbol> = std::sync::OnceLock::new();
        DTYPE.get_or_init(domains::bool)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Host
    }

    fn cell(&self, index: usize) -> Result<Value> {
        if index >= self.len {
            return Err(sim_kernel::Error::Eval(
                "tensor cell index was out of bounds".to_owned(),
            ));
        }
        let bit = ((self.words[index / 64] >> (index % 64)) & 1) == 1;
        bool_value(bit)
            .ok_or_else(|| sim_kernel::Error::Eval("bool tensor cell encode failed".to_owned()))
    }

    fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        Ok(Arc::new(Self {
            len: self.len,
            words: self.words.clone(),
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Debug for BitTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitTensor")
            .field("shape", &self.shape())
            .field("bits", &self.to_bools())
            .finish()
    }
}

impl PartialEq for BitTensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.to_bools() == other.to_bools()
    }
}

impl Eq for BitTensor {}

impl SpecTensor for BitTensor {
    fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    fn dtype(&self) -> Symbol {
        domains::bool()
    }

    fn to_uniform(&self) -> Tensor {
        self.tensor.clone()
    }

    fn from_uniform(tensor: &Tensor) -> Option<Self> {
        (tensor.dtype() == &domains::bool()).then_some(())?;
        if tensor.storage().as_any().is::<BitTensorStorage>() {
            return Some(Self {
                tensor: tensor.clone(),
            });
        }
        let bits = tensor
            .cells()
            .ok()?
            .iter()
            .map(parse_bool_cell)
            .collect::<Option<Vec<_>>>()?;
        Self::from_bools(tensor.shape().to_vec(), &bits)
    }
}

/// Registered library that installs the bit-packed boolean tensor backend.
///
/// Loading this [`Lib`] registers a [`SpecTensor`] descriptor binding the
/// `bool` element type to the [`BitTensor`] storage, so the base tensor domain
/// can construct and round-trip boolean tensors through packed `u64` words.
pub struct BitTensorLib;

impl BitTensorLib {
    /// Creates the bit-tensor library. The value is stateless; the spec-tensor
    /// descriptor is installed when it is loaded into a
    /// [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for BitTensorLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for BitTensorLib {
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
                    dtype: domains::bool(),
                    implementation: "BitTensor",
                    storage: "canonical Tensor storage over bit-packed u64 words",
                },
            )?,
        )
    }
}

/// The manifest id symbol for this library (`numbers/tensor-bit`).
pub fn tensor_lib_symbol() -> Symbol {
    domains::domain("tensor-bit")
}

/// The symbol under which the bit-tensor [`SpecTensor`] descriptor is exported.
pub fn tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("bit")
}

fn bool_value(value: bool) -> Option<Value> {
    DefaultFactory
        .number_literal(domains::bool(), value.to_string())
        .ok()
}

fn parse_bool_cell(value: &Value) -> Option<bool> {
    let mut cx = sim_kernel::Cx::new(
        std::sync::Arc::new(sim_kernel::NoopEvalPolicy),
        std::sync::Arc::new(DefaultFactory),
    );
    let literal = value
        .object()
        .as_number_value()?
        .number_literal(&mut cx)
        .ok()??;
    (literal.domain == domains::bool())
        .then(|| literal.canonical.parse::<bool>().ok())
        .flatten()
}

fn map_words(
    left: &BitTensor,
    right: &BitTensor,
    f: impl Fn(u64, u64) -> u64,
) -> Option<BitTensor> {
    if left.shape() != right.shape() {
        return None;
    }
    let storage = Arc::new(BitTensorStorage {
        len: left.storage().len,
        words: left
            .storage()
            .words
            .iter()
            .zip(right.storage().words.iter())
            .map(|(left, right)| f(*left, *right))
            .collect::<Vec<_>>()
            .into(),
    });
    Tensor::from_storage(left.shape().to_vec(), domains::bool(), storage)
        .ok()
        .map(|tensor| BitTensor { tensor })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sim_kernel::Lib;

    use super::{BitTensor, BitTensorLib, SpecTensor, tensor_spec_symbol};

    #[test]
    fn bit_tensor_and_matches_bool_and() {
        let left = BitTensor::from_bools(vec![4], &[true, false, true, true]).unwrap();
        let right = BitTensor::from_bools(vec![4], &[true, true, false, true]).unwrap();
        let out = left.bit_and(&right).unwrap();
        assert_eq!(out.to_bools(), vec![true, false, false, true]);
        let uniform = out.to_uniform();
        assert_eq!(uniform.shape(), &[4]);
    }

    #[test]
    fn uniform_roundtrip_preserves_bit_storage_identity() {
        let tensor = BitTensor::from_bools(vec![4], &[true, false, true, true]).unwrap();
        let uniform = tensor.to_uniform();
        assert!(Arc::ptr_eq(tensor.tensor.storage(), uniform.storage()));

        let roundtrip = BitTensor::from_uniform(&uniform).unwrap();
        assert!(Arc::ptr_eq(roundtrip.tensor.storage(), uniform.storage()));
        assert_eq!(roundtrip.to_bools(), vec![true, false, true, true]);
    }

    #[test]
    fn constructor_rejects_overflowing_shape() {
        assert!(BitTensor::from_bools(vec![usize::MAX, 2], &[]).is_none());
    }

    #[test]
    fn lib_exports_spec_tensor_descriptor() {
        assert_eq!(
            BitTensorLib::new().manifest().exports[0].symbol(),
            &tensor_spec_symbol()
        );
    }
}
