//! Bounded transforms over caller-supplied Table or Dir block storage.

use sim_kernel::{Cx, Symbol, Table};

use crate::{
    PlacementPolicy, SignalBuffer, SignalError, SignalView, TensorView, TransformKind,
    TransformPlan,
    block_io::{
        BlockCache, decode_complex_block, decode_real_block, encode_complex, encode_real,
        load_block, preflight_blocks, store_error,
    },
    io_plan::{
        TransformPrecision, TransformReport, TransformResources, scratch_bytes,
        transform_plan_digest,
    },
    multidimensional::{axis_plan, for_each_line, validate_nd_plan},
    tensor_view::transform_cell_count,
    transform,
};

/// Public descriptor for a row-major tensor stored as Table/Dir byte blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedTensor {
    namespace: Symbol,
    shape: Vec<usize>,
    precision: TransformPrecision,
    block_len: usize,
}

impl BlockedTensor {
    /// Describes an existing row-major tensor in caller-managed blocks.
    ///
    /// Construction does not read the store. [`transform_nd_blocked`] performs
    /// a complete bounded preflight before its first mutation.
    pub fn new(
        namespace: Symbol,
        shape: Vec<usize>,
        precision: TransformPrecision,
        block_len: usize,
    ) -> Result<Self, SignalError> {
        let _ = transform_cell_count(&shape)?;
        if block_len == 0 {
            return Err(SignalError::InvalidPolicy {
                policy: "block_len",
                reason: "external block length must be nonzero",
            });
        }
        Ok(Self {
            namespace,
            shape,
            precision,
            block_len,
        })
    }

    /// Stable caller-selected namespace used to derive Table block keys.
    pub fn namespace(&self) -> &Symbol {
        &self.namespace
    }

    /// Logical row-major tensor extents.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Stored cell representation.
    pub const fn precision(&self) -> TransformPrecision {
        self.precision
    }

    /// Maximum number of cells encoded in each block.
    pub const fn block_len(&self) -> usize {
        self.block_len
    }

    /// Logical cell count.
    pub fn len(&self) -> usize {
        transform_cell_count(&self.shape).expect("blocked tensor descriptor was validated")
    }

    /// Whether the descriptor contains no cells.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of external blocks in the descriptor.
    pub fn block_count(&self) -> usize {
        self.len().div_ceil(self.block_len)
    }

    /// Stable Table key for `block`, or `None` when it is out of range.
    pub fn block_key(&self, block: usize) -> Option<Symbol> {
        (block < self.block_count()).then(|| self.key(block))
    }

    /// Number of cells expected in `block`, or `None` when it is out of range.
    pub fn cells_in_block(&self, block: usize) -> Option<usize> {
        (block < self.block_count()).then(|| self.block_cell_count(block))
    }

    pub(crate) fn key(&self, block: usize) -> Symbol {
        Symbol::qualified(
            "numbers-signal-block-v1",
            format!("{}:{block:016x}", self.namespace.as_qualified_str()),
        )
    }
}

/// Writes one real-f64 block into an existing external tensor descriptor.
///
/// This incremental surface lets callers populate a tensor larger than memory
/// without ever constructing a complete [`TensorView`].
pub fn write_f64_block(
    cx: &mut Cx,
    store: &dyn Table,
    tensor: &BlockedTensor,
    block: usize,
    values: &[f64],
) -> Result<(), SignalError> {
    if tensor.precision != TransformPrecision::F64 {
        return Err(SignalError::InputKind {
            expected: "complex",
            actual: "real",
        });
    }
    validate_block_cells(tensor, block, values.len())?;
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "value",
            });
        }
        encode_real(&mut bytes, value);
    }
    set_block(cx, store, tensor, block, bytes)
}

/// Writes one complex-f64 block into an existing external tensor descriptor.
///
/// Components are encoded as portable little-endian real/imaginary f64 pairs.
pub fn write_complex_f64_block(
    cx: &mut Cx,
    store: &dyn Table,
    tensor: &BlockedTensor,
    block: usize,
    values: &[(f64, f64)],
) -> Result<(), SignalError> {
    if tensor.precision != TransformPrecision::ComplexF64 {
        return Err(SignalError::InputKind {
            expected: "real",
            actual: "complex",
        });
    }
    validate_block_cells(tensor, block, values.len())?;
    let mut bytes = Vec::with_capacity(values.len() * 2 * size_of::<f64>());
    for (index, &value) in values.iter().enumerate() {
        if !value.0.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "real",
            });
        }
        if !value.1.is_finite() {
            return Err(SignalError::NonFinite {
                index,
                component: "imag",
            });
        }
        encode_complex(&mut bytes, value);
    }
    set_block(cx, store, tensor, block, bytes)
}

/// Writes a borrowed tensor view into caller-supplied Table/Dir blocks.
///
/// Blocks are portable little-endian f64 byte strings. The function retains no
/// backend handle and accepts no filesystem path.
pub fn write_blocked_tensor(
    cx: &mut Cx,
    store: &dyn Table,
    namespace: Symbol,
    input: &TensorView<'_>,
    resources: TransformResources,
) -> Result<BlockedTensor, SignalError> {
    resources.validate()?;
    let precision = input_precision(input);
    let descriptor = BlockedTensor::new(
        namespace,
        input.shape().to_vec(),
        precision,
        resources.block_len,
    )?;
    let block_bytes = resources
        .block_len
        .checked_mul(precision.cell_bytes())
        .ok_or(SignalError::InvalidTensorView {
            reason: "block byte length overflowed",
        })?;
    if block_bytes > resources.max_scratch_bytes {
        return Err(SignalError::ScratchLimit {
            required: block_bytes,
            maximum: resources.max_scratch_bytes,
        });
    }
    validate_input_finite(input)?;
    for block in 0..descriptor.len().div_ceil(descriptor.block_len) {
        let start = block * descriptor.block_len;
        let end = descriptor.len().min(start + descriptor.block_len);
        let mut bytes = Vec::with_capacity((end - start) * precision.cell_bytes());
        for logical in start..end {
            let physical = input.physical_index(logical)?;
            match input {
                TensorView::Complex { values, .. } => encode_complex(&mut bytes, values[physical]),
                TensorView::Real { values, .. } => encode_real(&mut bytes, values[physical]),
            }
        }
        let value = cx
            .factory()
            .bytes(bytes)
            .map_err(|error| store_error("encode", error))?;
        store
            .set(cx, descriptor.key(block), value)
            .map_err(|error| store_error("set", error))?;
    }
    Ok(descriptor)
}

/// Materializes Table/Dir blocks back into a canonical tensor buffer.
pub fn read_blocked_tensor(
    cx: &mut Cx,
    store: &dyn Table,
    tensor: &BlockedTensor,
) -> Result<SignalBuffer, SignalError> {
    match tensor.precision {
        TransformPrecision::ComplexF64 => {
            let mut values = Vec::with_capacity(tensor.len());
            for block in 0..tensor.len().div_ceil(tensor.block_len) {
                let bytes = load_block(cx, store, tensor, block)?;
                values.extend(decode_complex_block(
                    block,
                    &bytes,
                    tensor.block_cell_count(block),
                )?);
            }
            sim_lib_numbers_tensor_cmplxf::ComplexFTensor::new(tensor.shape.clone(), values)
                .map(SignalBuffer::Complex)
                .ok_or(SignalError::InvalidTensorView {
                    reason: "blocked complex tensor shape overflowed",
                })
        }
        TransformPrecision::F64 => {
            let mut values = Vec::with_capacity(tensor.len());
            for block in 0..tensor.len().div_ceil(tensor.block_len) {
                let bytes = load_block(cx, store, tensor, block)?;
                values.extend(decode_real_block(
                    block,
                    &bytes,
                    tensor.block_cell_count(block),
                )?);
            }
            sim_lib_numbers_tensor_f64::F64Tensor::new(tensor.shape.clone(), values)
                .map(SignalBuffer::Real)
                .ok_or(SignalError::InvalidTensorView {
                    reason: "blocked real tensor shape overflowed",
                })
        }
    }
}

/// Applies a separable transform in place over caller-owned Table/Dir blocks.
///
/// One transform line and one external block are resident at a time. The
/// tensor may therefore exceed memory as long as its longest selected axis and
/// one block fit [`TransformResources::max_scratch_bytes`].
pub fn transform_nd_blocked(
    cx: &mut Cx,
    store: &dyn Table,
    tensor: &BlockedTensor,
    axes: &[usize],
    plan: &TransformPlan,
    resources: TransformResources,
) -> Result<TransformReport, SignalError> {
    resources.validate()?;
    if tensor.block_len != resources.block_len {
        return Err(SignalError::InvalidPolicy {
            policy: "block_len",
            reason: "resource block length differs from stored tensor descriptor",
        });
    }
    validate_nd_plan(&tensor.shape, axes, plan, PlacementPolicy::InPlace)?;
    validate_kind_precision(plan.kind, tensor.precision)?;
    let required = scratch_bytes(&tensor.shape, axes, tensor.precision, tensor.block_len)?;
    if required > resources.max_scratch_bytes {
        return Err(SignalError::ScratchLimit {
            required,
            maximum: resources.max_scratch_bytes,
        });
    }

    let preflight_io = preflight_blocks(cx, store, tensor)?;
    let mut cache = BlockCache::new(cx, store, tensor, preflight_io);
    for &axis in axes {
        for_each_line(&tensor.shape, axis, |start, stride, len| {
            match tensor.precision {
                TransformPrecision::ComplexF64 => {
                    let mut line = Vec::with_capacity(len);
                    for index in 0..len {
                        line.push(cache.read_complex(start + index * stride)?);
                    }
                    let axis_plan = axis_plan(plan, len, PlacementPolicy::OutOfPlace)?;
                    let SignalBuffer::Complex(output) =
                        transform(&axis_plan, SignalView::Complex(&line))?
                    else {
                        return Err(SignalError::InputKind {
                            expected: "complex",
                            actual: "real",
                        });
                    };
                    for (index, value) in output.as_slice().iter().copied().enumerate() {
                        cache.write_complex(start + index * stride, value)?;
                    }
                }
                TransformPrecision::F64 => {
                    let mut line = Vec::with_capacity(len);
                    for index in 0..len {
                        line.push(cache.read_real(start + index * stride)?);
                    }
                    let axis_plan = axis_plan(plan, len, PlacementPolicy::OutOfPlace)?;
                    let SignalBuffer::Real(output) =
                        transform(&axis_plan, SignalView::Real(&line))?
                    else {
                        return Err(SignalError::InputKind {
                            expected: "real",
                            actual: "complex",
                        });
                    };
                    for (index, value) in output.as_slice().iter().copied().enumerate() {
                        cache.write_real(start + index * stride, value)?;
                    }
                }
            }
            Ok(())
        })?;
    }
    cache.finish()?;
    Ok(TransformReport {
        scratch_bytes: required,
        passes: axes.len(),
        io_blocks: cache.io_blocks(),
        precision: tensor.precision,
        plan_digest: transform_plan_digest(
            &tensor.shape,
            axes,
            plan,
            tensor.precision,
            Some(resources),
        ),
    })
}

impl BlockedTensor {
    pub(crate) fn block_cell_count(&self, block: usize) -> usize {
        let start = block * self.block_len;
        self.len().saturating_sub(start).min(self.block_len)
    }
}

fn validate_block_cells(
    tensor: &BlockedTensor,
    block: usize,
    actual: usize,
) -> Result<(), SignalError> {
    let Some(expected) = tensor.cells_in_block(block) else {
        return Err(SignalError::BlockStore {
            operation: "set",
            message: format!("block {block} is outside descriptor"),
        });
    };
    if actual != expected {
        return Err(SignalError::BlockStore {
            operation: "set",
            message: format!("block {block} has {actual} cells, expected {expected}"),
        });
    }
    Ok(())
}

fn set_block(
    cx: &mut Cx,
    store: &dyn Table,
    tensor: &BlockedTensor,
    block: usize,
    bytes: Vec<u8>,
) -> Result<(), SignalError> {
    let value = cx
        .factory()
        .bytes(bytes)
        .map_err(|error| store_error("encode", error))?;
    store
        .set(cx, tensor.key(block), value)
        .map_err(|error| store_error("set", error))
}

fn input_precision(input: &TensorView<'_>) -> TransformPrecision {
    match input {
        TensorView::Complex { .. } => TransformPrecision::ComplexF64,
        TensorView::Real { .. } => TransformPrecision::F64,
    }
}

fn validate_input_finite(input: &TensorView<'_>) -> Result<(), SignalError> {
    for logical in 0..input.len() {
        let physical = input.physical_index(logical)?;
        match input {
            TensorView::Complex { values, .. } => {
                let (real, imag) = values[physical];
                if !real.is_finite() {
                    return Err(SignalError::NonFinite {
                        index: logical,
                        component: "real",
                    });
                }
                if !imag.is_finite() {
                    return Err(SignalError::NonFinite {
                        index: logical,
                        component: "imag",
                    });
                }
            }
            TensorView::Real { values, .. } if !values[physical].is_finite() => {
                return Err(SignalError::NonFinite {
                    index: logical,
                    component: "value",
                });
            }
            TensorView::Real { .. } => {}
        }
    }
    Ok(())
}

fn validate_kind_precision(
    kind: TransformKind,
    precision: TransformPrecision,
) -> Result<(), SignalError> {
    match (kind, precision) {
        (TransformKind::Dft | TransformKind::Fft, TransformPrecision::ComplexF64)
        | (TransformKind::Dct(_) | TransformKind::Dst(_), TransformPrecision::F64) => Ok(()),
        (TransformKind::RealFft, _) => Err(SignalError::InvalidPolicy {
            policy: "kind",
            reason: "blocked real FFT requires an explicit shape-changing packing plan",
        }),
        (TransformKind::Dft | TransformKind::Fft, TransformPrecision::F64) => {
            Err(SignalError::InputKind {
                expected: "complex",
                actual: "real",
            })
        }
        (TransformKind::Dct(_) | TransformKind::Dst(_), TransformPrecision::ComplexF64) => {
            Err(SignalError::InputKind {
                expected: "real",
                actual: "complex",
            })
        }
    }
}
