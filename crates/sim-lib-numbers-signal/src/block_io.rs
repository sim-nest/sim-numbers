//! Portable Table block encoding and one-block execution cache.

use sim_kernel::{Cx, Expr, Table};

use crate::{BlockedTensor, SignalError, TransformPrecision};

pub(crate) struct BlockCache<'a> {
    cx: &'a mut Cx,
    store: &'a dyn Table,
    tensor: &'a BlockedTensor,
    current: Option<(usize, Vec<u8>)>,
    dirty: bool,
    io_blocks: usize,
}

impl<'a> BlockCache<'a> {
    pub(crate) fn new(
        cx: &'a mut Cx,
        store: &'a dyn Table,
        tensor: &'a BlockedTensor,
        io_blocks: usize,
    ) -> Self {
        Self {
            cx,
            store,
            tensor,
            current: None,
            dirty: false,
            io_blocks,
        }
    }

    pub(crate) const fn io_blocks(&self) -> usize {
        self.io_blocks
    }

    fn load(&mut self, block: usize) -> Result<&mut Vec<u8>, SignalError> {
        if self.current.as_ref().map(|(index, _)| *index) != Some(block) {
            self.flush()?;
            let bytes = load_block(self.cx, self.store, self.tensor, block)?;
            self.io_blocks += 1;
            self.current = Some((block, bytes));
        }
        Ok(&mut self.current.as_mut().expect("cache just loaded").1)
    }

    fn flush(&mut self) -> Result<(), SignalError> {
        if !self.dirty {
            return Ok(());
        }
        let (block, bytes) = self.current.take().expect("dirty cache has a block");
        let value = self
            .cx
            .factory()
            .bytes(bytes)
            .map_err(|error| store_error("encode", error))?;
        self.store
            .set(self.cx, self.tensor.key(block), value)
            .map_err(|error| store_error("set", error))?;
        self.io_blocks += 1;
        self.dirty = false;
        Ok(())
    }

    pub(crate) fn read_complex(&mut self, flat: usize) -> Result<(f64, f64), SignalError> {
        let block = flat / self.tensor.block_len();
        let local = flat % self.tensor.block_len();
        let bytes = self.load(block)?;
        decode_complex(bytes, local, block)
    }

    pub(crate) fn write_complex(
        &mut self,
        flat: usize,
        value: (f64, f64),
    ) -> Result<(), SignalError> {
        let block = flat / self.tensor.block_len();
        let local = flat % self.tensor.block_len();
        let bytes = self.load(block)?;
        let start = local * (2 * size_of::<f64>());
        bytes[start..start + 8].copy_from_slice(&value.0.to_le_bytes());
        bytes[start + 8..start + 16].copy_from_slice(&value.1.to_le_bytes());
        self.dirty = true;
        Ok(())
    }

    pub(crate) fn read_real(&mut self, flat: usize) -> Result<f64, SignalError> {
        let block = flat / self.tensor.block_len();
        let local = flat % self.tensor.block_len();
        let bytes = self.load(block)?;
        decode_real(bytes, local, block)
    }

    pub(crate) fn write_real(&mut self, flat: usize, value: f64) -> Result<(), SignalError> {
        let block = flat / self.tensor.block_len();
        let local = flat % self.tensor.block_len();
        let bytes = self.load(block)?;
        let start = local * size_of::<f64>();
        bytes[start..start + 8].copy_from_slice(&value.to_le_bytes());
        self.dirty = true;
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<(), SignalError> {
        self.flush()
    }
}

pub(crate) fn preflight_blocks(
    cx: &mut Cx,
    store: &dyn Table,
    tensor: &BlockedTensor,
) -> Result<usize, SignalError> {
    let blocks = tensor.len().div_ceil(tensor.block_len());
    for block in 0..blocks {
        let bytes = load_block(cx, store, tensor, block)?;
        match tensor.precision() {
            TransformPrecision::ComplexF64 => {
                for local in 0..tensor.block_cell_count(block) {
                    let _ = decode_complex(&bytes, local, block)?;
                }
            }
            TransformPrecision::F64 => {
                for local in 0..tensor.block_cell_count(block) {
                    let _ = decode_real(&bytes, local, block)?;
                }
            }
        }
    }
    Ok(blocks)
}

pub(crate) fn load_block(
    cx: &mut Cx,
    store: &dyn Table,
    tensor: &BlockedTensor,
    block: usize,
) -> Result<Vec<u8>, SignalError> {
    let key = tensor.key(block);
    if !store
        .has(cx, key.clone())
        .map_err(|error| store_error("has", error))?
    {
        return Err(SignalError::BlockStore {
            operation: "get",
            message: format!("missing block {block}"),
        });
    }
    let value = store
        .get(cx, key)
        .map_err(|error| store_error("get", error))?;
    let Expr::Bytes(bytes) = value
        .object()
        .as_expr(cx)
        .map_err(|error| store_error("decode", error))?
    else {
        return Err(SignalError::BlockStore {
            operation: "decode",
            message: format!("block {block} is not bytes"),
        });
    };
    let expected = tensor.block_cell_count(block) * tensor.precision().cell_bytes();
    if bytes.len() != expected {
        return Err(SignalError::BlockStore {
            operation: "decode",
            message: format!(
                "block {block} has {} bytes, expected {expected}",
                bytes.len()
            ),
        });
    }
    Ok(bytes)
}

pub(crate) fn encode_complex(bytes: &mut Vec<u8>, value: (f64, f64)) {
    bytes.extend_from_slice(&value.0.to_le_bytes());
    bytes.extend_from_slice(&value.1.to_le_bytes());
}

pub(crate) fn encode_real(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn decode_complex_block(
    block: usize,
    bytes: &[u8],
    cells: usize,
) -> Result<Vec<(f64, f64)>, SignalError> {
    (0..cells)
        .map(|local| decode_complex(bytes, local, block))
        .collect()
}

pub(crate) fn decode_real_block(
    block: usize,
    bytes: &[u8],
    cells: usize,
) -> Result<Vec<f64>, SignalError> {
    (0..cells)
        .map(|local| decode_real(bytes, local, block))
        .collect()
}

fn decode_complex(bytes: &[u8], local: usize, block: usize) -> Result<(f64, f64), SignalError> {
    let start = local * (2 * size_of::<f64>());
    let real = decode_f64(&bytes[start..start + 8], block)?;
    let imag = decode_f64(&bytes[start + 8..start + 16], block)?;
    Ok((real, imag))
}

fn decode_real(bytes: &[u8], local: usize, block: usize) -> Result<f64, SignalError> {
    let start = local * size_of::<f64>();
    decode_f64(&bytes[start..start + 8], block)
}

fn decode_f64(bytes: &[u8], block: usize) -> Result<f64, SignalError> {
    let value = f64::from_le_bytes(bytes.try_into().expect("f64 cell has eight bytes"));
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SignalError::BlockStore {
            operation: "decode",
            message: format!("block {block} contains a non-finite component"),
        })
    }
}

pub(crate) fn store_error(operation: &'static str, error: impl std::fmt::Display) -> SignalError {
    SignalError::BlockStore {
        operation,
        message: error.to_string(),
    }
}
