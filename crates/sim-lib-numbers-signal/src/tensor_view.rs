//! Borrowed multidimensional tensor layouts.

use sim_lib_numbers_tensor::{SpecTensor, checked_element_count};
use sim_lib_numbers_tensor_cmplxf::ComplexFTensor;
use sim_lib_numbers_tensor_f64::F64Tensor;

use crate::SignalError;

/// Borrowed canonical real or complex tensor cells with an explicit layout.
///
/// A view owns only its small shape/stride descriptor. Cell storage remains
/// borrowed, so sliced, padded, and transposed layouts can be transformed
/// without first materializing a private input array.
#[derive(Clone, Debug)]
pub enum TensorView<'a> {
    /// Complex f64-pair cells.
    Complex {
        /// Borrowed physical storage.
        values: &'a [(f64, f64)],
        /// Logical tensor extents.
        shape: Vec<usize>,
        /// Physical cell stride for each logical axis.
        strides: Vec<usize>,
        /// Physical offset of the logical all-zero coordinate.
        offset: usize,
    },
    /// Real f64 cells.
    Real {
        /// Borrowed physical storage.
        values: &'a [f64],
        /// Logical tensor extents.
        shape: Vec<usize>,
        /// Physical cell stride for each logical axis.
        strides: Vec<usize>,
        /// Physical offset of the logical all-zero coordinate.
        offset: usize,
    },
}

impl<'a> TensorView<'a> {
    /// Builds a complex view after validating bounds and non-overlap.
    pub fn complex(
        values: &'a [(f64, f64)],
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, SignalError> {
        validate_layout(values.len(), &shape, &strides, offset)?;
        Ok(Self::Complex {
            values,
            shape,
            strides,
            offset,
        })
    }

    /// Builds a real view after validating bounds and non-overlap.
    pub fn real(
        values: &'a [f64],
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, SignalError> {
        validate_layout(values.len(), &shape, &strides, offset)?;
        Ok(Self::Real {
            values,
            shape,
            strides,
            offset,
        })
    }

    /// Borrows a canonical contiguous complex tensor.
    pub fn from_complex_tensor(tensor: &'a ComplexFTensor) -> Self {
        let shape = tensor.shape().to_vec();
        let strides = contiguous_strides(&shape)
            .expect("a constructed canonical tensor has a representable layout");
        Self::Complex {
            values: tensor.as_slice(),
            shape,
            strides,
            offset: 0,
        }
    }

    /// Borrows a canonical contiguous real tensor.
    pub fn from_real_tensor(tensor: &'a F64Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        let strides = contiguous_strides(&shape)
            .expect("a constructed canonical tensor has a representable layout");
        Self::Real {
            values: tensor.as_slice(),
            shape,
            strides,
            offset: 0,
        }
    }

    /// Logical tensor extents.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Complex { shape, .. } | Self::Real { shape, .. } => shape,
        }
    }

    /// Physical cell stride for each logical axis.
    pub fn strides(&self) -> &[usize] {
        match self {
            Self::Complex { strides, .. } | Self::Real { strides, .. } => strides,
        }
    }

    /// Number of logical cells.
    pub fn len(&self) -> usize {
        transform_cell_count(self.shape())
            .expect("validated tensor view has a representable cell count")
    }

    /// Whether the view contains no logical cells.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn complex_cells(&self) -> Option<&[(f64, f64)]> {
        match self {
            Self::Complex { values, .. } => Some(values),
            Self::Real { .. } => None,
        }
    }

    pub(crate) fn real_cells(&self) -> Option<&[f64]> {
        match self {
            Self::Real { values, .. } => Some(values),
            Self::Complex { .. } => None,
        }
    }

    pub(crate) fn physical_index(&self, logical_flat: usize) -> Result<usize, SignalError> {
        let (strides, offset) = match self {
            Self::Complex {
                strides, offset, ..
            }
            | Self::Real {
                strides, offset, ..
            } => (strides, *offset),
        };
        physical_index(logical_flat, self.shape(), strides, offset)
    }
}

pub(crate) fn transform_cell_count(shape: &[usize]) -> Result<usize, SignalError> {
    if shape.is_empty() {
        return Err(SignalError::InvalidTensorView {
            reason: "rank-zero tensors have no transform axis",
        });
    }
    if shape.contains(&0) {
        return Err(SignalError::InvalidTensorView {
            reason: "transform tensor extents must be nonzero",
        });
    }
    checked_element_count(shape).map_err(|_| SignalError::InvalidTensorView {
        reason: "tensor element count overflowed",
    })
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, SignalError> {
    let _ = transform_cell_count(shape)?;
    let mut strides = vec![1usize; shape.len()];
    for dimension in (0..shape.len().saturating_sub(1)).rev() {
        strides[dimension] = strides[dimension + 1]
            .checked_mul(shape[dimension + 1])
            .ok_or(SignalError::InvalidTensorView {
                reason: "contiguous tensor stride overflowed",
            })?;
    }
    Ok(strides)
}

fn validate_layout(
    physical_len: usize,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<(), SignalError> {
    let _ = transform_cell_count(shape)?;
    if shape.len() != strides.len() {
        return Err(SignalError::InvalidTensorView {
            reason: "shape and stride ranks differ",
        });
    }
    if strides.contains(&0) {
        return Err(SignalError::InvalidTensorView {
            reason: "tensor strides must be nonzero",
        });
    }
    let mut dimensions = shape
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|(extent, _)| *extent > 1)
        .collect::<Vec<_>>();
    dimensions.sort_by_key(|(_, stride)| *stride);
    let mut occupied_span = 1usize;
    for (extent, stride) in dimensions {
        if stride < occupied_span {
            return Err(SignalError::InvalidTensorView {
                reason: "tensor layout aliases logical cells",
            });
        }
        occupied_span = stride
            .checked_mul(extent - 1)
            .and_then(|delta| occupied_span.checked_add(delta))
            .ok_or(SignalError::InvalidTensorView {
                reason: "tensor layout span overflowed",
            })?;
    }
    let last = offset
        .checked_add(occupied_span - 1)
        .ok_or(SignalError::InvalidTensorView {
            reason: "tensor layout bound overflowed",
        })?;
    if last >= physical_len {
        return Err(SignalError::InvalidTensorView {
            reason: "tensor layout exceeds borrowed storage",
        });
    }
    Ok(())
}

fn physical_index(
    logical_flat: usize,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<usize, SignalError> {
    let mut remainder = logical_flat;
    let mut physical = offset;
    for dimension in (0..shape.len()).rev() {
        let coordinate = remainder % shape[dimension];
        remainder /= shape[dimension];
        physical = physical
            .checked_add(coordinate.checked_mul(strides[dimension]).ok_or(
                SignalError::InvalidTensorView {
                    reason: "tensor view index overflowed",
                },
            )?)
            .ok_or(SignalError::InvalidTensorView {
                reason: "tensor view index overflowed",
            })?;
    }
    Ok(physical)
}
