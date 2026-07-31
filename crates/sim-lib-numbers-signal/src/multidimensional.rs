//! Separable transforms over borrowed multidimensional tensor layouts.

use sim_lib_numbers_tensor_cmplxf::ComplexFTensor;
use sim_lib_numbers_tensor_f64::F64Tensor;

use crate::{
    LengthPolicy, PaddingPolicy, PlacementPolicy, SignalBuffer, SignalError, SignalView,
    SpectrumPacking, Stride, TransformKind, TransformPlan, TransformPrecision, TransformReport,
    io_plan::{scratch_bytes, transform_plan_digest},
    tensor_view::{TensorView, contiguous_strides, transform_cell_count},
    transform,
};

/// Result of an in-memory multidimensional transform.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorTransform {
    /// Canonical tensor buffer preserving the input's logical shape.
    pub output: SignalBuffer,
    /// Auditable execution resource and provenance facts.
    pub report: TransformReport,
}

/// Applies a separable transform over `axes` in declaration order.
///
/// Each axis extent supplies the one-dimensional plan length, which permits
/// rectangular tensors. Tensor layout strides select the borrowed input;
/// [`TransformPlan::stride`] must therefore remain contiguous. Complex DFT/FFT
/// and real DCT/DST keep the tensor shape. Real FFT is rejected because its
/// representation and packed axis extent can change.
pub fn transform_nd(
    input: TensorView<'_>,
    axes: &[usize],
    plan: &TransformPlan,
) -> Result<TensorTransform, SignalError> {
    validate_nd_plan(input.shape(), axes, plan, PlacementPolicy::OutOfPlace)?;
    let shape = input.shape().to_vec();
    let precision = match plan.kind {
        TransformKind::Dft | TransformKind::Fft => TransformPrecision::ComplexF64,
        TransformKind::Dct(_) | TransformKind::Dst(_) => TransformPrecision::F64,
        TransformKind::RealFft => TransformPrecision::ComplexF64,
    };
    let report = TransformReport {
        scratch_bytes: scratch_bytes(&shape, axes, precision, 0)?,
        passes: axes.len(),
        io_blocks: 0,
        precision,
        plan_digest: transform_plan_digest(&shape, axes, plan, precision, None),
    };
    let output = match plan.kind {
        TransformKind::Dft | TransformKind::Fft => {
            let mut values = collect_complex(&input)?;
            apply_complex_axes(&mut values, &shape, axes, plan)?;
            SignalBuffer::Complex(ComplexFTensor::new(shape, values).ok_or(
                SignalError::InvalidTensorView {
                    reason: "complex output shape overflowed",
                },
            )?)
        }
        TransformKind::Dct(_) | TransformKind::Dst(_) => {
            let mut values = collect_real(&input)?;
            apply_real_axes(&mut values, &shape, axes, plan)?;
            SignalBuffer::Real(F64Tensor::new(shape, values).ok_or(
                SignalError::InvalidTensorView {
                    reason: "real output shape overflowed",
                },
            )?)
        }
        TransformKind::RealFft => {
            return Err(SignalError::InvalidPolicy {
                policy: "kind",
                reason: "multidimensional real FFT requires an explicit shape-changing packing plan",
            });
        }
    };
    Ok(TensorTransform { output, report })
}

pub(crate) fn validate_nd_plan(
    shape: &[usize],
    axes: &[usize],
    plan: &TransformPlan,
    placement: PlacementPolicy,
) -> Result<(), SignalError> {
    if axes.is_empty() {
        return Err(SignalError::InvalidPolicy {
            policy: "axes",
            reason: "at least one transform axis is required",
        });
    }
    if plan.placement != placement {
        return Err(SignalError::InvalidPolicy {
            policy: "placement",
            reason: match placement {
                PlacementPolicy::OutOfPlace => {
                    "in-memory multidimensional transforms require OutOfPlace"
                }
                PlacementPolicy::InPlace => "blocked multidimensional transforms require InPlace",
            },
        });
    }
    let rank = shape.len();
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(SignalError::AxisOutOfBounds { axis, rank });
        }
        if seen[axis] {
            return Err(SignalError::DuplicateAxis { axis });
        }
        seen[axis] = true;
        let mut axis_plan = axis_plan(plan, shape[axis], placement)?;
        axis_plan.stride = Stride::contiguous();
        axis_plan.validate()?;
    }
    if plan.stride != Stride::contiguous() {
        return Err(SignalError::InvalidPolicy {
            policy: "stride",
            reason: "tensor layout owns multidimensional strides",
        });
    }
    if plan.length != LengthPolicy::Exact || plan.padding != PaddingPolicy::Reject {
        return Err(SignalError::InvalidPolicy {
            policy: "length",
            reason: "multidimensional axes have exact extents and cannot pad or truncate",
        });
    }
    if plan.packing != SpectrumPacking::Full {
        return Err(SignalError::InvalidPolicy {
            policy: "packing",
            reason: "multidimensional transforms preserve every declared axis extent",
        });
    }
    Ok(())
}

pub(crate) fn axis_plan(
    plan: &TransformPlan,
    len: usize,
    placement: PlacementPolicy,
) -> Result<TransformPlan, SignalError> {
    let mut axis_plan = plan.clone();
    axis_plan.len = len;
    axis_plan.placement = placement;
    axis_plan.stride = Stride::contiguous();
    axis_plan.validate()?;
    Ok(axis_plan)
}

fn collect_complex(input: &TensorView<'_>) -> Result<Vec<(f64, f64)>, SignalError> {
    let mut output = Vec::with_capacity(input.len());
    if let Some(values) = input.complex_cells() {
        for logical in 0..input.len() {
            output.push(values[input.physical_index(logical)?]);
        }
    } else if let Some(values) = input.real_cells() {
        for logical in 0..input.len() {
            output.push((values[input.physical_index(logical)?], 0.0));
        }
    }
    Ok(output)
}

fn collect_real(input: &TensorView<'_>) -> Result<Vec<f64>, SignalError> {
    let values = input.real_cells().ok_or(SignalError::InputKind {
        expected: "real",
        actual: "complex",
    })?;
    (0..input.len())
        .map(|logical| Ok(values[input.physical_index(logical)?]))
        .collect()
}

fn apply_complex_axes(
    values: &mut [(f64, f64)],
    shape: &[usize],
    axes: &[usize],
    plan: &TransformPlan,
) -> Result<(), SignalError> {
    for &axis in axes {
        for_each_line(shape, axis, |start, stride, len| {
            let axis_plan = axis_plan(plan, len, PlacementPolicy::OutOfPlace)?;
            let line = (0..len)
                .map(|index| values[start + index * stride])
                .collect::<Vec<_>>();
            let SignalBuffer::Complex(output) = transform(&axis_plan, SignalView::Complex(&line))?
            else {
                return Err(SignalError::InputKind {
                    expected: "complex",
                    actual: "real",
                });
            };
            for (index, value) in output.as_slice().iter().copied().enumerate() {
                values[start + index * stride] = value;
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn apply_real_axes(
    values: &mut [f64],
    shape: &[usize],
    axes: &[usize],
    plan: &TransformPlan,
) -> Result<(), SignalError> {
    for &axis in axes {
        for_each_line(shape, axis, |start, stride, len| {
            let axis_plan = axis_plan(plan, len, PlacementPolicy::OutOfPlace)?;
            let line = (0..len)
                .map(|index| values[start + index * stride])
                .collect::<Vec<_>>();
            let SignalBuffer::Real(output) = transform(&axis_plan, SignalView::Real(&line))? else {
                return Err(SignalError::InputKind {
                    expected: "real",
                    actual: "complex",
                });
            };
            for (index, value) in output.as_slice().iter().copied().enumerate() {
                values[start + index * stride] = value;
            }
            Ok(())
        })?;
    }
    Ok(())
}

pub(crate) fn for_each_line(
    shape: &[usize],
    axis: usize,
    mut apply: impl FnMut(usize, usize, usize) -> Result<(), SignalError>,
) -> Result<(), SignalError> {
    let strides = contiguous_strides(shape)?;
    let axis_len = shape[axis];
    let line_count = transform_cell_count(shape)? / axis_len;
    for line in 0..line_count {
        let mut remainder = line;
        let mut start = 0usize;
        for dimension in (0..shape.len()).rev() {
            if dimension == axis {
                continue;
            }
            let coordinate = remainder % shape[dimension];
            remainder /= shape[dimension];
            start = start
                .checked_add(coordinate.checked_mul(strides[dimension]).ok_or(
                    SignalError::InvalidTensorView {
                        reason: "line offset overflowed",
                    },
                )?)
                .ok_or(SignalError::InvalidTensorView {
                    reason: "line offset overflowed",
                })?;
        }
        apply(start, strides[axis], axis_len)?;
    }
    Ok(())
}
