//! Ranked regular-grid interpolation by repeated use of the one-dimensional owner.

use crate::{
    DuplicateXPolicy, InterpolationMethod, InterpolationPlan, SampleInterpolator, SignalError,
};

/// Ranked-grid interpolation policy with explicit resource ceilings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridInterpolationPlan {
    /// Order in which tensor axes are contracted; every axis must appear once.
    pub axis_order: Vec<usize>,
    /// Shared one-dimensional interpolation and extrapolation policy.
    pub interpolation: InterpolationPlan,
    /// Maximum scalar interpolation work units per query.
    pub max_work: u64,
    /// Maximum temporary scalar values per query.
    pub max_memory_values: usize,
}

/// Evidence for a ranked-grid query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridInterpolationReport {
    /// Tensor dimensions, in storage order.
    pub shape: Vec<usize>,
    /// Applied contraction order.
    pub axis_order: Vec<usize>,
    /// Conservative scalar work units.
    pub work_units: u64,
    /// Peak temporary scalar count.
    pub peak_memory_values: usize,
    /// Whether the result makes a multidimensional monotonicity guarantee.
    pub multidimensional_monotonicity_claimed: bool,
}

/// One interpolated scalar and its bounded-work evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct GridInterpolationResult {
    /// Interpolated value.
    pub value: f64,
    /// Shape, order, and limits evidence.
    pub report: GridInterpolationReport,
}

/// Reusable dense row-major interpolator over explicit monotone coordinate axes.
#[derive(Clone, Debug, PartialEq)]
pub struct RegularGridInterpolator {
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    values: Vec<f64>,
    plan: GridInterpolationPlan,
    work: u64,
}

impl RegularGridInterpolator {
    /// Validates axes, tensor shape, order, and resource limits once.
    pub fn new(
        mut axes: Vec<Vec<f64>>,
        shape: Vec<usize>,
        mut values: Vec<f64>,
        plan: GridInterpolationPlan,
    ) -> Result<Self, SignalError> {
        let rank = axes.len();
        if rank == 0 || shape.len() != rank {
            return Err(SignalError::InvalidTensorView {
                reason: "grid axes and tensor shape must have equal nonzero rank",
            });
        }
        let size = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or(SignalError::InvalidTensorView {
                reason: "grid tensor size overflow",
            })?;
        if size != values.len() {
            return Err(SignalError::LengthMismatch {
                expected: size,
                actual: values.len(),
            });
        }
        if plan.axis_order.len() != rank {
            return Err(SignalError::InvalidPolicy {
                policy: "axis order",
                reason: "every tensor axis must appear exactly once",
            });
        }
        let mut seen = vec![false; rank];
        for &axis in &plan.axis_order {
            if axis >= rank {
                return Err(SignalError::AxisOutOfBounds { axis, rank });
            }
            if seen[axis] {
                return Err(SignalError::DuplicateAxis { axis });
            }
            seen[axis] = true;
        }
        if plan.interpolation.duplicates != DuplicateXPolicy::Reject {
            return Err(SignalError::InvalidPolicy {
                policy: "grid duplicate coordinates",
                reason: "ranked tensor axes require strict coordinates; use the existing reject policy",
            });
        }
        for (axis, (coordinates, &dimension)) in axes.iter_mut().zip(&shape).enumerate() {
            if coordinates.len() != dimension {
                return Err(SignalError::InvalidLength {
                    len: coordinates.len(),
                    reason: "coordinate count must equal tensor axis extent",
                });
            }
            if dimension < 2 {
                return Err(SignalError::InvalidLength {
                    len: dimension,
                    reason: "regular-grid axes require at least two coordinates",
                });
            }
            if coordinates.iter().any(|x| !x.is_finite()) {
                return Err(SignalError::NonFinite {
                    index: axis,
                    component: "coordinate",
                });
            }
            let increasing = coordinates.windows(2).all(|p| p[0] < p[1]);
            let decreasing = coordinates.windows(2).all(|p| p[0] > p[1]);
            if !increasing && !decreasing {
                return Err(SignalError::InvalidPolicy {
                    policy: "grid coordinates",
                    reason: "each axis must be strictly monotone",
                });
            }
            if decreasing {
                coordinates.reverse();
                reverse_axis(&mut values, &shape, axis);
            }
        }
        let work = (size as u64)
            .checked_mul(rank as u64)
            .ok_or(SignalError::WorkLimit {
                required: u64::MAX,
                maximum: plan.max_work,
            })?;
        if work > plan.max_work {
            return Err(SignalError::WorkLimit {
                required: work,
                maximum: plan.max_work,
            });
        }
        if size > plan.max_memory_values {
            return Err(SignalError::ScratchLimit {
                required: size,
                maximum: plan.max_memory_values,
            });
        }
        Ok(Self {
            axes,
            shape,
            values,
            plan,
            work,
        })
    }
    /// Evaluates one coordinate tuple by tensor-product contraction.
    pub fn evaluate(&self, point: &[f64]) -> Result<GridInterpolationResult, SignalError> {
        if point.len() != self.shape.len() {
            return Err(SignalError::LengthMismatch {
                expected: self.shape.len(),
                actual: point.len(),
            });
        }
        let mut data = self.values.clone();
        let mut shape = self.shape.clone();
        let mut labels = (0..shape.len()).collect::<Vec<_>>();
        for &original_axis in &self.plan.axis_order {
            let position = labels
                .iter()
                .position(|&a| a == original_axis)
                .expect("validated axis remains");
            data = contract(
                &data,
                &shape,
                position,
                &self.axes[original_axis],
                point[original_axis],
                self.plan.interpolation,
            )?;
            shape.remove(position);
            labels.remove(position);
        }
        Ok(GridInterpolationResult {
            value: data[0],
            report: GridInterpolationReport {
                shape: self.shape.clone(),
                axis_order: self.plan.axis_order.clone(),
                work_units: self.work,
                peak_memory_values: self.values.len(),
                multidimensional_monotonicity_claimed: self.shape.len() == 1
                    && self.plan.interpolation.method == InterpolationMethod::Monotone,
            },
        })
    }
}
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        out[i] = out[i + 1] * shape[i + 1];
    }
    out
}
fn reverse_axis(data: &mut [f64], shape: &[usize], axis: usize) {
    let stride = strides(shape)[axis];
    let width = shape[axis];
    let block = stride * width;
    for base in (0..data.len()).step_by(block) {
        for offset in 0..stride {
            for left in 0..width / 2 {
                data.swap(
                    base + left * stride + offset,
                    base + (width - 1 - left) * stride + offset,
                );
            }
        }
    }
}
fn contract(
    data: &[f64],
    shape: &[usize],
    axis: usize,
    coordinates: &[f64],
    query: f64,
    plan: InterpolationPlan,
) -> Result<Vec<f64>, SignalError> {
    let stride = strides(shape)[axis];
    let width = shape[axis];
    let block = stride * width;
    let mut output = Vec::with_capacity(data.len() / width);
    for base in (0..data.len()).step_by(block) {
        for offset in 0..stride {
            let line = (0..width)
                .map(|i| data[base + i * stride + offset])
                .collect::<Vec<_>>();
            output.push(
                SampleInterpolator::new(coordinates, &line, plan)?
                    .evaluate(&[query])?
                    .values[0],
            );
        }
    }
    Ok(output)
}
