//! Linear, natural-cubic, and shape-preserving interpolation of sampled data.

use sim_lib_numbers_tensor_linalg::{DenseSolveOptions, DenseSolveReport, solve_dense_f64};

use crate::{SignalError, linalg_support::dense_error};

/// Interpolant constructed between unique, increasing sample coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Piecewise linear interpolation.
    Linear,
    /// Natural cubic spline interpolation.
    Cubic,
    /// Shape-preserving piecewise cubic Hermite interpolation.
    #[default]
    Monotone,
}

/// Policy for repeated adjacent abscissas in non-decreasing input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DuplicateXPolicy {
    /// Reject every repeated coordinate.
    #[default]
    Reject,
    /// Retain the first ordinate at a repeated coordinate.
    KeepFirst,
    /// Retain the final ordinate at a repeated coordinate.
    KeepLast,
    /// Replace repeated ordinates with their arithmetic mean.
    Average,
}

/// Policy for query coordinates beyond the first or last sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExtrapolationPolicy {
    /// Reject the out-of-domain query.
    #[default]
    Reject,
    /// Return the nearest endpoint ordinate.
    Clamp,
    /// Continue the nearest endpoint secant line.
    Linear,
}

/// Complete policy for sampled-data interpolation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterpolationPlan {
    /// Within-domain interpolant.
    pub method: InterpolationMethod,
    /// Repeated-coordinate handling.
    pub duplicates: DuplicateXPolicy,
    /// Out-of-domain handling.
    pub extrapolation: ExtrapolationPolicy,
}

/// Evidence retained for one interpolation request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolationReport {
    /// Number of caller-supplied sample pairs.
    pub input_points: usize,
    /// Number of unique sample coordinates after duplicate policy.
    pub unique_points: usize,
    /// Number of repeated points consumed by duplicate policy.
    pub duplicates_resolved: usize,
    /// Number of query coordinates outside the sampled domain.
    pub extrapolated_points: usize,
    /// Interpolation method.
    pub method: InterpolationMethod,
    /// Duplicate-coordinate policy.
    pub duplicates: DuplicateXPolicy,
    /// Extrapolation policy.
    pub extrapolation: ExtrapolationPolicy,
    /// Dense-solve evidence for natural cubic construction, when required.
    pub cubic_solve: Option<DenseSolveReport>,
}

/// Interpolated values and policy evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct InterpolationResult {
    /// Values corresponding to query order.
    pub values: Vec<f64>,
    /// Input, duplicate, extrapolation, and construction evidence.
    pub report: InterpolationReport,
}

#[derive(Clone, Debug, PartialEq)]
enum Model {
    Linear,
    Cubic { second_derivatives: Vec<f64> },
    Monotone { tangents: Vec<f64> },
}

/// Reusable interpolant over finite, non-decreasing sample coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleInterpolator {
    x: Vec<f64>,
    y: Vec<f64>,
    model: Model,
    plan: InterpolationPlan,
    input_points: usize,
    duplicates_resolved: usize,
    cubic_solve: Option<DenseSolveReport>,
}

impl SampleInterpolator {
    /// Builds an interpolator, applying the declared duplicate policy once.
    pub fn new(x: &[f64], y: &[f64], plan: InterpolationPlan) -> Result<Self, SignalError> {
        if x.len() != y.len() {
            return Err(SignalError::LengthMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }
        if x.len() < 2 {
            return Err(SignalError::InvalidLength {
                len: x.len(),
                reason: "sampled interpolation requires at least two points",
            });
        }
        validate_finite(x, "sample coordinate")?;
        validate_finite(y, "sample value")?;
        let input_points = x.len();
        let (x, y, duplicates_resolved) = resolve_duplicates(x, y, plan.duplicates)?;
        if x.len() < 2 {
            return Err(SignalError::InvalidLength {
                len: x.len(),
                reason: "duplicate resolution left fewer than two unique coordinates",
            });
        }
        let (model, cubic_solve) = match plan.method {
            InterpolationMethod::Linear => (Model::Linear, None),
            InterpolationMethod::Cubic => {
                let (second_derivatives, report) = cubic_model(&x, &y)?;
                (Model::Cubic { second_derivatives }, report)
            }
            InterpolationMethod::Monotone => (
                Model::Monotone {
                    tangents: monotone_tangents(&x, &y),
                },
                None,
            ),
        };
        Ok(Self {
            x,
            y,
            model,
            plan,
            input_points,
            duplicates_resolved,
            cubic_solve,
        })
    }

    /// Returns the unique coordinates retained by duplicate policy.
    pub fn coordinates(&self) -> &[f64] {
        &self.x
    }

    /// Returns the ordinates retained or combined by duplicate policy.
    pub fn samples(&self) -> &[f64] {
        &self.y
    }

    /// Evaluates the interpolator at every query coordinate.
    pub fn evaluate(&self, coordinates: &[f64]) -> Result<InterpolationResult, SignalError> {
        validate_finite(coordinates, "query coordinate")?;
        let minimum = self.x[0];
        let maximum = self.x[self.x.len() - 1];
        let mut extrapolated_points = 0;
        let mut values = Vec::with_capacity(coordinates.len());
        for (index, coordinate) in coordinates.iter().copied().enumerate() {
            if coordinate < minimum || coordinate > maximum {
                extrapolated_points += 1;
                values.push(self.extrapolate(index, coordinate)?);
            } else {
                values.push(self.interpolate_one(coordinate));
            }
        }
        validate_finite(&values, "interpolated value")?;
        Ok(InterpolationResult {
            values,
            report: InterpolationReport {
                input_points: self.input_points,
                unique_points: self.x.len(),
                duplicates_resolved: self.duplicates_resolved,
                extrapolated_points,
                method: self.plan.method,
                duplicates: self.plan.duplicates,
                extrapolation: self.plan.extrapolation,
                cubic_solve: self.cubic_solve,
            },
        })
    }

    fn interpolate_one(&self, coordinate: f64) -> f64 {
        if coordinate == self.x[self.x.len() - 1] {
            return self.y[self.y.len() - 1];
        }
        let upper = self.x.partition_point(|sample| *sample <= coordinate);
        let interval = upper.saturating_sub(1).min(self.x.len() - 2);
        let x0 = self.x[interval];
        let x1 = self.x[interval + 1];
        let y0 = self.y[interval];
        let y1 = self.y[interval + 1];
        let width = x1 - x0;
        let t = (coordinate - x0) / width;
        match &self.model {
            Model::Linear => y0 + t * (y1 - y0),
            Model::Cubic { second_derivatives } => {
                let a = 1.0 - t;
                let b = t;
                a * y0
                    + b * y1
                    + ((a * a * a - a) * second_derivatives[interval]
                        + (b * b * b - b) * second_derivatives[interval + 1])
                        * width
                        * width
                        / 6.0
            }
            Model::Monotone { tangents } => {
                let t2 = t * t;
                let t3 = t2 * t;
                (2.0 * t3 - 3.0 * t2 + 1.0) * y0
                    + (t3 - 2.0 * t2 + t) * width * tangents[interval]
                    + (-2.0 * t3 + 3.0 * t2) * y1
                    + (t3 - t2) * width * tangents[interval + 1]
            }
        }
    }

    fn extrapolate(&self, index: usize, coordinate: f64) -> Result<f64, SignalError> {
        let last = self.x.len() - 1;
        match self.plan.extrapolation {
            ExtrapolationPolicy::Reject => Err(SignalError::OutOfDomain {
                index,
                value: coordinate,
                minimum: self.x[0],
                maximum: self.x[last],
            }),
            ExtrapolationPolicy::Clamp => Ok(if coordinate < self.x[0] {
                self.y[0]
            } else {
                self.y[last]
            }),
            ExtrapolationPolicy::Linear => {
                let interval = if coordinate < self.x[0] { 0 } else { last - 1 };
                let slope = (self.y[interval + 1] - self.y[interval])
                    / (self.x[interval + 1] - self.x[interval]);
                Ok(self.y[interval] + slope * (coordinate - self.x[interval]))
            }
        }
    }
}

/// Builds and evaluates a sampled-data interpolator in one call.
pub fn interpolate_samples(
    x: &[f64],
    y: &[f64],
    coordinates: &[f64],
    plan: InterpolationPlan,
) -> Result<InterpolationResult, SignalError> {
    SampleInterpolator::new(x, y, plan)?.evaluate(coordinates)
}

fn resolve_duplicates(
    x: &[f64],
    y: &[f64],
    policy: DuplicateXPolicy,
) -> Result<(Vec<f64>, Vec<f64>, usize), SignalError> {
    let mut unique_x = Vec::with_capacity(x.len());
    let mut unique_y = Vec::with_capacity(y.len());
    let mut counts = Vec::with_capacity(x.len());
    let mut duplicates = 0;
    for (index, (&x, &y)) in x.iter().zip(y).enumerate() {
        if let Some(previous) = unique_x.last().copied() {
            if x < previous {
                return Err(SignalError::InvalidPolicy {
                    policy: "sample coordinate order",
                    reason: "sample coordinates must be non-decreasing",
                });
            }
            if x == previous {
                duplicates += 1;
                match policy {
                    DuplicateXPolicy::Reject => {
                        return Err(SignalError::DuplicateCoordinate { index, value: x });
                    }
                    DuplicateXPolicy::KeepFirst => {}
                    DuplicateXPolicy::KeepLast => {
                        *unique_y
                            .last_mut()
                            .expect("a repeated point has a predecessor") = y;
                    }
                    DuplicateXPolicy::Average => {
                        let count = counts
                            .last_mut()
                            .expect("a repeated point has a predecessor");
                        *count += 1;
                        let mean = unique_y
                            .last_mut()
                            .expect("a repeated point has a predecessor");
                        *mean += (y - *mean) / *count as f64;
                    }
                }
                continue;
            }
        }
        unique_x.push(x);
        unique_y.push(y);
        counts.push(1usize);
    }
    Ok((unique_x, unique_y, duplicates))
}

fn cubic_model(x: &[f64], y: &[f64]) -> Result<(Vec<f64>, Option<DenseSolveReport>), SignalError> {
    if x.len() == 2 {
        return Ok((vec![0.0; 2], None));
    }
    let dimension = x.len();
    let mut matrix = vec![0.0; dimension * dimension];
    let mut rhs = vec![0.0; dimension];
    matrix[0] = 1.0;
    matrix[dimension * dimension - 1] = 1.0;
    for row in 1..dimension - 1 {
        let left_width = x[row] - x[row - 1];
        let right_width = x[row + 1] - x[row];
        matrix[row * dimension + row - 1] = left_width;
        matrix[row * dimension + row] = 2.0 * (left_width + right_width);
        matrix[row * dimension + row + 1] = right_width;
        rhs[row] = 6.0 * ((y[row + 1] - y[row]) / right_width - (y[row] - y[row - 1]) / left_width);
    }
    let solution = solve_dense_f64(&matrix, &rhs, DenseSolveOptions::default())
        .map_err(|error| dense_error(error, "natural cubic interpolation"))?;
    Ok((solution.values, Some(solution.report)))
}

fn monotone_tangents(x: &[f64], y: &[f64]) -> Vec<f64> {
    let intervals = x.len() - 1;
    let widths = (0..intervals)
        .map(|index| x[index + 1] - x[index])
        .collect::<Vec<_>>();
    let slopes = (0..intervals)
        .map(|index| (y[index + 1] - y[index]) / widths[index])
        .collect::<Vec<_>>();
    if x.len() == 2 {
        return vec![slopes[0], slopes[0]];
    }
    let mut tangents = vec![0.0; x.len()];
    tangents[0] = endpoint_tangent(widths[0], widths[1], slopes[0], slopes[1]);
    for index in 1..x.len() - 1 {
        let left = slopes[index - 1];
        let right = slopes[index];
        if left == 0.0 || right == 0.0 || left.signum() != right.signum() {
            tangents[index] = 0.0;
        } else {
            let left_weight = 2.0 * widths[index] + widths[index - 1];
            let right_weight = widths[index] + 2.0 * widths[index - 1];
            tangents[index] =
                (left_weight + right_weight) / (left_weight / left + right_weight / right);
        }
    }
    let last = x.len() - 1;
    tangents[last] = endpoint_tangent(
        widths[intervals - 1],
        widths[intervals - 2],
        slopes[intervals - 1],
        slopes[intervals - 2],
    );
    tangents
}

fn endpoint_tangent(width: f64, neighbor_width: f64, slope: f64, neighbor_slope: f64) -> f64 {
    let candidate = ((2.0 * width + neighbor_width) * slope - width * neighbor_slope)
        / (width + neighbor_width);
    if candidate.signum() != slope.signum() {
        0.0
    } else if slope.signum() != neighbor_slope.signum() && candidate.abs() > 3.0 * slope.abs() {
        3.0 * slope
    } else {
        candidate
    }
}

fn validate_finite(values: &[f64], component: &'static str) -> Result<(), SignalError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SignalError::NonFinite { index, component });
        }
    }
    Ok(())
}
