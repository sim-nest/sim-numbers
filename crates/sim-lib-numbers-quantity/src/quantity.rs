use core::cmp::Ordering;
use core::fmt;

use crate::{Dimension, DimensionError, Exponent, MeasureKind, MeasureRole, Scalar, Unit};

/// A scalar carrying independent physical and semantic metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantity<S> {
    scalar: S,
    dimension: Dimension,
    kind: Option<MeasureKind>,
    unit: Option<Unit>,
    role: MeasureRole,
}

/// Quantity construction or algebra failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QuantityError {
    /// The supplied kind does not admit the quantity dimension.
    KindDimensionMismatch,
    /// The supplied unit does not describe the quantity.
    UnitMismatch,
    /// Addition, subtraction, comparison, or conversion was incompatible.
    Incompatible {
        /// Human-readable refusal reason.
        reason: &'static str,
    },
    /// Multiplication, division, or powers were attempted on an affine point.
    AffinePointInLinearOperation,
    /// Exact dimension exponent arithmetic failed.
    Dimension(DimensionError),
    /// The scalar domain refused or overflowed an operation.
    Scalar(String),
}

impl fmt::Display for QuantityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KindDimensionMismatch => {
                f.write_str("measure kind does not admit quantity dimension")
            }
            Self::UnitMismatch => {
                f.write_str("unit dimension, kind, or role does not match quantity")
            }
            Self::Incompatible { reason } => write!(f, "incompatible quantities: {reason}"),
            Self::AffinePointInLinearOperation => f.write_str(
                "affine points cannot participate in linear multiplication, division, or powers",
            ),
            Self::Dimension(error) => error.fmt(f),
            Self::Scalar(error) => write!(f, "scalar operation failed: {error}"),
        }
    }
}
impl std::error::Error for QuantityError {}
impl From<DimensionError> for QuantityError {
    fn from(value: DimensionError) -> Self {
        Self::Dimension(value)
    }
}

impl<S: Scalar> Quantity<S> {
    /// Constructs a checked quantity.
    pub fn new(
        scalar: S,
        dimension: Dimension,
        kind: Option<MeasureKind>,
        unit: Option<Unit>,
        role: MeasureRole,
    ) -> Result<Self, QuantityError> {
        if kind
            .as_ref()
            .is_some_and(|kind| kind.dimension() != &dimension)
        {
            return Err(QuantityError::KindDimensionMismatch);
        }
        if let Some(unit) = &unit
            && (unit.dimension() != &dimension
                || unit.role() != role
                || unit.kind() != kind.as_ref())
        {
            return Err(QuantityError::UnitMismatch);
        }
        Ok(Self {
            scalar,
            dimension,
            kind,
            unit,
            role,
        })
    }

    /// Scalar in its installed domain.
    pub fn scalar(&self) -> &S {
        &self.scalar
    }
    /// Exact physical dimension.
    pub fn dimension(&self) -> &Dimension {
        &self.dimension
    }
    /// Optional semantic kind.
    pub fn kind(&self) -> Option<&MeasureKind> {
        self.kind.as_ref()
    }
    /// Optional display/conversion unit.
    pub fn unit(&self) -> Option<&Unit> {
        self.unit.as_ref()
    }
    /// Point or interval role.
    pub const fn role(&self) -> MeasureRole {
        self.role
    }

    /// Converts through exact scale and offset constants while retaining `S`.
    pub fn convert(&self, target: &Unit) -> Result<Self, QuantityError> {
        let source = self.unit.as_ref().ok_or(QuantityError::Incompatible {
            reason: "source quantity has no named unit",
        })?;
        if source.family() != target.family()
            || source.dimension() != target.dimension()
            || source.kind() != target.kind()
            || source.role() != target.role()
        {
            return Err(QuantityError::Incompatible {
                reason: "unit family, dimension, kind, or role differs",
            });
        }
        let (sn, sd) = source.scale();
        let (on, od) = source.offset();
        let canonical = self
            .scalar
            .mul_ratio(sn, sd)
            .map_err(scalar_error::<S>)?
            .add_ratio(on, od)
            .map_err(scalar_error::<S>)?;
        let (tn, td) = target.scale();
        let (ton, tod) = target.offset();
        let scalar = canonical
            .add_ratio(-ton, tod)
            .map_err(scalar_error::<S>)?
            .mul_ratio(td, tn)
            .map_err(scalar_error::<S>)?;
        Self::new(
            scalar,
            self.dimension.clone(),
            self.kind.clone(),
            Some(target.clone()),
            self.role,
        )
    }

    /// Adds compatible intervals, or an interval displacement to a point.
    pub fn add(&self, rhs: &Self) -> Result<Self, QuantityError> {
        self.require_dimension_kind(rhs)?;
        if self.role == MeasureRole::Point && rhs.role == MeasureRole::Point {
            return Err(QuantityError::Incompatible {
                reason: "affine points cannot be added",
            });
        }
        let rhs = self.in_left_unit(rhs)?;
        let role = if self.role == MeasureRole::Point || rhs.role == MeasureRole::Point {
            MeasureRole::Point
        } else {
            MeasureRole::Interval
        };
        let unit = if role == self.role {
            self.unit.clone()
        } else {
            rhs.unit.clone()
        };
        Self::new(
            self.scalar.add(&rhs.scalar).map_err(scalar_error::<S>)?,
            self.dimension.clone(),
            self.kind.clone(),
            unit,
            role,
        )
    }

    /// Subtracts compatible values; point minus point is an interval.
    pub fn sub(&self, rhs: &Self) -> Result<Self, QuantityError> {
        self.require_dimension_kind(rhs)?;
        if self.role == MeasureRole::Interval && rhs.role == MeasureRole::Point {
            return Err(QuantityError::Incompatible {
                reason: "an affine point cannot be subtracted from an interval",
            });
        }
        let rhs = self.in_left_unit(rhs)?;
        let role = if self.role == MeasureRole::Point && rhs.role == MeasureRole::Point {
            MeasureRole::Interval
        } else {
            self.role
        };
        let unit = self
            .unit
            .as_ref()
            .filter(|unit| unit.role() == role)
            .cloned();
        Self::new(
            self.scalar.sub(&rhs.scalar).map_err(scalar_error::<S>)?,
            self.dimension.clone(),
            self.kind.clone(),
            unit,
            role,
        )
    }

    /// Compares compatible quantities after exact conversion.
    pub fn partial_compare(&self, rhs: &Self) -> Result<Option<Ordering>, QuantityError>
    where
        S: PartialOrd,
    {
        self.require_dimension_kind(rhs)?;
        if self.role != rhs.role {
            return Err(QuantityError::Incompatible {
                reason: "measure roles differ",
            });
        }
        Ok(self.scalar.partial_cmp(&self.in_left_unit(rhs)?.scalar))
    }

    /// Multiplies linear quantities. A semantic kind is retained only for a
    /// dimensionless factor; otherwise the result is honestly unnamed.
    pub fn mul(&self, rhs: &Self) -> Result<Self, QuantityError> {
        self.require_linear(rhs)?;
        let dimension = self.dimension.product(&rhs.dimension)?;
        let kind = if self.dimension.is_dimensionless() {
            rhs.kind.clone()
        } else if rhs.dimension.is_dimensionless() {
            self.kind.clone()
        } else {
            None
        };
        Self::new(
            self.scalar.mul(&rhs.scalar).map_err(scalar_error::<S>)?,
            dimension,
            kind,
            None,
            MeasureRole::Interval,
        )
    }

    /// Divides linear quantities, retaining kind only across a dimensionless divisor.
    pub fn div(&self, rhs: &Self) -> Result<Self, QuantityError> {
        self.require_linear(rhs)?;
        let dimension = self.dimension.quotient(&rhs.dimension)?;
        let kind = if rhs.dimension.is_dimensionless() {
            self.kind.clone()
        } else {
            None
        };
        Self::new(
            self.scalar.div(&rhs.scalar).map_err(scalar_error::<S>)?,
            dimension,
            kind,
            None,
            MeasureRole::Interval,
        )
    }

    /// Raises a linear quantity to an exact rational power. Non-identity
    /// powers intentionally drop the semantic kind unless a caller assigns a
    /// separately declared kind after checking the resulting dimension.
    pub fn pow(&self, power: Exponent) -> Result<Self, QuantityError> {
        if self.role == MeasureRole::Point {
            return Err(QuantityError::AffinePointInLinearOperation);
        }
        let scalar = self
            .scalar
            .pow_ratio(power.numerator(), power.denominator())
            .map_err(scalar_error::<S>)?;
        let kind = if power == Exponent::new(1, 1)? {
            self.kind.clone()
        } else {
            None
        };
        Self::new(
            scalar,
            self.dimension.power(power)?,
            kind,
            None,
            MeasureRole::Interval,
        )
    }

    fn require_dimension_kind(&self, rhs: &Self) -> Result<(), QuantityError> {
        if self.dimension != rhs.dimension {
            return Err(QuantityError::Incompatible {
                reason: "dimensions differ",
            });
        }
        if self.kind != rhs.kind {
            return Err(QuantityError::Incompatible {
                reason: "semantic kinds differ",
            });
        }
        Ok(())
    }
    fn require_linear(&self, rhs: &Self) -> Result<(), QuantityError> {
        if self.role == MeasureRole::Point || rhs.role == MeasureRole::Point {
            Err(QuantityError::AffinePointInLinearOperation)
        } else {
            Ok(())
        }
    }
    fn in_left_unit(&self, rhs: &Self) -> Result<Self, QuantityError> {
        match (&self.unit, &rhs.unit) {
            (Some(left), Some(_)) => rhs.convert(left),
            (None, None) => Ok(rhs.clone()),
            _ => Err(QuantityError::Incompatible {
                reason: "named and unnamed units do not mix implicitly",
            }),
        }
    }
}

fn scalar_error<S: Scalar>(error: S::Error) -> QuantityError {
    QuantityError::Scalar(error.to_string())
}

/// Declarative constraints for matching quantity metadata.
#[derive(Clone, Debug, Default)]
pub struct QuantityShape {
    /// Required dimension, if any.
    pub dimension: Option<Dimension>,
    /// Required semantic kind content id, if any.
    pub kind_id: Option<String>,
    /// Required unit family, if any.
    pub unit_family: Option<String>,
    /// Required point/interval role, if any.
    pub role: Option<MeasureRole>,
}

impl QuantityShape {
    /// Checks metadata without inspecting or coercing the scalar domain.
    pub fn matches<S: Scalar>(&self, quantity: &Quantity<S>) -> bool {
        self.dimension
            .as_ref()
            .is_none_or(|value| value == quantity.dimension())
            && self
                .kind_id
                .as_ref()
                .is_none_or(|value| quantity.kind().is_some_and(|kind| kind.id() == value))
            && self
                .unit_family
                .as_ref()
                .is_none_or(|value| quantity.unit().is_some_and(|unit| unit.family() == value))
            && self.role.is_none_or(|value| value == quantity.role())
    }
}
