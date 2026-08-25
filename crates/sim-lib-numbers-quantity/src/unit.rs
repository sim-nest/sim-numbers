use core::fmt;

use crate::Dimension;

/// Semantic distinction between a location on an affine scale and a linear
/// displacement. Points may be subtracted but never multiplied or divided.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MeasureRole {
    /// A linear interval or difference.
    Interval,
    /// An affine point (for example a Celsius temperature reading).
    Point,
}

/// An open, content-identified semantic measure kind.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MeasureKind {
    id: String,
    name: String,
    dimension: Dimension,
}

impl MeasureKind {
    /// Creates an open kind whose collision-free canonical identifier includes
    /// its namespace, name, and admitted dimension.
    pub fn new(namespace: &str, name: &str, dimension: Dimension) -> Result<Self, UnitError> {
        if namespace.is_empty()
            || name.is_empty()
            || namespace.contains(char::is_whitespace)
            || name.contains(char::is_whitespace)
        {
            return Err(UnitError::InvalidIdentifier);
        }
        let id = format!("kind:{namespace}:{name}@{}", dimension.canonical());
        Ok(Self {
            id,
            name: name.to_owned(),
            dimension,
        })
    }

    /// Returns the canonical content identifier.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Returns the human-facing name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns the sole dimension admitted by this kind.
    pub fn dimension(&self) -> &Dimension {
        &self.dimension
    }
}

/// A unit is an exact affine transform to its canonical SI representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Unit {
    symbol: String,
    family: String,
    scale: (i128, i128),
    offset: (i128, i128),
    dimension: Dimension,
    kind: Option<MeasureKind>,
    role: MeasureRole,
}

/// Invalid unit metadata or compatibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnitError {
    /// Empty or whitespace-bearing stable identifier.
    InvalidIdentifier,
    /// Scale denominator was zero or scale was not positive.
    InvalidScale,
    /// Offset denominator was zero.
    InvalidOffset,
    /// A kind admitted a different dimension.
    KindDimensionMismatch,
    /// An interval unit attempted to carry an affine offset.
    IntervalHasOffset,
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidIdentifier => "invalid unit or kind identifier",
            Self::InvalidScale => "unit scale must be positive with nonzero denominator",
            Self::InvalidOffset => "unit offset denominator must not be zero",
            Self::KindDimensionMismatch => "measure kind does not admit the unit dimension",
            Self::IntervalHasOffset => "interval units cannot carry affine offsets",
        })
    }
}
impl std::error::Error for UnitError {}

impl Unit {
    /// Constructs a checked exact unit transform `canonical = scalar * scale + offset`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: &str,
        family: &str,
        scale: (i128, i128),
        offset: (i128, i128),
        dimension: Dimension,
        kind: Option<MeasureKind>,
        role: MeasureRole,
    ) -> Result<Self, UnitError> {
        if symbol.is_empty()
            || family.is_empty()
            || symbol.contains(char::is_whitespace)
            || family.contains(char::is_whitespace)
        {
            return Err(UnitError::InvalidIdentifier);
        }
        if scale.0 <= 0 || scale.1 <= 0 {
            return Err(UnitError::InvalidScale);
        }
        if offset.1 == 0 {
            return Err(UnitError::InvalidOffset);
        }
        if role == MeasureRole::Interval && offset.0 != 0 {
            return Err(UnitError::IntervalHasOffset);
        }
        if kind
            .as_ref()
            .is_some_and(|kind| kind.dimension() != &dimension)
        {
            return Err(UnitError::KindDimensionMismatch);
        }
        Ok(Self {
            symbol: symbol.to_owned(),
            family: family.to_owned(),
            scale: reduce(scale),
            offset: reduce(offset),
            dimension,
            kind,
            role,
        })
    }

    /// Stable unit symbol.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
    /// Conversion family; only units in one family convert directly.
    pub fn family(&self) -> &str {
        &self.family
    }
    /// Exact multiplicative scale to canonical units.
    pub const fn scale(&self) -> (i128, i128) {
        self.scale
    }
    /// Exact affine offset to canonical units.
    pub const fn offset(&self) -> (i128, i128) {
        self.offset
    }
    /// Physical dimension.
    pub fn dimension(&self) -> &Dimension {
        &self.dimension
    }
    /// Optional semantic kind.
    pub fn kind(&self) -> Option<&MeasureKind> {
        self.kind.as_ref()
    }
    /// Point or interval role.
    pub const fn role(&self) -> MeasureRole {
        self.role
    }
}

fn reduce((mut n, mut d): (i128, i128)) -> (i128, i128) {
    if d < 0 {
        n = -n;
        d = -d;
    }
    let mut a = n.unsigned_abs();
    let mut b = d as u128;
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    let gcd = a.max(1) as i128;
    (n / gcd, d / gcd)
}
