#![allow(non_snake_case)]

use crate::{BaseDimension, Dimension, Exponent, MeasureKind, MeasureRole, Unit};

fn energy_dimension() -> Dimension {
    Dimension::base(BaseDimension::Mass)
        .product(
            &Dimension::base(BaseDimension::Length)
                .power(Exponent::new(2, 1).expect("constant exponent"))
                .expect("bounded constant"),
        )
        .expect("bounded constant")
        .quotient(
            &Dimension::base(BaseDimension::Time)
                .power(Exponent::new(2, 1).expect("constant exponent"))
                .expect("bounded constant"),
        )
        .expect("bounded constant")
}

/// The SI energy kind.
pub fn energy_kind() -> MeasureKind {
    MeasureKind::new("si", "energy", energy_dimension()).expect("static kind")
}
/// The distinct SI torque kind, despite sharing energy's dimension.
pub fn torque_kind() -> MeasureKind {
    MeasureKind::new("si", "torque", energy_dimension()).expect("static kind")
}

/// Metre interval unit.
pub fn METRE() -> Unit {
    Unit::new(
        "m",
        "si:length",
        (1, 1),
        (0, 1),
        Dimension::base(BaseDimension::Length),
        None,
        MeasureRole::Interval,
    )
    .expect("static unit")
}
/// Joule energy unit.
pub fn JOULE() -> Unit {
    Unit::new(
        "J",
        "si:energy",
        (1, 1),
        (0, 1),
        energy_dimension(),
        Some(energy_kind()),
        MeasureRole::Interval,
    )
    .expect("static unit")
}
/// Newton-metre torque unit; dimensionally equal to a joule but semantically distinct.
pub fn NEWTON_METRE() -> Unit {
    Unit::new(
        "N*m",
        "si:torque",
        (1, 1),
        (0, 1),
        energy_dimension(),
        Some(torque_kind()),
        MeasureRole::Interval,
    )
    .expect("static unit")
}
/// Kelvin affine temperature-point unit.
pub fn KELVIN() -> Unit {
    Unit::new(
        "K",
        "si:temperature",
        (1, 1),
        (0, 1),
        Dimension::base(BaseDimension::Temperature),
        None,
        MeasureRole::Point,
    )
    .expect("static unit")
}
/// Celsius affine temperature-point unit, exactly offset from kelvin by 27315/100.
pub fn CELSIUS() -> Unit {
    Unit::new(
        "degC",
        "si:temperature",
        (1, 1),
        (27315, 100),
        Dimension::base(BaseDimension::Temperature),
        None,
        MeasureRole::Point,
    )
    .expect("static unit")
}
