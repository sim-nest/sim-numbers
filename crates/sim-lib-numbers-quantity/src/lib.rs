#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Semantic physical quantities for SIM.
//!
//! A [`Quantity`] retains four independent facts: its scalar, physical
//! [`Dimension`], optional semantic [`MeasureKind`], and display [`Unit`] with
//! a [`MeasureRole`]. Dimensions use bounded exact rational exponents. Kinds
//! are open content-identified records, so equal dimensions (notably energy
//! and torque) do not imply substitutable meaning. Scalar arithmetic is
//! generic and unit conversion never passes through `f64`.

mod dimension;
mod quantity;
mod runtime;
mod scalar;
mod si;
mod unit;

pub use dimension::{BaseDimension, Dimension, DimensionError, Exponent};
pub use quantity::{Quantity, QuantityError, QuantityShape};
pub use runtime::{QuantityLib, QuantityValue, quantity_class_symbol, quantity_shape_symbol};
pub use scalar::{ExactScalar, Scalar};
pub use si::{CELSIUS, JOULE, KELVIN, METRE, NEWTON_METRE, energy_kind, torque_kind};
pub use unit::{MeasureKind, MeasureRole, Unit, UnitError};

/// Cookbook recipes embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
