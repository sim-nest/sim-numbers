//! The `numbers/i64` literal value-shape symbol.
//!
//! The number-literal shape and class themselves are the shared
//! `sim_lib_numbers_core::{NumberLiteralShape, NumberLiteralClass}`; only the
//! domain-specific value-shape symbol lives here.

use sim_kernel::Symbol;

pub(crate) fn value_instance_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&super::number_domain())
}
