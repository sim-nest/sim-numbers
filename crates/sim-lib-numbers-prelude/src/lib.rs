#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Umbrella library for the sim-numbers family: `NumbersPreludeLib` installs the
//! standard number domains and tensor backends (arithmetic, f64/i64/bigint,
//! rational, complex, CAS, func, tensor and its specializations, numeric, quad,
//! and rk) into a runtime in one call.

use sim_kernel::{Cx, Lib, Result, Symbol};
use sim_lib_numbers_arith::NumbersArithmeticLib;
use sim_lib_numbers_bigint::BigIntNumbersLib;
use sim_lib_numbers_bool::BoolNumbersLib;
use sim_lib_numbers_cas::CasNumbersLib;
use sim_lib_numbers_cas_diff::CasDiffLib;
use sim_lib_numbers_cas_eval::CasEvalLib;
use sim_lib_numbers_complex::ComplexNumbersLib;
use sim_lib_numbers_core::domains;
use sim_lib_numbers_exotic::ExoticNumbersLib;
use sim_lib_numbers_f64::F64NumbersLib;
use sim_lib_numbers_fixed::FixedNumbersLib;
use sim_lib_numbers_float::F32NumbersLib;
use sim_lib_numbers_func::FuncNumbersLib;
use sim_lib_numbers_i64::I64NumbersLib;
use sim_lib_numbers_numeric::NumericNumbersLib;
use sim_lib_numbers_quad::QuadNumbersLib;
use sim_lib_numbers_rational::RationalNumbersLib;
use sim_lib_numbers_rk::RkNumbersLib;
use sim_lib_numbers_stats::StatsNumbersLib;
use sim_lib_numbers_tensor::TensorNumbersLib;
use sim_lib_numbers_tensor_bcast::TensorBroadcastLib;
use sim_lib_numbers_tensor_bit::BitTensorLib;
use sim_lib_numbers_tensor_cmplxf::ComplexFTensorLib;
use sim_lib_numbers_tensor_f64::F64TensorLib;
use sim_lib_numbers_tensor_i64::I64TensorLib;
use sim_lib_numbers_tensor_linalg::TensorLinalgLib;
use sim_lib_numbers_tensor_rat64::Rat64TensorLib;

/// Umbrella library that installs the standard sim-numbers domains and tensor
/// backends into a runtime in one call.
pub struct NumbersPreludeLib;

impl NumbersPreludeLib {
    /// Creates the prelude library.
    pub fn new() -> Self {
        Self
    }

    /// Installs every standard number domain and tensor backend into `cx`,
    /// skipping any domain that is already present.
    pub fn install_all(&self, cx: &mut Cx) -> Result<()> {
        install_if_missing(cx, domains::arith(), &NumbersArithmeticLib::new())?;
        install_if_missing(cx, domains::f64(), &F64NumbersLib::new())?;
        install_if_missing(cx, domains::i64(), &I64NumbersLib::new())?;
        install_if_missing(cx, domains::bool(), &BoolNumbersLib::new())?;
        install_if_missing(cx, domains::fixed(), &FixedNumbersLib::new())?;
        install_if_missing(cx, domains::f32(), &F32NumbersLib::new())?;
        install_if_missing(cx, domains::bigint(), &BigIntNumbersLib::new())?;
        install_if_missing(cx, domains::rational(), &RationalNumbersLib::new())?;
        install_if_missing(cx, domains::complex(), &ComplexNumbersLib::new())?;
        install_if_missing(cx, domains::continued_fraction(), &ExoticNumbersLib::new())?;
        install_if_missing(cx, domains::cas(), &CasNumbersLib::new())?;
        install_if_missing(cx, domains::cas_diff(), &CasDiffLib::new())?;
        install_if_missing(cx, domains::cas_eval(), &CasEvalLib::new())?;
        install_if_missing(cx, domains::func(), &FuncNumbersLib::new())?;
        install_if_missing(cx, domains::tensor(), &TensorNumbersLib::new())?;
        install_if_missing(cx, domains::tensor_bcast(), &TensorBroadcastLib::new())?;
        install_if_missing(cx, domains::tensor_linalg(), &TensorLinalgLib::new())?;
        install_if_missing(
            cx,
            sim_lib_numbers_tensor_f64::tensor_lib_symbol(),
            &F64TensorLib::new(),
        )?;
        install_if_missing(
            cx,
            sim_lib_numbers_tensor_i64::tensor_lib_symbol(),
            &I64TensorLib::new(),
        )?;
        install_if_missing(
            cx,
            sim_lib_numbers_tensor_rat64::tensor_lib_symbol(),
            &Rat64TensorLib::new(),
        )?;
        install_if_missing(
            cx,
            sim_lib_numbers_tensor_cmplxf::tensor_lib_symbol(),
            &ComplexFTensorLib::new(),
        )?;
        install_if_missing(
            cx,
            sim_lib_numbers_tensor_bit::tensor_lib_symbol(),
            &BitTensorLib::new(),
        )?;
        install_if_missing(cx, domains::numeric(), &NumericNumbersLib::new())?;
        install_if_missing(cx, domains::quad(), &QuadNumbersLib::new())?;
        install_if_missing(cx, domains::rk(), &RkNumbersLib::new())?;
        install_if_missing(
            cx,
            Symbol::qualified("numbers", "stats"),
            &StatsNumbersLib::new(),
        )?;
        Ok(())
    }
}

impl Default for NumbersPreludeLib {
    fn default() -> Self {
        Self::new()
    }
}

fn install_if_missing(cx: &mut Cx, symbol: Symbol, lib: &dyn Lib) -> Result<()> {
    if cx.registry().lib(&symbol).is_none() {
        cx.load_lib(lib)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
