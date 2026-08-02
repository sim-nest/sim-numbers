use sim_kernel::Symbol;
use sim_lib_numbers_tensor::{domains, spec_tensor_symbol};

/// The manifest id symbol for this library (`numbers/tensor-f64`).
pub fn tensor_lib_symbol() -> Symbol {
    domains::domain("tensor-f64")
}

/// The symbol under which the `f64`-tensor spec descriptor is exported.
pub fn tensor_spec_symbol() -> Symbol {
    spec_tensor_symbol("f64")
}

/// Cookbook recipes for this lib, embedded at build time.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));
