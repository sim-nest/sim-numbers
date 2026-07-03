#![cfg_attr(not(feature = "native-export"), forbid(unsafe_code))]
#![cfg_attr(feature = "native-export", deny(unsafe_code))]
#![deny(missing_docs)]
#![allow(deprecated)]

//! The `numbers/f64` domain: double-precision floating-point literals and
//! values, their scalar arithmetic, and promotion into `complex`.

mod implementation;
#[cfg(feature = "native-export")]
mod loaders;
#[cfg(feature = "native-export")]
mod native;
#[cfg(feature = "native-export")]
extern crate self as sim;

#[cfg(feature = "native-export")]
use sim_codec_binary as codec_binary;
#[cfg(feature = "native-export")]
use sim_kernel as kernel;
#[cfg(feature = "native-export")]
use sim_macros::{sim_lib, sim_number_domain};

/// The cookbook recipes embedded for the `numbers/f64` domain, packaged at
/// build time for runtime help and browse surfaces.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

pub use implementation::*;
