//! Canonical symbols for the number-domain family.
//!
//! Concrete number crates should import these helpers instead of spelling
//! `Symbol::qualified("numbers", "...")` locally. Keeping the symbols in one
//! place makes promotion edges auditable: a typo cannot silently create an
//! unreachable domain.
//!
//! # Promotion lattice
//!
//! The default number prelude installs a directed promotion graph. Literal
//! promotion rules are a fast path; every edge that participates in dispatch
//! also has a value-promotion rule so opaque number values can move through the
//! same lattice.
//!
//! Intended scalar reachability:
//!
//! - `bool -> u8 -> ...`, plus direct `bool -> i64` and `bool -> f64` edges.
//! - Signed fixed-width integers widen through `i8 -> i16 -> i32 -> i64 ->
//!   i128`; unsigned fixed-width integers widen through `u8 -> u16 -> u32 ->
//!   u64 -> u128` and can cross to the next wider signed domain.
//! - Fixed-width integers reach `f32`, `f64`, and `rational`; `i64` also
//!   provides direct `i64 -> f64` and `i64 -> rational` edges.
//! - `bigint` is the arbitrary-precision integer domain and reaches
//!   `rational`.
//! - `f32 -> f64`; `f64 <-> rational` when decimal rationalization succeeds.
//! - `i64`, `f64`, and `rational` reach `complex`; `complex` is the scalar
//!   numeric sink before symbolic, function, or tensor lifting.
//! - `cas` absorbs scalar domains through value promotion so symbolic
//!   arithmetic can mix CAS values with installed numeric domains.
//! - `tensor` is value-only; tensor broadcast and typed tensor fast paths use
//!   explicit value descriptors rather than scalar promotion.

use sim_kernel::Symbol;

/// Build a symbol in the canonical `numbers` namespace.
pub fn domain(name: impl Into<String>) -> Symbol {
    Symbol::qualified("numbers", name.into())
}

/// The `numbers/arith` symbol: the cross-domain arithmetic op namespace.
pub fn arith() -> Symbol {
    domain("arith")
}

/// The `numbers/bool` domain symbol.
pub fn bool() -> Symbol {
    domain("bool")
}

/// The `numbers/i8` domain symbol.
pub fn i8() -> Symbol {
    domain("i8")
}

/// The `numbers/u8` domain symbol.
pub fn u8() -> Symbol {
    domain("u8")
}

/// The `numbers/i16` domain symbol.
pub fn i16() -> Symbol {
    domain("i16")
}

/// The `numbers/u16` domain symbol.
pub fn u16() -> Symbol {
    domain("u16")
}

/// The `numbers/i32` domain symbol.
pub fn i32() -> Symbol {
    domain("i32")
}

/// The `numbers/u32` domain symbol.
pub fn u32() -> Symbol {
    domain("u32")
}

/// The `numbers/i64` domain symbol.
pub fn i64() -> Symbol {
    domain("i64")
}

/// The `numbers/u64` domain symbol.
pub fn u64() -> Symbol {
    domain("u64")
}

/// The `numbers/i128` domain symbol.
pub fn i128() -> Symbol {
    domain("i128")
}

/// The `numbers/u128` domain symbol.
pub fn u128() -> Symbol {
    domain("u128")
}

/// The `numbers/isize` domain symbol.
pub fn isize() -> Symbol {
    domain("isize")
}

/// The `numbers/usize` domain symbol.
pub fn usize() -> Symbol {
    domain("usize")
}

/// The `numbers/f32` domain symbol.
pub fn f32() -> Symbol {
    domain("f32")
}

/// The `numbers/f64` domain symbol.
pub fn f64() -> Symbol {
    domain("f64")
}

/// The `numbers/fixed` fixed-point domain symbol.
pub fn fixed() -> Symbol {
    domain("fixed")
}

/// The `numbers/bigint` arbitrary-precision integer domain symbol.
pub fn bigint() -> Symbol {
    domain("bigint")
}

/// The `numbers/rational` domain symbol.
pub fn rational() -> Symbol {
    domain("rational")
}

/// The `numbers/complex` domain symbol (the scalar numeric sink).
pub fn complex() -> Symbol {
    domain("complex")
}

/// The `numbers/cf` continued-fraction domain symbol.
pub fn continued_fraction() -> Symbol {
    domain("cf")
}

/// The `numbers/cas` symbolic (computer-algebra) domain symbol.
pub fn cas() -> Symbol {
    domain("cas")
}

/// The `numbers/cas-diff` symbolic-differentiation domain symbol.
pub fn cas_diff() -> Symbol {
    domain("cas-diff")
}

/// The `numbers/cas-eval` symbolic-evaluation domain symbol.
pub fn cas_eval() -> Symbol {
    domain("cas-eval")
}

/// The `numbers/func` function-value domain symbol.
pub fn func() -> Symbol {
    domain("func")
}

/// The `numbers/tensor` domain symbol (value-only lift over scalar domains).
pub fn tensor() -> Symbol {
    domain("tensor")
}

/// The `numbers/tensor-bcast` broadcasting tensor domain symbol.
pub fn tensor_bcast() -> Symbol {
    domain("tensor-bcast")
}

/// The `numbers/tensor-linalg` linear-algebra tensor domain symbol.
pub fn tensor_linalg() -> Symbol {
    domain("tensor-linalg")
}

/// The `numbers/numeric` namespace symbol for numeric utilities.
pub fn numeric() -> Symbol {
    domain("numeric")
}

/// The `numbers/quad` quadrature (numeric integration) domain symbol.
pub fn quad() -> Symbol {
    domain("quad")
}

/// The `numbers/rk` Runge-Kutta (numeric ODE) domain symbol.
pub fn rk() -> Symbol {
    domain("rk")
}

/// The literal-class symbol for a domain, e.g. `numbers/i64-literal`.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_core::domains;
///
/// assert_eq!(
///     domains::literal_class("i64"),
///     domains::domain("i64-literal"),
/// );
/// ```
pub fn literal_class(domain_name: impl AsRef<str>) -> Symbol {
    domain(format!("{}-literal", domain_name.as_ref()))
}

/// The value-shape symbol for `domain`, e.g. `numbers/i64/value-shape`.
pub fn value_shape(domain: &Symbol) -> Symbol {
    Symbol::qualified(domain.to_string(), "value-shape")
}

/// The `numbers/Rational` value-class symbol.
pub fn rational_value_class() -> Symbol {
    domain("Rational")
}

/// The `numbers/Complex` value-class symbol.
pub fn complex_value_class() -> Symbol {
    domain("Complex")
}

/// The `numbers/Cas` value-class symbol.
pub fn cas_value_class() -> Symbol {
    domain("Cas")
}

/// The `numbers/Tensor` value-class symbol.
pub fn tensor_value_class() -> Symbol {
    domain("Tensor")
}

/// All fixed-width integer domain symbols (`i8`..`usize`), in lattice order.
pub fn fixed_integer_domains() -> Vec<Symbol> {
    [
        i8(),
        u8(),
        i16(),
        u16(),
        i32(),
        u32(),
        i64(),
        u64(),
        i128(),
        u128(),
        isize(),
        usize(),
    ]
    .into()
}

/// All integer domain symbols: the fixed-width set plus `numbers/bigint`.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_core::domains;
///
/// let ints = domains::integer_domains();
/// assert_eq!(ints.len(), domains::fixed_integer_domains().len() + 1);
/// assert!(ints.contains(&domains::bigint()));
/// ```
pub fn integer_domains() -> Vec<Symbol> {
    let mut domains = fixed_integer_domains();
    domains.push(bigint());
    domains
}
