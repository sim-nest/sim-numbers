#![forbid(unsafe_code)]

//! The cross-domain arithmetic library: the `math/*` operator symbols and the
//! `Lib` that installs the generic add/sub/mul/div/compare and reduction entry
//! points over installed numeric domains.
//!
//! # Examples
//!
//! Loading [`NumbersArithmeticLib`] registers the operator aliases on the
//! [`Cx`] registry:
//!
//! ```
//! use std::sync::Arc;
//! use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy, Symbol};
//! use sim_lib_numbers_arith::NumbersArithmeticLib;
//!
//! let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
//! cx.load_lib(&NumbersArithmeticLib::new()).unwrap();
//! assert!(cx.registry().function_by_symbol(&Symbol::new("+")).is_some());
//! assert!(cx.registry().function_by_symbol(&Symbol::new("*")).is_some());
//! ```

use std::sync::Arc;

mod cas_route;
mod function_builders;
mod ops;

use sim_kernel::{
    AbiVersion, Dependency, Export, Lib, LibManifest, LibTarget, Linker, Result, Symbol, Version,
};
use sim_lib_numbers_core::domains;

use self::function_builders::{
    AliasCase, build_alias_function, build_binary_function, build_cmp_function,
    build_reduction_function, build_unary_function,
};

/// The `numbers/arith` library identity symbol.
pub fn lib_symbol() -> Symbol {
    domains::arith()
}
/// The `math/add` symbol: the cross-domain addition entry point.
///
/// # Examples
///
/// ```
/// use sim_kernel::Symbol;
/// use sim_lib_numbers_arith::add_symbol;
///
/// assert_eq!(add_symbol(), Symbol::qualified("math", "add"));
/// ```
pub fn add_symbol() -> Symbol {
    Symbol::qualified("math", "add")
}
/// The `math/sub` symbol: the cross-domain subtraction entry point.
pub fn sub_symbol() -> Symbol {
    Symbol::qualified("math", "sub")
}
/// The `math/mul` symbol: the cross-domain multiplication entry point.
pub fn mul_symbol() -> Symbol {
    Symbol::qualified("math", "mul")
}
/// The `math/div` symbol: the cross-domain division entry point.
pub fn div_symbol() -> Symbol {
    Symbol::qualified("math", "div")
}
/// The `math/rem` symbol: the cross-domain remainder entry point.
pub fn rem_symbol() -> Symbol {
    Symbol::qualified("math", "rem")
}
/// The `math/pow` symbol: the cross-domain exponentiation entry point.
pub fn pow_symbol() -> Symbol {
    Symbol::qualified("math", "pow")
}
/// The `math/cmp` symbol: the cross-domain comparison entry point.
pub fn cmp_symbol() -> Symbol {
    Symbol::qualified("math", "cmp")
}
/// The `math/neg` symbol: the cross-domain unary negation entry point.
pub fn neg_symbol() -> Symbol {
    Symbol::qualified("math", "neg")
}
/// The `math/sum` symbol: the variadic additive reduction entry point.
pub fn sum_symbol() -> Symbol {
    Symbol::qualified("math", "sum")
}
/// The `math/product` symbol: the variadic multiplicative reduction entry point.
pub fn product_symbol() -> Symbol {
    Symbol::qualified("math", "product")
}

#[derive(Clone, Copy)]
enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

impl BinaryOpKind {
    fn symbol(self) -> Symbol {
        match self {
            Self::Add => add_symbol(),
            Self::Sub => sub_symbol(),
            Self::Mul => mul_symbol(),
            Self::Div => div_symbol(),
            Self::Rem => rem_symbol(),
            Self::Pow => pow_symbol(),
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Rem => "rem",
            Self::Pow => "pow",
        }
    }
}

#[derive(Clone, Copy)]
enum UnaryOpKind {
    Neg,
}

impl UnaryOpKind {
    fn symbol(self) -> Symbol {
        match self {
            Self::Neg => neg_symbol(),
        }
    }
}

#[derive(Clone, Copy)]
enum ReductionOpKind {
    Sum,
    Product,
}

impl ReductionOpKind {
    fn symbol(self) -> Symbol {
        match self {
            Self::Sum => sum_symbol(),
            Self::Product => product_symbol(),
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Product => "product",
        }
    }
}

/// The cross-domain arithmetic library.
///
/// Loading this [`Lib`] installs the `math/*` operator functions (`add`, `sub`,
/// `mul`, `div`, `rem`, `pow`, `cmp`, `neg`) and the `sum`/`product`
/// reductions, plus their `+ - * / % ^` aliases. Each entry point coerces
/// mixed-domain operands through the promotion lattice and routes symbolic
/// arithmetic inputs to the CAS when it is loaded.
///
/// # Examples
///
/// ```
/// use sim_kernel::Lib;
/// use sim_lib_numbers_arith::{lib_symbol, NumbersArithmeticLib};
///
/// let manifest = NumbersArithmeticLib::new().manifest();
/// assert_eq!(manifest.id, lib_symbol());
/// // The `math/*` operators and their `+ - * / % ^` aliases are all exported.
/// assert_eq!(manifest.exports.len(), 16);
/// ```
pub struct NumbersArithmeticLib;

impl NumbersArithmeticLib {
    /// Construct the arithmetic library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for NumbersArithmeticLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for NumbersArithmeticLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: lib_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![
                Export::Function {
                    symbol: add_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: sub_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: mul_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: div_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: rem_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: pow_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: cmp_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: neg_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: sum_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: product_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: Symbol::new("+"),
                    function_id: None,
                },
                Export::Function {
                    symbol: Symbol::new("-"),
                    function_id: None,
                },
                Export::Function {
                    symbol: Symbol::new("*"),
                    function_id: None,
                },
                Export::Function {
                    symbol: Symbol::new("/"),
                    function_id: None,
                },
                Export::Function {
                    symbol: Symbol::new("%"),
                    function_id: None,
                },
                Export::Function {
                    symbol: Symbol::new("^"),
                    function_id: None,
                },
            ],
        }
    }

    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        for (symbol, function) in [
            build_binary_function(cx, BinaryOpKind::Add),
            build_binary_function(cx, BinaryOpKind::Sub),
            build_binary_function(cx, BinaryOpKind::Mul),
            build_binary_function(cx, BinaryOpKind::Div),
            build_binary_function(cx, BinaryOpKind::Rem),
            build_binary_function(cx, BinaryOpKind::Pow),
            build_cmp_function(cx),
            build_unary_function(cx, UnaryOpKind::Neg),
            build_reduction_function(cx, ReductionOpKind::Sum),
            build_reduction_function(cx, ReductionOpKind::Product),
            build_alias_function(
                cx,
                Symbol::new("+"),
                &[AliasCase::Binary(BinaryOpKind::Add)],
            ),
            build_alias_function(
                cx,
                Symbol::new("-"),
                &[
                    AliasCase::Unary(UnaryOpKind::Neg),
                    AliasCase::Binary(BinaryOpKind::Sub),
                ],
            ),
            build_alias_function(
                cx,
                Symbol::new("*"),
                &[AliasCase::Binary(BinaryOpKind::Mul)],
            ),
            build_alias_function(
                cx,
                Symbol::new("/"),
                &[AliasCase::Binary(BinaryOpKind::Div)],
            ),
            build_alias_function(
                cx,
                Symbol::new("%"),
                &[AliasCase::Binary(BinaryOpKind::Rem)],
            ),
            build_alias_function(
                cx,
                Symbol::new("^"),
                &[AliasCase::Binary(BinaryOpKind::Pow)],
            ),
        ] {
            let value = cx
                .factory()
                .opaque(Arc::new(function))
                .expect("function should be boxable");
            linker.function_value(symbol, value)?;
        }
        Ok(())
    }
}
