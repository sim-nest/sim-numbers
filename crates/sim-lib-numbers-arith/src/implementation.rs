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

use sim_kernel::{
    AbiVersion, Cx, Dependency, Error, Export, Lib, LibManifest, LibTarget, Linker, PreparedArgs,
    Result, Symbol, Value, Version,
};
use sim_lib_numbers_core::domains;
use sim_shape::{AnyShape, Bindings, CaptureShape, FunctionCase, FunctionObject, ListShape};

use self::cas_route::{coerce_arith_argument, route_via_cas, should_route_via_cas};

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

fn number_shape() -> Arc<AnyShape> {
    Arc::new(AnyShape)
}

fn captured_number(name: &str) -> Arc<CaptureShape> {
    Arc::new(CaptureShape::new(Symbol::new(name), number_shape()))
}

fn build_binary_function(
    cx: &mut sim_kernel::LoadCx,
    kind: BinaryOpKind,
) -> (Symbol, FunctionObject) {
    build_binary_named_function(cx, kind.symbol(), kind)
}

fn build_binary_named_function(
    cx: &mut sim_kernel::LoadCx,
    symbol: Symbol,
    kind: BinaryOpKind,
) -> (Symbol, FunctionObject) {
    let case = FunctionCase {
        id: cx.fresh_case_id(),
        name: Symbol::qualified(symbol.to_string(), "numbers"),
        args: Arc::new(ListShape::with_rest(
            vec![captured_number("left"), captured_number("right")],
            captured_number("rest"),
        )),
        result: Some(number_shape()),
        demand: vec![sim_kernel::Demand::Value, sim_kernel::Demand::Value],
        priority: 10,
        implementation: match kind {
            BinaryOpKind::Add => add_impl,
            BinaryOpKind::Sub => sub_impl,
            BinaryOpKind::Mul => mul_impl,
            BinaryOpKind::Div => div_impl,
            BinaryOpKind::Rem => rem_impl,
            BinaryOpKind::Pow => pow_impl,
        },
    };
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, vec![case]),
    )
}

fn build_cmp_function(cx: &mut sim_kernel::LoadCx) -> (Symbol, FunctionObject) {
    let symbol = cmp_symbol();
    let case = FunctionCase {
        id: cx.fresh_case_id(),
        name: Symbol::qualified(symbol.to_string(), "numbers"),
        args: Arc::new(ListShape::new(vec![
            captured_number("left"),
            captured_number("right"),
        ])),
        result: Some(number_shape()),
        demand: vec![sim_kernel::Demand::Value, sim_kernel::Demand::Value],
        priority: 10,
        implementation: cmp_impl,
    };
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, vec![case]),
    )
}

fn build_unary_function(
    cx: &mut sim_kernel::LoadCx,
    kind: UnaryOpKind,
) -> (Symbol, FunctionObject) {
    build_unary_named_function(cx, kind.symbol(), kind)
}

fn build_unary_named_function(
    cx: &mut sim_kernel::LoadCx,
    symbol: Symbol,
    _kind: UnaryOpKind,
) -> (Symbol, FunctionObject) {
    let case = FunctionCase {
        id: cx.fresh_case_id(),
        name: Symbol::qualified(symbol.to_string(), "numbers"),
        args: Arc::new(ListShape::new(vec![captured_number("value")])),
        result: Some(number_shape()),
        demand: vec![sim_kernel::Demand::Value],
        priority: 10,
        implementation: neg_impl,
    };
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, vec![case]),
    )
}

#[derive(Clone, Copy)]
enum AliasCase {
    Binary(BinaryOpKind),
    Unary(UnaryOpKind),
}

fn build_alias_function(
    cx: &mut sim_kernel::LoadCx,
    symbol: Symbol,
    cases: &[AliasCase],
) -> (Symbol, FunctionObject) {
    let mut built = Vec::with_capacity(cases.len());
    for case in cases {
        built.push(match case {
            AliasCase::Binary(kind) => build_binary_named_function(cx, symbol.clone(), *kind).1,
            AliasCase::Unary(kind) => build_unary_named_function(cx, symbol.clone(), *kind).1,
        });
    }
    let cases = built
        .into_iter()
        .flat_map(|function| function.cases)
        .collect();
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, cases),
    )
}

fn build_reduction_function(
    cx: &mut sim_kernel::LoadCx,
    kind: ReductionOpKind,
) -> (Symbol, FunctionObject) {
    let symbol = kind.symbol();
    let case = FunctionCase {
        id: cx.fresh_case_id(),
        name: Symbol::qualified(symbol.to_string(), "numbers"),
        args: Arc::new(ListShape::with_rest(
            vec![captured_number("first")],
            captured_number("rest"),
        )),
        result: Some(number_shape()),
        demand: vec![sim_kernel::Demand::Value],
        priority: 10,
        implementation: match kind {
            ReductionOpKind::Sum => sum_impl,
            ReductionOpKind::Product => product_impl,
        },
    };
    (
        symbol.clone(),
        FunctionObject::new(cx.fresh_function_id(), symbol, vec![case]),
    )
}

fn add_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Add)
}

fn sub_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Sub)
}

fn mul_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Mul)
}

fn div_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Div)
}

fn rem_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Rem)
}

fn pow_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    fold_binary_impl(cx, prepared, BinaryOpKind::Pow)
}

fn cmp_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let [left, right] = prepared.values() else {
        return Err(Error::Eval("cmp expects exactly two arguments".to_owned()));
    };
    let left = coerce_arith_argument(cx, left.clone())?;
    let right = coerce_arith_argument(cx, right.clone())?;
    require_number_arg(cx, left.clone(), cmp_symbol(), 0)?;
    require_number_arg(cx, right.clone(), cmp_symbol(), 1)?;
    cx.apply_value_number_binary_op(&cmp_symbol(), left, right)
}

fn neg_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    let operand = prepared
        .get(0)
        .cloned()
        .ok_or_else(|| Error::Eval("neg expects exactly one argument".to_owned()))?;
    require_number_arg(cx, operand, neg_symbol(), 0)?;
    cx.apply_value_number_unary_op(
        &UnaryOpKind::Neg.symbol(),
        prepared.get(0).cloned().unwrap(),
    )
}

fn sum_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    reduction_impl(cx, prepared, ReductionOpKind::Sum)
}

fn product_impl(cx: &mut Cx, prepared: &PreparedArgs, _bindings: Bindings) -> Result<Value> {
    reduction_impl(cx, prepared, ReductionOpKind::Product)
}

fn fold_binary_impl(cx: &mut Cx, prepared: &PreparedArgs, kind: BinaryOpKind) -> Result<Value> {
    let values = prepared
        .values()
        .iter()
        .map(|value| coerce_arith_argument(cx, value.clone()))
        .collect::<Result<Vec<_>>>()?;
    if should_route_via_cas(cx, &values)?
        && let Some(value) = route_via_cas(cx, kind.symbol(), &values)?
    {
        return Ok(value);
    }
    let [first, second, rest @ ..] = values.as_slice() else {
        return Err(Error::Eval(format!(
            "{} expects at least two arguments",
            kind.display_name()
        )));
    };
    require_number_arg(cx, first.clone(), kind.symbol(), 0)?;
    require_number_arg(cx, second.clone(), kind.symbol(), 1)?;
    let mut acc = cx.apply_value_number_binary_op(&kind.symbol(), first.clone(), second.clone())?;
    for (offset, value) in rest.iter().enumerate() {
        require_number_arg(cx, value.clone(), kind.symbol(), offset + 2)?;
        acc = cx.apply_value_number_binary_op(&kind.symbol(), acc, value.clone())?;
    }
    Ok(acc)
}

fn reduction_impl(cx: &mut Cx, prepared: &PreparedArgs, kind: ReductionOpKind) -> Result<Value> {
    let values = prepared
        .values()
        .iter()
        .map(|value| coerce_arith_argument(cx, value.clone()))
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        return Err(Error::Eval(format!(
            "{} expects at least one argument",
            kind.display_name()
        )));
    }
    let operands = number_args(cx, &values, kind.symbol())?;
    cx.apply_value_number_reduction_op(&kind.symbol(), operands)
}

fn require_number_arg(cx: &mut Cx, value: Value, function: Symbol, index: usize) -> Result<()> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        });
    };
    if cx
        .registry()
        .number_domain_by_symbol(&number.domain)
        .is_none()
    {
        return Err(Error::Eval(format!(
            "{} arg {} uses unloaded number domain {}",
            function, index, number.domain
        )));
    }
    Ok(())
}

fn number_args(cx: &mut Cx, values: &[Value], function: Symbol) -> Result<Vec<Value>> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            require_number_arg(cx, value.clone(), function.clone(), index)?;
            Ok(value.clone())
        })
        .collect()
}
