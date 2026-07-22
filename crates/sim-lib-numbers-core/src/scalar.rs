//! Scalar-domain spec, literal matcher, and the shared op-loop installer.
//!
//! Each scalar number domain crate repeats the same `load()` registration loop
//! (binary/unary/reduction ops, each in both literal and value form). This is
//! the shared installer: a domain crate describes its ops as data
//! ([`ScalarOps`]) and calls [`install_scalar_ops`].

use sim_kernel::{
    Cx, Expr, Factory, Linker, NumberBinaryOp, NumberLiteral, NumberReductionOp, NumberUnaryOp,
    Symbol, Value, ValueNumberBinaryOp, ValueNumberReductionOp, ValueNumberUnaryOp,
};

/// The `ObjectCompat::class` stub every number-domain object returns: the
/// registered `core/NumberDomain` class, or a fresh stub for it.
///
/// Scalar domain implementations delegate here so the runtime sees one shared
/// number-domain class shape across every concrete scalar domain.
pub fn number_domain_class_stub(cx: &mut Cx) -> sim_kernel::Result<sim_kernel::ClassRef> {
    if let Some(value) = cx
        .registry()
        .class_by_symbol(&Symbol::qualified("core", "NumberDomain"))
    {
        return Ok(value.clone());
    }
    sim_kernel::DefaultFactory.class_stub(
        sim_kernel::CORE_NUMBER_DOMAIN_CLASS_ID,
        Symbol::qualified("core", "NumberDomain"),
    )
}

use crate::domains;

/// Tests whether an expression is a literal in some scalar domain.
pub trait ScalarLiteralMatcher {
    /// Whether `expr` is a number literal this matcher accepts.
    fn matches_expr(&self, expr: &Expr) -> bool;
}

/// A matcher accepting `Expr::Number` literals in exactly one domain.
///
/// # Examples
///
/// ```
/// use sim_kernel::{Expr, NumberLiteral};
/// use sim_lib_numbers_core::{DomainLiteralMatcher, ScalarLiteralMatcher, domains};
///
/// let matcher = DomainLiteralMatcher::new(domains::i64());
/// let lit = Expr::Number(NumberLiteral {
///     domain: domains::i64(),
///     canonical: "42".to_owned(),
/// });
/// assert!(matcher.matches_expr(&lit));
/// assert!(!matcher.matches_expr(&Expr::String("42".to_owned())));
/// ```
pub struct DomainLiteralMatcher {
    domain: Symbol,
}

impl DomainLiteralMatcher {
    /// Build a matcher accepting only literals in `domain`.
    pub fn new(domain: Symbol) -> Self {
        Self { domain }
    }

    /// The domain this matcher accepts.
    pub fn domain(&self) -> &Symbol {
        &self.domain
    }
}

impl ScalarLiteralMatcher for DomainLiteralMatcher {
    fn matches_expr(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Number(number) if number.domain == self.domain)
    }
}

/// Static identity of a scalar number domain (data only).
///
/// A concrete domain crate fills this in once and derives its stable
/// literal-class and instance-shape symbols from it, rather than spelling them
/// out by hand.
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_core::{ScalarDomainSpec, domains};
///
/// let spec = ScalarDomainSpec {
///     domain: domains::i64(),
///     numeric_family: "integer",
///     canonical_form: "i64",
///     parse_priority: 20,
/// };
/// assert_eq!(spec.literal_class_symbol(), domains::literal_class("i64"));
/// ```
pub struct ScalarDomainSpec {
    /// The domain symbol, e.g. `numbers/i64`.
    pub domain: Symbol,
    /// The numeric family label, e.g. `"integer"`.
    pub numeric_family: &'static str,
    /// The canonical form label, e.g. `"i64"`.
    pub canonical_form: &'static str,
    /// The literal parse priority.
    pub parse_priority: i32,
}

impl ScalarDomainSpec {
    /// A literal matcher for this domain.
    pub fn matcher(&self) -> DomainLiteralMatcher {
        DomainLiteralMatcher::new(self.domain.clone())
    }

    /// The literal class symbol, e.g. `numbers/i64-literal`.
    pub fn literal_class_symbol(&self) -> Symbol {
        domains::literal_class(self.canonical_form)
    }

    /// The literal instance-shape symbol, e.g. `numbers/i64-literal/instance-shape`.
    pub fn literal_instance_shape_symbol(&self) -> Symbol {
        Symbol::qualified(self.literal_class_symbol().to_string(), "instance-shape")
    }
}

/// One binary op in both literal and value form.
pub struct ScalarBinaryOp {
    /// The operator symbol this op implements (e.g. `+`).
    pub operator: Symbol,
    /// Dispatch cost of the literal (parsed-form) implementation.
    pub literal_cost: u16,
    /// The literal-form implementation over two same-domain number literals.
    pub literal_apply: fn(&mut Cx, NumberLiteral, NumberLiteral) -> sim_kernel::Result<Value>,
    /// Dispatch cost of the value (opaque-object) implementation.
    pub value_cost: u16,
    /// The value-form implementation over two same-domain number values.
    pub value_apply: fn(&mut Cx, Value, Value) -> sim_kernel::Result<Value>,
}

/// One unary op in both literal and value form.
pub struct ScalarUnaryOp {
    /// The operator symbol this op implements (e.g. `neg`).
    pub operator: Symbol,
    /// Dispatch cost of the literal (parsed-form) implementation.
    pub literal_cost: u16,
    /// The literal-form implementation over one number literal.
    pub literal_apply: fn(&mut Cx, NumberLiteral) -> sim_kernel::Result<Value>,
    /// Dispatch cost of the value (opaque-object) implementation.
    pub value_cost: u16,
    /// The value-form implementation over one number value.
    pub value_apply: fn(&mut Cx, Value) -> sim_kernel::Result<Value>,
}

/// One reduction op in both literal and value form.
pub struct ScalarReductionOp {
    /// The operator symbol this op implements (e.g. `sum`).
    pub operator: Symbol,
    /// Dispatch cost of the literal (parsed-form) implementation.
    pub literal_cost: u16,
    /// The literal-form implementation over a vector of number literals.
    pub literal_apply: fn(&mut Cx, Vec<NumberLiteral>) -> sim_kernel::Result<Value>,
    /// Dispatch cost of the value (opaque-object) implementation.
    pub value_cost: u16,
    /// The value-form implementation over a vector of number values.
    pub value_apply: fn(&mut Cx, Vec<Value>) -> sim_kernel::Result<Value>,
}

/// The full op set for one scalar domain.
pub struct ScalarOps {
    /// The domain all ops in this set operate within.
    pub domain: Symbol,
    /// The binary ops to register for this domain.
    pub binary: Vec<ScalarBinaryOp>,
    /// The unary ops to register for this domain.
    pub unary: Vec<ScalarUnaryOp>,
    /// The reduction ops to register for this domain.
    pub reduction: Vec<ScalarReductionOp>,
}

/// Register every op in `ops` (literal and value form) against `linker`.
pub fn install_scalar_ops(linker: &mut Linker<'_>, ops: &ScalarOps) {
    for op in &ops.binary {
        linker.number_binary_op(NumberBinaryOp {
            operator: op.operator.clone(),
            left_domain: ops.domain.clone(),
            right_domain: ops.domain.clone(),
            cost: op.literal_cost,
            apply: op.literal_apply,
        });
        linker.value_number_binary_op(ValueNumberBinaryOp {
            operator: op.operator.clone(),
            left_domain: ops.domain.clone(),
            right_domain: ops.domain.clone(),
            cost: op.value_cost,
            apply: op.value_apply,
        });
    }
    for op in &ops.unary {
        linker.number_unary_op(NumberUnaryOp {
            operator: op.operator.clone(),
            operand_domain: ops.domain.clone(),
            cost: op.literal_cost,
            apply: op.literal_apply,
        });
        linker.value_number_unary_op(ValueNumberUnaryOp {
            operator: op.operator.clone(),
            operand_domain: ops.domain.clone(),
            cost: op.value_cost,
            apply: op.value_apply,
        });
    }
    for op in &ops.reduction {
        linker.number_reduction_op(NumberReductionOp {
            operator: op.operator.clone(),
            operand_domain: ops.domain.clone(),
            cost: op.literal_cost,
            apply: op.literal_apply,
        });
        linker.value_number_reduction_op(ValueNumberReductionOp {
            operator: op.operator.clone(),
            operand_domain: ops.domain.clone(),
            cost: op.value_cost,
            apply: op.value_apply,
        });
    }
}
