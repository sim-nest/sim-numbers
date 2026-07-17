//! The `numbers/complex` domain object and `ComplexNumbersLib`: the domain and
//! operator symbols and the `Lib` that registers the domain, its shapes, value
//! class, ops, and inbound promotion edges.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, ClassRef, Cx, DefaultFactory, Dependency, Export, Expr, Factory, Lib, LibManifest,
    LibTarget, Linker, NumberDomain, NumberLiteral, Object, PromotionRule, Result, Symbol, Value,
    ValuePromotionRule, Version,
};
use sim_lib_numbers_core::{
    NumberLiteralClass, NumberLiteralShape, ScalarBinaryOp, ScalarOps, ScalarReductionOp,
    ScalarUnaryOp, class_surface_or_symbol, domains, install_scalar_ops, shape_surface_or_symbol,
};
use sim_shape::shape_value;

use super::ops::{
    ComplexRuleFn, ValueRuleFn, canonical_complex, parse_complex_literal, register_promotions,
};
use super::surface::NumberValueShape;
use super::value::{build_complex_value_class, complex_value_class_symbol};

/// The `numbers/complex` domain symbol shared by this crate's literals, values,
/// and ops.
pub fn number_domain() -> Symbol {
    domains::complex()
}

/// The symbol of the complex literal class (the `Expr::Number` literal shape in
/// canonical `a+bi` form).
pub fn literal_class_symbol() -> Symbol {
    domains::literal_class("complex")
}

/// The symbol of the shape matching individual complex literals, derived from
/// [`literal_class_symbol`].
pub fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

/// The symbol of the shape matching opaque complex values in the
/// `numbers/complex` domain.
pub fn value_shape_symbol() -> Symbol {
    domains::value_shape(&number_domain())
}

/// The `numbers/f64` domain symbol, source of the f64 -> complex promotion edge.
pub fn f64_domain() -> Symbol {
    domains::f64()
}

/// The `numbers/i64` domain symbol, source of the i64 -> complex promotion edge.
pub fn i64_domain() -> Symbol {
    domains::i64()
}

/// The `numbers/rational` domain symbol, source of the rational -> complex
/// promotion edge.
pub fn rational_domain() -> Symbol {
    domains::rational()
}

/// The `math/add` operator symbol this domain installs a complex rule for.
pub fn add_symbol() -> Symbol {
    Symbol::qualified("math", "add")
}

/// The `math/sub` operator symbol this domain installs a complex rule for.
pub fn sub_symbol() -> Symbol {
    Symbol::qualified("math", "sub")
}

/// The `math/mul` operator symbol this domain installs a complex rule for.
pub fn mul_symbol() -> Symbol {
    Symbol::qualified("math", "mul")
}

/// The `math/div` operator symbol this domain installs a complex rule for.
pub fn div_symbol() -> Symbol {
    Symbol::qualified("math", "div")
}

/// The `math/neg` operator symbol this domain installs a complex rule for.
pub fn neg_symbol() -> Symbol {
    Symbol::qualified("math", "neg")
}

/// The `math/sum` reduction operator symbol this domain installs a complex rule
/// for.
pub fn sum_symbol() -> Symbol {
    Symbol::qualified("math", "sum")
}

/// The `math/product` reduction operator symbol this domain installs a complex
/// rule for.
pub fn product_symbol() -> Symbol {
    Symbol::qualified("math", "product")
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/complex number-domain marker; reconstruct by loading the complex number lib",
    kind = "marker",
    descriptor = "numbers/complex"
)]
/// The complex number domain at the sink of the scalar promotion lattice:
/// parses `a+bi` literals and accepts the widening edges from `f64`, `i64`, and
/// `rational`.
pub struct ComplexNumberDomain;

impl NumberDomain for ComplexNumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }

    fn parse_priority(&self) -> i32 {
        -10
    }

    fn parse_literal(&self, cx: &mut Cx, text: &str) -> Result<Option<Value>> {
        let Some((real, imag)) = parse_complex_literal(text) else {
            return Ok(None);
        };
        cx.factory()
            .number_literal(number_domain(), canonical_complex(real, imag))
            .map(Some)
    }

    fn encode_literal(&self, cx: &mut Cx, value: Value) -> Result<Option<NumberLiteral>> {
        let expr = value.object().as_expr(cx)?;
        match expr {
            Expr::Number(number) if number.domain == self.symbol() => Ok(Some(number)),
            _ => Ok(None),
        }
    }

    fn promotions(&self) -> Vec<PromotionRule> {
        Vec::new()
    }
}

impl Object for ComplexNumberDomain {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("#<number-domain numbers/complex>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ComplexNumberDomain {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        sim_lib_numbers_core::number_domain_class_stub(cx)
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(number_domain()))
    }
    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        let literal_class = class_surface_or_symbol(cx, literal_class_symbol())?;
        let instance_shape = shape_surface_or_symbol(cx, literal_instance_shape_symbol())?;
        let value_shape = shape_surface_or_symbol(cx, value_shape_symbol())?;
        cx.factory().table(vec![
            (Symbol::new("symbol"), cx.factory().symbol(number_domain())?),
            (
                Symbol::new("kind"),
                cx.factory().string("number-domain".to_owned())?,
            ),
            (
                Symbol::new("numeric-family"),
                cx.factory().string("complex".to_owned())?,
            ),
            (
                Symbol::new("canonical-form"),
                cx.factory().string("a+bi".to_owned())?,
            ),
            (
                Symbol::new("parse-priority"),
                cx.factory().string("-10".to_owned())?,
            ),
            (Symbol::new("literal-class"), literal_class),
            (Symbol::new("instance-shape"), instance_shape),
            (Symbol::new("value-shape"), value_shape),
        ])
    }
    fn as_number_domain(&self) -> Option<&dyn NumberDomain> {
        Some(self)
    }
}

/// The library that installs the `numbers/complex` domain: its literal class
/// and shapes, the `ComplexValue` class, the complex ops, and the inbound
/// promotion rules from `f64`, `i64`, and `rational`.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy};
/// use sim_lib_numbers_complex::{ComplexNumbersLib, number_domain, complex_value};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// cx.load_lib(&ComplexNumbersLib::new()).unwrap();
///
/// let value = complex_value(&mut cx, 3.0, -4.0).unwrap();
/// let number = cx.number_value_ref(value).unwrap().unwrap();
/// assert_eq!(number.domain, number_domain());
/// ```
pub struct ComplexNumbersLib;

impl ComplexNumbersLib {
    /// Creates a new `numbers/complex` domain library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ComplexNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for ComplexNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: number_domain(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![
                Export::NumberDomain {
                    symbol: number_domain(),
                    number_domain_id: None,
                },
                Export::Class {
                    symbol: literal_class_symbol(),
                    class_id: None,
                },
                Export::Class {
                    symbol: complex_value_class_symbol(),
                    class_id: None,
                },
                Export::Shape {
                    symbol: literal_instance_shape_symbol(),
                    shape_id: None,
                },
                Export::Shape {
                    symbol: value_shape_symbol(),
                    shape_id: None,
                },
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        let instance_shape = Arc::new(NumberLiteralShape::new(
            number_domain(),
            "ComplexLiteral",
            [
                "number literal in the numbers/complex domain",
                "matches Expr::Number where domain == numbers/complex",
            ],
        ));
        let literal_class = Arc::new(NumberLiteralClass::new(
            literal_class_symbol(),
            number_domain(),
            "complex",
            "a+bi",
            literal_instance_shape_symbol(),
            instance_shape.clone(),
        ));
        let value_shape = Arc::new(NumberValueShape::new(
            number_domain(),
            "ComplexValue",
            [
                "number value in the numbers/complex domain",
                "accepts any NumberValue where domain == numbers/complex",
            ],
        ));
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(ComplexNumberDomain))
                .expect("number domain should be boxable"),
        )?;
        let class_id = linker.class_value(
            literal_class_symbol(),
            DefaultFactory
                .opaque(literal_class.clone())
                .expect("number literal class should be boxable"),
        )?;
        literal_class.set_id(class_id);
        register_complex_value_class(linker)?;
        linker.shape_value(
            literal_instance_shape_symbol(),
            shape_value(literal_instance_shape_symbol(), instance_shape),
        )?;
        linker.shape_value(
            value_shape_symbol(),
            shape_value(value_shape_symbol(), value_shape),
        )?;
        register_promotions(linker);
        for rule in [
            ValuePromotionRule {
                from_domain: f64_domain(),
                to_domain: number_domain(),
                cost: 1,
                convert: super::ops::promote_f64_value_to_complex,
            },
            ValuePromotionRule {
                from_domain: i64_domain(),
                to_domain: number_domain(),
                cost: 1,
                convert: super::ops::promote_i64_value_to_complex,
            },
            ValuePromotionRule {
                from_domain: rational_domain(),
                to_domain: number_domain(),
                cost: 1,
                convert: super::ops::promote_rational_value_to_complex,
            },
        ] {
            linker.value_promotion_rule(rule);
        }
        let binary = [
            (
                add_symbol(),
                super::ops::complex_add_rule as ComplexRuleFn,
                super::ops::complex_add_value_rule as ValueRuleFn,
            ),
            (
                sub_symbol(),
                super::ops::complex_sub_rule,
                super::ops::complex_sub_value_rule,
            ),
            (
                mul_symbol(),
                super::ops::complex_mul_rule,
                super::ops::complex_mul_value_rule,
            ),
            (
                div_symbol(),
                super::ops::complex_div_rule,
                super::ops::complex_div_value_rule,
            ),
        ]
        .into_iter()
        .map(|(operator, literal_apply, value_apply)| ScalarBinaryOp {
            operator,
            literal_cost: 0,
            literal_apply,
            value_cost: 1,
            value_apply,
        })
        .collect();
        let ops = ScalarOps {
            domain: number_domain(),
            binary,
            unary: vec![ScalarUnaryOp {
                operator: neg_symbol(),
                literal_cost: 0,
                literal_apply: super::ops::complex_neg_rule,
                value_cost: 1,
                value_apply: super::ops::complex_neg_value_rule,
            }],
            reduction: vec![
                ScalarReductionOp {
                    operator: sum_symbol(),
                    literal_cost: 0,
                    literal_apply: super::ops::complex_sum_rule,
                    value_cost: 1,
                    value_apply: super::ops::complex_sum_value_rule,
                },
                ScalarReductionOp {
                    operator: product_symbol(),
                    literal_cost: 0,
                    literal_apply: super::ops::complex_product_rule,
                    value_cost: 1,
                    value_apply: super::ops::complex_product_value_rule,
                },
            ],
        };
        install_scalar_ops(linker, &ops);
        Ok(())
    }
}

fn register_complex_value_class(linker: &mut Linker<'_>) -> Result<()> {
    let complex_class = build_complex_value_class();
    let class_id = linker.class_value(
        complex_value_class_symbol(),
        DefaultFactory
            .opaque(complex_class.clone())
            .expect("complex value class should be boxable"),
    )?;
    complex_class.set_id(class_id);
    Ok(())
}

fn install_complex_value_citizen(linker: &mut Linker<'_>) -> Result<()> {
    register_complex_value_class(linker)
}

fn conformance_complex_value_citizen(cx: &mut sim_kernel::Cx) -> Result<()> {
    let value = super::value::complex_value(cx, 1.5, -2.25)?;
    sim_citizen::check_value_fixture(cx, value)
}

sim_citizen::inventory::submit! {
    sim_citizen::CitizenInfo {
        symbol: "numbers/Complex",
        version: 1,
        crate_name: env!("CARGO_PKG_NAME"),
        arity: 2,
        install: install_complex_value_citizen,
        conformance: conformance_complex_value_citizen,
    }
}
