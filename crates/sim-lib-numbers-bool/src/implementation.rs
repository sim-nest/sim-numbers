#![forbid(unsafe_code)]

//! The `numbers/bool` library: its domain object, literal and value shapes, and
//! the `Lib` that installs the bool ops and promotions into the integer and
//! float domains.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Expr, Factory, Lib, LibManifest, LibTarget,
    Linker, NumberDomain, NumberLiteral, Object, PromotionRule, Result, Symbol, Value,
    ValuePromotionRule, Version,
};
use sim_lib_numbers_core::{
    DomainNumberValueShape, NumberDomainTableSpec, ScalarBinaryOp, ScalarOps, domains,
    install_scalar_ops, number_domain_table,
};
use sim_shape::shape_value;

use crate::literal::{
    NumberLiteralClass, NumberLiteralShape, class_surface_or_symbol, shape_surface_or_symbol,
    value_instance_shape_symbol,
};

/// The `numbers/bool` domain symbol shared by this crate's literals, values,
/// and ops.
pub fn number_domain() -> Symbol {
    domains::bool()
}

fn literal_class_symbol() -> Symbol {
    domains::literal_class("bool")
}

pub(crate) fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

fn value_shape_symbol() -> Symbol {
    value_instance_shape_symbol()
}

fn u8_domain() -> Symbol {
    domains::u8()
}

fn i64_domain() -> Symbol {
    domains::i64()
}

fn f64_domain() -> Symbol {
    domains::f64()
}

fn add_symbol() -> Symbol {
    Symbol::qualified("math", "add")
}

fn sub_symbol() -> Symbol {
    Symbol::qualified("math", "sub")
}

fn mul_symbol() -> Symbol {
    Symbol::qualified("math", "mul")
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/bool number-domain marker; reconstruct by loading the bool number lib",
    kind = "marker"
)]
/// The boolean number domain at the base of the promotion lattice: parses
/// `true`/`false` literals and declares the widening edges into the integer
/// and float domains.
pub struct BoolNumberDomain;

impl NumberDomain for BoolNumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }

    fn parse_priority(&self) -> i32 {
        -10
    }

    fn parse_literal(&self, cx: &mut sim_kernel::Cx, text: &str) -> Result<Option<Value>> {
        match text {
            "true" => cx
                .factory()
                .number_literal(number_domain(), "true".to_owned())
                .map(Some),
            "false" => cx
                .factory()
                .number_literal(number_domain(), "false".to_owned())
                .map(Some),
            _ => Ok(None),
        }
    }

    fn encode_literal(
        &self,
        cx: &mut sim_kernel::Cx,
        value: Value,
    ) -> Result<Option<NumberLiteral>> {
        match value.object().as_expr(cx)? {
            Expr::Number(number) if number.domain == number_domain() => Ok(Some(number)),
            Expr::Bool(value) => Ok(Some(NumberLiteral {
                domain: number_domain(),
                canonical: if value { "true" } else { "false" }.to_owned(),
            })),
            _ => Ok(None),
        }
    }

    fn promotions(&self) -> Vec<PromotionRule> {
        vec![
            PromotionRule {
                from_domain: number_domain(),
                to_domain: u8_domain(),
                cost: 1,
                convert: promote_bool_to_u8,
            },
            PromotionRule {
                from_domain: number_domain(),
                to_domain: i64_domain(),
                cost: 2,
                convert: promote_bool_to_i64,
            },
            PromotionRule {
                from_domain: number_domain(),
                to_domain: f64_domain(),
                cost: 4,
                convert: promote_bool_to_f64,
            },
        ]
    }
}

impl Object for BoolNumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/bool>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for BoolNumberDomain {
    fn class(&self, cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        sim_lib_numbers_core::number_domain_class_stub(cx)
    }
    fn as_expr(&self, _cx: &mut sim_kernel::Cx) -> Result<Expr> {
        Ok(Expr::Symbol(number_domain()))
    }
    fn as_table(&self, cx: &mut sim_kernel::Cx) -> Result<Value> {
        let literal_class = class_surface_or_symbol(cx, literal_class_symbol())?;
        let instance_shape = shape_surface_or_symbol(cx, literal_instance_shape_symbol())?;
        let value_shape = shape_surface_or_symbol(cx, value_shape_symbol())?;
        number_domain_table(
            cx,
            NumberDomainTableSpec::new(
                number_domain(),
                "boolean",
                "true|false",
                -10,
                literal_class,
                instance_shape,
                value_shape,
            ),
        )
    }
    fn as_number_domain(&self) -> Option<&dyn NumberDomain> {
        Some(self)
    }
}

/// The library that installs the `numbers/bool` domain: its literal class and
/// shapes, value shape, boolean ops, and widening promotion rules.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy};
/// use sim_lib_numbers_bool::{BoolNumbersLib, number_domain};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// cx.load_lib(&BoolNumbersLib::new()).unwrap();
///
/// let value = cx.factory().bool(true).unwrap();
/// let number = cx.number_value_ref(value).unwrap().unwrap();
/// assert_eq!(number.domain, number_domain());
/// ```
pub struct BoolNumbersLib;

impl BoolNumbersLib {
    /// Construct the bool library installer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BoolNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for BoolNumbersLib {
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
            "BoolLiteral",
            [
                "number literal in the numbers/bool domain",
                "matches Expr::Bool or Expr::Number where domain == numbers/bool",
            ],
        ));
        let literal_class = Arc::new(NumberLiteralClass::new(
            literal_class_symbol(),
            number_domain(),
            "boolean",
            "true|false",
            instance_shape.clone(),
        ));
        let value_shape = Arc::new(DomainNumberValueShape::new(
            number_domain(),
            "BoolValue",
            [
                "number value in the numbers/bool domain",
                "accepts any NumberValue where domain == numbers/bool",
            ],
        ));
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(BoolNumberDomain))
                .expect("number domain should be boxable"),
        )?;
        let class_id = linker.class_value(
            literal_class_symbol(),
            DefaultFactory
                .opaque(literal_class.clone())
                .expect("number literal class should be boxable"),
        )?;
        literal_class.set_id(class_id);
        linker.shape_value(
            literal_instance_shape_symbol(),
            shape_value(literal_instance_shape_symbol(), instance_shape),
        )?;
        linker.shape_value(
            value_shape_symbol(),
            shape_value(value_shape_symbol(), value_shape),
        )?;
        for rule in BoolNumberDomain.promotions() {
            linker.promotion_rule(rule.clone());
            linker.value_promotion_rule(bool_value_promotion_rule(&rule));
        }
        let binary = [
            (
                add_symbol(),
                bool_add_rule as BoolRuleFn,
                bool_add_value_rule as ValueRuleFn,
            ),
            (sub_symbol(), bool_sub_rule, bool_sub_value_rule),
            (mul_symbol(), bool_mul_rule, bool_mul_value_rule),
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
            unary: Vec::new(),
            reduction: Vec::new(),
        };
        install_scalar_ops(linker, &ops);
        Ok(())
    }
}

type BoolRuleFn = fn(&mut sim_kernel::Cx, NumberLiteral, NumberLiteral) -> Result<Value>;
type ValueRuleFn = fn(&mut sim_kernel::Cx, Value, Value) -> Result<Value>;

fn bool_add_rule(
    cx: &mut sim_kernel::Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let out = parse_bool_literal(left, "left")? || parse_bool_literal(right, "right")?;
    cx.factory().bool(out)
}

fn bool_sub_rule(
    cx: &mut sim_kernel::Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let out = parse_bool_literal(left, "left")? ^ parse_bool_literal(right, "right")?;
    cx.factory().bool(out)
}

fn bool_mul_rule(
    cx: &mut sim_kernel::Cx,
    left: NumberLiteral,
    right: NumberLiteral,
) -> Result<Value> {
    let out = parse_bool_literal(left, "left")? && parse_bool_literal(right, "right")?;
    cx.factory().bool(out)
}

fn bool_add_value_rule(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_bool_literal(cx, left, "left")?;
    let right = expect_bool_literal(cx, right, "right")?;
    bool_add_rule(cx, left, right)
}

fn bool_sub_value_rule(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_bool_literal(cx, left, "left")?;
    let right = expect_bool_literal(cx, right, "right")?;
    bool_sub_rule(cx, left, right)
}

fn bool_mul_value_rule(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    let left = expect_bool_literal(cx, left, "left")?;
    let right = expect_bool_literal(cx, right, "right")?;
    bool_mul_rule(cx, left, right)
}

fn parse_bool_literal(number: NumberLiteral, side: &str) -> Result<bool> {
    if number.domain != number_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    match number.canonical.as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(sim_kernel::Error::Eval(format!(
            "{side} operand was not a valid bool literal: {}",
            other
        ))),
    }
}

fn expect_bool_literal(cx: &mut sim_kernel::Cx, value: Value, side: &str) -> Result<NumberLiteral> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found non-number",
            number_domain()
        )));
    };
    if number.domain != number_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "{side} operand expected number domain {}, found {}",
            number_domain(),
            number.domain
        )));
    }
    match number.literal {
        Some(literal) => Ok(literal),
        None => Err(sim_kernel::Error::Eval(format!(
            "{side} operand in {} does not have a canonical literal form",
            number_domain()
        ))),
    }
}

fn bool_value_promotion_rule(rule: &PromotionRule) -> ValuePromotionRule {
    let convert = if rule.to_domain == u8_domain() {
        promote_bool_value_to_u8
    } else if rule.to_domain == i64_domain() {
        promote_bool_value_to_i64
    } else {
        promote_bool_value_to_f64
    };
    ValuePromotionRule {
        from_domain: rule.from_domain.clone(),
        to_domain: rule.to_domain.clone(),
        cost: rule.cost,
        convert,
    }
}

fn promote_bool_value_to_u8(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    let literal = expect_bool_literal(cx, value, "operand")?;
    let promoted = promote_bool_to_u8(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

fn promote_bool_value_to_i64(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    let literal = expect_bool_literal(cx, value, "operand")?;
    let promoted = promote_bool_to_i64(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

fn promote_bool_value_to_f64(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    let literal = expect_bool_literal(cx, value, "operand")?;
    let promoted = promote_bool_to_f64(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

fn promote_bool_to_u8(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    Ok(NumberLiteral {
        domain: u8_domain(),
        canonical: if parse_bool_literal(number, "operand")? {
            "1"
        } else {
            "0"
        }
        .to_owned(),
    })
}

fn promote_bool_to_i64(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    Ok(NumberLiteral {
        domain: i64_domain(),
        canonical: if parse_bool_literal(number, "operand")? {
            "1"
        } else {
            "0"
        }
        .to_owned(),
    })
}

fn promote_bool_to_f64(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    Ok(NumberLiteral {
        domain: f64_domain(),
        canonical: if parse_bool_literal(number, "operand")? {
            "1"
        } else {
            "0"
        }
        .to_owned(),
    })
}
