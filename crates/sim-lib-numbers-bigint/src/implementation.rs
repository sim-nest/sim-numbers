#![forbid(unsafe_code)]

//! The `numbers/bigint` library: its domain object, literal and value shapes,
//! and the `Lib` that installs the bigint ops and the promotion into
//! `rational`.

use std::sync::Arc;

use num_bigint::BigInt;
use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Expr, Factory, Lib, LibManifest, LibTarget,
    Linker, NumberDomain, NumberLiteral, Object, PromotionRule, Result, Symbol, Value,
    ValuePromotionRule, Version,
};
use sim_lib_numbers_core::{
    DomainNumberValueShape, NumberDomainTableSpec, NumberLiteralClass, NumberLiteralShape,
    ScalarBinaryOp, ScalarOps, ScalarReductionOp, ScalarUnaryOp, class_surface_or_symbol, domains,
    install_scalar_ops, number_domain_table, shape_surface_or_symbol,
};
use sim_shape::shape_value;

use crate::literal::value_instance_shape_symbol;
use crate::ops::{
    BigIntRuleFn, ValueRuleFn, bigint_add_rule, bigint_cmp_rule, bigint_div_rule, bigint_mul_rule,
    bigint_neg_rule, bigint_pow_rule, bigint_product_rule, bigint_rem_rule, bigint_sub_rule,
    bigint_sum_rule, canonical_bigint, precheck_bigint_decimal_text, promote_bigint_to_rational,
    promote_integer_to_bigint,
};

/// The `numbers/bigint` domain symbol shared by this crate's literals, values,
/// and ops.
pub fn number_domain() -> Symbol {
    domains::bigint()
}

fn literal_class_symbol() -> Symbol {
    domains::literal_class("bigint")
}

pub(crate) fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

fn value_shape_symbol() -> Symbol {
    value_instance_shape_symbol()
}

fn i128_domain() -> Symbol {
    domains::i128()
}

fn u128_domain() -> Symbol {
    domains::u128()
}

fn i64_domain() -> Symbol {
    domains::i64()
}

fn u64_domain() -> Symbol {
    domains::u64()
}

pub(crate) fn rational_domain() -> Symbol {
    domains::rational()
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/bigint number-domain marker; reconstruct by loading the bigint number lib",
    kind = "marker",
    descriptor = "numbers/bigint"
)]
/// The arbitrary-precision signed-integer number domain: parses budgeted
/// integer literals, performs exact arithmetic, and declares the promotion edge
/// into [`rational`](domains::rational).
pub struct BigIntNumberDomain;

impl NumberDomain for BigIntNumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }

    fn parse_priority(&self) -> i32 {
        15
    }

    fn parse_literal(&self, cx: &mut sim_kernel::Cx, text: &str) -> Result<Option<Value>> {
        if text.contains(['.', '/']) || text.is_empty() {
            return Ok(None);
        }
        precheck_bigint_decimal_text("bigint literal", text)?;
        let Some(_) = text.parse::<BigInt>().ok() else {
            return Ok(None);
        };
        cx.factory()
            .number_literal(number_domain(), canonical_bigint(text)?)
            .map(Some)
    }

    fn encode_literal(
        &self,
        cx: &mut sim_kernel::Cx,
        value: Value,
    ) -> Result<Option<NumberLiteral>> {
        match value.object().as_expr(cx)? {
            Expr::Number(number) if number.domain == number_domain() => Ok(Some(number)),
            _ => Ok(None),
        }
    }

    fn promotions(&self) -> Vec<PromotionRule> {
        vec![
            PromotionRule {
                from_domain: i128_domain(),
                to_domain: number_domain(),
                cost: 1,
                convert: promote_integer_to_bigint,
            },
            PromotionRule {
                from_domain: u128_domain(),
                to_domain: number_domain(),
                cost: 1,
                convert: promote_integer_to_bigint,
            },
            PromotionRule {
                from_domain: i64_domain(),
                to_domain: number_domain(),
                cost: 1,
                convert: promote_integer_to_bigint,
            },
            PromotionRule {
                from_domain: u64_domain(),
                to_domain: number_domain(),
                cost: 1,
                convert: promote_integer_to_bigint,
            },
            PromotionRule {
                from_domain: number_domain(),
                to_domain: rational_domain(),
                cost: 1,
                convert: promote_bigint_to_rational,
            },
        ]
    }
}

impl Object for BigIntNumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/bigint>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for BigIntNumberDomain {
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
                "integer",
                "bigint",
                15,
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

/// The library that installs the `numbers/bigint` domain: its literal class and
/// shapes, value shape, exact arithmetic ops, and promotion rules.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy};
/// use sim_lib_numbers_bigint::{BigIntNumbersLib, number_domain};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// cx.load_lib(&BigIntNumbersLib::new()).unwrap();
///
/// let value = cx
///     .factory()
///     .number_literal(number_domain(), "1267650600228229401496703205376".to_owned())
///     .unwrap();
/// let number = cx.number_value_ref(value).unwrap().unwrap();
/// assert_eq!(number.domain, number_domain());
/// ```
pub struct BigIntNumbersLib;

impl BigIntNumbersLib {
    /// Construct the bigint library installer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BigIntNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for BigIntNumbersLib {
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
            "BigIntLiteral",
            [
                "number literal in the numbers/bigint domain",
                "matches Expr::Number where domain == numbers/bigint",
            ],
        ));
        let literal_class = Arc::new(NumberLiteralClass::new(
            literal_class_symbol(),
            number_domain(),
            "integer",
            "bigint",
            literal_instance_shape_symbol(),
            instance_shape.clone(),
        ));
        let value_shape = Arc::new(DomainNumberValueShape::new(
            number_domain(),
            "BigIntValue",
            [
                "number value in the numbers/bigint domain",
                "accepts any NumberValue where domain == numbers/bigint",
            ],
        ));
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(BigIntNumberDomain))
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
        for rule in BigIntNumberDomain.promotions() {
            linker.promotion_rule(rule.clone());
            linker.value_promotion_rule(bigint_value_promotion_rule(&rule));
        }
        let binary = [
            (
                Symbol::qualified("math", "add"),
                bigint_add_rule as BigIntRuleFn,
                crate::ops::bigint_add_value_rule as ValueRuleFn,
            ),
            (
                Symbol::qualified("math", "sub"),
                bigint_sub_rule,
                crate::ops::bigint_sub_value_rule,
            ),
            (
                Symbol::qualified("math", "mul"),
                bigint_mul_rule,
                crate::ops::bigint_mul_value_rule,
            ),
            (
                Symbol::qualified("math", "div"),
                bigint_div_rule,
                crate::ops::bigint_div_value_rule,
            ),
            (
                Symbol::qualified("math", "rem"),
                bigint_rem_rule,
                crate::ops::bigint_rem_value_rule,
            ),
            (
                Symbol::qualified("math", "pow"),
                bigint_pow_rule,
                crate::ops::bigint_pow_value_rule,
            ),
            (
                Symbol::qualified("math", "cmp"),
                bigint_cmp_rule,
                crate::ops::bigint_cmp_value_rule,
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
                operator: Symbol::qualified("math", "neg"),
                literal_cost: 0,
                literal_apply: bigint_neg_rule,
                value_cost: 1,
                value_apply: crate::ops::bigint_neg_value_rule,
            }],
            reduction: vec![
                ScalarReductionOp {
                    operator: Symbol::qualified("math", "sum"),
                    literal_cost: 0,
                    literal_apply: bigint_sum_rule,
                    value_cost: 1,
                    value_apply: crate::ops::bigint_sum_value_rule,
                },
                ScalarReductionOp {
                    operator: Symbol::qualified("math", "product"),
                    literal_cost: 0,
                    literal_apply: bigint_product_rule,
                    value_cost: 1,
                    value_apply: crate::ops::bigint_product_value_rule,
                },
            ],
        };
        install_scalar_ops(linker, &ops);
        Ok(())
    }
}

fn bigint_value_promotion_rule(rule: &PromotionRule) -> ValuePromotionRule {
    let convert = if rule.to_domain == number_domain() {
        promote_integer_value_to_bigint
    } else {
        promote_bigint_value_to_rational
    };
    ValuePromotionRule {
        from_domain: rule.from_domain.clone(),
        to_domain: rule.to_domain.clone(),
        cost: rule.cost,
        convert,
    }
}

fn promote_integer_value_to_bigint(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    let literal = expect_value_literal(cx, value, "integer promotion")?;
    let promoted = promote_integer_to_bigint(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

fn promote_bigint_value_to_rational(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    let literal = expect_value_literal(cx, value, "bigint promotion")?;
    if literal.domain != number_domain() {
        return Err(sim_kernel::Error::Eval(format!(
            "bigint promotion expected number domain {}, found {}",
            number_domain(),
            literal.domain
        )));
    }
    let promoted = promote_bigint_to_rational(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

fn expect_value_literal(
    cx: &mut sim_kernel::Cx,
    value: Value,
    context: &str,
) -> Result<NumberLiteral> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(sim_kernel::Error::Eval(format!(
            "{context} expected a number value"
        )));
    };
    number.literal.ok_or_else(|| {
        sim_kernel::Error::Eval(format!(
            "{context} in {} does not have a canonical literal form",
            number.domain
        ))
    })
}
