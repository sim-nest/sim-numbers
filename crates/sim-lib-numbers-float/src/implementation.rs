#![forbid(unsafe_code)]

//! The `numbers/f32` library: its domain object, literal and value shapes, and
//! the `Lib` that installs the f32 ops and the promotion into `f64`.

use std::sync::Arc;

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
    F32RuleFn, ValueRuleFn, canonical_f32, f32_add_rule, f32_div_rule, f32_mul_rule, f32_neg_rule,
    f32_product_rule, f32_sub_rule, f32_sum_rule,
};

/// The `numbers/f32` domain symbol shared by this crate's literals, values,
/// and ops.
pub fn number_domain() -> Symbol {
    domains::f32()
}

fn literal_class_symbol() -> Symbol {
    domains::literal_class("f32")
}

pub(crate) fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

fn value_shape_symbol() -> Symbol {
    value_instance_shape_symbol()
}

pub(crate) fn f64_domain() -> Symbol {
    domains::f64()
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/f32 number-domain marker; reconstruct by loading the float number lib",
    kind = "marker"
)]
/// The single-precision 32-bit floating-point number domain: parses decimal
/// literals and declares the widening promotion edge into [`f64`](domains::f64).
pub struct F32NumberDomain;

impl NumberDomain for F32NumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }

    fn parse_priority(&self) -> i32 {
        -1
    }

    fn parse_literal(&self, cx: &mut sim_kernel::Cx, text: &str) -> Result<Option<Value>> {
        if text.parse::<f32>().is_err() {
            return Ok(None);
        }
        cx.factory()
            .number_literal(self.symbol(), canonical_f32(text))
            .map(Some)
    }

    fn encode_literal(
        &self,
        cx: &mut sim_kernel::Cx,
        value: Value,
    ) -> Result<Option<NumberLiteral>> {
        match value.object().as_expr(cx)? {
            Expr::Number(number) if number.domain == self.symbol() => Ok(Some(number)),
            _ => Ok(None),
        }
    }

    fn promotions(&self) -> Vec<PromotionRule> {
        vec![PromotionRule {
            from_domain: number_domain(),
            to_domain: f64_domain(),
            cost: 1,
            convert: crate::ops::promote_f32_to_f64,
        }]
    }
}

impl Object for F32NumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/f32>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for F32NumberDomain {
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
                "real",
                "f32",
                -1,
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

/// The library that installs the `numbers/f32` domain: its literal class and
/// shapes, value shape, scalar ops, and promotion rules.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy};
/// use sim_lib_numbers_float::{F32NumbersLib, number_domain};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// cx.load_lib(&F32NumbersLib::new()).unwrap();
///
/// let value = cx.factory().number_literal(number_domain(), "1.5".to_owned()).unwrap();
/// let number = cx.number_value_ref(value).unwrap().unwrap();
/// assert_eq!(number.domain, number_domain());
/// ```
pub struct F32NumbersLib;

impl F32NumbersLib {
    /// Construct the f32 library installer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for F32NumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for F32NumbersLib {
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
            "F32Literal",
            [
                "number literal in the numbers/f32 domain",
                "matches Expr::Number where domain == numbers/f32",
            ],
        ));
        let literal_class = Arc::new(NumberLiteralClass::new(
            literal_class_symbol(),
            number_domain(),
            "real",
            "f32",
            literal_instance_shape_symbol(),
            instance_shape.clone(),
        ));
        let value_shape = Arc::new(DomainNumberValueShape::new(
            number_domain(),
            "F32Value",
            [
                "number value in the numbers/f32 domain",
                "accepts any NumberValue where domain == numbers/f32",
            ],
        ));
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(F32NumberDomain))
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
        for rule in F32NumberDomain.promotions() {
            linker.promotion_rule(rule.clone());
            linker.value_promotion_rule(ValuePromotionRule {
                from_domain: rule.from_domain,
                to_domain: rule.to_domain,
                cost: rule.cost,
                convert: crate::ops::promote_f32_value_to_f64,
            });
        }
        let binary = [
            (
                Symbol::qualified("math", "add"),
                f32_add_rule as F32RuleFn,
                crate::ops::f32_add_value_rule as ValueRuleFn,
            ),
            (
                Symbol::qualified("math", "sub"),
                f32_sub_rule,
                crate::ops::f32_sub_value_rule,
            ),
            (
                Symbol::qualified("math", "mul"),
                f32_mul_rule,
                crate::ops::f32_mul_value_rule,
            ),
            (
                Symbol::qualified("math", "div"),
                f32_div_rule,
                crate::ops::f32_div_value_rule,
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
                literal_apply: f32_neg_rule,
                value_cost: 1,
                value_apply: crate::ops::f32_neg_value_rule,
            }],
            reduction: vec![
                ScalarReductionOp {
                    operator: Symbol::qualified("math", "sum"),
                    literal_cost: 0,
                    literal_apply: f32_sum_rule,
                    value_cost: 1,
                    value_apply: crate::ops::f32_sum_value_rule,
                },
                ScalarReductionOp {
                    operator: Symbol::qualified("math", "product"),
                    literal_cost: 0,
                    literal_apply: f32_product_rule,
                    value_cost: 1,
                    value_apply: crate::ops::f32_product_value_rule,
                },
            ],
        };
        install_scalar_ops(linker, &ops);
        Ok(())
    }
}
