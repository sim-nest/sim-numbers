#![forbid(unsafe_code)]

//! The `numbers/i64` library: its domain object, literal and value shapes, and
//! the `Lib` that installs the i64 ops and promotion rules.

mod literal;
mod ops;

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

use literal::value_instance_shape_symbol;
use ops::{
    I64RuleFn, ValueRuleFn, i64_add_rule, i64_cmp_rule, i64_div_rule, i64_mul_rule, i64_neg_rule,
    i64_pow_rule, i64_product_rule, i64_rem_rule, i64_sub_rule, i64_sum_rule, promote_i64_to_f64,
    promote_i64_to_rational,
};

/// The `numbers/i64` domain symbol shared by this crate's literals, values,
/// and ops.
pub fn number_domain() -> Symbol {
    domains::i64()
}

pub(crate) fn literal_class_symbol() -> Symbol {
    domains::literal_class("i64")
}

pub(crate) fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

pub(crate) fn value_shape_symbol() -> Symbol {
    value_instance_shape_symbol()
}

pub(crate) fn f64_domain() -> Symbol {
    domains::f64()
}

pub(crate) fn rational_domain() -> Symbol {
    domains::rational()
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

fn div_symbol() -> Symbol {
    Symbol::qualified("math", "div")
}

fn rem_symbol() -> Symbol {
    Symbol::qualified("math", "rem")
}

fn pow_symbol() -> Symbol {
    Symbol::qualified("math", "pow")
}

fn cmp_symbol() -> Symbol {
    Symbol::qualified("math", "cmp")
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/i64 number-domain marker; reconstruct by loading the i64 number lib",
    kind = "marker",
    descriptor = "numbers/i64"
)]
/// The exact 64-bit signed-integer number domain: parses integer literals and
/// declares promotion edges into [`f64`](domains::f64) and
/// [`rational`](domains::rational).
pub struct I64NumberDomain;

impl NumberDomain for I64NumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }

    fn parse_priority(&self) -> i32 {
        20
    }

    fn parse_literal(&self, cx: &mut sim_kernel::Cx, text: &str) -> Result<Option<Value>> {
        if text.contains('.') {
            return Ok(None);
        }
        match text.parse::<i64>() {
            Ok(value) => cx
                .factory()
                .number_literal(self.symbol(), value.to_string())
                .map(Some),
            Err(_) => Ok(None),
        }
    }

    fn encode_literal(
        &self,
        cx: &mut sim_kernel::Cx,
        value: Value,
    ) -> Result<Option<NumberLiteral>> {
        let expr = value.object().as_expr(cx)?;
        match expr {
            Expr::Number(number) if number.domain == self.symbol() => Ok(Some(number)),
            _ => Ok(None),
        }
    }

    fn promotions(&self) -> Vec<PromotionRule> {
        vec![
            PromotionRule {
                from_domain: number_domain(),
                to_domain: f64_domain(),
                cost: 1,
                convert: promote_i64_to_f64,
            },
            PromotionRule {
                from_domain: number_domain(),
                to_domain: rational_domain(),
                cost: 1,
                convert: promote_i64_to_rational,
            },
        ]
    }
}

impl Object for I64NumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/i64>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for I64NumberDomain {
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
                "i64",
                20,
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

/// The library that installs the `numbers/i64` domain: its literal class and
/// shapes, value shape, scalar ops, and promotion rules.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy};
/// use sim_lib_numbers_i64::{I64NumbersLib, number_domain};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// cx.load_lib(&I64NumbersLib::new()).unwrap();
///
/// let value = cx.factory().number_literal(number_domain(), "42".to_owned()).unwrap();
/// let number = cx.number_value_ref(value).unwrap().unwrap();
/// assert_eq!(number.domain, number_domain());
/// ```
pub struct I64NumbersLib;

impl I64NumbersLib {
    /// Construct the i64 library installer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for I64NumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for I64NumbersLib {
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
        let domain = I64NumberDomain;
        let instance_shape = Arc::new(NumberLiteralShape::new(
            number_domain(),
            "I64Literal",
            [
                "number literal in the numbers/i64 domain",
                "matches Expr::Number where domain == numbers/i64",
            ],
        ));
        let literal_class = Arc::new(NumberLiteralClass::new(
            literal_class_symbol(),
            number_domain(),
            "integer",
            "i64",
            literal_instance_shape_symbol(),
            instance_shape.clone(),
        ));
        let value_shape = Arc::new(DomainNumberValueShape::new(
            number_domain(),
            "I64Value",
            [
                "number value in the numbers/i64 domain",
                "accepts any NumberValue where domain == numbers/i64",
            ],
        ));
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(domain))
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
        for rule in I64NumberDomain.promotions() {
            if rule.to_domain == f64_domain() {
                linker.promotion_rule(rule.clone());
                linker.value_promotion_rule(ValuePromotionRule {
                    from_domain: rule.from_domain,
                    to_domain: rule.to_domain,
                    cost: rule.cost,
                    convert: ops::promote_i64_value_to_f64,
                });
            } else {
                linker.promotion_rule(rule.clone());
                linker.value_promotion_rule(ValuePromotionRule {
                    from_domain: rule.from_domain,
                    to_domain: rule.to_domain,
                    cost: rule.cost,
                    convert: ops::promote_i64_value_to_rational,
                });
            }
        }
        let binary = [
            (
                add_symbol(),
                i64_add_rule as I64RuleFn,
                ops::i64_add_value_rule as ValueRuleFn,
            ),
            (sub_symbol(), i64_sub_rule, ops::i64_sub_value_rule),
            (mul_symbol(), i64_mul_rule, ops::i64_mul_value_rule),
            (div_symbol(), i64_div_rule, ops::i64_div_value_rule),
            (rem_symbol(), i64_rem_rule, ops::i64_rem_value_rule),
            (pow_symbol(), i64_pow_rule, ops::i64_pow_value_rule),
            (cmp_symbol(), i64_cmp_rule, ops::i64_cmp_value_rule),
        ]
        .into_iter()
        .map(|(operator, literal_apply, value_apply)| ScalarBinaryOp {
            literal_cost: if operator == div_symbol() { 10 } else { 0 },
            operator,
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
                literal_apply: i64_neg_rule,
                value_cost: 1,
                value_apply: ops::i64_neg_value_rule,
            }],
            reduction: vec![
                ScalarReductionOp {
                    operator: Symbol::qualified("math", "sum"),
                    literal_cost: 0,
                    literal_apply: i64_sum_rule,
                    value_cost: 1,
                    value_apply: ops::i64_sum_value_rule,
                },
                ScalarReductionOp {
                    operator: Symbol::qualified("math", "product"),
                    literal_cost: 0,
                    literal_apply: i64_product_rule,
                    value_cost: 1,
                    value_apply: ops::i64_product_value_rule,
                },
            ],
        };
        install_scalar_ops(linker, &ops);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn value_shape_symbol_matches_numbers_core_helper() {
        sim_lib_numbers_core::assert_value_shape_symbol(
            super::number_domain(),
            super::value_shape_symbol(),
        );
    }

    #[test]
    fn div_min_by_neg_one_errors_instead_of_panicking() {
        use std::sync::Arc;

        use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy, NumberLiteral};

        let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
        let left = NumberLiteral {
            domain: super::number_domain(),
            canonical: i64::MIN.to_string(),
        };
        let right = NumberLiteral {
            domain: super::number_domain(),
            canonical: "-1".to_owned(),
        };
        // i64::MIN / -1 overflows i64; with no wider domain loaded this must
        // fail closed rather than panic on the raw `/` operator.
        let result = super::ops::i64_div_rule(&mut cx, left, right);
        assert!(result.is_err());
    }
}
