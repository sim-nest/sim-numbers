//! The `numbers/rational` domain object and `RationalNumbersLib`: the domain
//! and operator symbols and the `Lib` that registers the domain, its shapes,
//! value class, ops, and promotions to and from the integer and `f64` domains.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, ClassId, DefaultFactory, Dependency, Export, Expr, Factory, Lib, LibManifest,
    LibTarget, Linker, NumberDomain, NumberLiteral, Object, Result, Symbol, Value,
    ValuePromotionRule, Version,
};
use sim_lib_numbers_core::{
    DomainNumberValueShape, NumberDomainTableSpec, NumberLiteralClass, NumberLiteralShape,
    ScalarBinaryOp, ScalarOps, ScalarReductionOp, ScalarUnaryOp, class_surface_or_symbol, domains,
    install_scalar_ops, number_domain_table, shape_surface_or_symbol,
};
use sim_shape::shape_value;

use super::integer::compact_canonical;
use super::ops::{
    RationalRuleFn, ValueRuleFn, promote_f64_literal_to_rational, promote_f64_value_to_rational,
    promote_integer_literal_to_rational, promote_integer_value_to_rational,
    promote_rational_literal_to_f64, promote_rational_value_to_f64, rational_add_rule,
    rational_add_value_rule, rational_div_rule, rational_div_value_rule, rational_mul_rule,
    rational_mul_value_rule, rational_neg_rule, rational_neg_value_rule, rational_pow_rule,
    rational_pow_value_rule, rational_product_rule, rational_product_value_rule, rational_sub_rule,
    rational_sub_value_rule, rational_sum_rule, rational_sum_value_rule,
};
use super::value::{Rational, RationalValueClass};

/// The `numbers/rational` domain symbol shared by this crate's literals,
/// values, and ops.
pub fn number_domain() -> Symbol {
    domains::rational()
}

/// The symbol of the rational literal class (the `Expr::Number` literal shape
/// in canonical `num/den` form).
pub fn literal_class_symbol() -> Symbol {
    domains::literal_class("rational")
}

/// The symbol of the shape matching individual rational literals, derived from
/// [`literal_class_symbol`].
pub fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

/// The symbol of the `numbers/rational` value class, used to register the class
/// and to tag the extension encoding of non-compact rational values.
pub fn rational_value_class_symbol() -> Symbol {
    domains::rational_value_class()
}

pub(crate) fn value_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&number_domain())
}

/// The `numbers/f64` domain symbol, the far end of the rational <-> f64
/// promotion edges.
pub fn f64_domain() -> Symbol {
    domains::f64()
}

/// The `math/add` operator symbol this domain installs a rational rule for.
pub fn add_symbol() -> Symbol {
    Symbol::qualified("math", "add")
}

/// The `math/sub` operator symbol this domain installs a rational rule for.
pub fn sub_symbol() -> Symbol {
    Symbol::qualified("math", "sub")
}

/// The `math/mul` operator symbol this domain installs a rational rule for.
pub fn mul_symbol() -> Symbol {
    Symbol::qualified("math", "mul")
}

/// The `math/div` operator symbol this domain installs a rational rule for.
pub fn div_symbol() -> Symbol {
    Symbol::qualified("math", "div")
}

/// The `math/pow` operator symbol this domain installs a rational rule for.
pub fn pow_symbol() -> Symbol {
    Symbol::qualified("math", "pow")
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/rational number-domain marker; reconstruct by loading the rational number lib",
    kind = "marker"
)]
/// The exact rational number domain: parses `num/den` literals and declares the
/// promotion edges to and from the integer and `f64` domains.
pub struct RationalNumberDomain;

impl NumberDomain for RationalNumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }

    fn parse_literal(&self, cx: &mut sim_kernel::Cx, text: &str) -> Result<Option<Value>> {
        let Some((numerator, denominator)) = super::ops::parse_rational_parts(text) else {
            return Ok(None);
        };
        cx.factory()
            .number_literal(number_domain(), format!("{numerator}/{denominator}"))
            .map(Some)
    }

    fn encode_literal(
        &self,
        cx: &mut sim_kernel::Cx,
        value: Value,
    ) -> Result<Option<NumberLiteral>> {
        match value.object().as_expr(cx)? {
            Expr::Number(number) if number.domain == number_domain() => Ok(Some(number)),
            _ => Ok(value
                .object()
                .downcast_ref::<Rational>()
                .cloned()
                .map(|rational| compact_canonical(cx, &rational.num, &rational.den))
                .transpose()?
                .flatten()
                .map(|canonical| NumberLiteral {
                    domain: number_domain(),
                    canonical,
                })),
        }
    }
}

impl Object for RationalNumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/rational>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for RationalNumberDomain {
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
                "rational",
                "numerator/denominator",
                0,
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

/// The library that installs the `numbers/rational` domain: its literal class
/// and shapes, the `Rational` value class, the reduced arithmetic ops, and the
/// promotion rules to and from the integer and `f64` domains.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy};
/// use sim_lib_numbers_rational::{RationalNumbersLib, number_domain};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// cx.load_lib(&RationalNumbersLib::new()).unwrap();
///
/// // Rational literals parse into the domain even before any base scalar lib
/// // is loaded; arithmetic over them additionally needs the integer libs.
/// let value = cx
///     .factory()
///     .number_literal(number_domain(), "1/2".to_owned())
///     .unwrap();
/// let number = cx.number_value_ref(value).unwrap().unwrap();
/// assert_eq!(number.domain, number_domain());
/// ```
pub struct RationalNumbersLib;

impl RationalNumbersLib {
    /// Creates a new `numbers/rational` domain library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RationalNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for RationalNumbersLib {
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
                Export::Class {
                    symbol: rational_value_class_symbol(),
                    class_id: None,
                },
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        let instance_shape = Arc::new(NumberLiteralShape::new(
            number_domain(),
            "RationalLiteral",
            [
                "number literal in the numbers/rational domain",
                "matches Expr::Number where domain == numbers/rational",
            ],
        ));
        let literal_class = Arc::new(NumberLiteralClass::new(
            literal_class_symbol(),
            number_domain(),
            "rational",
            "numerator/denominator",
            literal_instance_shape_symbol(),
            instance_shape.clone(),
        ));
        let value_shape = Arc::new(DomainNumberValueShape::new(
            number_domain(),
            "RationalValue",
            [
                "number value in the numbers/rational domain",
                "accepts any NumberValue where domain == numbers/rational",
            ],
        ));

        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(RationalNumberDomain))
                .expect("number domain should be boxable"),
        )?;
        let literal_class_id = linker.class_value(
            literal_class_symbol(),
            DefaultFactory
                .opaque(literal_class.clone())
                .expect("number literal class should be boxable"),
        )?;
        literal_class.set_id(literal_class_id);
        register_rational_value_class(linker)?;

        linker.shape_value(
            literal_instance_shape_symbol(),
            shape_value(literal_instance_shape_symbol(), instance_shape),
        )?;
        linker.shape_value(
            value_shape_symbol(),
            shape_value(value_shape_symbol(), value_shape),
        )?;

        register_promotions(linker);
        let binary = [
            (
                add_symbol(),
                rational_add_rule as RationalRuleFn,
                rational_add_value_rule as ValueRuleFn,
            ),
            (sub_symbol(), rational_sub_rule, rational_sub_value_rule),
            (mul_symbol(), rational_mul_rule, rational_mul_value_rule),
            (div_symbol(), rational_div_rule, rational_div_value_rule),
            (pow_symbol(), rational_pow_rule, rational_pow_value_rule),
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
                literal_apply: rational_neg_rule,
                value_cost: 1,
                value_apply: rational_neg_value_rule,
            }],
            reduction: vec![
                ScalarReductionOp {
                    operator: Symbol::qualified("math", "sum"),
                    literal_cost: 0,
                    literal_apply: rational_sum_rule,
                    value_cost: 1,
                    value_apply: rational_sum_value_rule,
                },
                ScalarReductionOp {
                    operator: Symbol::qualified("math", "product"),
                    literal_cost: 0,
                    literal_apply: rational_product_rule,
                    value_cost: 1,
                    value_apply: rational_product_value_rule,
                },
            ],
        };
        install_scalar_ops(linker, &ops);
        Ok(())
    }
}

fn register_rational_value_class(linker: &mut Linker<'_>) -> Result<ClassId> {
    let rational_value_class = Arc::new(RationalValueClass::new());
    let rational_class_id = linker.class_value(
        rational_value_class_symbol(),
        DefaultFactory
            .opaque(rational_value_class.clone())
            .expect("rational value class should be boxable"),
    )?;
    rational_value_class.set_id(rational_class_id);
    Ok(rational_class_id)
}

fn install_rational_value_citizen(linker: &mut Linker<'_>) -> Result<()> {
    register_rational_value_class(linker).map(|_| ())
}

fn conformance_rational_value_citizen(cx: &mut sim_kernel::Cx) -> Result<()> {
    let num = cx
        .factory()
        .number_literal(domains::i64(), "3".to_owned())?;
    let den = cx
        .factory()
        .number_literal(domains::bigint(), "5".to_owned())?;
    let value = super::value::make_rational(cx, num, den)?;
    sim_citizen::check_value_fixture(cx, value)
}

sim_citizen::inventory::submit! {
    sim_citizen::CitizenInfo {
        symbol: "numbers/Rational",
        version: 0,
        crate_name: env!("CARGO_PKG_NAME"),
        arity: 2,
        install: install_rational_value_citizen,
        conformance: conformance_rational_value_citizen,
    }
}

fn register_promotions(linker: &mut Linker<'_>) {
    for domain in integer_domains() {
        linker.promotion_rule(sim_kernel::PromotionRule {
            from_domain: domain.clone(),
            to_domain: number_domain(),
            cost: 1,
            convert: promote_integer_literal_to_rational,
        });
        linker.value_promotion_rule(ValuePromotionRule {
            from_domain: domain,
            to_domain: number_domain(),
            cost: 1,
            convert: promote_integer_value_to_rational,
        });
    }
    linker.promotion_rule(sim_kernel::PromotionRule {
        from_domain: f64_domain(),
        to_domain: number_domain(),
        cost: 1,
        convert: promote_f64_literal_to_rational,
    });
    linker.value_promotion_rule(ValuePromotionRule {
        from_domain: f64_domain(),
        to_domain: number_domain(),
        cost: 1,
        convert: promote_f64_value_to_rational,
    });
    linker.promotion_rule(sim_kernel::PromotionRule {
        from_domain: number_domain(),
        to_domain: f64_domain(),
        cost: 50,
        convert: promote_rational_literal_to_f64,
    });
    linker.value_promotion_rule(ValuePromotionRule {
        from_domain: number_domain(),
        to_domain: f64_domain(),
        cost: 50,
        convert: promote_rational_value_to_f64,
    });
}

fn integer_domains() -> Vec<Symbol> {
    domains::integer_domains()
}
