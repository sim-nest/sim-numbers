//! The `numbers/cf` domain object and `ExoticNumbersLib`: the domain symbols,
//! the builtin continued-fraction constants, and the `Lib` that registers the
//! domain, its shapes, the `as-f64` function, and value promotions.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Expr, Factory, Lib, LibManifest, LibTarget,
    Linker, NumberDomain, Object, Result, Symbol, Value, ValuePromotionRule, Version,
};
use sim_lib_numbers_core::{
    DomainNumberValueShape, NumberDomainTableSpec, domains, number_domain_table,
};
use sim_shape::shape_value;

use super::{
    function::{AsF64Function, builtin_cf, plain_as_f64_symbol},
    literal::{
        NumberLiteralClass, NumberLiteralShape, class_surface_or_symbol, shape_surface_or_symbol,
    },
    value::{CfTail, ContinuedFraction, ExoticReal},
};

/// The `numbers/cf` domain symbol shared by this crate's continued-fraction
/// values and builtins.
pub fn number_domain() -> Symbol {
    domains::continued_fraction()
}

pub fn literal_class_symbol() -> Symbol {
    domains::literal_class("cf")
}

pub fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

pub fn value_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&number_domain())
}

/// The symbol naming a builtin continued-fraction constant (for example `phi`
/// or `e`) installed by this domain.
pub fn builtin_symbol(name: &str) -> Symbol {
    Symbol::new(name)
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/cf number-domain marker; reconstruct by loading the exotic number lib",
    kind = "marker",
    descriptor = "numbers/cf"
)]
pub struct ExoticNumberDomain;

impl NumberDomain for ExoticNumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }

    fn parse_priority(&self) -> i32 {
        -100
    }

    fn parse_literal(&self, _cx: &mut sim_kernel::Cx, _text: &str) -> Result<Option<Value>> {
        Ok(None)
    }

    fn encode_literal(
        &self,
        _cx: &mut sim_kernel::Cx,
        _value: Value,
    ) -> Result<Option<sim_kernel::NumberLiteral>> {
        Ok(None)
    }
}

impl Object for ExoticNumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/cf>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ExoticNumberDomain {
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
                "continued-fraction",
                "value-only",
                -100,
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

/// The library that installs the `numbers/cf` continued-fraction domain: its
/// value shape, the builtin continued-fraction constants (registered as
/// registry values), and the `as-f64` truncation function.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy, Symbol};
/// use sim_lib_numbers_exotic::{ExoticNumbersLib, number_domain};
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// cx.load_lib(&ExoticNumbersLib::new()).unwrap();
///
/// // The builtin continued-fraction constants are registered as values in
/// // the numbers/cf domain.
/// let sqrt2 = cx.registry().value_by_symbol(&Symbol::new("cf-sqrt2")).unwrap().clone();
/// let number = cx.number_value_ref(sqrt2).unwrap().unwrap();
/// assert_eq!(number.domain, number_domain());
/// ```
pub struct ExoticNumbersLib;

impl ExoticNumbersLib {
    /// Creates a new `numbers/cf` continued-fraction domain library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExoticNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for ExoticNumbersLib {
    fn manifest(&self) -> LibManifest {
        let mut exports = vec![
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
            Export::Function {
                symbol: super::function::as_f64_symbol(),
                function_id: None,
            },
            Export::Function {
                symbol: plain_as_f64_symbol(),
                function_id: None,
            },
        ];
        for name in ["cf-sqrt2", "cf-phi", "cf-e", "cf-pi"] {
            exports.push(Export::Value {
                symbol: builtin_symbol(name),
            });
            exports.push(Export::Value {
                symbol: domains::domain(name),
            });
        }
        LibManifest {
            id: number_domain(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports,
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        let instance_shape = Arc::new(NumberLiteralShape::new(
            number_domain(),
            "ContinuedFractionLiteral",
            [
                "placeholder literal shape for the numbers/cf domain",
                "continued fractions are value-level number objects rather than parsed literals",
            ],
        ));
        let literal_class = Arc::new(NumberLiteralClass::new(
            literal_class_symbol(),
            number_domain(),
            "continued-fraction",
            "value-only",
            instance_shape.clone(),
        ));
        let value_shape = Arc::new(DomainNumberValueShape::new(
            number_domain(),
            "ContinuedFractionValue",
            [
                "number value in the numbers/cf domain",
                "accepts any continued fraction NumberValue",
            ],
        ));
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(ExoticNumberDomain))
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

        for symbol in [super::function::as_f64_symbol(), plain_as_f64_symbol()] {
            linker.function_value(
                symbol.clone(),
                DefaultFactory
                    .opaque(Arc::new(AsF64Function { symbol }))
                    .expect("function should be boxable"),
            )?;
        }

        for (name, value) in builtins() {
            let value = DefaultFactory
                .opaque(value)
                .expect("continued fraction value should be boxable");
            linker.value(builtin_symbol(name), value.clone())?;
            linker.value(domains::domain(name), value)?;
        }

        for rule in promotions() {
            linker.value_promotion_rule(rule);
        }
        Ok(())
    }
}

fn builtins() -> Vec<(&'static str, Arc<ContinuedFraction>)> {
    vec![
        (
            "cf-sqrt2",
            builtin_cf("cf-sqrt2", vec![1], Some(CfTail::endless(|| Some(2)))),
        ),
        (
            "cf-phi",
            builtin_cf("cf-phi", vec![1], Some(CfTail::endless(|| Some(1)))),
        ),
        (
            "cf-e",
            builtin_cf(
                "cf-e",
                vec![2],
                Some(CfTail::endless({
                    let mut index = 0usize;
                    move || {
                        let out = match index % 3 {
                            0 => 1,
                            1 => 2 * ((index / 3) as i128 + 1),
                            _ => 1,
                        };
                        index += 1;
                        Some(out)
                    }
                })),
            ),
        ),
        (
            "cf-pi",
            builtin_cf(
                "cf-pi",
                vec![3, 7, 15, 1, 292],
                Some(CfTail::endless(|| Some(1))),
            ),
        ),
    ]
}

fn promotions() -> Vec<ValuePromotionRule> {
    vec![
        ValuePromotionRule {
            from_domain: number_domain(),
            to_domain: domains::f64(),
            cost: 30,
            convert: promote_cf_to_f64,
        },
        ValuePromotionRule {
            from_domain: number_domain(),
            to_domain: domains::rational(),
            cost: 200,
            convert: promote_cf_to_rational,
        },
    ]
}

fn promote_cf_to_f64(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    let Some(cf) = value.object().downcast_ref::<ContinuedFraction>() else {
        return Err(sim_kernel::Error::Eval(
            "numbers/cf promotion expected a continued fraction value".to_owned(),
        ));
    };
    let (approx, _) = cf.as_f64(64);
    cx.factory()
        .number_literal(domains::f64(), approx.to_string())
}

fn promote_cf_to_rational(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    let Some(cf) = value.object().downcast_ref::<ContinuedFraction>() else {
        return Err(sim_kernel::Error::Eval(
            "numbers/cf promotion expected a continued fraction value".to_owned(),
        ));
    };
    cf.truncate_rational(cx, 32)?.ok_or_else(|| {
        sim_kernel::Error::Eval("numbers/cf could not promote to numbers/rational".to_owned())
    })
}
