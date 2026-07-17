#![forbid(unsafe_code)]

//! The fixed-width integer library: the per-domain spec table and the `Lib`
//! that installs every `i8`..`i128` and `u8`..`u128` domain with its literal
//! and value shapes and widening promotion edges.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Expr, Factory, Lib, LibManifest, LibTarget,
    Linker, NumberDomain, NumberLiteral, Object, PromotionRule, Result, Symbol, Value,
    ValuePromotionRule, Version,
};
use sim_lib_numbers_core::{
    DomainNumberValueShape, NumberDomainTableSpec, domains, number_domain_table,
};
use sim_shape::shape_value;

use crate::literal::{
    NumberLiteralClass, NumberLiteralShape, class_surface_or_symbol, shape_surface_or_symbol,
};

#[derive(Clone, Copy)]
struct DomainSpec {
    name: &'static str,
    parse_priority: i32,
}

const FIXED_DOMAINS: [DomainSpec; 11] = [
    DomainSpec {
        name: "i8",
        parse_priority: 1,
    },
    DomainSpec {
        name: "u8",
        parse_priority: 1,
    },
    DomainSpec {
        name: "i16",
        parse_priority: 1,
    },
    DomainSpec {
        name: "u16",
        parse_priority: 1,
    },
    DomainSpec {
        name: "i32",
        parse_priority: 1,
    },
    DomainSpec {
        name: "u32",
        parse_priority: 1,
    },
    DomainSpec {
        name: "u64",
        parse_priority: 1,
    },
    DomainSpec {
        name: "i128",
        parse_priority: 5,
    },
    DomainSpec {
        name: "u128",
        parse_priority: 5,
    },
    DomainSpec {
        name: "isize",
        parse_priority: 1,
    },
    DomainSpec {
        name: "usize",
        parse_priority: 1,
    },
];

/// The library that installs every fixed-width integer domain (`numbers/i8`
/// through `numbers/i128`, `numbers/u8` through `numbers/u128`, and the
/// pointer-width `isize`/`usize`): their literal classes and shapes, value
/// shapes, and the widening promotion edges through the integer lattice.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use sim_kernel::{Cx, DefaultFactory, NoopEvalPolicy};
/// use sim_lib_numbers_core::domains;
/// use sim_lib_numbers_fixed::FixedNumbersLib;
///
/// let mut cx = Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
/// cx.load_lib(&FixedNumbersLib::new()).unwrap();
///
/// let edges = cx.registry().value_promotion_rules();
/// assert!(edges.iter().any(|rule| {
///     rule.from_domain == domains::i8() && rule.to_domain == domains::i16()
/// }));
/// ```
pub struct FixedNumbersLib;

impl FixedNumbersLib {
    /// Construct the fixed-width integer library installer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FixedNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for FixedNumbersLib {
    fn manifest(&self) -> LibManifest {
        let mut exports = Vec::new();
        for spec in FIXED_DOMAINS {
            exports.push(Export::NumberDomain {
                symbol: domain_symbol(spec),
                number_domain_id: None,
            });
            exports.push(Export::Class {
                symbol: literal_class_symbol(spec),
                class_id: None,
            });
            exports.push(Export::Shape {
                symbol: literal_instance_shape_symbol(spec),
                shape_id: None,
            });
            exports.push(Export::Shape {
                symbol: value_shape_symbol(spec),
                shape_id: None,
            });
        }
        LibManifest {
            id: domains::fixed(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports,
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        for spec in FIXED_DOMAINS {
            install_domain(linker, spec)?;
        }
        for rule in promotion_rules() {
            linker.promotion_rule(rule);
        }
        for rule in value_promotion_rules() {
            linker.value_promotion_rule(rule);
        }
        Ok(())
    }
}

fn install_domain(linker: &mut Linker<'_>, spec: DomainSpec) -> Result<()> {
    let domain = Arc::new(FixedNumberDomain { spec });
    let literal_shape = Arc::new(NumberLiteralShape::new(
        domain_symbol(spec),
        "FixedIntegerLiteral",
        [
            "number literal in a fixed integer domain",
            "matches Expr::Number where domain matches the fixed integer domain",
        ],
    ));
    let literal_class = Arc::new(NumberLiteralClass::new(
        literal_class_symbol(spec),
        domain_symbol(spec),
        "integer",
        spec.name,
        literal_shape.clone(),
    ));
    let value_shape = Arc::new(DomainNumberValueShape::new(
        domain_symbol(spec),
        "FixedIntegerValue",
        [
            "number value in a fixed integer domain",
            "accepts any NumberValue where domain matches the fixed integer domain",
        ],
    ));
    linker.number_domain_value(
        domain_symbol(spec),
        DefaultFactory
            .opaque(domain)
            .expect("number domain should be boxable"),
    )?;
    let class_id = linker.class_value(
        literal_class_symbol(spec),
        DefaultFactory
            .opaque(literal_class.clone())
            .expect("number literal class should be boxable"),
    )?;
    literal_class.set_id(class_id);
    linker.shape_value(
        literal_instance_shape_symbol(spec),
        shape_value(literal_instance_shape_symbol(spec), literal_shape),
    )?;
    linker.shape_value(
        value_shape_symbol(spec),
        shape_value(value_shape_symbol(spec), value_shape),
    )?;
    Ok(())
}

#[sim_citizen_derive::non_citizen(
    reason = "fixed-width number-domain marker; reconstruct by loading the fixed number lib",
    kind = "marker",
    descriptor = "numbers/fixed"
)]
struct FixedNumberDomain {
    spec: DomainSpec,
}

impl NumberDomain for FixedNumberDomain {
    fn symbol(&self) -> Symbol {
        domain_symbol(self.spec)
    }

    fn parse_priority(&self) -> i32 {
        self.spec.parse_priority
    }

    fn parse_literal(&self, cx: &mut sim_kernel::Cx, text: &str) -> Result<Option<Value>> {
        if text.contains(['.', '/']) {
            return Ok(None);
        }
        let canonical = match self.spec.name {
            "i8" => text.parse::<i8>().ok().map(|value| value.to_string()),
            "u8" => text.parse::<u8>().ok().map(|value| value.to_string()),
            "i16" => text.parse::<i16>().ok().map(|value| value.to_string()),
            "u16" => text.parse::<u16>().ok().map(|value| value.to_string()),
            "i32" => text.parse::<i32>().ok().map(|value| value.to_string()),
            "u32" => text.parse::<u32>().ok().map(|value| value.to_string()),
            "u64" => text.parse::<u64>().ok().map(|value| value.to_string()),
            "i128" => text.parse::<i128>().ok().map(|value| value.to_string()),
            "u128" => text.parse::<u128>().ok().map(|value| value.to_string()),
            "isize" => text.parse::<isize>().ok().map(|value| value.to_string()),
            "usize" => text.parse::<usize>().ok().map(|value| value.to_string()),
            _ => None,
        };
        match canonical {
            Some(canonical) => cx
                .factory()
                .number_literal(self.symbol(), canonical)
                .map(Some),
            None => Ok(None),
        }
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
}

impl Object for FixedNumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok(format!("#<number-domain {}>", domain_symbol(self.spec)))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FixedNumberDomain {
    fn class(&self, cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        sim_lib_numbers_core::number_domain_class_stub(cx)
    }
    fn as_expr(&self, _cx: &mut sim_kernel::Cx) -> Result<Expr> {
        Ok(Expr::Symbol(domain_symbol(self.spec)))
    }
    fn as_table(&self, cx: &mut sim_kernel::Cx) -> Result<Value> {
        let literal_class = class_surface_or_symbol(cx, literal_class_symbol(self.spec))?;
        let instance_shape = shape_surface_or_symbol(cx, literal_instance_shape_symbol(self.spec))?;
        let value_shape = shape_surface_or_symbol(cx, value_shape_symbol(self.spec))?;
        number_domain_table(
            cx,
            NumberDomainTableSpec::new(
                domain_symbol(self.spec),
                "integer",
                self.spec.name,
                self.spec.parse_priority,
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

#[derive(Clone, Copy)]
struct PromotionSpec {
    from: &'static str,
    to: &'static str,
    cost: u16,
    literal_convert: fn(&mut sim_kernel::Cx, NumberLiteral) -> Result<NumberLiteral>,
    value_convert: fn(&mut sim_kernel::Cx, Value) -> Result<Value>,
}

fn promotion_specs() -> Vec<PromotionSpec> {
    vec![
        spec("i8", "i16", 1, promote_to_i16, promote_value_to_i16),
        spec("i16", "i32", 1, promote_to_i32, promote_value_to_i32),
        spec("i32", "i64", 1, promote_to_i64, promote_value_to_i64),
        spec("i64", "i128", 1, promote_to_i128, promote_value_to_i128),
        spec("u8", "u16", 1, promote_to_u16, promote_value_to_u16),
        spec("u16", "u32", 1, promote_to_u32, promote_value_to_u32),
        spec("u32", "u64", 1, promote_to_u64, promote_value_to_u64),
        spec("u64", "u128", 1, promote_to_u128, promote_value_to_u128),
        spec("u8", "i16", 1, promote_to_i16, promote_value_to_i16),
        spec("u16", "i32", 1, promote_to_i32, promote_value_to_i32),
        spec("u32", "i64", 1, promote_to_i64, promote_value_to_i64),
        spec("u64", "i128", 1, promote_to_i128, promote_value_to_i128),
        spec("isize", "i128", 1, promote_to_i128, promote_value_to_i128),
        spec("usize", "u128", 1, promote_to_u128, promote_value_to_u128),
        spec("i8", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("u8", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("i16", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("u16", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("i32", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("u32", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("i64", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("u64", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("i128", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("u128", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("isize", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("usize", "f64", 50, promote_to_f64, promote_value_to_f64),
        spec("i8", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("u8", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("i16", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("u16", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("i32", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("u32", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("i64", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("u64", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("i128", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("u128", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("isize", "f32", 100, promote_to_f32, promote_value_to_f32),
        spec("usize", "f32", 100, promote_to_f32, promote_value_to_f32),
    ]
}

fn spec(
    from: &'static str,
    to: &'static str,
    cost: u16,
    literal_convert: fn(&mut sim_kernel::Cx, NumberLiteral) -> Result<NumberLiteral>,
    value_convert: fn(&mut sim_kernel::Cx, Value) -> Result<Value>,
) -> PromotionSpec {
    PromotionSpec {
        from,
        to,
        cost,
        literal_convert,
        value_convert,
    }
}

fn promotion_rules() -> Vec<PromotionRule> {
    promotion_specs()
        .into_iter()
        .map(|spec| PromotionRule {
            from_domain: domains::domain(spec.from),
            to_domain: domains::domain(spec.to),
            cost: spec.cost,
            convert: spec.literal_convert,
        })
        .collect()
}

fn value_promotion_rules() -> Vec<ValuePromotionRule> {
    promotion_specs()
        .into_iter()
        .map(|spec| ValuePromotionRule {
            from_domain: domains::domain(spec.from),
            to_domain: domains::domain(spec.to),
            cost: spec.cost,
            convert: spec.value_convert,
        })
        .collect()
}

fn promote_value_to_i16(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "i16", promote_to_i16)
}
fn promote_value_to_i32(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "i32", promote_to_i32)
}
fn promote_value_to_i64(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "i64", promote_to_i64)
}
fn promote_value_to_i128(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "i128", promote_to_i128)
}
fn promote_value_to_u16(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "u16", promote_to_u16)
}
fn promote_value_to_u32(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "u32", promote_to_u32)
}
fn promote_value_to_u64(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "u64", promote_to_u64)
}
fn promote_value_to_u128(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "u128", promote_to_u128)
}
fn promote_value_to_f32(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "f32", promote_to_f32)
}
fn promote_value_to_f64(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    promote_value_to_target(cx, value, "f64", promote_to_f64)
}

fn promote_value_to_target(
    cx: &mut sim_kernel::Cx,
    value: Value,
    target: &str,
    convert: fn(&mut sim_kernel::Cx, NumberLiteral) -> Result<NumberLiteral>,
) -> Result<Value> {
    let Some(number) = cx.number_value_ref(value)? else {
        return Err(sim_kernel::Error::Eval(format!(
            "fixed promotion to {} expected a number value",
            domains::domain(target)
        )));
    };
    let literal = number.literal.ok_or_else(|| {
        sim_kernel::Error::Eval(format!(
            "fixed promotion from {} to {} requires a canonical literal form",
            number.domain,
            domains::domain(target)
        ))
    })?;
    let promoted = convert(cx, literal)?;
    cx.factory()
        .number_literal(promoted.domain, promoted.canonical)
}

fn promote_to_i16(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "i16")
}
fn promote_to_i32(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "i32")
}
fn promote_to_i64(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "i64")
}
fn promote_to_i128(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "i128")
}
fn promote_to_u16(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "u16")
}
fn promote_to_u32(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "u32")
}
fn promote_to_u64(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "u64")
}
fn promote_to_u128(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "u128")
}
fn promote_to_f32(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "f32")
}
fn promote_to_f64(_cx: &mut sim_kernel::Cx, number: NumberLiteral) -> Result<NumberLiteral> {
    promote_to_target(number, "f64")
}

fn promote_to_target(number: NumberLiteral, target: &str) -> Result<NumberLiteral> {
    Ok(NumberLiteral {
        domain: domains::domain(target),
        canonical: number.canonical,
    })
}

fn domain_symbol(spec: DomainSpec) -> Symbol {
    domains::domain(spec.name)
}

fn literal_class_symbol(spec: DomainSpec) -> Symbol {
    domains::literal_class(spec.name)
}

fn literal_instance_shape_symbol(spec: DomainSpec) -> Symbol {
    Symbol::qualified(literal_class_symbol(spec).to_string(), "instance-shape")
}

fn value_shape_symbol(spec: DomainSpec) -> Symbol {
    domains::value_shape(&domain_symbol(spec))
}
