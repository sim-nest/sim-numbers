//! The `numbers/cas` domain object and the `CasNumbersLib` that registers the
//! domain, its literal and value shapes, the value class, and the `cas/var` and
//! `cas/simplify` functions.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Error, Export, Expr, Factory, Lib, LibManifest,
    LibTarget, Linker, NumberDomain, Object, Result, Symbol, Value, ValueNumberBinaryOp,
    ValuePromotionRule, Version,
};
use sim_lib_numbers_core::{
    DomainNumberValueShape, NumberDomainTableSpec, domains, number_domain_table,
};
use sim_shape::shape_value;

use super::{
    citizen::{cas_value_class_symbol, register_cas_value_class},
    function::{CasFunction, cas_simplify_symbol, cas_var_symbol},
    literal::{
        NumberLiteralClass, NumberLiteralShape, class_surface_or_symbol, shape_surface_or_symbol,
    },
    simplify::{simplify_expr, value_to_cas_expr},
    value::{CasExpr, cas_expr_to_value},
};

/// The `numbers/cas` domain symbol.
pub fn cas_domain_symbol() -> Symbol {
    domains::cas()
}

fn literal_class_symbol() -> Symbol {
    domains::literal_class("cas")
}

fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

fn value_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&cas_domain_symbol())
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/cas number-domain marker; reconstruct by loading the CAS number lib",
    kind = "marker",
    descriptor = "numbers/cas"
)]
pub struct CasNumberDomain;

impl NumberDomain for CasNumberDomain {
    fn symbol(&self) -> Symbol {
        cas_domain_symbol()
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

impl Object for CasNumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/cas>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for CasNumberDomain {
    fn class(&self, cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        sim_lib_numbers_core::number_domain_class_stub(cx)
    }
    fn as_expr(&self, _cx: &mut sim_kernel::Cx) -> Result<Expr> {
        Ok(Expr::Symbol(cas_domain_symbol()))
    }
    fn as_table(&self, cx: &mut sim_kernel::Cx) -> Result<Value> {
        let literal_class = class_surface_or_symbol(cx, literal_class_symbol())?;
        let instance_shape = shape_surface_or_symbol(cx, literal_instance_shape_symbol())?;
        let value_shape = shape_surface_or_symbol(cx, value_shape_symbol())?;
        number_domain_table(
            cx,
            NumberDomainTableSpec::new(
                cas_domain_symbol(),
                "cas",
                "value-only canonical lisp form",
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

/// The `numbers/cas` domain library.
///
/// Loading this [`Lib`] registers the CAS number domain, its literal and value
/// shapes, the symbolic value class, the `cas/var` and `cas/simplify`
/// functions, the value-promotion edges that lift scalar domains into the CAS,
/// and the symbolic `math/*` arithmetic operators over CAS values.
pub struct CasNumbersLib;

impl CasNumbersLib {
    /// Construct the CAS domain library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CasNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for CasNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: cas_domain_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![
                Export::NumberDomain {
                    symbol: cas_domain_symbol(),
                    number_domain_id: None,
                },
                Export::Class {
                    symbol: literal_class_symbol(),
                    class_id: None,
                },
                Export::Class {
                    symbol: cas_value_class_symbol(),
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
                    symbol: cas_var_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: cas_simplify_symbol(),
                    function_id: None,
                },
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        install_shapes(linker)?;
        register_cas_value_class(linker)?;
        linker.number_domain_value(
            cas_domain_symbol(),
            DefaultFactory
                .opaque(Arc::new(CasNumberDomain))
                .expect("number domain should be boxable"),
        )?;
        for symbol in [cas_var_symbol(), cas_simplify_symbol()] {
            linker.function_value(
                symbol.clone(),
                DefaultFactory
                    .opaque(Arc::new(CasFunction { symbol }))
                    .expect("function should be boxable"),
            )?;
        }
        for rule in promotion_rules() {
            linker.value_promotion_rule(rule);
        }
        for operator in arithmetic_symbols() {
            let apply = cas_binary_apply_for(&operator)?;
            linker.value_number_binary_op(ValueNumberBinaryOp {
                operator,
                left_domain: cas_domain_symbol(),
                right_domain: cas_domain_symbol(),
                cost: 0,
                apply,
            });
        }
        Ok(())
    }
}

fn install_shapes(linker: &mut Linker<'_>) -> Result<()> {
    let literal_shape = Arc::new(NumberLiteralShape::new(
        cas_domain_symbol(),
        "CasLiteral",
        [
            "placeholder literal shape for the numbers/cas domain",
            "CAS values print as canonical Lisp lists rather than parsed literals",
        ],
    ));
    let literal_class = Arc::new(NumberLiteralClass::new(
        literal_class_symbol(),
        cas_domain_symbol(),
        "cas",
        "value-only",
        literal_shape.clone(),
    ));
    let value_shape = Arc::new(DomainNumberValueShape::new(
        cas_domain_symbol(),
        "CasValue",
        [
            "number value in the numbers/cas domain",
            "accepts symbolic CAS values and canonical Lisp-shaped arithmetic forms",
        ],
    ));
    let class_id = linker.class_value(
        literal_class_symbol(),
        DefaultFactory
            .opaque(literal_class.clone())
            .expect("number literal class should be boxable"),
    )?;
    literal_class.set_id(class_id);
    linker.shape_value(
        literal_instance_shape_symbol(),
        shape_value(literal_instance_shape_symbol(), literal_shape),
    )?;
    linker.shape_value(
        value_shape_symbol(),
        shape_value(value_shape_symbol(), value_shape),
    )?;
    Ok(())
}

type CasBinaryApply = fn(&mut sim_kernel::Cx, Value, Value) -> Result<Value>;

#[derive(Clone, Copy)]
struct CasBinarySpec {
    name: &'static str,
    apply: CasBinaryApply,
}

impl CasBinarySpec {
    fn symbol(self) -> Symbol {
        Symbol::qualified("math", self.name)
    }
}

fn cas_binary_specs() -> [CasBinarySpec; 5] {
    [
        CasBinarySpec {
            name: "add",
            apply: apply_cas_add,
        },
        CasBinarySpec {
            name: "sub",
            apply: apply_cas_sub,
        },
        CasBinarySpec {
            name: "mul",
            apply: apply_cas_mul,
        },
        CasBinarySpec {
            name: "div",
            apply: apply_cas_div,
        },
        CasBinarySpec {
            name: "pow",
            apply: apply_cas_pow,
        },
    ]
}

fn arithmetic_symbols() -> Vec<Symbol> {
    cas_binary_specs()
        .into_iter()
        .map(CasBinarySpec::symbol)
        .collect()
}

fn cas_binary_apply_for(symbol: &Symbol) -> Result<CasBinaryApply> {
    cas_binary_specs()
        .into_iter()
        .find(|spec| spec.symbol() == *symbol)
        .map(|spec| spec.apply)
        .ok_or_else(|| Error::Eval(format!("unsupported CAS arithmetic operator {symbol}")))
}

fn promotion_rules() -> Vec<ValuePromotionRule> {
    let mut rules = value_promotions_for(
        &[
            "bool", "i8", "u8", "i16", "u16", "i32", "u32", "u64", "i64", "i128", "u128", "isize",
            "usize", "f32", "f64", "bigint", "complex",
        ],
        1,
    );
    rules.push(ValuePromotionRule {
        from_domain: domains::rational(),
        to_domain: cas_domain_symbol(),
        cost: 5,
        convert: lift_value_to_cas,
    });
    rules.push(ValuePromotionRule {
        from_domain: domains::continued_fraction(),
        to_domain: cas_domain_symbol(),
        cost: 10,
        convert: lift_value_to_cas,
    });
    rules
}

fn value_promotions_for(names: &[&str], cost: u16) -> Vec<ValuePromotionRule> {
    names
        .iter()
        .map(|name| ValuePromotionRule {
            from_domain: domains::domain(*name),
            to_domain: cas_domain_symbol(),
            cost,
            convert: lift_value_to_cas,
        })
        .collect()
}

fn lift_value_to_cas(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    let expr = CasExpr::num(cx, value)?;
    cas_expr_to_value(cx, expr)
}

fn apply_cas_add(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    apply_cas_binary(cx, Symbol::qualified("math", "add"), left, right)
}

fn apply_cas_sub(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    apply_cas_binary(cx, Symbol::qualified("math", "sub"), left, right)
}

fn apply_cas_mul(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    apply_cas_binary(cx, Symbol::qualified("math", "mul"), left, right)
}

fn apply_cas_div(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    apply_cas_binary(cx, Symbol::qualified("math", "div"), left, right)
}

fn apply_cas_pow(cx: &mut sim_kernel::Cx, left: Value, right: Value) -> Result<Value> {
    apply_cas_binary(cx, Symbol::qualified("math", "pow"), left, right)
}

fn apply_cas_binary(
    cx: &mut sim_kernel::Cx,
    operator: Symbol,
    left: Value,
    right: Value,
) -> Result<Value> {
    let expr = CasExpr::Op(
        operator,
        vec![value_to_cas_expr(cx, left)?, value_to_cas_expr(cx, right)?],
    );
    let simplified = simplify_expr(cx, expr)?;
    cas_expr_to_value(cx, simplified)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_cas_operator_returns_error() {
        let result = cas_binary_apply_for(&Symbol::qualified("math", "unknown"));
        assert!(matches!(result, Err(Error::Eval(_))));
    }
}
