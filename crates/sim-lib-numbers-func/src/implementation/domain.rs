//! `Func` number-domain registration: the domain library, class symbols, and
//! value-shape wiring that install the function domain into the runtime.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, ClassId, DefaultFactory, Dependency, Export, Expr, Factory, Lib, LibManifest,
    LibTarget, Linker, NumberDomain, NumberLiteral, Object, Result, Symbol, Value,
    ValuePromotionRule, Version,
};
use sim_lib_numbers_cas::CasExpr;
use sim_lib_numbers_core::{DomainNumberValueShape, domains};
use sim_shape::shape_value;

use super::function::{
    CallFunction, FnBuilder, GradFunction, build_func_class, call_symbol, fn_symbol, grad_symbol,
};
use super::value::{Func, build_constant_func_value, build_func_value};

/// Returns the domain symbol that names the `Func` number domain (`numbers/func`).
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_func::func_domain_symbol;
///
/// assert_eq!(func_domain_symbol().to_string(), "numbers/func");
/// ```
pub fn func_domain_symbol() -> Symbol {
    domains::func()
}

/// Returns the class symbol for the constructible `Func` value class (`numbers/Func`).
///
/// # Examples
///
/// ```
/// use sim_lib_numbers_func::func_class_symbol;
///
/// assert_eq!(func_class_symbol().to_string(), "numbers/Func");
/// ```
pub fn func_class_symbol() -> Symbol {
    domains::domain("Func")
}

pub fn value_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&func_domain_symbol())
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/func number-domain marker; reconstruct by loading the function number lib",
    kind = "marker"
)]
pub struct FuncNumberDomain;

impl NumberDomain for FuncNumberDomain {
    fn symbol(&self) -> Symbol {
        func_domain_symbol()
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
    ) -> Result<Option<NumberLiteral>> {
        Ok(None)
    }
}

impl Object for FuncNumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/func>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for FuncNumberDomain {
    fn class(&self, cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        sim_lib_numbers_core::number_domain_class_stub(cx)
    }
    fn as_expr(&self, _cx: &mut sim_kernel::Cx) -> Result<Expr> {
        Ok(Expr::Symbol(func_domain_symbol()))
    }
    fn as_table(&self, cx: &mut sim_kernel::Cx) -> Result<Value> {
        let value_shape = cx
            .registry()
            .shape_by_symbol(&value_shape_symbol())
            .cloned()
            .unwrap_or(cx.factory().symbol(value_shape_symbol())?);
        let func_class = cx
            .registry()
            .class_by_symbol(&func_class_symbol())
            .cloned()
            .unwrap_or(cx.factory().symbol(func_class_symbol())?);
        cx.factory().table(vec![
            (
                Symbol::new("symbol"),
                cx.factory().symbol(func_domain_symbol())?,
            ),
            (
                Symbol::new("kind"),
                cx.factory().string("number-domain".to_owned())?,
            ),
            (
                Symbol::new("numeric-family"),
                cx.factory().string("function".to_owned())?,
            ),
            (
                Symbol::new("canonical-form"),
                cx.factory()
                    .string("callable symbolic function".to_owned())?,
            ),
            (
                Symbol::new("parse-priority"),
                cx.factory().string("-100".to_owned())?,
            ),
            (Symbol::new("constructor-class"), func_class),
            (Symbol::new("value-shape"), value_shape),
            (Symbol::new("builder"), cx.factory().symbol(fn_symbol())?),
            (Symbol::new("call"), cx.factory().symbol(call_symbol())?),
            (Symbol::new("grad"), cx.factory().symbol(grad_symbol())?),
        ])
    }
    fn as_number_domain(&self) -> Option<&dyn NumberDomain> {
        Some(self)
    }
}

/// Library that installs the `Func` number domain: its domain object, value
/// class and shape, the `fn`/`call`/`grad` callables, and the promotion rules
/// that lift scalar number values into constant functions.
pub struct FuncNumbersLib;

impl FuncNumbersLib {
    /// Creates a new `FuncNumbersLib` ready to be loaded into a runtime.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FuncNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for FuncNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: func_domain_symbol(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![
                Export::NumberDomain {
                    symbol: func_domain_symbol(),
                    number_domain_id: None,
                },
                Export::Class {
                    symbol: func_class_symbol(),
                    class_id: None,
                },
                Export::Shape {
                    symbol: value_shape_symbol(),
                    shape_id: None,
                },
                Export::Function {
                    symbol: fn_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: call_symbol(),
                    function_id: None,
                },
                Export::Function {
                    symbol: grad_symbol(),
                    function_id: None,
                },
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        let value_shape = Arc::new(DomainNumberValueShape::new(
            func_domain_symbol(),
            "FuncValue",
            [
                "callable number value in the numbers/func domain",
                "accepts any NumberValue where domain == numbers/func",
            ],
        ));

        linker.number_domain_value(
            func_domain_symbol(),
            DefaultFactory
                .opaque(Arc::new(FuncNumberDomain))
                .expect("number domain should be boxable"),
        )?;
        register_func_value_class(linker)?;
        linker.shape_value(
            value_shape_symbol(),
            shape_value(value_shape_symbol(), value_shape),
        )?;
        for (symbol, value) in [
            (
                fn_symbol(),
                DefaultFactory
                    .opaque(Arc::new(FnBuilder))
                    .expect("fn builder should be boxable"),
            ),
            (
                call_symbol(),
                DefaultFactory
                    .opaque(Arc::new(CallFunction))
                    .expect("call helper should be boxable"),
            ),
            (
                grad_symbol(),
                DefaultFactory
                    .opaque(Arc::new(GradFunction))
                    .expect("grad helper should be boxable"),
            ),
        ] {
            linker.function_value(symbol, value)?;
        }
        for from_domain in promoted_domains() {
            linker.value_promotion_rule(ValuePromotionRule {
                from_domain,
                to_domain: func_domain_symbol(),
                cost: 1,
                convert: promote_value_to_func,
            });
        }
        super::value::register_value_ops(linker);
        Ok(())
    }
}

fn register_func_value_class(linker: &mut Linker<'_>) -> Result<ClassId> {
    let func_class = build_func_class();
    let class_id = linker.class_value(
        func_class_symbol(),
        DefaultFactory
            .opaque(func_class.clone())
            .expect("function class should be boxable"),
    )?;
    func_class.set_id(class_id);
    Ok(class_id)
}

fn install_func_value_citizen(linker: &mut Linker<'_>) -> Result<()> {
    register_func_value_class(linker).map(|_| ())
}

fn conformance_func_value_citizen(cx: &mut sim_kernel::Cx) -> Result<()> {
    let var = Symbol::new("x");
    let value = build_func_value(cx, Func::symbolic(vec![var.clone()], CasExpr::Var(var)))?;
    sim_citizen::check_value_fixture(cx, value)
}

sim_citizen::inventory::submit! {
    sim_citizen::CitizenInfo {
        symbol: "numbers/Func",
        version: 0,
        crate_name: env!("CARGO_PKG_NAME"),
        arity: 2,
        install: install_func_value_citizen,
        conformance: conformance_func_value_citizen,
    }
}

fn promoted_domains() -> Vec<Symbol> {
    vec![
        domains::bool(),
        domains::f32(),
        domains::f64(),
        domains::i64(),
        domains::bigint(),
        domains::rational(),
        domains::complex(),
        domains::cas(),
        domains::continued_fraction(),
    ]
}

fn promote_value_to_func(cx: &mut sim_kernel::Cx, value: Value) -> Result<Value> {
    build_constant_func_value(cx, value)
}
