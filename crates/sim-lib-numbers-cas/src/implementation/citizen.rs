//! The CAS value class: the runtime class and read-constructor that let a
//! symbolic `CasValue` participate as a first-class citizen object.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use sim_kernel::{
    Args, Callable, Class, ClassId, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, Linker,
    Object, ReadConstructor, ReadConstructorRef, Result, ShapeRef, Symbol, TableRef, Value,
};
use sim_lib_numbers_core::domains;

use super::{
    domain::cas_domain_symbol,
    simplify::expr_to_cas_expr,
    value::{CasExpr, cas_expr_to_value},
};

/// The class symbol of the symbolic CAS value object.
pub fn cas_value_class_symbol() -> Symbol {
    domains::cas_value_class()
}

fn value_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&cas_domain_symbol())
}

struct CasValueClass {
    id: AtomicU32,
}

impl CasValueClass {
    fn new() -> Self {
        Self {
            id: AtomicU32::new(0),
        }
    }

    fn set_id(&self, id: ClassId) {
        self.id.store(id.0, Ordering::Relaxed);
    }
}

impl Object for CasValueClass {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<class {}>", cas_value_class_symbol()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for CasValueClass {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Class"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_CLASS_CLASS_ID,
            Symbol::qualified("core", "Class"),
        )
    }

    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(cas_value_class_symbol()))
    }

    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }

    fn as_class(&self) -> Option<&dyn Class> {
        Some(self)
    }

    fn as_read_constructor(&self) -> Option<&dyn ReadConstructor> {
        Some(self)
    }
}

impl Callable for CasValueClass {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [version, expr] = values.as_slice() else {
            return Err(sim_citizen::arity_error(
                cas_value_class_symbol(),
                2,
                values.len(),
            ));
        };
        sim_citizen::decode_version(cx, version.clone(), 1, cas_value_class_symbol())?;
        let expr = expr.object().as_expr(cx)?;
        let cas_expr = expr_to_cas_expr(cx, &expr)?.ok_or_else(|| {
            Error::Eval(format!(
                "class {} expects a CAS-compatible expression",
                cas_value_class_symbol()
            ))
        })?;
        cas_expr_to_value(cx, cas_expr)
    }
}

impl Class for CasValueClass {
    fn id(&self) -> ClassId {
        ClassId(self.id.load(Ordering::Relaxed))
    }

    fn symbol(&self) -> Symbol {
        cas_value_class_symbol()
    }

    fn constructor_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }

    fn instance_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        Ok(cx
            .registry()
            .shape_by_symbol(&value_shape_symbol())
            .cloned()
            .unwrap_or(cx.factory().symbol(value_shape_symbol())?))
    }

    fn read_constructor(&self, cx: &mut Cx) -> Result<Option<ReadConstructorRef>> {
        Ok(cx
            .registry()
            .class_by_symbol(&cas_value_class_symbol())
            .cloned())
    }

    fn members(&self, cx: &mut Cx) -> Result<TableRef> {
        cx.factory().table(vec![
            (
                Symbol::new("version"),
                cx.factory()
                    .number_literal(Symbol::qualified("citizen", "int"), "1".to_owned())?,
            ),
            (
                Symbol::new("fields"),
                cx.factory()
                    .list(vec![cx.factory().symbol(Symbol::new("expr"))?])?,
            ),
        ])
    }
}

impl ReadConstructor for CasValueClass {
    fn symbol(&self) -> Symbol {
        cas_value_class_symbol()
    }

    fn args_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }

    fn construct_read(&self, cx: &mut Cx, args: Vec<Value>) -> Result<Value> {
        if args.len() != 2 {
            return Err(sim_citizen::arity_error(
                cas_value_class_symbol(),
                2,
                args.len(),
            ));
        }
        self.call(cx, Args::new(args))
    }
}

pub(crate) fn register_cas_value_class(linker: &mut Linker<'_>) -> Result<()> {
    let class = Arc::new(CasValueClass::new());
    let id = linker.class_value(
        cas_value_class_symbol(),
        DefaultFactory
            .opaque(class.clone())
            .expect("CAS value class should be boxable"),
    )?;
    class.set_id(id);
    Ok(())
}

fn install_cas_value_citizen(linker: &mut Linker<'_>) -> Result<()> {
    register_cas_value_class(linker)
}

fn conformance_cas_value_citizen(cx: &mut Cx) -> Result<()> {
    let one = cx
        .factory()
        .number_literal(domains::i64(), "1".to_owned())?;
    let expr = CasExpr::Op(
        Symbol::qualified("math", "add"),
        vec![CasExpr::Var(Symbol::new("x")), CasExpr::num(cx, one)?],
    );
    let value = cas_expr_to_value(cx, expr)?;
    sim_citizen::check_value_fixture_with_wrong_version(
        cx,
        value,
        Some(vec![
            Expr::Symbol(Symbol::new("v999")),
            Expr::Symbol(Symbol::new("x")),
        ]),
    )
}

sim_citizen::inventory::submit! {
    sim_citizen::CitizenInfo {
        symbol: "numbers/Cas",
        version: 1,
        crate_name: env!("CARGO_PKG_NAME"),
        arity: 1,
        install: install_cas_value_citizen,
        conformance: conformance_cas_value_citizen,
    }
}
