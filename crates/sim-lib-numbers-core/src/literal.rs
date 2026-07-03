//! The shared number-literal shape and class.
//!
//! Every scalar number domain (`numbers/i64`, `numbers/f64`, `numbers/bool`,
//! `numbers/rational`, ...) had a byte-identical copy of these two types. This
//! is their one home; a domain crate constructs them with its own domain symbol
//! and instance-shape symbol.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use sim_kernel::{
    Args, Callable, Class, ClassId, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, Object,
    ReadConstructorRef, Result, ShapeRef, Symbol, TableRef, Value,
};
use sim_shape::{MatchScore, Shape, ShapeDoc, ShapeMatch, shape_value};

/// A shape matching number literals (`Expr::Number`) in one domain.
pub struct NumberLiteralShape {
    domain: Symbol,
    name: &'static str,
    details: Vec<&'static str>,
}

impl NumberLiteralShape {
    /// Build a literal shape for `domain` with a browse `name` and detail lines.
    pub fn new(
        domain: Symbol,
        name: &'static str,
        details: impl IntoIterator<Item = &'static str>,
    ) -> Self {
        Self {
            domain,
            name,
            details: details.into_iter().collect(),
        }
    }
}

impl Shape for NumberLiteralShape {
    fn check_value(&self, cx: &mut Cx, value: Value) -> Result<ShapeMatch> {
        let expr = value.object().as_expr(cx)?;
        self.check_expr(cx, &expr)
    }

    fn check_expr(&self, _cx: &mut Cx, expr: &Expr) -> Result<ShapeMatch> {
        match expr {
            Expr::Number(number) if number.domain == self.domain => {
                Ok(ShapeMatch::accept(MatchScore::exact(20)))
            }
            _ => Ok(ShapeMatch::reject(format!(
                "expected number literal in {}",
                self.domain
            ))),
        }
    }

    fn describe(&self, _cx: &mut Cx) -> Result<ShapeDoc> {
        let mut doc = ShapeDoc::new(self.name);
        for detail in &self.details {
            doc = doc.with_detail(*detail);
        }
        Ok(doc)
    }
}

/// The class object for a domain's number literals.
pub struct NumberLiteralClass {
    id: AtomicU32,
    symbol: Symbol,
    domain: Symbol,
    numeric_family: &'static str,
    canonical_form: &'static str,
    instance_shape_symbol: Symbol,
    instance_shape: Arc<dyn Shape>,
}

impl NumberLiteralClass {
    /// Build the literal class for one domain from its identifying metadata and
    /// the shape matching its instances.
    pub fn new(
        symbol: Symbol,
        domain: Symbol,
        numeric_family: &'static str,
        canonical_form: &'static str,
        instance_shape_symbol: Symbol,
        instance_shape: Arc<dyn Shape>,
    ) -> Self {
        Self {
            id: AtomicU32::new(0),
            symbol,
            domain,
            numeric_family,
            canonical_form,
            instance_shape_symbol,
            instance_shape,
        }
    }

    /// Record the kernel-assigned [`ClassId`] after registration.
    pub fn set_id(&self, id: ClassId) {
        self.id.store(id.0, Ordering::Relaxed);
    }
}

impl Object for NumberLiteralClass {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<class {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for NumberLiteralClass {
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
        Ok(Expr::Symbol(self.symbol.clone()))
    }
    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        let instance_shape = shape_surface_or_symbol(cx, self.instance_shape_symbol.clone())?;
        cx.factory().table(vec![
            (
                Symbol::new("symbol"),
                cx.factory().symbol(self.symbol.clone())?,
            ),
            (
                Symbol::new("domain"),
                cx.factory().symbol(self.domain.clone())?,
            ),
            (
                Symbol::new("numeric-family"),
                cx.factory().string(self.numeric_family.to_owned())?,
            ),
            (
                Symbol::new("canonical-form"),
                cx.factory().string(self.canonical_form.to_owned())?,
            ),
            (Symbol::new("instance-shape"), instance_shape),
        ])
    }
    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
    fn as_class(&self) -> Option<&dyn Class> {
        Some(self)
    }
}

impl Callable for NumberLiteralClass {
    fn call(&self, _cx: &mut Cx, _args: Args) -> Result<Value> {
        Err(Error::Eval(format!(
            "class {} does not construct values directly; parse or compute a number literal instead",
            self.symbol
        )))
    }
}

impl Class for NumberLiteralClass {
    fn id(&self) -> ClassId {
        ClassId(self.id.load(Ordering::Relaxed))
    }

    fn symbol(&self) -> Symbol {
        self.symbol.clone()
    }

    fn constructor_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }

    fn instance_shape(&self, _cx: &mut Cx) -> Result<ShapeRef> {
        Ok(shape_value(
            self.instance_shape_symbol.clone(),
            self.instance_shape.clone(),
        ))
    }

    fn read_constructor(&self, _cx: &mut Cx) -> Result<Option<ReadConstructorRef>> {
        Ok(None)
    }

    fn members(&self, cx: &mut Cx) -> Result<TableRef> {
        cx.factory().table(Vec::new())
    }
}

/// Resolve a class symbol to its registered surface value, or the bare symbol.
pub fn class_surface_or_symbol(cx: &mut Cx, symbol: Symbol) -> Result<Value> {
    Ok(cx
        .registry()
        .class_by_symbol(&symbol)
        .cloned()
        .unwrap_or(cx.factory().symbol(symbol)?))
}

/// Resolve a shape symbol to its registered surface value, or the bare symbol.
pub fn shape_surface_or_symbol(cx: &mut Cx, symbol: Symbol) -> Result<Value> {
    Ok(cx
        .registry()
        .shape_by_symbol(&symbol)
        .cloned()
        .unwrap_or(cx.factory().symbol(symbol)?))
}
