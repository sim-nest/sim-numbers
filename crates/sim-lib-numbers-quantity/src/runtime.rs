use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use sim_kernel::{
    AbiVersion, Args, Callable, Class, ClassId, ClassRef, Cx, Datum, DefaultFactory, Dependency,
    Export, Expr, Factory, Lib, LibManifest, LibTarget, Linker, MatchScore, Object, ObjectEncode,
    ObjectEncoding, ReadConstructor, ReadConstructorRef, Result, Shape, ShapeDoc, ShapeMatch,
    ShapeRef, Symbol, TableRef, Value, Version, value_from_datum,
};
use sim_shape::shape_value;

/// Symbol of the canonical `Quantity` read constructor.
pub fn quantity_class_symbol() -> Symbol {
    Symbol::qualified("numbers", "Quantity")
}
/// Symbol of the runtime quantity Shape.
pub fn quantity_shape_symbol() -> Symbol {
    Symbol::qualified("numbers", "QuantityShape")
}

/// Runtime quantity backed by canonical pure data. The `scalar` field remains
/// a codec-visible number datum, retaining its installed domain tag.
#[derive(Clone, Debug)]
pub struct QuantityValue {
    datum: Datum,
}

impl QuantityValue {
    /// Validates bounded canonical quantity data.
    pub fn from_datum(datum: Datum) -> Result<Self> {
        validate_quantity_datum(&datum).map_err(sim_kernel::Error::Eval)?;
        Ok(Self { datum })
    }
    /// Returns canonical codec/read-construct data.
    pub const fn datum(&self) -> &Datum {
        &self.datum
    }
    /// Boxes this quantity as a runtime value.
    pub fn into_value(self) -> Result<Value> {
        DefaultFactory.opaque(Arc::new(self))
    }
}

impl Object for QuantityValue {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok("#<quantity>".to_owned())
    }
    fn snapshot(&self, _cx: &mut Cx) -> Result<Option<Datum>> {
        Ok(Some(self.datum.clone()))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl sim_kernel::ObjectCompat for QuantityValue {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        cx.registry()
            .class_by_symbol(&quantity_class_symbol())
            .cloned()
            .map_or_else(
                || {
                    cx.factory()
                        .class_stub(sim_kernel::CORE_EXPR_CLASS_ID, quantity_class_symbol())
                },
                Ok,
            )
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Call {
            operator: Box::new(Expr::Symbol(quantity_class_symbol())),
            args: vec![
                Expr::Symbol(Symbol::new("v1")),
                Expr::from(self.datum.clone()),
            ],
        })
    }
    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        let Datum::Node { fields, .. } = &self.datum else {
            unreachable!()
        };
        let mut entries = Vec::with_capacity(fields.len());
        for (key, datum) in fields {
            entries.push((key.clone(), value_from_datum(cx, datum.clone())?));
        }
        cx.factory().table(entries)
    }
    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}
impl ObjectEncode for QuantityValue {
    fn object_encoding(&self, _cx: &mut Cx) -> Result<ObjectEncoding> {
        Ok(ObjectEncoding::Constructor {
            class: quantity_class_symbol(),
            args: vec![
                Expr::Symbol(Symbol::new("v1")),
                Expr::from(self.datum.clone()),
            ],
        })
    }
}

struct QuantityClass {
    id: AtomicU32,
}
impl QuantityClass {
    fn new() -> Self {
        Self {
            id: AtomicU32::new(0),
        }
    }
}
impl Object for QuantityClass {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<class {}>", quantity_class_symbol()))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl sim_kernel::ObjectCompat for QuantityClass {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        cx.factory().class_stub(
            sim_kernel::CORE_CLASS_CLASS_ID,
            Symbol::qualified("core", "Class"),
        )
    }
    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(quantity_class_symbol()))
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
impl Callable for QuantityClass {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [version, datum] = values.as_slice() else {
            return Err(sim_kernel::Error::Eval(
                "Quantity expects version and one data record".to_owned(),
            ));
        };
        if version.object().as_expr(cx)? != Expr::Symbol(Symbol::new("v1")) {
            return Err(sim_kernel::Error::Eval(
                "Quantity requires constructor version v1".to_owned(),
            ));
        }
        let datum = datum.object().snapshot(cx)?.ok_or_else(|| {
            sim_kernel::Error::Eval("Quantity constructor data must be pure".to_owned())
        })?;
        QuantityValue::from_datum(datum)?.into_value()
    }
}
impl Class for QuantityClass {
    fn id(&self) -> ClassId {
        ClassId(self.id.load(Ordering::Relaxed))
    }
    fn symbol(&self) -> Symbol {
        quantity_class_symbol()
    }
    fn constructor_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }
    fn instance_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        Ok(cx
            .registry()
            .shape_by_symbol(&quantity_shape_symbol())
            .cloned()
            .unwrap_or(cx.factory().symbol(quantity_shape_symbol())?))
    }
    fn read_constructor(&self, cx: &mut Cx) -> Result<Option<ReadConstructorRef>> {
        Ok(cx
            .registry()
            .class_by_symbol(&quantity_class_symbol())
            .cloned())
    }
    fn members(&self, cx: &mut Cx) -> Result<TableRef> {
        cx.factory().table(Vec::new())
    }
}
impl ReadConstructor for QuantityClass {
    fn symbol(&self) -> Symbol {
        quantity_class_symbol()
    }
    fn args_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }
    fn construct_read(&self, cx: &mut Cx, args: Vec<Value>) -> Result<Value> {
        self.call(cx, Args::new(args))
    }
}

struct RuntimeQuantityShape;
impl Shape for RuntimeQuantityShape {
    fn symbol(&self) -> Option<Symbol> {
        Some(quantity_shape_symbol())
    }
    fn check_value(&self, cx: &mut Cx, value: Value) -> Result<ShapeMatch> {
        Ok(value.object().snapshot(cx)?.as_ref().map_or_else(
            || ShapeMatch::reject("quantity must be inspectable data"),
            |datum| match validate_quantity_datum(datum) {
                Ok(()) => ShapeMatch::accept(MatchScore::exact(100)),
                Err(error) => ShapeMatch::reject(error),
            },
        ))
    }
    fn check_expr(&self, _cx: &mut Cx, expr: &Expr) -> Result<ShapeMatch> {
        Ok(Datum::try_from(expr.clone()).ok().as_ref().map_or_else(
            || ShapeMatch::reject("quantity expression must be pure data"),
            |datum| match validate_quantity_datum(datum) {
                Ok(()) => ShapeMatch::accept(MatchScore::exact(90)),
                Err(error) => ShapeMatch::reject(error),
            },
        ))
    }
    fn describe(&self, _cx: &mut Cx) -> Result<ShapeDoc> {
        Ok(ShapeDoc::new("semantic quantity")
            .with_detail("bounded seven-axis rational dimension vector")
            .with_detail("independent scalar, kind, unit, and point/interval role"))
    }
}

/// Loadable runtime surface registering the quantity class and Shape.
pub struct QuantityLib;
impl QuantityLib {
    /// Creates the quantity runtime library.
    pub const fn new() -> Self {
        Self
    }
}
impl Default for QuantityLib {
    fn default() -> Self {
        Self::new()
    }
}
impl Lib for QuantityLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "quantity"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![
                Export::Class {
                    symbol: quantity_class_symbol(),
                    class_id: None,
                },
                Export::Shape {
                    symbol: quantity_shape_symbol(),
                    shape_id: None,
                },
            ],
        }
    }
    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        let class = Arc::new(QuantityClass::new());
        let id = linker.class_value(
            quantity_class_symbol(),
            DefaultFactory.opaque(class.clone())?,
        )?;
        class.id.store(id.0, Ordering::Relaxed);
        linker.shape_value(
            quantity_shape_symbol(),
            shape_value(quantity_shape_symbol(), Arc::new(RuntimeQuantityShape)),
        )?;
        Ok(())
    }
}

fn validate_quantity_datum(datum: &Datum) -> std::result::Result<(), String> {
    let Datum::Node { tag, fields } = datum else {
        return Err("Quantity data must be a node".to_owned());
    };
    if *tag != quantity_class_symbol() {
        return Err(format!("expected {}, found {tag}", quantity_class_symbol()));
    }
    if fields.len() != 5 {
        return Err(
            "Quantity data requires exactly scalar, dimension, kind, unit, and role".to_owned(),
        );
    }
    for required in ["scalar", "dimension", "kind", "unit", "role"] {
        if !fields.iter().any(|(key, _)| key == &Symbol::new(required)) {
            return Err(format!("Quantity data lacks {required}"));
        }
    }
    let dimension = fields
        .iter()
        .find(|(key, _)| key == &Symbol::new("dimension"))
        .map(|(_, value)| value)
        .expect("checked");
    let Datum::Vector(exponents) = dimension else {
        return Err("Quantity dimension must be a seven-element vector".to_owned());
    };
    if exponents.len() != 7 {
        return Err("Quantity dimension must have exactly seven exponents".to_owned());
    }
    for exponent in exponents {
        let Datum::Vector(pair) = exponent else {
            return Err("dimension exponent must be [numerator denominator]".to_owned());
        };
        if pair.len() != 2 {
            return Err("dimension exponent must have exactly two parts".to_owned());
        }
        let numerator = bounded_integer(&pair[0])?;
        let denominator = bounded_integer(&pair[1])?;
        if denominator == 0 {
            return Err("dimension exponent denominator must not be zero".to_owned());
        }
        if numerator == 0 && denominator != 1 {
            return Err("zero dimension exponents must use canonical denominator one".to_owned());
        }
    }
    let role = fields
        .iter()
        .find(|(key, _)| key == &Symbol::new("role"))
        .map(|(_, value)| value)
        .expect("checked");
    if !matches!(role, Datum::Symbol(value) if value == &Symbol::qualified("measure", "point") || value == &Symbol::qualified("measure", "interval"))
    {
        return Err("Quantity role must be measure:point or measure:interval".to_owned());
    }
    Ok(())
}

fn bounded_integer(datum: &Datum) -> std::result::Result<i64, String> {
    let Datum::Number(number) = datum else {
        return Err("dimension exponent parts must be number literals".to_owned());
    };
    if number.domain != Symbol::qualified("numbers", "i64") {
        return Err("dimension exponent parts must use numbers/i64".to_owned());
    }
    let value = number
        .canonical
        .parse::<i64>()
        .map_err(|_| "invalid i64 dimension exponent".to_owned())?;
    if value.unsigned_abs() > 1024 {
        return Err("dimension exponent exceeds magnitude bound 1024".to_owned());
    }
    Ok(value)
}
