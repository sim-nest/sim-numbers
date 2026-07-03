//! The `ComplexValue` object and its value class: the real/imaginary
//! representation, its citizen encoding, and the read-constructor that rebuilds
//! complex values.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use sim_citizen::{CitizenField, arity_error, decode_version};
use sim_kernel::{
    Args, Callable, Class, ClassId, ClassRef, Cx, DefaultFactory, Error, Expr, Factory,
    NumberLiteral, NumberValue, Object, ObjectEncode, ObjectEncoding, ReadConstructor,
    ReadConstructorRef, Result, ShapeRef, Symbol, TableRef, Value,
};
use sim_lib_numbers_core::domains;

use super::literal::{number_domain, value_shape_symbol};
use super::ops::canonical_complex;

/// The symbol of the `numbers/complex` value class, used to register the class
/// and to tag the read-constructor encoding of complex values.
pub fn complex_value_class_symbol() -> Symbol {
    domains::complex_value_class()
}

/// A complex number value: real and imaginary parts over an `f64` base scalar.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexValue {
    /// The real part.
    real: f64,
    /// The imaginary part.
    imag: f64,
}

impl ComplexValue {
    /// Creates a complex value from its real and imaginary parts.
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    fn literal(&self) -> NumberLiteral {
        NumberLiteral {
            domain: number_domain(),
            canonical: canonical_complex(self.real, self.imag),
        }
    }
}

impl Object for ComplexValue {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(self.literal().canonical)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ComplexValue {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx.registry().class_by_symbol(&complex_value_class_symbol()) {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_NUMBER_CLASS_ID,
            Symbol::qualified("core", "Number"),
        )
    }

    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Number(self.literal()))
    }

    fn as_table(&self, cx: &mut Cx) -> Result<Value> {
        cx.factory().table(vec![
            (
                Symbol::new("kind"),
                cx.factory().string("complex".to_owned())?,
            ),
            (Symbol::new("domain"), cx.factory().symbol(number_domain())?),
            (
                Symbol::new("real"),
                cx.factory()
                    .number_literal(domains::f64(), self.real.to_string())?,
            ),
            (
                Symbol::new("imag"),
                cx.factory()
                    .number_literal(domains::f64(), self.imag.to_string())?,
            ),
        ])
    }

    fn as_number_value(&self) -> Option<&dyn NumberValue> {
        Some(self)
    }

    fn as_object_encoder(&self) -> Option<&dyn ObjectEncode> {
        Some(self)
    }
}

impl NumberValue for ComplexValue {
    fn number_domain(&self, _cx: &mut Cx) -> Result<Symbol> {
        Ok(number_domain())
    }

    fn number_literal(&self, _cx: &mut Cx) -> Result<Option<NumberLiteral>> {
        Ok(Some(self.literal()))
    }
}

impl ObjectEncode for ComplexValue {
    fn object_encoding(&self, _cx: &mut Cx) -> Result<ObjectEncoding> {
        Ok(ObjectEncoding::Constructor {
            class: complex_value_class_symbol(),
            args: vec![
                Expr::Symbol(Symbol::new("v1")),
                self.real.encode_field(),
                self.imag.encode_field(),
            ],
        })
    }
}

impl sim_citizen::Citizen for ComplexValue {
    fn citizen_symbol() -> Symbol {
        complex_value_class_symbol()
    }

    fn citizen_version() -> u32 {
        1
    }

    fn citizen_arity() -> usize {
        2
    }

    fn citizen_fields() -> &'static [&'static str] {
        &["real", "imag"]
    }
}

pub(crate) struct ComplexValueClass {
    id: AtomicU32,
}

pub(crate) fn build_complex_value_class() -> Arc<ComplexValueClass> {
    Arc::new(ComplexValueClass {
        id: AtomicU32::new(0),
    })
}

impl ComplexValueClass {
    pub(crate) fn set_id(&self, id: ClassId) {
        self.id.store(id.0, Ordering::Relaxed);
    }
}

impl Object for ComplexValueClass {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<class {}>", complex_value_class_symbol()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ComplexValueClass {
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
        Ok(Expr::Symbol(complex_value_class_symbol()))
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

impl Callable for ComplexValueClass {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [version, real, imag] = values.as_slice() else {
            return Err(arity_error(complex_value_class_symbol(), 3, values.len()));
        };
        decode_version(cx, version.clone(), 1, complex_value_class_symbol())?;
        let real = f64::decode_field_value(cx, real.clone(), "real")?;
        let imag = f64::decode_field_value(cx, imag.clone(), "imag")?;
        complex_value(cx, real, imag)
    }
}

impl Class for ComplexValueClass {
    fn id(&self) -> ClassId {
        ClassId(self.id.load(Ordering::Relaxed))
    }

    fn symbol(&self) -> Symbol {
        complex_value_class_symbol()
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
            .class_by_symbol(&complex_value_class_symbol())
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
                cx.factory().list(vec![
                    cx.factory().symbol(Symbol::new("real"))?,
                    cx.factory().symbol(Symbol::new("imag"))?,
                ])?,
            ),
        ])
    }
}

impl ReadConstructor for ComplexValueClass {
    fn symbol(&self) -> Symbol {
        complex_value_class_symbol()
    }

    fn args_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }

    fn construct_read(&self, cx: &mut Cx, args: Vec<Value>) -> Result<Value> {
        if args.len() != 3 {
            return Err(arity_error(complex_value_class_symbol(), 3, args.len()));
        }
        self.call(cx, Args::new(args))
    }
}

/// Constructs an opaque `numbers/complex` value from finite real and imaginary
/// parts, erroring when either part is non-finite.
pub fn complex_value(cx: &mut Cx, real: f64, imag: f64) -> Result<Value> {
    if !real.is_finite() || !imag.is_finite() {
        return Err(Error::Eval(
            "numbers/Complex constructor requires finite f64 parts".to_owned(),
        ));
    }
    cx.factory().opaque(Arc::new(ComplexValue::new(real, imag)))
}
