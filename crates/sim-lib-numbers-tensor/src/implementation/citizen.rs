//! The tensor value class as a runtime citizen: its class registration and the
//! read-constructor that reconstructs tensor values from encoded form.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use sim_citizen::{arity_error, decode_version};
use sim_kernel::{
    Args, Callable, Class, ClassId, ClassRef, Cx, DefaultFactory, Error, Expr, Factory, Linker,
    Object, ReadConstructor, ReadConstructorRef, Result, ShapeRef, Symbol, TableRef, Value,
    force_list_to_vec,
};
use sim_lib_numbers_core::domains;

use super::{dimension::extract_dims, domain::number_domain, value::build_tensor_value};

/// The symbol naming the tensor value class (`numbers/Tensor`) under which
/// tensor values register and reconstruct.
pub fn tensor_value_class_symbol() -> Symbol {
    domains::tensor_value_class()
}

fn value_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&number_domain())
}

struct TensorValueClass {
    id: AtomicU32,
}

impl TensorValueClass {
    fn new() -> Self {
        Self {
            id: AtomicU32::new(0),
        }
    }

    fn set_id(&self, id: ClassId) {
        self.id.store(id.0, Ordering::Relaxed);
    }
}

impl Object for TensorValueClass {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<class {}>", tensor_value_class_symbol()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for TensorValueClass {
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
        Ok(Expr::Symbol(tensor_value_class_symbol()))
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

impl Callable for TensorValueClass {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        let values = args.into_vec();
        let [version, shape, data, domain] = values.as_slice() else {
            return Err(arity_error(tensor_value_class_symbol(), 4, values.len()));
        };
        decode_version(cx, version.clone(), 1, tensor_value_class_symbol())?;
        let shape = extract_dims(cx, shape, "numbers/Tensor shape")?;
        let data = decode_data(cx, data)?;
        let domain = decode_domain(cx, domain)?;
        if domain == number_domain() {
            return Err(Error::Eval(
                "numbers/Tensor domain field must name a scalar number domain".to_owned(),
            ));
        }
        build_tensor_value(cx, shape, Some(domain), data)
    }
}

impl Class for TensorValueClass {
    fn id(&self) -> ClassId {
        ClassId(self.id.load(Ordering::Relaxed))
    }

    fn symbol(&self) -> Symbol {
        tensor_value_class_symbol()
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
            .class_by_symbol(&tensor_value_class_symbol())
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
                    cx.factory().symbol(Symbol::new("shape"))?,
                    cx.factory().symbol(Symbol::new("data"))?,
                    cx.factory().symbol(Symbol::new("domain"))?,
                ])?,
            ),
        ])
    }
}

impl ReadConstructor for TensorValueClass {
    fn symbol(&self) -> Symbol {
        tensor_value_class_symbol()
    }

    fn args_shape(&self, cx: &mut Cx) -> Result<ShapeRef> {
        cx.factory().nil()
    }

    fn construct_read(&self, cx: &mut Cx, args: Vec<Value>) -> Result<Value> {
        if args.len() != 4 {
            return Err(arity_error(tensor_value_class_symbol(), 4, args.len()));
        }
        self.call(cx, Args::new(args))
    }
}

fn decode_data(cx: &mut Cx, value: &Value) -> Result<Vec<Value>> {
    let list = value
        .object()
        .as_list()
        .ok_or_else(|| Error::Eval("numbers/Tensor data field must be a list".to_owned()))?;
    force_list_to_vec(cx, list, "numbers/Tensor data")
}

fn decode_domain(cx: &mut Cx, value: &Value) -> Result<Symbol> {
    match value.object().as_expr(cx)? {
        Expr::Symbol(symbol) => Ok(symbol),
        _ => Err(Error::Eval(
            "numbers/Tensor domain field must be a symbol".to_owned(),
        )),
    }
}

pub(crate) fn register_tensor_value_class(linker: &mut Linker<'_>) -> Result<()> {
    let class = Arc::new(TensorValueClass::new());
    let id = linker.class_value(
        tensor_value_class_symbol(),
        DefaultFactory
            .opaque(class.clone())
            .expect("tensor value class should be boxable"),
    )?;
    class.set_id(id);
    Ok(())
}

fn install_tensor_value_citizen(linker: &mut Linker<'_>) -> Result<()> {
    register_tensor_value_class(linker)
}

fn conformance_tensor_value_citizen(cx: &mut Cx) -> Result<()> {
    let dtype = domains::i64();
    let value = build_tensor_value(
        cx,
        vec![2],
        Some(dtype.clone()),
        vec![i64_cell("1")?, i64_cell("2")?],
    )?;
    sim_citizen::check_value_fixture_with_wrong_version(
        cx,
        value,
        Some(vec![
            Expr::Symbol(Symbol::new("v999")),
            Expr::List(vec![int_expr("2")]),
            Expr::List(vec![
                Expr::Number(sim_kernel::NumberLiteral {
                    domain: dtype.clone(),
                    canonical: "1".to_owned(),
                }),
                Expr::Number(sim_kernel::NumberLiteral {
                    domain: dtype.clone(),
                    canonical: "2".to_owned(),
                }),
            ]),
            Expr::Symbol(dtype),
        ]),
    )
}

fn i64_cell(canonical: &str) -> Result<Value> {
    DefaultFactory.number_literal(domains::i64(), canonical.to_owned())
}

fn int_expr(canonical: &str) -> Expr {
    Expr::Number(sim_kernel::NumberLiteral {
        domain: Symbol::qualified("citizen", "int"),
        canonical: canonical.to_owned(),
    })
}

sim_citizen::inventory::submit! {
    sim_citizen::CitizenInfo {
        symbol: "numbers/Tensor",
        version: 1,
        crate_name: env!("CARGO_PKG_NAME"),
        arity: 3,
        install: install_tensor_value_citizen,
        conformance: conformance_tensor_value_citizen,
    }
}
