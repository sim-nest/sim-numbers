//! Tensor number-domain registration: the `TensorNumbersLib` that installs the
//! tensor domain, its value class, and its constructor operations.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, DefaultFactory, Dependency, Export, Expr, Factory, Lib, LibManifest, LibTarget,
    Linker, NumberDomain, Object, Result, Symbol, Value, Version,
};
use sim_lib_numbers_core::{
    DomainNumberValueShape, NumberDomainTableSpec, domains, number_domain_table,
};
use sim_shape::shape_value;

use super::{
    citizen::{register_tensor_value_class, tensor_value_class_symbol},
    function::{
        TensorFunction, index_symbol, map_symbol, mat_symbol, reshape_symbol, scalar_symbol,
        slice_symbol, tensor_symbol, vec_symbol,
    },
};

/// The symbol naming the tensor number domain (`numbers/tensor`).
pub fn number_domain() -> Symbol {
    domains::tensor()
}

fn literal_class_symbol() -> Symbol {
    domains::literal_class("tensor")
}

fn literal_instance_shape_symbol() -> Symbol {
    Symbol::qualified(literal_class_symbol().to_string(), "instance-shape")
}

fn value_shape_symbol() -> Symbol {
    sim_lib_numbers_core::value_shape_symbol(&number_domain())
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/tensor number-domain marker; reconstruct by loading the tensor number lib",
    kind = "marker"
)]
struct TensorNumberDomain;

impl NumberDomain for TensorNumberDomain {
    fn symbol(&self) -> Symbol {
        number_domain()
    }

    fn parse_priority(&self) -> i32 {
        -200
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

impl Object for TensorNumberDomain {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok("#<number-domain numbers/tensor>".to_owned())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for TensorNumberDomain {
    fn class(&self, cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        sim_lib_numbers_core::number_domain_class_stub(cx)
    }
    fn as_expr(&self, _cx: &mut sim_kernel::Cx) -> Result<Expr> {
        Ok(Expr::Symbol(number_domain()))
    }
    fn as_table(&self, cx: &mut sim_kernel::Cx) -> Result<Value> {
        let literal_class = cx
            .registry()
            .class_by_symbol(&literal_class_symbol())
            .cloned()
            .unwrap_or(cx.factory().symbol(literal_class_symbol())?);
        let instance_shape = cx
            .registry()
            .shape_by_symbol(&literal_instance_shape_symbol())
            .cloned()
            .unwrap_or(cx.factory().symbol(literal_instance_shape_symbol())?);
        let value_shape = cx
            .registry()
            .shape_by_symbol(&value_shape_symbol())
            .cloned()
            .unwrap_or(cx.factory().symbol(value_shape_symbol())?);
        number_domain_table(
            cx,
            NumberDomainTableSpec::new(
                number_domain(),
                "tensor",
                "value-only",
                -200,
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

struct TensorLiteralShape;

impl sim_shape::Shape for TensorLiteralShape {
    fn check_value(
        &self,
        _cx: &mut sim_kernel::Cx,
        _value: Value,
    ) -> Result<sim_shape::ShapeMatch> {
        Ok(sim_shape::ShapeMatch::reject(
            "numbers/tensor has no parsed literal surface".to_owned(),
        ))
    }

    fn check_expr(&self, _cx: &mut sim_kernel::Cx, _expr: &Expr) -> Result<sim_shape::ShapeMatch> {
        Ok(sim_shape::ShapeMatch::reject(
            "numbers/tensor has no parsed literal surface".to_owned(),
        ))
    }

    fn describe(&self, _cx: &mut sim_kernel::Cx) -> Result<sim_shape::ShapeDoc> {
        Ok(sim_shape::ShapeDoc::new("TensorLiteral")
            .with_detail("placeholder literal shape for the numbers/tensor domain")
            .with_detail("tensor values are constructed by functions rather than parsed literals"))
    }
}

#[sim_citizen_derive::non_citizen(
    reason = "numbers/tensor literal class marker; tensor values use the numbers/Tensor citizen descriptor",
    kind = "marker"
)]
struct TensorLiteralClass;

impl Object for TensorLiteralClass {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok(format!("#<class {}>", literal_class_symbol()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl sim_kernel::ObjectCompat for TensorLiteralClass {
    fn class(&self, cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
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
    fn as_expr(&self, _cx: &mut sim_kernel::Cx) -> Result<Expr> {
        Ok(Expr::Symbol(literal_class_symbol()))
    }
}

/// Registered number-domain library that installs the `numbers/tensor` domain.
///
/// Loading this [`Lib`] registers the tensor number domain and its value class,
/// the placeholder literal and value shapes, and the tensor constructor
/// operations (`tensor`, `scalar`, `vec`, `mat`, `index`, `reshape`, `slice`,
/// `map`). Specialized element-type backends layer on top through the
/// [`SpecTensor`](crate::SpecTensor) interface.
pub struct TensorNumbersLib;

impl TensorNumbersLib {
    /// Creates the tensor domain library. The value is stateless; the domain,
    /// classes, shapes, and functions are installed when it is loaded into a
    /// [`Cx`](sim_kernel::Cx).
    pub fn new() -> Self {
        Self
    }
}

impl Default for TensorNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for TensorNumbersLib {
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
                Export::Class {
                    symbol: tensor_value_class_symbol(),
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
                export_function(tensor_symbol()),
                export_function(scalar_symbol()),
                export_function(vec_symbol()),
                export_function(mat_symbol()),
                export_function(index_symbol()),
                export_function(reshape_symbol()),
                export_function(slice_symbol()),
                export_function(map_symbol()),
            ],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.number_domain_value(
            number_domain(),
            DefaultFactory
                .opaque(Arc::new(TensorNumberDomain))
                .expect("tensor domain should be boxable"),
        )?;
        linker.class_value(
            literal_class_symbol(),
            DefaultFactory
                .opaque(Arc::new(TensorLiteralClass))
                .expect("tensor literal class should be boxable"),
        )?;
        register_tensor_value_class(linker)?;
        linker.shape_value(
            literal_instance_shape_symbol(),
            shape_value(
                literal_instance_shape_symbol(),
                Arc::new(TensorLiteralShape),
            ),
        )?;
        linker.shape_value(
            value_shape_symbol(),
            shape_value(
                value_shape_symbol(),
                Arc::new(DomainNumberValueShape::new(
                    number_domain(),
                    "TensorValue",
                    [
                        "number value in the numbers/tensor domain",
                        "accepts tensor-shaped collections of scalar number cells",
                    ],
                )),
            ),
        )?;

        for symbol in [
            tensor_symbol(),
            scalar_symbol(),
            vec_symbol(),
            mat_symbol(),
            index_symbol(),
            reshape_symbol(),
            slice_symbol(),
            map_symbol(),
        ] {
            linker.function_value(
                symbol.clone(),
                DefaultFactory
                    .opaque(Arc::new(TensorFunction { symbol }))
                    .expect("tensor function should be boxable"),
            )?;
        }
        Ok(())
    }
}

fn export_function(symbol: Symbol) -> Export {
    Export::Function {
        symbol,
        function_id: None,
    }
}
