//! Loadable Lisp-facing transform callable.

use std::{any::Any, sync::Arc};

use sim_kernel::{
    AbiVersion, Args, Callable, ClassRef, Cx, DefaultFactory, Dependency, Error, Export, Expr,
    Factory, Lib, LibManifest, LibTarget, Linker, NumberLiteral, Object, RawArgs, Result, Symbol,
    Value, Version, force_list_to_vec,
};

use crate::{
    DctType, Direction, DstType, Normalization, SignalBuffer, SignalError, SignalView,
    SpectrumPacking, TransformKind, TransformPlan, transform,
};

/// Symbol of the Lisp-facing transform operation (`signal/transform`).
pub fn signal_transform_symbol() -> Symbol {
    Symbol::qualified("signal", "transform")
}

#[derive(Clone)]
struct SignalTransformFunction;

impl Object for SignalTransformFunction {
    fn display(&self, _cx: &mut Cx) -> Result<String> {
        Ok(format!("#<function {}>", signal_transform_symbol()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for SignalTransformFunction {
    fn class(&self, cx: &mut Cx) -> Result<ClassRef> {
        if let Some(value) = cx
            .registry()
            .class_by_symbol(&Symbol::qualified("core", "Function"))
        {
            return Ok(value.clone());
        }
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }

    fn as_expr(&self, _cx: &mut Cx) -> Result<Expr> {
        Ok(Expr::Symbol(signal_transform_symbol()))
    }

    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for SignalTransformFunction {
    fn call(&self, cx: &mut Cx, args: Args) -> Result<Value> {
        call_signal_transform(cx, args)
    }

    fn call_exprs(&self, cx: &mut Cx, args: RawArgs) -> Result<Value> {
        let values = args.into_exprs();
        let [kind, direction, normalization, packing, len, input] = values.as_slice() else {
            return Err(argument_error());
        };
        let options = ParsedOptions {
            kind: option_expr(kind, "kind")?,
            direction: option_expr(direction, "direction")?,
            normalization: option_expr(normalization, "normalization")?,
            packing: option_expr(packing, "packing")?,
        };
        let len = cx.eval_expr(len.clone())?;
        let input = cx.eval_expr(input.clone())?;
        execute(cx, options, &len, &input)
    }
}

/// Calls `signal/transform` with evaluated symbol options, a length, and a
/// real list or list of complex `(real imag)` pairs.
pub fn call_signal_transform(cx: &mut Cx, args: Args) -> Result<Value> {
    let values = args.into_vec();
    let [kind, direction, normalization, packing, len, input] = values.as_slice() else {
        return Err(argument_error());
    };
    let options = ParsedOptions {
        kind: option_value(cx, kind, "kind")?,
        direction: option_value(cx, direction, "direction")?,
        normalization: option_value(cx, normalization, "normalization")?,
        packing: option_value(cx, packing, "packing")?,
    };
    execute(cx, options, len, input)
}

struct ParsedOptions {
    kind: String,
    direction: String,
    normalization: String,
    packing: String,
}

fn execute(cx: &mut Cx, options: ParsedOptions, len: &Value, input: &Value) -> Result<Value> {
    let kind = parse_kind(&options.kind)?;
    let direction = parse_direction(&options.direction)?;
    let normalization = parse_normalization(&options.normalization)?;
    let packing = parse_packing(&options.packing)?;
    let len = value_to_usize(cx, len, "len")?;
    let mut plan = TransformPlan::new(kind, len);
    plan.direction = direction;
    plan.normalization = normalization;
    plan.packing = packing;

    let output = match (kind, direction) {
        (TransformKind::Dft | TransformKind::Fft, _)
        | (TransformKind::RealFft, Direction::Inverse) => {
            let input = value_to_complex_list(cx, input)?;
            transform(&plan, SignalView::Complex(&input))
        }
        (TransformKind::RealFft | TransformKind::Dct(_) | TransformKind::Dst(_), _) => {
            let input = value_to_real_list(cx, input)?;
            transform(&plan, SignalView::Real(&input))
        }
    }
    .map_err(signal_error_to_kernel)?;
    buffer_to_value(cx, output)
}

/// Loadable runtime library exporting [`signal_transform_symbol`].
pub struct SignalNumbersLib;

impl SignalNumbersLib {
    /// Creates the stateless signal-transform library.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SignalNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for SignalNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "signal"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Function {
                symbol: signal_transform_symbol(),
                function_id: None,
            }],
        }
    }

    fn load(&self, _cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        linker.function_value(
            signal_transform_symbol(),
            DefaultFactory.opaque(Arc::new(SignalTransformFunction))?,
        )?;
        Ok(())
    }
}

fn parse_kind(name: &str) -> Result<TransformKind> {
    match name {
        "dft" => Ok(TransformKind::Dft),
        "fft" => Ok(TransformKind::Fft),
        "rfft" => Ok(TransformKind::RealFft),
        "dct-i" => Ok(TransformKind::Dct(DctType::I)),
        "dct-ii" => Ok(TransformKind::Dct(DctType::II)),
        "dct-iii" => Ok(TransformKind::Dct(DctType::III)),
        "dct-iv" => Ok(TransformKind::Dct(DctType::IV)),
        "dst-i" => Ok(TransformKind::Dst(DstType::I)),
        "dst-ii" => Ok(TransformKind::Dst(DstType::II)),
        "dst-iii" => Ok(TransformKind::Dst(DstType::III)),
        "dst-iv" => Ok(TransformKind::Dst(DstType::IV)),
        _ => Err(Error::Eval(format!(
            "signal/transform unsupported kind {name}"
        ))),
    }
}

fn parse_direction(name: &str) -> Result<Direction> {
    match name {
        "forward" => Ok(Direction::Forward),
        "inverse" => Ok(Direction::Inverse),
        _ => Err(Error::Eval(format!(
            "signal/transform direction must be forward or inverse, got {name}"
        ))),
    }
}

fn parse_normalization(name: &str) -> Result<Normalization> {
    match name {
        "none" => Ok(Normalization::None),
        "forward" => Ok(Normalization::Forward),
        "inverse" => Ok(Normalization::Inverse),
        "orthonormal" => Ok(Normalization::Orthonormal),
        _ => Err(Error::Eval(format!(
            "signal/transform unsupported normalization {name}"
        ))),
    }
}

fn parse_packing(name: &str) -> Result<SpectrumPacking> {
    match name {
        "full" => Ok(SpectrumPacking::Full),
        "hermitian-half" => Ok(SpectrumPacking::HermitianHalf),
        _ => Err(Error::Eval(format!(
            "signal/transform packing must be full or hermitian-half, got {name}"
        ))),
    }
}

fn option_expr(expr: &Expr, name: &str) -> Result<String> {
    let Expr::Symbol(symbol) = expr else {
        return Err(Error::Eval(format!(
            "signal/transform {name} must be an unquoted option symbol"
        )));
    };
    Ok(symbol.as_qualified_str().to_owned())
}

fn option_value(cx: &mut Cx, value: &Value, name: &str) -> Result<String> {
    let Expr::Symbol(symbol) = value.object().as_expr(cx)? else {
        return Err(Error::Eval(format!(
            "signal/transform {name} must be a symbol"
        )));
    };
    Ok(symbol.as_qualified_str().to_owned())
}

fn value_to_usize(cx: &mut Cx, value: &Value, name: &str) -> Result<usize> {
    let literal = value_to_number(cx, value, name)?;
    literal.canonical.parse::<usize>().map_err(|_| {
        Error::Eval(format!(
            "signal/transform {name} must be a non-negative integer"
        ))
    })
}

fn value_to_real_list(cx: &mut Cx, value: &Value) -> Result<Vec<f64>> {
    value_to_list(cx, value, "input")?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value_to_number(cx, value, &format!("input[{index}]"))?
                .canonical
                .parse::<f64>()
                .map_err(|_| {
                    Error::Eval(format!(
                        "signal/transform input[{index}] must be an f64-compatible number"
                    ))
                })
        })
        .collect()
}

fn value_to_complex_list(cx: &mut Cx, value: &Value) -> Result<Vec<(f64, f64)>> {
    value_to_list(cx, value, "input")?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let pair = value_to_list(cx, value, &format!("input[{index}]"))?;
            let [real, imag] = pair.as_slice() else {
                return Err(Error::Eval(format!(
                    "signal/transform input[{index}] must contain real and imaginary components"
                )));
            };
            Ok((
                value_to_number(cx, real, &format!("input[{index}].real"))?
                    .canonical
                    .parse::<f64>()
                    .map_err(|_| Error::Eval("complex real component must be f64".to_owned()))?,
                value_to_number(cx, imag, &format!("input[{index}].imag"))?
                    .canonical
                    .parse::<f64>()
                    .map_err(|_| {
                        Error::Eval("complex imaginary component must be f64".to_owned())
                    })?,
            ))
        })
        .collect()
}

fn value_to_number(cx: &mut Cx, value: &Value, name: &str) -> Result<NumberLiteral> {
    value
        .object()
        .as_number_value()
        .ok_or(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        })?
        .number_literal(cx)?
        .ok_or_else(|| Error::Eval(format!("signal/transform {name} has no numeric literal")))
}

fn value_to_list(cx: &mut Cx, value: &Value, name: &str) -> Result<Vec<Value>> {
    let list = value.object().as_list().ok_or(Error::TypeMismatch {
        expected: "list",
        found: "non-list",
    })?;
    force_list_to_vec(cx, list, &format!("signal/transform {name}"))
}

fn buffer_to_value(cx: &mut Cx, output: SignalBuffer) -> Result<Value> {
    match output {
        SignalBuffer::Real(values) => {
            let values = values
                .as_slice()
                .iter()
                .map(|value| f64_value(cx, *value))
                .collect::<Result<Vec<_>>>()?;
            cx.factory().list(values)
        }
        SignalBuffer::Complex(values) => {
            let values = values
                .as_slice()
                .iter()
                .map(|(real, imag)| {
                    let real = f64_value(cx, *real)?;
                    let imag = f64_value(cx, *imag)?;
                    cx.factory().list(vec![real, imag])
                })
                .collect::<Result<Vec<_>>>()?;
            cx.factory().list(values)
        }
    }
}

fn f64_value(cx: &mut Cx, value: f64) -> Result<Value> {
    let canonical = if value == 0.0 {
        "0".to_owned()
    } else {
        value.to_string()
    };
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "f64"), canonical)
}

fn signal_error_to_kernel(error: SignalError) -> Error {
    Error::Eval(format!("signal/transform: {error}"))
}

fn argument_error() -> Error {
    Error::Eval(
        "signal/transform expects kind, direction, normalization, packing, len, and input"
            .to_owned(),
    )
}
