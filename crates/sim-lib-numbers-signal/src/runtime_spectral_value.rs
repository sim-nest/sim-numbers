//! Parsing and rendering helpers shared by spectral runtime operations.

use std::collections::BTreeMap;

use sim_kernel::{Cx, Error, Expr, NumberLiteral, Result, Symbol, Value, force_list_to_vec};

use crate::{
    ArOrderCriterion, BurgTermination, EndpointConvention, Normalization, NyquistConvention,
    Periodicity, SignConvention,
    runtime_convolution_render::f64_value,
    runtime_spectral_callable::{SpectralOperation, argument_error},
};

pub(crate) type Options = BTreeMap<String, Value>;

pub(crate) fn expr_options(
    cx: &mut Cx,
    operation: SpectralOperation,
    exprs: &[Expr],
) -> Result<Options> {
    if !exprs.len().is_multiple_of(2) {
        return Err(argument_error(operation));
    }
    let mut options = Options::new();
    for pair in exprs.chunks(2) {
        let key = keyword_expr(operation, &pair[0])?;
        insert_option(operation, &mut options, key, cx.eval_expr(pair[1].clone())?)?;
    }
    Ok(options)
}

pub(crate) fn value_options(
    cx: &mut Cx,
    operation: SpectralOperation,
    values: &[Value],
) -> Result<Options> {
    if !values.len().is_multiple_of(2) {
        return Err(argument_error(operation));
    }
    let mut options = Options::new();
    for pair in values.chunks(2) {
        let key = keyword_expr(operation, &pair[0].object().as_expr(cx)?)?;
        insert_option(operation, &mut options, key, pair[1].clone())?;
    }
    Ok(options)
}

fn insert_option(
    operation: SpectralOperation,
    options: &mut Options,
    key: String,
    value: Value,
) -> Result<()> {
    if options.insert(key.clone(), value).is_some() {
        return Err(Error::Eval(format!(
            "{}: duplicate option :{key}",
            operation.name()
        )));
    }
    Ok(())
}

fn keyword_expr(operation: SpectralOperation, expression: &Expr) -> Result<String> {
    let Expr::Symbol(symbol) = expression else {
        return Err(argument_error(operation));
    };
    symbol
        .name
        .strip_prefix(':')
        .map(str::to_owned)
        .ok_or_else(|| argument_error(operation))
}

pub(crate) fn reject_unknown(
    operation: SpectralOperation,
    options: &Options,
    allowed: &[&str],
) -> Result<()> {
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::Eval(format!(
                "{}: unknown option :{key}",
                operation.name()
            )));
        }
    }
    Ok(())
}

pub(crate) fn option_symbol(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<String>> {
    options
        .get(key)
        .map(|value| {
            let Expr::Symbol(symbol) = value.object().as_expr(cx)? else {
                return Err(Error::Eval(format!("option :{key} must be a symbol")));
            };
            Ok(symbol.as_qualified_str().to_owned())
        })
        .transpose()
}

pub(crate) fn option_f64(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<f64>> {
    options
        .get(key)
        .map(|value| {
            parse_number(cx, value, key)?
                .canonical
                .parse::<f64>()
                .map_err(|_| Error::Eval(format!("option :{key} must be f64")))
        })
        .transpose()
}

pub(crate) fn option_usize(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<usize>> {
    options
        .get(key)
        .map(|value| {
            parse_number(cx, value, key)?
                .canonical
                .parse::<usize>()
                .map_err(|_| Error::Eval(format!("option :{key} must be a non-negative integer")))
        })
        .transpose()
}

pub(crate) fn option_u64(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<u64>> {
    options
        .get(key)
        .map(|value| {
            parse_number(cx, value, key)?
                .canonical
                .parse::<u64>()
                .map_err(|_| Error::Eval(format!("option :{key} must be a non-negative integer")))
        })
        .transpose()
}

fn parse_number(cx: &mut Cx, value: &Value, name: &str) -> Result<NumberLiteral> {
    value
        .object()
        .as_number_value()
        .ok_or(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        })?
        .number_literal(cx)?
        .ok_or_else(|| Error::Eval(format!("signal value {name} has no numeric literal")))
}

fn value_list(
    cx: &mut Cx,
    value: &Value,
    name: &str,
    operation: SpectralOperation,
) -> Result<Vec<Value>> {
    if let Some(list) = value.object().as_list() {
        return force_list_to_vec(cx, list, &format!("{} {name}", operation.name()));
    }
    match value.object().as_expr(cx)? {
        Expr::List(items) | Expr::Block(items) => {
            items.into_iter().map(|item| cx.eval_expr(item)).collect()
        }
        expression => Err(Error::Eval(format!(
            "{} {name} must be a list, got {expression:?}",
            operation.name()
        ))),
    }
}

pub(crate) fn real_list(
    cx: &mut Cx,
    value: &Value,
    name: &str,
    operation: SpectralOperation,
) -> Result<Vec<f64>> {
    value_list(cx, value, name, operation)?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            parse_number(cx, value, &format!("{name}[{index}]"))?
                .canonical
                .parse::<f64>()
                .map_err(|_| {
                    Error::Eval(format!("{} {name}[{index}] must be f64", operation.name()))
                })
        })
        .collect()
}

pub(crate) fn complex_list(
    cx: &mut Cx,
    value: &Value,
    name: &str,
    operation: SpectralOperation,
) -> Result<Vec<(f64, f64)>> {
    value_list(cx, value, name, operation)?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let pair = real_list(cx, value, &format!("{name}[{index}]"), operation)?;
            let [real, imag] = pair.as_slice() else {
                return Err(Error::Eval(format!(
                    "{} {name}[{index}] must contain real and imaginary components",
                    operation.name()
                )));
            };
            Ok((*real, *imag))
        })
        .collect()
}

pub(crate) fn complex_values(cx: &mut Cx, values: &[(f64, f64)]) -> Result<Value> {
    let values = values
        .iter()
        .map(|(real, imag)| {
            let real = f64_value(cx, *real)?;
            let imag = f64_value(cx, *imag)?;
            cx.factory().list(vec![real, imag])
        })
        .collect::<Result<Vec<_>>>()?;
    cx.factory().list(values)
}

pub(crate) fn u64_value(cx: &mut Cx, value: u64) -> Result<Value> {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
}

pub(crate) fn parse_normalization(name: &str) -> Result<Normalization> {
    match name {
        "none" => Ok(Normalization::None),
        "forward" => Ok(Normalization::Forward),
        "inverse" => Ok(Normalization::Inverse),
        "orthonormal" | "unitary" => Ok(Normalization::Orthonormal),
        _ => Err(Error::Eval(format!("unsupported DFT normalization {name}"))),
    }
}

pub(crate) fn criterion_name(value: ArOrderCriterion) -> &'static str {
    match value {
        ArOrderCriterion::Fixed => "fixed",
        ArOrderCriterion::Akaike => "aic",
        ArOrderCriterion::Bayesian => "bic",
        ArOrderCriterion::FinalPredictionError => "fpe",
    }
}

pub(crate) fn termination_name(value: BurgTermination) -> &'static str {
    match value {
        BurgTermination::RequestedOrder => "requested-order",
        BurgTermination::SingularAt(_) => "singular-reduction",
        BurgTermination::UnstableAt(_) => "unstable-reduction",
    }
}

pub(crate) fn periodicity_name(value: Periodicity) -> &'static str {
    match value {
        Periodicity::Wrap => "wrap",
        Periodicity::PrincipalPeriod => "principal-period",
    }
}

pub(crate) fn endpoint_name(value: EndpointConvention) -> &'static str {
    match value {
        EndpointConvention::Excluded => "excluded",
        EndpointConvention::Included => "included",
    }
}

pub(crate) fn normalization_name(value: Normalization) -> &'static str {
    match value {
        Normalization::None => "none",
        Normalization::Forward => "forward",
        Normalization::Inverse => "inverse",
        Normalization::Orthonormal => "unitary",
    }
}

pub(crate) fn sign_name(value: SignConvention) -> &'static str {
    match value {
        SignConvention::NegativeForward => "negative-forward",
        SignConvention::PositiveForward => "positive-forward",
    }
}

pub(crate) fn nyquist_name(value: NyquistConvention) -> &'static str {
    match value {
        NyquistConvention::Positive => "positive",
        NyquistConvention::Negative => "negative",
    }
}
