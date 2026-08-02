//! Lisp-facing convolution, correlation, and guarded deconvolution execution.

use std::collections::BTreeMap;

use sim_kernel::{Cx, Error, Expr, Result, Symbol, Value};

use crate::{
    BoundaryPolicy, ConvolutionAlgorithm, ConvolutionMode, ConvolutionNormalization,
    ConvolutionPlan, CorrelationNormalization, CorrelationPlan, DeconvolutionMode,
    DeconvolutionPlan, LagOrder, LinearOutput, Regularization, SignalError, convolve, correlate,
    deconvolve,
    runtime_convolution_callable::{Operation, argument_error},
    runtime_convolution_render::{
        algorithm_name, correlation_normalization_name, f64_value, lag_order_name, real_values,
        regularization_name, signed_value, symbol_value, usize_value,
    },
    runtime_convolution_value::{number_value, real_list},
};

type Options = BTreeMap<String, Value>;

pub(crate) fn execute_exprs(cx: &mut Cx, operation: Operation, exprs: Vec<Expr>) -> Result<Value> {
    let [left, right, rest @ ..] = exprs.as_slice() else {
        return Err(argument_error(operation));
    };
    let left = cx.eval_expr(left.clone())?;
    let right = cx.eval_expr(right.clone())?;
    let options = expr_options(cx, operation, rest)?;
    execute(cx, operation, &left, &right, &options)
}

pub(crate) fn execute_values(
    cx: &mut Cx,
    operation: Operation,
    values: Vec<Value>,
) -> Result<Value> {
    let [left, right, rest @ ..] = values.as_slice() else {
        return Err(argument_error(operation));
    };
    let options = value_options(cx, operation, rest)?;
    execute(cx, operation, left, right, &options)
}

fn execute(
    cx: &mut Cx,
    operation: Operation,
    left: &Value,
    right: &Value,
    options: &Options,
) -> Result<Value> {
    let left = real_list(cx, operation, left, "signal")?;
    let right = real_list(cx, operation, right, "kernel")?;
    match operation {
        Operation::Convolve => execute_convolve(cx, &left, &right, options),
        Operation::Correlate => execute_correlate(cx, &left, &right, options),
        Operation::Deconvolve => execute_deconvolve(cx, &left, &right, options),
    }
}

fn execute_convolve(
    cx: &mut Cx,
    signal: &[f64],
    kernel: &[f64],
    options: &Options,
) -> Result<Value> {
    reject_unknown(
        Operation::Convolve,
        options,
        &["mode", "period", "algorithm", "boundary", "normalization"],
    )?;
    let mode = convolution_mode(cx, options, signal.len())?;
    let plan = ConvolutionPlan {
        mode,
        algorithm: algorithm(cx, options)?,
        boundary: boundary(cx, options, mode)?,
        normalization: convolution_normalization(cx, options)?,
    };
    let result = convolve(signal, kernel, &plan).map_err(operation_error(Operation::Convolve))?;
    let entries = vec![
        (
            Symbol::new("samples"),
            real_values(cx, result.samples.as_slice())?,
        ),
        (
            Symbol::new("algorithm"),
            symbol_value(cx, algorithm_name(result.report.cost.selected))?,
        ),
        (
            Symbol::new("direct-cost-units"),
            usize_value(cx, result.report.cost.direct_cost_units)?,
        ),
        (
            Symbol::new("fft-cost-units"),
            usize_value(cx, result.report.cost.fft_cost_units)?,
        ),
        (
            Symbol::new("fft-len"),
            usize_value(cx, result.report.cost.fft_len)?,
        ),
        (
            Symbol::new("retained-start"),
            usize_value(cx, result.report.retained_start)?,
        ),
        (
            Symbol::new("retained-len"),
            usize_value(cx, result.report.retained_len)?,
        ),
    ];
    cx.factory().table(entries)
}

fn execute_correlate(cx: &mut Cx, left: &[f64], right: &[f64], options: &Options) -> Result<Value> {
    reject_unknown(
        Operation::Correlate,
        options,
        &[
            "mode",
            "period",
            "algorithm",
            "boundary",
            "normalization",
            "lag-order",
        ],
    )?;
    let mode = convolution_mode(cx, options, left.len())?;
    let plan = CorrelationPlan {
        mode,
        algorithm: algorithm(cx, options)?,
        boundary: boundary(cx, options, mode)?,
        normalization: correlation_normalization(cx, options)?,
        lag_order: lag_order(cx, options)?,
    };
    let result = correlate(left, right, &plan).map_err(operation_error(Operation::Correlate))?;
    let lags = result
        .lags
        .iter()
        .map(|lag| signed_value(cx, *lag))
        .collect::<Result<Vec<_>>>()?;
    let entries = vec![
        (
            Symbol::new("samples"),
            real_values(cx, result.samples.as_slice())?,
        ),
        (Symbol::new("lags"), cx.factory().list(lags)?),
        (
            Symbol::new("algorithm"),
            symbol_value(cx, algorithm_name(result.convolution.cost.selected))?,
        ),
        (
            Symbol::new("normalization"),
            symbol_value(cx, correlation_normalization_name(result.normalization))?,
        ),
        (
            Symbol::new("lag-order"),
            symbol_value(cx, lag_order_name(result.lag_order))?,
        ),
    ];
    cx.factory().table(entries)
}

fn execute_deconvolve(
    cx: &mut Cx,
    observation: &[f64],
    kernel: &[f64],
    options: &Options,
) -> Result<Value> {
    reject_unknown(
        Operation::Deconvolve,
        options,
        &["mode", "period", "regularization", "singular-threshold"],
    )?;
    let plan = DeconvolutionPlan {
        mode: deconvolution_mode(cx, options, observation.len())?,
        regularization: regularization(cx, options)?,
        singular_threshold: option_f64(cx, options, "singular-threshold")?.unwrap_or(1.0e-12),
    };
    let result =
        deconvolve(observation, kernel, &plan).map_err(operation_error(Operation::Deconvolve))?;
    let singular_bins = result
        .report
        .singular_bins
        .iter()
        .map(|bin| usize_value(cx, *bin))
        .collect::<Result<Vec<_>>>()?;
    let entries = vec![
        (
            Symbol::new("samples"),
            real_values(cx, result.samples.as_slice())?,
        ),
        (
            Symbol::new("fft-len"),
            usize_value(cx, result.report.fft_len)?,
        ),
        (
            Symbol::new("singular-bins"),
            cx.factory().list(singular_bins)?,
        ),
        (
            Symbol::new("minimum-kernel-magnitude"),
            f64_value(cx, result.report.minimum_kernel_magnitude)?,
        ),
        (
            Symbol::new("maximum-inverse-gain"),
            f64_value(cx, result.report.maximum_inverse_gain)?,
        ),
        (
            Symbol::new("residual-l2"),
            f64_value(cx, result.report.residual_l2)?,
        ),
        (
            Symbol::new("regularization"),
            symbol_value(cx, regularization_name(result.report.regularization))?,
        ),
    ];
    cx.factory().table(entries)
}

fn expr_options(cx: &mut Cx, operation: Operation, expressions: &[Expr]) -> Result<Options> {
    if !expressions.len().is_multiple_of(2) {
        return Err(argument_error(operation));
    }
    let mut options = Options::new();
    for pair in expressions.chunks(2) {
        let key = keyword_expr(operation, &pair[0])?;
        insert_option(
            operation,
            &mut options,
            key,
            eval_option_expr(cx, operation, &pair[1])?,
        )?;
    }
    Ok(options)
}

fn eval_option_expr(cx: &mut Cx, operation: Operation, expression: &Expr) -> Result<Value> {
    let pairs = match expression {
        Expr::Map(entries) => entries
            .iter()
            .map(|(key, value)| (key, value))
            .collect::<Vec<_>>(),
        Expr::Block(items) if items.len().is_multiple_of(2) => items
            .chunks(2)
            .map(|pair| (&pair[0], &pair[1]))
            .collect::<Vec<_>>(),
        _ => return cx.eval_expr(expression.clone()),
    };
    let mut values = Vec::with_capacity(pairs.len());
    for (key, value) in pairs {
        let key = keyword_expr(operation, key)?;
        values.push((Symbol::new(format!(":{key}")), cx.eval_expr(value.clone())?));
    }
    cx.factory().table(values)
}

fn value_options(cx: &mut Cx, operation: Operation, values: &[Value]) -> Result<Options> {
    if !values.len().is_multiple_of(2) {
        return Err(argument_error(operation));
    }
    let mut options = Options::new();
    for pair in values.chunks(2) {
        let key = keyword_value(cx, operation, &pair[0])?;
        insert_option(operation, &mut options, key, pair[1].clone())?;
    }
    Ok(options)
}

fn insert_option(
    operation: Operation,
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

fn convolution_mode(cx: &mut Cx, options: &Options, signal_len: usize) -> Result<ConvolutionMode> {
    let name = option_symbol(cx, options, "mode")?.unwrap_or_else(|| "linear-full".into());
    match name.as_str() {
        "linear-full" => Ok(ConvolutionMode::Linear(LinearOutput::Full)),
        "linear-same" => Ok(ConvolutionMode::Linear(LinearOutput::Same)),
        "linear-valid" => Ok(ConvolutionMode::Linear(LinearOutput::Valid)),
        "circular" => Ok(ConvolutionMode::Circular {
            period: option_usize(cx, options, "period")?.unwrap_or(signal_len),
        }),
        _ => Err(Error::Eval(format!("unsupported convolution mode {name}"))),
    }
}

fn deconvolution_mode(
    cx: &mut Cx,
    options: &Options,
    observation_len: usize,
) -> Result<DeconvolutionMode> {
    let name = option_symbol(cx, options, "mode")?.unwrap_or_else(|| "linear-full".into());
    match name.as_str() {
        "linear-full" => Ok(DeconvolutionMode::LinearFull),
        "circular" => Ok(DeconvolutionMode::Circular {
            period: option_usize(cx, options, "period")?.unwrap_or(observation_len),
        }),
        _ => Err(Error::Eval(format!(
            "unsupported deconvolution mode {name}"
        ))),
    }
}

fn algorithm(cx: &mut Cx, options: &Options) -> Result<ConvolutionAlgorithm> {
    match option_symbol(cx, options, "algorithm")?
        .unwrap_or_else(|| "auto".into())
        .as_str()
    {
        "auto" => Ok(ConvolutionAlgorithm::Auto),
        "direct" => Ok(ConvolutionAlgorithm::Direct),
        "fft" => Ok(ConvolutionAlgorithm::Fft),
        name => Err(Error::Eval(format!(
            "unsupported convolution algorithm {name}"
        ))),
    }
}

fn boundary(cx: &mut Cx, options: &Options, mode: ConvolutionMode) -> Result<BoundaryPolicy> {
    let default = match mode {
        ConvolutionMode::Linear(_) => "zero",
        ConvolutionMode::Circular { .. } => "periodic",
    };
    match option_symbol(cx, options, "boundary")?
        .unwrap_or_else(|| default.into())
        .as_str()
    {
        "zero" | "zero-pad" => Ok(BoundaryPolicy::ZeroPad),
        "periodic" => Ok(BoundaryPolicy::Periodic),
        name => Err(Error::Eval(format!("unsupported signal boundary {name}"))),
    }
}

fn convolution_normalization(cx: &mut Cx, options: &Options) -> Result<ConvolutionNormalization> {
    match option_symbol(cx, options, "normalization")?
        .unwrap_or_else(|| "none".into())
        .as_str()
    {
        "none" => Ok(ConvolutionNormalization::None),
        "kernel-sum" => Ok(ConvolutionNormalization::KernelSum),
        name => Err(Error::Eval(format!(
            "unsupported convolution normalization {name}"
        ))),
    }
}

fn correlation_normalization(cx: &mut Cx, options: &Options) -> Result<CorrelationNormalization> {
    match option_symbol(cx, options, "normalization")?
        .unwrap_or_else(|| "none".into())
        .as_str()
    {
        "none" => Ok(CorrelationNormalization::None),
        "biased" => Ok(CorrelationNormalization::Biased),
        "unbiased" => Ok(CorrelationNormalization::Unbiased),
        "coefficient" => Ok(CorrelationNormalization::Coefficient),
        name => Err(Error::Eval(format!(
            "unsupported correlation normalization {name}"
        ))),
    }
}

fn lag_order(cx: &mut Cx, options: &Options) -> Result<LagOrder> {
    match option_symbol(cx, options, "lag-order")?
        .unwrap_or_else(|| "ascending".into())
        .as_str()
    {
        "ascending" => Ok(LagOrder::Ascending),
        "descending" => Ok(LagOrder::Descending),
        name => Err(Error::Eval(format!("unsupported lag order {name}"))),
    }
}

fn regularization(cx: &mut Cx, options: &Options) -> Result<Regularization> {
    let Some(value) = options.get("regularization") else {
        return Ok(Regularization::Tikhonov { lambda: 1.0e-8 });
    };
    match value.object().as_expr(cx)? {
        Expr::Symbol(symbol) => regularization_from_parts(&symbol.as_qualified_str(), None),
        Expr::Map(entries) => {
            let mut nested = Options::new();
            for (key, value) in entries {
                let key = keyword_expr(Operation::Deconvolve, &key)?;
                insert_option(
                    Operation::Deconvolve,
                    &mut nested,
                    key,
                    cx.eval_expr(value)?,
                )?;
            }
            reject_unknown(Operation::Deconvolve, &nested, &["kind", "lambda"])?;
            let kind = option_symbol(cx, &nested, "kind")?
                .ok_or_else(|| Error::Eval("regularization table requires :kind".into()))?;
            regularization_from_parts(&kind, option_f64(cx, &nested, "lambda")?)
        }
        other => Err(Error::Eval(format!(
            "signal/deconvolve :regularization must be a symbol or table, got {other:?}"
        ))),
    }
}

fn regularization_from_parts(kind: &str, lambda: Option<f64>) -> Result<Regularization> {
    match kind {
        "tikhonov" => Ok(Regularization::Tikhonov {
            lambda: lambda.unwrap_or(1.0e-8),
        }),
        "truncated" => {
            if lambda.is_some() {
                return Err(Error::Eval(
                    "truncated regularization does not accept :lambda".into(),
                ));
            }
            Ok(Regularization::Truncated)
        }
        _ => Err(Error::Eval(format!("unsupported regularization {kind}"))),
    }
}

fn option_symbol(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<String>> {
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

fn option_f64(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<f64>> {
    options
        .get(key)
        .map(|value| {
            number_value(cx, value, key)?
                .canonical
                .parse::<f64>()
                .map_err(|_| Error::Eval(format!("option :{key} must be f64")))
        })
        .transpose()
}

fn option_usize(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<usize>> {
    options
        .get(key)
        .map(|value| {
            number_value(cx, value, key)?
                .canonical
                .parse::<usize>()
                .map_err(|_| Error::Eval(format!("option :{key} must be a non-negative integer")))
        })
        .transpose()
}

fn keyword_expr(operation: Operation, expression: &Expr) -> Result<String> {
    let Expr::Symbol(symbol) = expression else {
        return Err(argument_error(operation));
    };
    symbol
        .name
        .strip_prefix(':')
        .map(str::to_owned)
        .ok_or_else(|| argument_error(operation))
}

fn keyword_value(cx: &mut Cx, operation: Operation, value: &Value) -> Result<String> {
    keyword_expr(operation, &value.object().as_expr(cx)?)
}

fn reject_unknown(operation: Operation, options: &Options, allowed: &[&str]) -> Result<()> {
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

fn operation_error(operation: Operation) -> impl FnOnce(SignalError) -> Error {
    move |error| Error::Eval(format!("{}: {error}", operation.name()))
}
