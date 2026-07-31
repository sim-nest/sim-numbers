//! Lisp-facing Burg and DFT-series execution with explicit policy evidence.

use sim_kernel::{Cx, Error, Expr, Result, Symbol, Value};

use crate::{
    ArOrderCriterion, BurgPlan, BurgStability, DftSeriesPlan, EndpointConvention,
    NyquistConvention, Periodicity, SignConvention, burg, dft_interpolate,
    runtime_convolution_render::{f64_value, real_values, symbol_value, usize_value},
    runtime_spectral_callable::{SpectralOperation, argument_error},
    runtime_spectral_value::{
        Options, complex_list, complex_values, criterion_name, endpoint_name, expr_options,
        normalization_name, nyquist_name, option_f64, option_symbol, option_u64, option_usize,
        parse_normalization, periodicity_name, real_list, reject_unknown, sign_name,
        termination_name, u64_value, value_options,
    },
};

pub(crate) fn execute_exprs(
    cx: &mut Cx,
    operation: SpectralOperation,
    expressions: Vec<Expr>,
) -> Result<Value> {
    let [input, rest @ ..] = expressions.as_slice() else {
        return Err(argument_error(operation));
    };
    let input = cx.eval_expr(input.clone())?;
    let options = expr_options(cx, operation, rest)?;
    execute(cx, operation, &input, &options)
}

pub(crate) fn execute_values(
    cx: &mut Cx,
    operation: SpectralOperation,
    values: Vec<Value>,
) -> Result<Value> {
    let [input, rest @ ..] = values.as_slice() else {
        return Err(argument_error(operation));
    };
    let options = value_options(cx, operation, rest)?;
    execute(cx, operation, input, &options)
}

fn execute(
    cx: &mut Cx,
    operation: SpectralOperation,
    input: &Value,
    options: &Options,
) -> Result<Value> {
    match operation {
        SpectralOperation::Burg => execute_burg(cx, input, options),
        SpectralOperation::DftInterpolate => execute_interpolate(cx, input, options),
    }
}

fn execute_burg(cx: &mut Cx, input: &Value, options: &Options) -> Result<Value> {
    reject_unknown(
        SpectralOperation::Burg,
        options,
        &[
            "order",
            "criterion",
            "stability",
            "singular-tolerance",
            "stability-margin",
            "max-work",
        ],
    )?;
    let order = option_usize(cx, options, "order")?
        .ok_or_else(|| Error::Eval("signal/burg requires :order".into()))?;
    let mut plan = BurgPlan::new(order);
    plan.criterion = match option_symbol(cx, options, "criterion")?
        .unwrap_or_else(|| "fixed".into())
        .as_str()
    {
        "fixed" => ArOrderCriterion::Fixed,
        "aic" | "akaike" => ArOrderCriterion::Akaike,
        "bic" | "bayesian" => ArOrderCriterion::Bayesian,
        "fpe" | "final-prediction-error" => ArOrderCriterion::FinalPredictionError,
        name => {
            return Err(Error::Eval(format!(
                "signal/burg unsupported criterion {name}"
            )));
        }
    };
    plan.stability = match option_symbol(cx, options, "stability")?
        .unwrap_or_else(|| "reject".into())
        .as_str()
    {
        "reject" => BurgStability::Reject,
        "reduce-order" => BurgStability::ReduceOrder,
        name => {
            return Err(Error::Eval(format!(
                "signal/burg unsupported stability {name}"
            )));
        }
    };
    plan.singular_tolerance =
        option_f64(cx, options, "singular-tolerance")?.unwrap_or(plan.singular_tolerance);
    plan.stability_margin =
        option_f64(cx, options, "stability-margin")?.unwrap_or(plan.stability_margin);
    plan.max_work = option_u64(cx, options, "max-work")?.unwrap_or(plan.max_work);
    let samples = real_list(cx, input, "samples", SpectralOperation::Burg)?;
    let model = burg(&samples, &plan).map_err(operation_error(SpectralOperation::Burg))?;
    let entries = vec![
        (
            Symbol::new("coefficients"),
            real_values(cx, &model.coefficients)?,
        ),
        (
            Symbol::new("reflection-coefficients"),
            real_values(cx, &model.reflection_coefficients)?,
        ),
        (Symbol::new("mean"), f64_value(cx, model.mean)?),
        (
            Symbol::new("innovation-variance"),
            f64_value(cx, model.innovation_variance)?,
        ),
        (
            Symbol::new("residual-energy"),
            f64_value(cx, model.evidence.residual_energy)?,
        ),
        (
            Symbol::new("requested-order"),
            usize_value(cx, model.evidence.requested_order)?,
        ),
        (
            Symbol::new("effective-order"),
            usize_value(cx, model.evidence.effective_order)?,
        ),
        (
            Symbol::new("criterion"),
            symbol_value(cx, criterion_name(model.evidence.criterion))?,
        ),
        (
            Symbol::new("termination"),
            symbol_value(cx, termination_name(model.evidence.termination))?,
        ),
        (
            Symbol::new("minimum-reflection-margin"),
            f64_value(cx, model.evidence.minimum_reflection_margin)?,
        ),
        (
            Symbol::new("work-units"),
            u64_value(cx, model.evidence.work_units)?,
        ),
        (
            Symbol::new("work-limit"),
            u64_value(cx, model.evidence.work_limit)?,
        ),
    ];
    cx.factory().table(entries)
}

fn execute_interpolate(cx: &mut Cx, input: &Value, options: &Options) -> Result<Value> {
    reject_unknown(
        SpectralOperation::DftInterpolate,
        options,
        &[
            "at",
            "origin",
            "period",
            "periodicity",
            "endpoint",
            "normalization",
            "sign",
            "nyquist",
            "max-points",
            "max-work",
        ],
    )?;
    let bins = complex_list(cx, input, "bins", SpectralOperation::DftInterpolate)?;
    let coordinates = options
        .get("at")
        .ok_or_else(|| Error::Eval("signal/dft-interpolate requires :at".into()))
        .and_then(|value| real_list(cx, value, "at", SpectralOperation::DftInterpolate))?;
    let mut plan = DftSeriesPlan::default();
    plan.origin = option_f64(cx, options, "origin")?.unwrap_or(plan.origin);
    plan.period = option_f64(cx, options, "period")?.unwrap_or(plan.period);
    plan.periodicity = match option_symbol(cx, options, "periodicity")?
        .unwrap_or_else(|| "wrap".into())
        .as_str()
    {
        "wrap" => Periodicity::Wrap,
        "principal" | "principal-period" => Periodicity::PrincipalPeriod,
        name => return Err(Error::Eval(format!("unsupported DFT periodicity {name}"))),
    };
    plan.endpoint = match option_symbol(cx, options, "endpoint")?
        .unwrap_or_else(|| "excluded".into())
        .as_str()
    {
        "excluded" => EndpointConvention::Excluded,
        "included" => EndpointConvention::Included,
        name => return Err(Error::Eval(format!("unsupported DFT endpoint {name}"))),
    };
    plan.normalization = parse_normalization(
        &option_symbol(cx, options, "normalization")?.unwrap_or_else(|| "inverse".into()),
    )?;
    plan.sign = match option_symbol(cx, options, "sign")?
        .unwrap_or_else(|| "negative-forward".into())
        .as_str()
    {
        "negative-forward" => SignConvention::NegativeForward,
        "positive-forward" => SignConvention::PositiveForward,
        name => return Err(Error::Eval(format!("unsupported DFT sign {name}"))),
    };
    plan.nyquist = match option_symbol(cx, options, "nyquist")?
        .unwrap_or_else(|| "positive".into())
        .as_str()
    {
        "positive" => NyquistConvention::Positive,
        "negative" => NyquistConvention::Negative,
        name => {
            return Err(Error::Eval(format!(
                "unsupported Nyquist convention {name}"
            )));
        }
    };
    plan.max_points = option_usize(cx, options, "max-points")?.unwrap_or(plan.max_points);
    plan.max_work = option_u64(cx, options, "max-work")?.unwrap_or(plan.max_work);
    let result = dft_interpolate(&bins, &coordinates, &plan)
        .map_err(operation_error(SpectralOperation::DftInterpolate))?;
    let entries = vec![
        (Symbol::new("values"), complex_values(cx, &result.values)?),
        (
            Symbol::new("phase-cycles"),
            real_values(cx, &result.phase_cycles)?,
        ),
        (Symbol::new("bins"), usize_value(cx, result.report.bins)?),
        (
            Symbol::new("points"),
            usize_value(cx, result.report.points)?,
        ),
        (Symbol::new("origin"), f64_value(cx, result.report.origin)?),
        (Symbol::new("period"), f64_value(cx, result.report.period)?),
        (
            Symbol::new("periodicity"),
            symbol_value(cx, periodicity_name(result.report.periodicity))?,
        ),
        (
            Symbol::new("endpoint"),
            symbol_value(cx, endpoint_name(result.report.endpoint))?,
        ),
        (
            Symbol::new("normalization"),
            symbol_value(cx, normalization_name(result.report.normalization))?,
        ),
        (
            Symbol::new("sign"),
            symbol_value(cx, sign_name(result.report.sign))?,
        ),
        (
            Symbol::new("nyquist"),
            symbol_value(cx, nyquist_name(result.report.nyquist))?,
        ),
        (
            Symbol::new("work-units"),
            u64_value(cx, result.report.work_units)?,
        ),
        (
            Symbol::new("work-limit"),
            u64_value(cx, result.report.work_limit)?,
        ),
    ];
    cx.factory().table(entries)
}

fn operation_error(operation: SpectralOperation) -> impl FnOnce(crate::SignalError) -> Error {
    move |error| Error::Eval(format!("{}: {error}", operation.name()))
}
