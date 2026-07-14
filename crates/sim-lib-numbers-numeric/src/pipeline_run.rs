//! Execution for first-class composed numeric pipeline values.

use std::{collections::BTreeMap, sync::Arc};

use sim_kernel::{Args, Cx, Error, Expr, Result, Symbol, Value, value_from_ref};
use sim_lib_numbers_core::domains;

use super::{
    options::parse_symbolish_value,
    pipeline::{ComposedPipeline, PipelineKind, StateKind},
    registry::global_numeric_registry,
    traits::{NumericCallable, NumericKind, OdeOpts, OdeProblem, QuadOpts, Quadrature},
};

pub fn call_numeric_run_composed(cx: &mut Cx, args: Args) -> Result<Value> {
    let values = args.into_vec();
    let input = RunComposedInput::from_values(cx, &values)?;
    match input.pipeline.kind.clone() {
        PipelineKind::OdeSolve => run_ode_composed(cx, input),
        PipelineKind::Quadrature => run_quad_composed(cx, input),
    }
}

pub fn call_numeric_run_composed_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    let values = args
        .into_iter()
        .map(|expr| eval_run_arg(cx, expr))
        .collect::<Result<Vec<_>>>()?;
    let input = RunComposedInput::from_values(cx, &values)?;
    match input.pipeline.kind.clone() {
        PipelineKind::OdeSolve => run_ode_composed(cx, input),
        PipelineKind::Quadrature => run_quad_composed(cx, input),
    }
}

struct RunComposedInput {
    pipeline: ComposedPipeline,
    args: RunArgs,
}

enum RunArgs {
    Ode {
        t0: Value,
        t1: Value,
        y0: Value,
        dt: f64,
    },
    Quadrature {
        a: Value,
        b: Value,
        n: Option<usize>,
        tol: Option<f64>,
    },
}

impl RunComposedInput {
    fn from_values(cx: &mut Cx, values: &[Value]) -> Result<Self> {
        let Some((pipeline_value, rest)) = values.split_first() else {
            return Err(Error::Eval(
                "numeric/run-composed expects a pipeline and run arguments".to_owned(),
            ));
        };
        let pipeline = require_composed_pipeline(pipeline_value)?;
        match pipeline.kind.clone() {
            PipelineKind::OdeSolve => Self::ode_from_values(cx, pipeline, rest),
            PipelineKind::Quadrature => Self::quad_from_values(cx, pipeline, rest),
        }
    }

    fn ode_from_values(cx: &mut Cx, pipeline: ComposedPipeline, values: &[Value]) -> Result<Self> {
        let args = match values {
            [t0, t1, y0, dt] => RunArgs::Ode {
                t0: t0.clone(),
                t1: t1.clone(),
                y0: y0.clone(),
                dt: value_to_f64(cx, dt, "numeric/run-composed :dt")?,
            },
            keyed if keyed.len().is_multiple_of(2) => ode_args_from_keyed(cx, keyed)?,
            _ => {
                return Err(Error::Eval(
                    "numeric/run-composed expects pipeline, t0, t1, y0, dt or keyword pairs"
                        .to_owned(),
                ));
            }
        };
        Ok(Self { pipeline, args })
    }

    fn quad_from_values(cx: &mut Cx, pipeline: ComposedPipeline, values: &[Value]) -> Result<Self> {
        let args = match values {
            [a, b, n] => RunArgs::Quadrature {
                a: a.clone(),
                b: b.clone(),
                n: Some(value_to_usize(cx, n, "numeric/run-composed :n")?),
                tol: None,
            },
            keyed if keyed.len().is_multiple_of(2) => quad_args_from_keyed(cx, keyed)?,
            _ => {
                return Err(Error::Eval(
                    "numeric/run-composed quadrature expects a, b, n or keyword pairs".to_owned(),
                ));
            }
        };
        Ok(Self { pipeline, args })
    }
}

fn ode_args_from_keyed(cx: &mut Cx, values: &[Value]) -> Result<RunArgs> {
    let options = keyed_run_options(cx, values)?;
    reject_unknown_run_keys(&options, &["t0", "t1", "y0", "dt"])?;
    Ok(RunArgs::Ode {
        t0: require_run_value(&options, "t0")?.clone(),
        t1: require_run_value(&options, "t1")?.clone(),
        y0: require_run_value(&options, "y0")?.clone(),
        dt: value_to_f64(
            cx,
            require_run_value(&options, "dt")?,
            "numeric/run-composed :dt",
        )?,
    })
}

fn quad_args_from_keyed(cx: &mut Cx, values: &[Value]) -> Result<RunArgs> {
    let options = keyed_run_options(cx, values)?;
    reject_unknown_run_keys(&options, &["a", "b", "n", "tol"])?;
    let n = options
        .get("n")
        .map(|value| value_to_usize(cx, value, "numeric/run-composed :n"))
        .transpose()?;
    let tol = options
        .get("tol")
        .map(|value| value_to_f64(cx, value, "numeric/run-composed :tol"))
        .transpose()?;
    if n.is_some() == tol.is_some() {
        return Err(Error::Eval(
            "numeric/run-composed quadrature expects exactly one of :n or :tol".to_owned(),
        ));
    }
    Ok(RunArgs::Quadrature {
        a: require_run_value(&options, "a")?.clone(),
        b: require_run_value(&options, "b")?.clone(),
        n,
        tol,
    })
}

fn keyed_run_options(cx: &mut Cx, values: &[Value]) -> Result<BTreeMap<String, Value>> {
    let mut options = BTreeMap::<String, Value>::new();
    for pair in values.chunks(2) {
        let [key, value] = pair else {
            unreachable!("chunks over even input yield pairs");
        };
        let symbol = parse_symbolish_value(cx, key)?.ok_or_else(|| {
            Error::Eval("numeric/run-composed expected keyword argument".to_owned())
        })?;
        options.insert(keyword_name(&symbol), value.clone());
    }
    Ok(options)
}

fn run_ode_composed(cx: &mut Cx, input: RunComposedInput) -> Result<Value> {
    let RunComposedInput { pipeline, args } = input;
    ensure_f64_state(&pipeline)?;
    let RunArgs::Ode { t0, t1, y0, dt } = args else {
        unreachable!("ODE pipeline parser builds ODE run args");
    };
    let func_value = value_from_ref(cx, &pipeline.func_ref)?;
    let dy = NumericCallable::sampled_binary(func_value, Symbol::new("x"), Symbol::new("y"))?;
    let method = resolve_ode_method(&pipeline.method);
    let plugin = {
        let registry = global_numeric_registry()
            .read()
            .map_err(|_| Error::PoisonedLock("numeric registry"))?;
        registry
            .ode_fixed(&method)
            .or_else(|| registry.ode_adaptive(&method))
    };
    let Some(plugin) = plugin else {
        return Err(Error::Eval(format!(
            "UnknownNumericMethod: ode method {method}"
        )));
    };
    let plugin_kind = plugin.kind();
    let tol = if plugin_kind == NumericKind::OdeAdaptive {
        Some(1.0e-8)
    } else {
        None
    };
    let points = plugin.solve(
        cx,
        OdeProblem {
            dy: &dy,
            var: &dy.vars()[0],
            y_var: &dy.vars()[1],
            x0: &t0,
            y0: &y0,
            x_end: &t1,
        },
        OdeOpts {
            method: method.clone(),
            h: Some(dt),
            tol,
            max_steps: None,
        },
    )?;
    let steps = points.len().saturating_sub(1);
    let steps_value = if plugin_kind == NumericKind::OdeAdaptive {
        cx.factory()
            .number_literal(domains::f64(), tol.unwrap_or(1.0e-8).to_string())?
    } else {
        cx.factory()
            .number_literal(domains::i64(), steps.to_string())?
    };
    let value = points
        .last()
        .map(|(_, y)| y.clone())
        .ok_or_else(|| Error::Eval("numeric/run-composed solver returned no points".to_owned()))?;
    cx.factory().table(vec![
        (Symbol::new("value"), value),
        (Symbol::new("method"), cx.factory().symbol(method)?),
        (
            Symbol::new("domain"),
            cx.factory().symbol(pipeline.kind.symbol())?,
        ),
        (
            Symbol::new("state-kind"),
            cx.factory().symbol(pipeline.state.symbol())?,
        ),
        (Symbol::new("steps"), steps_value),
    ])
}

fn run_quad_composed(cx: &mut Cx, input: RunComposedInput) -> Result<Value> {
    let RunComposedInput { pipeline, args } = input;
    ensure_f64_state(&pipeline)?;
    let RunArgs::Quadrature { a, b, n, tol } = args else {
        unreachable!("quadrature pipeline parser builds quadrature run args");
    };
    let func_value = value_from_ref(cx, &pipeline.func_ref)?;
    let func = NumericCallable::sampled_unary(func_value, Symbol::new("x"))?;
    let selection = select_quad_plugin(&pipeline.method, n, tol)?;
    let value = selection.plugin.integrate(
        cx,
        &func,
        &func.vars()[0],
        &a,
        &b,
        QuadOpts {
            method: selection.method.clone(),
            n,
            tol,
        },
    )?;
    let mut entries = vec![
        (Symbol::new("value"), value),
        (
            Symbol::new("method"),
            cx.factory().symbol(selection.method.clone())?,
        ),
        (
            Symbol::new("domain"),
            cx.factory().symbol(pipeline.kind.symbol())?,
        ),
        (
            Symbol::new("state-kind"),
            cx.factory().symbol(pipeline.state.symbol())?,
        ),
    ];
    if let Some(n) = n {
        entries.push((
            Symbol::new("n"),
            cx.factory().number_literal(domains::i64(), n.to_string())?,
        ));
    }
    if let Some(tol) = tol {
        entries.push((
            Symbol::new("tol"),
            cx.factory()
                .number_literal(domains::f64(), tol.to_string())?,
        ));
    }
    cx.push_info(format!(
        "numeric/run-composed method={} domain={:?}",
        selection.method, selection.kind
    ));
    cx.factory().table(entries)
}

fn require_composed_pipeline(value: &Value) -> Result<ComposedPipeline> {
    value
        .object()
        .downcast_ref::<ComposedPipeline>()
        .cloned()
        .ok_or_else(|| Error::Eval("numeric/run-composed expects a ComposedPipeline".to_owned()))
}

fn require_run_value<'a>(options: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a Value> {
    options
        .get(key)
        .ok_or_else(|| Error::Eval(format!("numeric/run-composed missing :{key}")))
}

fn reject_unknown_run_keys(options: &BTreeMap<String, Value>, allowed: &[&str]) -> Result<()> {
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::Eval(format!(
                "numeric/run-composed: unknown option :{key}"
            )));
        }
    }
    Ok(())
}

fn eval_run_arg(cx: &mut Cx, expr: Expr) -> Result<Value> {
    match &expr {
        Expr::Symbol(symbol) if symbol.name.starts_with(':') => cx.factory().symbol(symbol.clone()),
        _ => cx.eval_expr(expr),
    }
}

fn value_to_f64(cx: &mut Cx, value: &Value, context: &str) -> Result<f64> {
    value
        .object()
        .display(cx)?
        .parse::<f64>()
        .map_err(|_| Error::Eval(format!("{context} expected an f64-compatible value")))
}

fn value_to_usize(cx: &mut Cx, value: &Value, context: &str) -> Result<usize> {
    value
        .object()
        .display(cx)?
        .parse::<usize>()
        .map_err(|_| Error::Eval(format!("{context} expected a non-negative integer")))
}

fn resolve_ode_method(method: &Symbol) -> Symbol {
    if *method == Symbol::new("auto") {
        Symbol::new("rkf45")
    } else {
        method.clone()
    }
}

fn ensure_f64_state(pipeline: &ComposedPipeline) -> Result<()> {
    if pipeline.state == StateKind::F64 {
        Ok(())
    } else {
        Err(Error::Eval("NotYetSupported: tensor state".to_owned()))
    }
}

struct QuadSelection {
    method: Symbol,
    kind: NumericKind,
    plugin: Arc<dyn Quadrature>,
}

fn select_quad_plugin(
    method: &Symbol,
    n: Option<usize>,
    tol: Option<f64>,
) -> Result<QuadSelection> {
    let method = resolve_quad_method(method, n, tol);
    let registry = global_numeric_registry()
        .read()
        .map_err(|_| Error::PoisonedLock("numeric registry"))?;
    let fixed = registry.quadrature_fixed(&method);
    let adaptive = registry.quadrature_adaptive(&method);
    match (n.is_some(), tol.is_some(), fixed, adaptive) {
        (true, false, Some(plugin), _) => Ok(QuadSelection {
            method,
            kind: NumericKind::QuadratureFixed,
            plugin,
        }),
        (false, true, _, Some(plugin)) => Ok(QuadSelection {
            method,
            kind: NumericKind::QuadratureAdaptive,
            plugin,
        }),
        (false, false, Some(plugin), _) => Ok(QuadSelection {
            method,
            kind: NumericKind::QuadratureFixed,
            plugin,
        }),
        (false, false, None, Some(plugin)) => Ok(QuadSelection {
            method,
            kind: NumericKind::QuadratureAdaptive,
            plugin,
        }),
        (false, false, None, None) => Err(unknown_numeric_method("quadrature", &method)),
        (true, true, _, _) => Err(Error::Eval(
            "numeric/run-composed quadrature expects either :n or :tol, not both".to_owned(),
        )),
        (true, false, None, _) => Err(Error::Eval(format!(
            "quadrature method {method} does not accept :n"
        ))),
        (false, true, _, None) => Err(Error::Eval(format!(
            "quadrature method {method} does not accept :tol"
        ))),
    }
}

fn resolve_quad_method(method: &Symbol, n: Option<usize>, tol: Option<f64>) -> Symbol {
    if *method != Symbol::new("auto") {
        return method.clone();
    }
    if tol.is_some() && n.is_none() {
        Symbol::new("adaptive-gauss-kronrod")
    } else {
        Symbol::new("simpson")
    }
}

fn keyword_name(symbol: &Symbol) -> String {
    symbol
        .name
        .strip_prefix(':')
        .unwrap_or(&symbol.name)
        .to_owned()
}

fn unknown_numeric_method(kind: &str, method: &Symbol) -> Error {
    Error::Eval(format!("UnknownNumericMethod: {kind} method {method}"))
}
