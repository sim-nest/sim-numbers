//! Runtime dispatch for the numeric operations, resolving arguments and options
//! to the selected registry plugin and invoking it.

use std::sync::Arc;

use sim_kernel::{Args, Cx, Error, Expr, Result, Symbol, Value};
use sim_lib_numbers_core::domains;

use super::{
    options::{
        option_f64, option_symbol, option_usize, parse_diff_exprs, parse_integrate_exprs,
        parse_ode_exprs, parse_symbolish_value, parse_table_options, reject_unknown,
    },
    registry::global_numeric_registry,
    traits::{
        DiffOpts, Differentiator, NumericCallable, NumericKind, OdeOpts, OdeProblem, QuadOpts,
    },
};

pub fn call_numeric_diff(cx: &mut Cx, args: Args) -> Result<Value> {
    let values = args.into_vec();
    let (func_value, var, point, opts) = match values.as_slice() {
        [func, var, point] => (
            func.clone(),
            extract_var(cx, var)?,
            point.clone(),
            DiffOpts::auto(),
        ),
        [func, var, point, options] => (
            func.clone(),
            extract_var(cx, var)?,
            point.clone(),
            diff_opts_from_table(cx, options)?,
        ),
        _ => {
            return Err(Error::Eval(
                "numeric-diff expects func, var, point, and optional options table".to_owned(),
            ));
        }
    };
    diff_dispatch(cx, func_value, var, point, opts)
}

pub fn call_numeric_diff_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    let (func, var, point, opts) = parse_diff_exprs(cx, args)?;
    diff_dispatch(cx, func, var, point, opts)
}

pub fn call_integrate(cx: &mut Cx, args: Args) -> Result<Value> {
    call_integrate_values(cx, args, false)
}

pub fn call_integrate_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    let (func, var, lo, hi, opts) = parse_integrate_exprs(cx, "integrate", args, false)?;
    integrate_dispatch(cx, func, var, lo, hi, opts, NumericKind::QuadratureFixed)
}

pub fn call_integrate_adapt(cx: &mut Cx, args: Args) -> Result<Value> {
    call_integrate_values(cx, args, true)
}

pub fn call_integrate_adapt_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    let (func, var, lo, hi, opts) = parse_integrate_exprs(cx, "integrate-adapt", args, true)?;
    integrate_dispatch(cx, func, var, lo, hi, opts, NumericKind::QuadratureAdaptive)
}

pub fn call_ode_solve(cx: &mut Cx, args: Args) -> Result<Value> {
    let values = args.into_vec();
    let (dy, var, y_var, x0, y0, x_end, opts) = match values.as_slice() {
        [dy, var, y_var, x0, y0, x_end] => (
            dy.clone(),
            extract_var(cx, var)?,
            extract_var(cx, y_var)?,
            x0.clone(),
            y0.clone(),
            x_end.clone(),
            OdeOpts::default_adaptive(),
        ),
        [dy, var, y_var, x0, y0, x_end, options] => (
            dy.clone(),
            extract_var(cx, var)?,
            extract_var(cx, y_var)?,
            x0.clone(),
            y0.clone(),
            x_end.clone(),
            ode_opts_from_table(cx, options)?,
        ),
        _ => {
            return Err(Error::Eval(
                "ode-solve expects dy/dx, x var, y var, x0, y0, x-end, and optional options table"
                    .to_owned(),
            ));
        }
    };
    ode_dispatch(
        cx,
        OdeDispatch {
            dy_value: dy,
            var,
            y_var,
            x0,
            y0,
            x_end,
            opts,
        },
    )
}

pub fn call_ode_solve_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    let (dy, var, y_var, x0, y0, x_end, opts) = parse_ode_exprs(cx, args)?;
    ode_dispatch(
        cx,
        OdeDispatch {
            dy_value: dy,
            var,
            y_var,
            x0,
            y0,
            x_end,
            opts,
        },
    )
}

pub fn call_numeric_compose(cx: &mut Cx, args: Args) -> Result<Value> {
    super::pipeline::call_numeric_compose(cx, args)
}

pub fn call_numeric_compose_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    super::pipeline::call_numeric_compose_exprs(cx, args)
}

pub fn call_numeric_run_composed(cx: &mut Cx, args: Args) -> Result<Value> {
    super::pipeline::call_numeric_run_composed(cx, args)
}

pub fn call_numeric_run_composed_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    super::pipeline::call_numeric_run_composed_exprs(cx, args)
}

fn call_integrate_values(cx: &mut Cx, args: Args, adaptive: bool) -> Result<Value> {
    let values = args.into_vec();
    let (func, var, lo, hi, opts) = match values.as_slice() {
        [func, var, lo, hi] => (
            func.clone(),
            extract_var(cx, var)?,
            lo.clone(),
            hi.clone(),
            if adaptive {
                QuadOpts::adaptive_default()
            } else {
                QuadOpts::fixed_default()
            },
        ),
        [func, var, lo, hi, options] => (
            func.clone(),
            extract_var(cx, var)?,
            lo.clone(),
            hi.clone(),
            quad_opts_from_table(
                cx,
                if adaptive {
                    "integrate-adapt"
                } else {
                    "integrate"
                },
                options,
                adaptive,
            )?,
        ),
        _ => {
            return Err(Error::Eval(
                "integrate expects func, var, lo, hi, and optional options table".to_owned(),
            ));
        }
    };
    integrate_dispatch(
        cx,
        func,
        var,
        lo,
        hi,
        opts,
        if adaptive {
            NumericKind::QuadratureAdaptive
        } else {
            NumericKind::QuadratureFixed
        },
    )
}

/// Dispatches numeric differentiation.
///
/// The `auto` method prefers, in order, a symbolic CAS body, a native
/// function's registered `differentiator_hint`, then central finite difference.
fn diff_dispatch(
    cx: &mut Cx,
    func_value: Value,
    var: Symbol,
    point: Value,
    opts: DiffOpts,
) -> Result<Value> {
    let func = NumericCallable::unary(func_value, var.clone())?;
    let auto = Symbol::new("auto");
    let is_auto = opts.method == auto;
    if is_auto && func.body_cas().is_some() {
        let derivative = cx.call_function(
            &Symbol::new("diff"),
            Args::new(vec![
                func.value().clone(),
                cx.factory().symbol(var.clone())?,
            ]),
        )?;
        let out = cx.call_value(derivative, Args::new(vec![point]))?;
        cx.push_info("numeric-diff method=auto steps=1");
        return Ok(out);
    }
    if is_auto
        && let Some(hint) = func.differentiator_hint()
        && let Some(plugin) = registered_differentiator(hint)?
    {
        let out = plugin.diff_callable_at(cx, &func, &var, &point, opts.clone())?;
        cx.push_info(format!("numeric-diff method=auto->{} steps=exact", hint));
        return Ok(out);
    }
    if let Some(plugin) = registered_differentiator(&opts.method)? {
        let out = plugin.diff_callable_at(cx, &func, &var, &point, opts.clone())?;
        cx.push_info(format!(
            "numeric-diff method={} steps=2 h={}",
            opts.method, opts.h
        ));
        return Ok(out);
    }
    if is_auto {
        let out = finite_diff_central(cx, &func, point, opts.h)?;
        cx.push_info(format!("numeric-diff method=auto steps=2 h={}", opts.h));
        return Ok(out);
    }
    Err(unknown_method("differentiator", &opts.method))
}

fn registered_differentiator(method: &Symbol) -> Result<Option<Arc<dyn Differentiator>>> {
    Ok(global_numeric_registry()
        .read()
        .map_err(|_| Error::PoisonedLock("numeric registry"))?
        .differentiator(method))
}

fn integrate_dispatch(
    cx: &mut Cx,
    func_value: Value,
    var: Symbol,
    lo: Value,
    hi: Value,
    opts: QuadOpts,
    kind: NumericKind,
) -> Result<Value> {
    let func = NumericCallable::unary(func_value, var.clone())?;
    let registry = global_numeric_registry()
        .read()
        .map_err(|_| Error::PoisonedLock("numeric registry"))?;
    let method = resolve_quad_method(kind, &opts.method);
    let plugin = match kind {
        NumericKind::QuadratureFixed => registry.quadrature_fixed(&method),
        NumericKind::QuadratureAdaptive => registry.quadrature_adaptive(&method),
        _ => None,
    };
    let Some(plugin) = plugin else {
        return Err(unknown_method("quadrature", &method));
    };
    let mut resolved = opts.clone();
    resolved.method = method;
    let out = plugin.integrate(cx, &func, &var, &lo, &hi, resolved.clone())?;
    cx.push_info(format!(
        "integrate method={} steps={} tol={}",
        resolved.method,
        resolved
            .n
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_owned()),
        resolved
            .tol
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_owned())
    ));
    Ok(out)
}

struct OdeDispatch {
    dy_value: Value,
    var: Symbol,
    y_var: Symbol,
    x0: Value,
    y0: Value,
    x_end: Value,
    opts: OdeOpts,
}

fn ode_dispatch(cx: &mut Cx, dispatch: OdeDispatch) -> Result<Value> {
    let dy = NumericCallable::binary(
        dispatch.dy_value,
        dispatch.var.clone(),
        dispatch.y_var.clone(),
    )?;
    let registry = global_numeric_registry()
        .read()
        .map_err(|_| Error::PoisonedLock("numeric registry"))?;
    let method = resolve_ode_method(&dispatch.opts.method);
    let plugin = registry
        .ode_fixed(&method)
        .or_else(|| registry.ode_adaptive(&method));
    let Some(plugin) = plugin else {
        return Err(unknown_method("ode", &method));
    };
    let mut resolved = dispatch.opts.clone();
    resolved.method = method;
    let points = plugin.solve(
        cx,
        OdeProblem {
            dy: &dy,
            var: &dispatch.var,
            y_var: &dispatch.y_var,
            x0: &dispatch.x0,
            y0: &dispatch.y0,
            x_end: &dispatch.x_end,
        },
        resolved.clone(),
    )?;
    let values = points
        .into_iter()
        .map(|(x, y)| cx.factory().list(vec![x, y]))
        .collect::<Result<Vec<_>>>()?;
    cx.push_info(format!(
        "ode-solve method={} steps={} tol={}",
        resolved.method,
        resolved
            .max_steps
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_owned()),
        resolved
            .tol
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_owned())
    ));
    cx.factory().list(values)
}

fn diff_opts_from_table(cx: &mut Cx, options: &Value) -> Result<DiffOpts> {
    let table = parse_table_options(cx, "numeric-diff", options)?;
    let method = option_symbol(&table, "method")?.unwrap_or(Symbol::new("auto"));
    let h = option_f64(&table, "h")?.unwrap_or(1.0e-6);
    reject_unknown("numeric-diff", &table, &["method", "h"])?;
    Ok(DiffOpts { method, h })
}

fn quad_opts_from_table(
    cx: &mut Cx,
    name: &str,
    options: &Value,
    adaptive: bool,
) -> Result<QuadOpts> {
    let table = parse_table_options(cx, name, options)?;
    let defaults = if adaptive {
        QuadOpts::adaptive_default()
    } else {
        QuadOpts::fixed_default()
    };
    let method = option_symbol(&table, "method")?.unwrap_or(defaults.method);
    let n = option_usize(&table, "n")?;
    let tol = option_f64(&table, "tol")?.or(defaults.tol);
    reject_unknown(name, &table, &["method", "n", "tol"])?;
    Ok(QuadOpts { method, n, tol })
}

fn ode_opts_from_table(cx: &mut Cx, options: &Value) -> Result<OdeOpts> {
    let table = parse_table_options(cx, "ode-solve", options)?;
    let method = option_symbol(&table, "method")?.unwrap_or(Symbol::new("auto"));
    let h = option_f64(&table, "h")?;
    let tol = option_f64(&table, "tol")?;
    let max_steps = option_usize(&table, "max-steps")?;
    reject_unknown("ode-solve", &table, &["method", "h", "tol", "max-steps"])?;
    Ok(OdeOpts {
        method,
        h,
        tol,
        max_steps,
    })
}

fn extract_var(cx: &mut Cx, value: &Value) -> Result<Symbol> {
    parse_symbolish_value(cx, value)?
        .ok_or_else(|| Error::Eval("expected symbol or quoted symbol variable".to_owned()))
}

fn finite_diff_central(cx: &mut Cx, func: &NumericCallable, point: Value, h: f64) -> Result<Value> {
    let h_value = cx.factory().number_literal(domains::f64(), h.to_string())?;
    let neg_h_value = cx
        .factory()
        .number_literal(domains::f64(), (-h).to_string())?;
    let denom = cx
        .factory()
        .number_literal(domains::f64(), (2.0 * h).to_string())?;
    let x_plus =
        cx.apply_value_number_binary_op(&Symbol::qualified("math", "add"), point.clone(), h_value)?;
    let x_minus =
        cx.apply_value_number_binary_op(&Symbol::qualified("math", "add"), point, neg_h_value)?;
    let f_plus = func.call(cx, vec![x_plus])?;
    let f_minus = func.call(cx, vec![x_minus])?;
    let numerator =
        cx.apply_value_number_binary_op(&Symbol::qualified("math", "sub"), f_plus, f_minus)?;
    cx.apply_value_number_binary_op(&Symbol::qualified("math", "div"), numerator, denom)
}

fn unknown_method(kind: &str, method: &Symbol) -> Error {
    Error::Eval(format!("UnknownNumericMethod: {kind} method {method}"))
}

fn resolve_quad_method(kind: NumericKind, method: &Symbol) -> Symbol {
    if *method != Symbol::new("auto") {
        return method.clone();
    }
    match kind {
        NumericKind::QuadratureFixed => Symbol::new("simpson"),
        NumericKind::QuadratureAdaptive => Symbol::new("adaptive-gauss-kronrod"),
        _ => method.clone(),
    }
}

fn resolve_ode_method(method: &Symbol) -> Symbol {
    if *method == Symbol::new("auto") {
        Symbol::new("rkf45")
    } else {
        method.clone()
    }
}
