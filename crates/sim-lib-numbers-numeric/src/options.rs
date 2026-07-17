//! Argument and option parsing for the numeric operations, turning expression
//! and table inputs into typed `DiffOpts`, `QuadOpts`, and `OdeOpts`.

use std::{collections::BTreeMap, sync::Arc};

use sim_kernel::{Cx, DefaultFactory, EagerPolicy, Error, Expr, QuoteMode, Result, Symbol, Value};

use super::traits::{DiffOpts, OdeOpts, QuadOpts};

/// Parsed numeric option values keyed by keyword name without the leading `:`.
pub type ParsedOptions = BTreeMap<String, Value>;

pub fn parse_diff_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<(Value, Symbol, Value, DiffOpts)> {
    let [func_expr, var_expr, point_expr, rest @ ..] = args.as_slice() else {
        return Err(Error::Eval(
            "numeric-diff expects func, var, point, and optional key/value options".to_owned(),
        ));
    };
    let options = parse_expr_options(cx, "numeric-diff", rest)?;
    let func = cx.eval_expr(func_expr.clone())?;
    let var = parse_symbolish_expr(var_expr)
        .ok_or_else(|| Error::Eval("numeric-diff expects a symbol variable".to_owned()))?;
    let point = cx.eval_expr(point_expr.clone())?;
    let method = option_symbol(&options, "method")?.unwrap_or(Symbol::new("auto"));
    let h = option_f64(&options, "h")?.unwrap_or(1.0e-6);
    reject_unknown("numeric-diff", &options, &["method", "h"])?;
    Ok((func, var, point, DiffOpts { method, h }))
}

pub fn parse_integrate_exprs(
    cx: &mut Cx,
    name: &str,
    args: Vec<Expr>,
    adaptive: bool,
) -> Result<(Value, Symbol, Value, Value, QuadOpts)> {
    let [func_expr, var_expr, lo_expr, hi_expr, rest @ ..] = args.as_slice() else {
        return Err(Error::Eval(format!(
            "{name} expects func, var, lo, hi, and optional key/value options"
        )));
    };
    let options = parse_expr_options(cx, name, rest)?;
    let func = cx.eval_expr(func_expr.clone())?;
    let var = parse_symbolish_expr(var_expr)
        .ok_or_else(|| Error::Eval(format!("{name} expects a symbol variable")))?;
    let lo = cx.eval_expr(lo_expr.clone())?;
    let hi = cx.eval_expr(hi_expr.clone())?;
    let defaults = if adaptive {
        QuadOpts::adaptive_default()
    } else {
        QuadOpts::fixed_default()
    };
    let method = option_symbol(&options, "method")?.unwrap_or(defaults.method);
    let n = option_usize(&options, "n")?;
    let tol = option_f64(&options, "tol")?.or(defaults.tol);
    reject_unknown(name, &options, &["method", "n", "tol"])?;
    Ok((func, var, lo, hi, QuadOpts { method, n, tol }))
}

pub fn parse_ode_exprs(
    cx: &mut Cx,
    args: Vec<Expr>,
) -> Result<(Value, Symbol, Symbol, Value, Value, Value, OdeOpts)> {
    let [
        dy_expr,
        var_expr,
        y_var_expr,
        x0_expr,
        y0_expr,
        x_end_expr,
        rest @ ..,
    ] = args.as_slice()
    else {
        return Err(Error::Eval(
            "ode-solve expects dy/dx, x var, y var, x0, y0, x-end, and optional key/value options"
                .to_owned(),
        ));
    };
    let options = parse_expr_options(cx, "ode-solve", rest)?;
    let dy = cx.eval_expr(dy_expr.clone())?;
    let var = parse_symbolish_expr(var_expr)
        .ok_or_else(|| Error::Eval("ode-solve expects a symbol x variable".to_owned()))?;
    let y_var = parse_symbolish_expr(y_var_expr)
        .ok_or_else(|| Error::Eval("ode-solve expects a symbol y variable".to_owned()))?;
    let x0 = cx.eval_expr(x0_expr.clone())?;
    let y0 = cx.eval_expr(y0_expr.clone())?;
    let x_end = cx.eval_expr(x_end_expr.clone())?;
    let method = option_symbol(&options, "method")?.unwrap_or(Symbol::new("auto"));
    let h = option_f64(&options, "h")?;
    let tol = option_f64(&options, "tol")?;
    let max_steps = option_usize(&options, "max-steps")?;
    reject_unknown("ode-solve", &options, &["method", "h", "tol", "max-steps"])?;
    Ok((
        dy,
        var,
        y_var,
        x0,
        y0,
        x_end,
        OdeOpts {
            method,
            h,
            tol,
            max_steps,
        },
    ))
}

pub fn parse_table_options(cx: &mut Cx, name: &str, value: &Value) -> Result<ParsedOptions> {
    let expr = value.object().as_expr(cx)?;
    let Expr::Map(entries) = expr else {
        return Err(Error::Eval(format!("{name} options must be a table")));
    };
    let mut options = ParsedOptions::new();
    for (key_expr, value_expr) in entries {
        let key = keyword(&key_expr)?;
        let value = cx.eval_expr(value_expr)?;
        insert_option(&mut options, name, key, value)?;
    }
    Ok(options)
}

pub fn parse_symbolish_value(cx: &mut Cx, value: &Value) -> Result<Option<Symbol>> {
    Ok(parse_symbolish_expr(&value.object().as_expr(cx)?))
}

pub fn option_symbol(options: &ParsedOptions, key: &str) -> Result<Option<Symbol>> {
    match options.get(key) {
        Some(value) => {
            let mut cx = dummy_cx();
            parse_symbolish_value(&mut cx, value)?
                .map(Some)
                .ok_or_else(|| Error::Eval(format!("expected symbol option :{key}")))
        }
        None => Ok(None),
    }
}

pub fn option_f64(options: &ParsedOptions, key: &str) -> Result<Option<f64>> {
    match options.get(key) {
        Some(value) => value
            .object()
            .display(&mut dummy_cx())?
            .parse::<f64>()
            .map(Some)
            .map_err(|_| Error::Eval(format!("expected numeric option :{key}"))),
        None => Ok(None),
    }
}

pub fn option_usize(options: &ParsedOptions, key: &str) -> Result<Option<usize>> {
    match options.get(key) {
        Some(value) => value
            .object()
            .display(&mut dummy_cx())?
            .parse::<usize>()
            .map(Some)
            .map_err(|_| Error::Eval(format!("expected integer option :{key}"))),
        None => Ok(None),
    }
}

pub fn reject_unknown(name: &str, options: &ParsedOptions, allowed: &[&str]) -> Result<()> {
    for key in options.keys() {
        if !allowed.iter().any(|allowed_key| key == allowed_key) {
            return Err(Error::Eval(format!("{name}: unknown option :{key}")));
        }
    }
    Ok(())
}

fn parse_expr_options(cx: &mut Cx, name: &str, exprs: &[Expr]) -> Result<ParsedOptions> {
    if !exprs.len().is_multiple_of(2) {
        return Err(Error::Eval(format!(
            "{name} options must be key/value pairs"
        )));
    }
    let mut options = ParsedOptions::new();
    for pair in exprs.chunks(2) {
        let key = keyword(&pair[0])?;
        let value = cx.eval_expr(pair[1].clone())?;
        insert_option(&mut options, name, key, value)?;
    }
    Ok(options)
}

fn insert_option(options: &mut ParsedOptions, name: &str, key: String, value: Value) -> Result<()> {
    if options.contains_key(&key) {
        return Err(Error::Eval(format!("{name}: duplicate option :{key}")));
    }
    options.insert(key, value);
    Ok(())
}

fn parse_symbolish_expr(expr: &Expr) -> Option<Symbol> {
    match expr {
        Expr::Symbol(symbol) => Some(symbol.clone()),
        Expr::Quote { mode, expr } if *mode == QuoteMode::Quote => match expr.as_ref() {
            Expr::Symbol(symbol) => Some(symbol.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn keyword(expr: &Expr) -> Result<String> {
    let Expr::Symbol(symbol) = expr else {
        return Err(Error::Eval("expected keyword option".to_owned()));
    };
    let Some(keyword) = symbol.name.strip_prefix(':') else {
        return Err(Error::Eval(format!(
            "expected keyword option, found {symbol}"
        )));
    };
    Ok(keyword.to_owned())
}

fn dummy_cx() -> Cx {
    Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory))
}
