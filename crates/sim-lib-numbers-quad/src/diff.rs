//! Finite-difference differentiator plugins (forward, backward, central, and
//! Richardson schemes) for the numeric domain's `numeric-diff` operation.

use std::sync::Arc;

use sim_kernel::{Cx, Result, Symbol, Value};
use sim_lib_numbers_func::Func;
use sim_lib_numbers_numeric::{
    DiffOpts, Differentiator, NumericCallable, NumericKind, NumericPlugin,
};

use super::support::{add, add_scaled, call_unary_callable, div, f64_value, scale, sub, zero_like};

#[derive(Clone, Copy)]
enum Scheme {
    Forward,
    Backward,
    Central3,
    Central5,
    Richardson,
}

pub fn differentiators() -> Vec<Arc<dyn Differentiator>> {
    vec![
        Arc::new(FiniteDiffPlugin::new("forward", Scheme::Forward)),
        Arc::new(FiniteDiffPlugin::new("backward", Scheme::Backward)),
        Arc::new(FiniteDiffPlugin::new("central-3", Scheme::Central3)),
        Arc::new(FiniteDiffPlugin::new("central-5", Scheme::Central5)),
        Arc::new(FiniteDiffPlugin::new("richardson", Scheme::Richardson)),
    ]
}

struct FiniteDiffPlugin {
    name: Symbol,
    scheme: Scheme,
}

impl FiniteDiffPlugin {
    fn new(name: &str, scheme: Scheme) -> Self {
        Self {
            name: Symbol::new(name),
            scheme,
        }
    }
}

impl NumericPlugin for FiniteDiffPlugin {
    fn name(&self) -> Symbol {
        self.name.clone()
    }

    fn kind(&self) -> NumericKind {
        NumericKind::Differentiator
    }
}

impl Differentiator for FiniteDiffPlugin {
    fn diff_at(
        &self,
        cx: &mut Cx,
        f: &Func,
        var: &Symbol,
        point: &Value,
        opt: DiffOpts,
    ) -> Result<Value> {
        let value = cx.factory().opaque(Arc::new(f.clone()))?;
        let callable = NumericCallable::unary(value, var.clone())?;
        self.diff_callable_at(cx, &callable, var, point, opt)
    }

    fn diff_callable_at(
        &self,
        cx: &mut Cx,
        f: &NumericCallable,
        _var: &Symbol,
        point: &Value,
        opt: DiffOpts,
    ) -> Result<Value> {
        match self.scheme {
            Scheme::Forward => {
                let fx = call_unary_callable(cx, f, point.clone())?;
                let point_h = offset(cx, point, opt.h)?;
                let fh = call_unary_callable(cx, f, point_h)?;
                let numerator = sub(cx, fh, fx)?;
                let denom = f64_value(cx, opt.h)?;
                div(cx, numerator, denom)
            }
            Scheme::Backward => {
                let fx = call_unary_callable(cx, f, point.clone())?;
                let point_h = offset(cx, point, -opt.h)?;
                let fh = call_unary_callable(cx, f, point_h)?;
                let numerator = sub(cx, fx, fh)?;
                let denom = f64_value(cx, opt.h)?;
                div(cx, numerator, denom)
            }
            Scheme::Central3 => central3(cx, f, point, opt.h),
            Scheme::Central5 => central5(cx, f, point, opt.h),
            Scheme::Richardson => {
                let coarse = central3(cx, f, point, opt.h)?;
                let fine = central3(cx, f, point, opt.h * 0.5)?;
                let fine4 = scale(cx, fine, 4.0)?;
                let numerator = sub(cx, fine4, coarse)?;
                let denom = f64_value(cx, 3.0)?;
                div(cx, numerator, denom)
            }
        }
    }
}

fn central3(cx: &mut Cx, f: &NumericCallable, point: &Value, h: f64) -> Result<Value> {
    let plus_point = offset(cx, point, h)?;
    let minus_point = offset(cx, point, -h)?;
    let plus = call_unary_callable(cx, f, plus_point)?;
    let minus = call_unary_callable(cx, f, minus_point)?;
    let numerator = sub(cx, plus, minus)?;
    let denom = f64_value(cx, 2.0 * h)?;
    div(cx, numerator, denom)
}

fn central5(cx: &mut Cx, f: &NumericCallable, point: &Value, h: f64) -> Result<Value> {
    let p2 = offset(cx, point, 2.0 * h)?;
    let p1 = offset(cx, point, h)?;
    let m1 = offset(cx, point, -h)?;
    let m2 = offset(cx, point, -2.0 * h)?;
    let p2 = call_unary_callable(cx, f, p2)?;
    let p1 = call_unary_callable(cx, f, p1)?;
    let m1 = call_unary_callable(cx, f, m1)?;
    let m2 = call_unary_callable(cx, f, m2)?;
    let mut acc = zero_like(cx, p1.clone())?;
    acc = add_scaled(cx, acc, p2, -1.0)?;
    acc = add_scaled(cx, acc, p1, 8.0)?;
    acc = add_scaled(cx, acc, m1, -8.0)?;
    acc = add_scaled(cx, acc, m2, 1.0)?;
    let denom = f64_value(cx, 12.0 * h)?;
    div(cx, acc, denom)
}

fn offset(cx: &mut Cx, point: &Value, delta: f64) -> Result<Value> {
    let delta = f64_value(cx, delta)?;
    add(cx, point.clone(), delta)
}
