//! The Runge-Kutta library and its ODE-solver backends, registering the
//! fixed-step and adaptive integrators as numeric plugins.

use std::sync::{Arc, OnceLock};

use sim_kernel::{
    AbiVersion, Cx, Dependency, Error, Export, Lib, LibManifest, LibTarget, Linker, Result, Symbol,
    Value, Version,
};
use sim_lib_numbers_codec::{numeric_plugin_descriptor_symbol, numeric_plugin_descriptor_value};
use sim_lib_numbers_core::domains;
use sim_lib_numbers_numeric::{
    NumericCallable, NumericKind, NumericPlugin, OdeOpts, OdeProblem, OdeSolver,
    register_ode_solver,
};

use super::support::{abs_error, add, add_scaled, call_rhs, f64_value, scale, value_to_f64};

/// Registered numeric plugin library that installs this crate's Runge-Kutta
/// ODE-solver backends.
///
/// Loading this [`Lib`] registers the fixed-step solvers (forward Euler,
/// backward Euler, midpoint, classic RK4) and the adaptive Runge-Kutta-Fehlberg
/// (RKF45) integrator as `ode-solve` plugins on the numeric surface, and
/// installs the plugin descriptor values that advertise each backend to the
/// registry.
///
/// # Examples
///
/// ```
/// use sim_kernel::Lib;
/// use sim_lib_numbers_rk::RkNumbersLib;
///
/// let lib = RkNumbersLib::new();
/// let manifest = lib.manifest();
/// // One descriptor export per registered solver (4 fixed-step plus RKF45).
/// assert_eq!(manifest.exports.len(), 5);
/// ```
pub struct RkNumbersLib;

impl RkNumbersLib {
    /// Creates the Runge-Kutta library. The value is stateless; all solver
    /// backends are installed when it is loaded into a [`Cx`].
    pub fn new() -> Self {
        Self
    }
}

impl Default for RkNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for RkNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: domains::rk(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: descriptor_exports(),
        }
    }

    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        register_plugins_once()?;
        install_descriptors(cx, linker)?;
        Ok(())
    }
}

static PLUGINS_REGISTERED: OnceLock<std::result::Result<(), String>> = OnceLock::new();

fn register_plugins_once() -> Result<()> {
    match PLUGINS_REGISTERED.get_or_init(|| register_plugins().map_err(|err| err.to_string())) {
        Ok(()) => Ok(()),
        Err(message) => Err(Error::Eval(message.clone())),
    }
}

fn register_plugins() -> Result<()> {
    for plugin in solvers() {
        register_ode_solver(plugin)?;
    }
    Ok(())
}

fn descriptor_exports() -> Vec<Export> {
    descriptor_specs()
        .into_iter()
        .map(|(name, _adaptive)| Export::Value {
            symbol: numeric_plugin_descriptor_symbol("numbers/rk", name),
        })
        .collect()
}

fn install_descriptors(cx: &sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
    for (name, adaptive) in descriptor_specs() {
        linker.value(
            numeric_plugin_descriptor_symbol("numbers/rk", name),
            numeric_plugin_descriptor_value(
                cx.factory(),
                Symbol::new(name),
                "ode",
                adaptive,
                domains::rk(),
            )?,
        )?;
    }
    Ok(())
}

fn descriptor_specs() -> Vec<(&'static str, bool)> {
    vec![
        ("forward-euler", false),
        ("backward-euler", false),
        ("midpoint", false),
        ("rk4", false),
        ("rkf45", true),
    ]
}

#[derive(Clone, Copy)]
enum Method {
    ForwardEuler,
    BackwardEuler,
    Midpoint,
    Rk4,
    Rkf45,
}

fn solvers() -> Vec<Arc<dyn OdeSolver>> {
    vec![
        Arc::new(RkPlugin::new(
            "forward-euler",
            NumericKind::OdeFixed,
            Method::ForwardEuler,
        )),
        Arc::new(RkPlugin::new(
            "backward-euler",
            NumericKind::OdeFixed,
            Method::BackwardEuler,
        )),
        Arc::new(RkPlugin::new(
            "midpoint",
            NumericKind::OdeFixed,
            Method::Midpoint,
        )),
        Arc::new(RkPlugin::new("rk4", NumericKind::OdeFixed, Method::Rk4)),
        Arc::new(RkPlugin::new(
            "rkf45",
            NumericKind::OdeAdaptive,
            Method::Rkf45,
        )),
    ]
}

struct RkPlugin {
    name: Symbol,
    kind: NumericKind,
    method: Method,
}

impl RkPlugin {
    fn new(name: &str, kind: NumericKind, method: Method) -> Self {
        Self {
            name: Symbol::new(name),
            kind,
            method,
        }
    }
}

impl NumericPlugin for RkPlugin {
    fn name(&self) -> Symbol {
        self.name.clone()
    }

    fn kind(&self) -> NumericKind {
        self.kind
    }
}

impl OdeSolver for RkPlugin {
    fn solve(
        &self,
        cx: &mut Cx,
        problem: OdeProblem<'_>,
        opt: OdeOpts,
    ) -> Result<Vec<(Value, Value)>> {
        let x0f = value_to_f64(cx, problem.x0, "ode x0")?;
        let x1f = value_to_f64(cx, problem.x_end, "ode x-end")?;
        match self.method {
            Method::ForwardEuler => fixed_step(
                cx,
                problem.dy,
                x0f,
                problem.y0.clone(),
                x1f,
                opt,
                step_forward_euler,
            ),
            Method::BackwardEuler => fixed_step(
                cx,
                problem.dy,
                x0f,
                problem.y0.clone(),
                x1f,
                opt,
                step_backward_euler,
            ),
            Method::Midpoint => fixed_step(
                cx,
                problem.dy,
                x0f,
                problem.y0.clone(),
                x1f,
                opt,
                step_midpoint,
            ),
            Method::Rk4 => fixed_step(cx, problem.dy, x0f, problem.y0.clone(), x1f, opt, step_rk4),
            Method::Rkf45 => adaptive_rkf45(cx, problem.dy, x0f, problem.y0.clone(), x1f, opt),
        }
    }
}

type FixedStepper = fn(&mut Cx, &NumericCallable, f64, &Value, f64, &OdeOpts) -> Result<Value>;

fn fixed_step(
    cx: &mut Cx,
    dy: &NumericCallable,
    x0: f64,
    y0: Value,
    x1: f64,
    opt: OdeOpts,
    stepper: FixedStepper,
) -> Result<Vec<(Value, Value)>> {
    let mut h = opt.h.unwrap_or(0.01);
    if h == 0.0 {
        return Err(Error::Eval(
            "ode-solve step size :h must be nonzero".to_owned(),
        ));
    }
    let direction = (x1 - x0).signum();
    h = h.abs() * if direction == 0.0 { 1.0 } else { direction };
    let max_steps = opt.max_steps.unwrap_or(100_000);
    let mut out = Vec::new();
    let mut x = x0;
    let mut y = y0;
    out.push((f64_value(cx, x)?, y.clone()));
    let mut steps = 0usize;
    while (x1 - x) * direction > 1.0e-12 {
        if steps >= max_steps {
            return Err(Error::Eval("ode-solve exceeded :max-steps".to_owned()));
        }
        let step = if (x + h - x1) * direction > 0.0 {
            x1 - x
        } else {
            h
        };
        y = stepper(cx, dy, x, &y, step, &opt)?;
        x += step;
        out.push((f64_value(cx, x)?, y.clone()));
        steps += 1;
    }
    Ok(out)
}

fn step_forward_euler(
    cx: &mut Cx,
    dy: &NumericCallable,
    x: f64,
    y: &Value,
    h: f64,
    _opt: &OdeOpts,
) -> Result<Value> {
    let k1 = rhs_at(cx, dy, x, y.clone())?;
    add_scaled(cx, y.clone(), k1, h)
}

fn step_backward_euler(
    cx: &mut Cx,
    dy: &NumericCallable,
    x: f64,
    y: &Value,
    h: f64,
    opt: &OdeOpts,
) -> Result<Value> {
    let x_next = x + h;
    let k1 = rhs_at(cx, dy, x, y.clone())?;
    let mut guess = add_scaled(cx, y.clone(), k1, h)?;
    let tol = opt.tol.unwrap_or(1.0e-10);
    for _ in 0..16 {
        let rhs = rhs_at(cx, dy, x_next, guess.clone())?;
        let next = add_scaled(cx, y.clone(), rhs, h)?;
        if abs_error(cx, next.clone(), guess.clone())? <= tol {
            return Ok(next);
        }
        guess = next;
    }
    Ok(guess)
}

fn step_midpoint(
    cx: &mut Cx,
    dy: &NumericCallable,
    x: f64,
    y: &Value,
    h: f64,
    _opt: &OdeOpts,
) -> Result<Value> {
    let k1 = rhs_at(cx, dy, x, y.clone())?;
    let mid = add_scaled(cx, y.clone(), k1, 0.5 * h)?;
    let k2 = rhs_at(cx, dy, x + 0.5 * h, mid)?;
    add_scaled(cx, y.clone(), k2, h)
}

fn step_rk4(
    cx: &mut Cx,
    dy: &NumericCallable,
    x: f64,
    y: &Value,
    h: f64,
    _opt: &OdeOpts,
) -> Result<Value> {
    let k1 = rhs_at(cx, dy, x, y.clone())?;
    let y2 = add_scaled(cx, y.clone(), k1.clone(), 0.5 * h)?;
    let k2 = rhs_at(cx, dy, x + 0.5 * h, y2)?;
    let y3 = add_scaled(cx, y.clone(), k2.clone(), 0.5 * h)?;
    let k3 = rhs_at(cx, dy, x + 0.5 * h, y3)?;
    let y4 = add_scaled(cx, y.clone(), k3.clone(), h)?;
    let k4 = rhs_at(cx, dy, x + h, y4)?;
    let mut sum = scale(cx, k1, 1.0)?;
    sum = add_scaled(cx, sum, k2, 2.0)?;
    sum = add_scaled(cx, sum, k3, 2.0)?;
    sum = add(cx, sum, k4)?;
    add_scaled(cx, y.clone(), sum, h / 6.0)
}

fn adaptive_rkf45(
    cx: &mut Cx,
    dy: &NumericCallable,
    x0: f64,
    y0: Value,
    x1: f64,
    opt: OdeOpts,
) -> Result<Vec<(Value, Value)>> {
    let direction = (x1 - x0).signum();
    let mut h = opt.h.unwrap_or(((x1 - x0).abs() / 16.0).max(1.0e-3));
    h = h.abs() * if direction == 0.0 { 1.0 } else { direction };
    let tol = opt.tol.unwrap_or(1.0e-8);
    let max_steps = opt.max_steps.unwrap_or(100_000);
    let mut out = Vec::new();
    let mut x = x0;
    let mut y = y0;
    out.push((f64_value(cx, x)?, y.clone()));
    let mut steps = 0usize;
    while (x1 - x) * direction > 1.0e-12 {
        if steps >= max_steps {
            return Err(Error::Eval("ode-solve exceeded :max-steps".to_owned()));
        }
        let step = if (x + h - x1) * direction > 0.0 {
            x1 - x
        } else {
            h
        };
        let (candidate, err) = rkf45_step(cx, dy, x, &y, step)?;
        if err <= tol || step.abs() <= 1.0e-12 {
            x += step;
            y = candidate;
            out.push((f64_value(cx, x)?, y.clone()));
            steps += 1;
            let factor = if err == 0.0 {
                2.0
            } else {
                (0.84 * (tol / err).powf(0.25)).clamp(0.1, 4.0)
            };
            h *= factor;
        } else {
            h *= (0.84 * (tol / err).powf(0.25)).clamp(0.1, 0.5);
        }
    }
    Ok(out)
}

fn rkf45_step(
    cx: &mut Cx,
    dy: &NumericCallable,
    x: f64,
    y: &Value,
    h: f64,
) -> Result<(Value, f64)> {
    let k1 = rhs_at(cx, dy, x, y.clone())?;
    let y2 = add_scaled(cx, y.clone(), k1.clone(), h * 0.25)?;
    let k2 = rhs_at(cx, dy, x + h * 0.25, y2)?;

    let mut y3 = add_scaled(cx, y.clone(), k1.clone(), h * (3.0 / 32.0))?;
    y3 = add_scaled(cx, y3, k2.clone(), h * (9.0 / 32.0))?;
    let k3 = rhs_at(cx, dy, x + h * (3.0 / 8.0), y3)?;

    let mut y4 = add_scaled(cx, y.clone(), k1.clone(), h * (1932.0 / 2197.0))?;
    y4 = add_scaled(cx, y4, k2.clone(), h * (-7200.0 / 2197.0))?;
    y4 = add_scaled(cx, y4, k3.clone(), h * (7296.0 / 2197.0))?;
    let k4 = rhs_at(cx, dy, x + h * (12.0 / 13.0), y4)?;

    let mut y5 = add_scaled(cx, y.clone(), k1.clone(), h * (439.0 / 216.0))?;
    y5 = add_scaled(cx, y5, k2.clone(), h * -8.0)?;
    y5 = add_scaled(cx, y5, k3.clone(), h * (3680.0 / 513.0))?;
    y5 = add_scaled(cx, y5, k4.clone(), h * (-845.0 / 4104.0))?;
    let k5 = rhs_at(cx, dy, x + h, y5)?;

    let mut y6 = add_scaled(cx, y.clone(), k1.clone(), h * (-8.0 / 27.0))?;
    y6 = add_scaled(cx, y6, k2.clone(), h * 2.0)?;
    y6 = add_scaled(cx, y6, k3.clone(), h * (-3544.0 / 2565.0))?;
    y6 = add_scaled(cx, y6, k4.clone(), h * (1859.0 / 4104.0))?;
    y6 = add_scaled(cx, y6, k5.clone(), h * (-11.0 / 40.0))?;
    let k6 = rhs_at(cx, dy, x + h * 0.5, y6)?;

    let mut fourth = add_scaled(cx, y.clone(), k1.clone(), h * (25.0 / 216.0))?;
    fourth = add_scaled(cx, fourth, k3.clone(), h * (1408.0 / 2565.0))?;
    fourth = add_scaled(cx, fourth, k4.clone(), h * (2197.0 / 4104.0))?;
    fourth = add_scaled(cx, fourth, k5.clone(), h * (-1.0 / 5.0))?;

    let mut fifth = add_scaled(cx, y.clone(), k1, h * (16.0 / 135.0))?;
    fifth = add_scaled(cx, fifth, k3, h * (6656.0 / 12825.0))?;
    fifth = add_scaled(cx, fifth, k4, h * (28561.0 / 56430.0))?;
    fifth = add_scaled(cx, fifth, k5, h * (-9.0 / 50.0))?;
    fifth = add_scaled(cx, fifth, k6, h * (2.0 / 55.0))?;

    let err = abs_error(cx, fifth.clone(), fourth)?;
    Ok((fifth, err))
}

fn rhs_at(cx: &mut Cx, dy: &NumericCallable, x: f64, y: Value) -> Result<Value> {
    let x = f64_value(cx, x)?;
    call_rhs(cx, dy, x, y)
}
