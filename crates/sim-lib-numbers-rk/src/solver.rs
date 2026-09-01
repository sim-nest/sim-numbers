//! The Runge-Kutta library and its ODE-solver backends, registering the
//! fixed-step and adaptive integrators as numeric plugins.

use std::sync::{Arc, OnceLock};

use sim_kernel::{
    AbiVersion, Cx, Dependency, Error, Export, Lib, LibManifest, LibTarget, Linker, Result, Symbol,
    Value, Version,
};
use sim_lib_numbers_codec::numeric_plugin_descriptor_symbol;
use sim_lib_numbers_core::domains;
use sim_lib_numbers_numeric::{
    AbsoluteTolerance, AcceptedStep, DenseSegment, EventDirection, LocatedEvent, MethodEvidence,
    NumericCallable, NumericKind, NumericPlugin, OdeCapabilities, OdePlan, OdeProblem, OdeSolution,
    OdeSolver, OdeTermination, Trajectory, register_ode_solver,
};

use super::dop853::adaptive_dop853;
use super::support::{abs_error, add, add_scaled, call_rhs, f64_value, scale};

#[path = "solver/registration.rs"]
mod registration;

use registration::Method;
pub use registration::RkNumbersLib;

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
    fn capabilities(&self) -> OdeCapabilities {
        OdeCapabilities {
            scalar_state: true,
            tensor_state: true,
            adaptive: matches!(self.method, Method::Rkf45 | Method::Dop853),
            fixed: !matches!(self.method, Method::Rkf45 | Method::Dop853),
            dense_path: true,
            events: true,
            jacobian: false,
            mass_matrix: false,
            dae_residual: false,
        }
    }

    fn solve(&self, cx: &mut Cx, problem: OdeProblem<'_>, plan: OdePlan) -> Result<OdeSolution> {
        validate_admission(&problem, &plan, self.capabilities())?;
        let x0f = problem.span.start;
        let x1f = problem.span.end;
        let (points, evidence) = match self.method {
            Method::ForwardEuler => (
                fixed_step(
                    cx,
                    problem.rhs,
                    x0f,
                    problem.initial.clone(),
                    x1f,
                    plan.clone(),
                    step_forward_euler,
                )?,
                None,
            ),
            Method::BackwardEuler => (
                fixed_step(
                    cx,
                    problem.rhs,
                    x0f,
                    problem.initial.clone(),
                    x1f,
                    plan.clone(),
                    step_backward_euler,
                )?,
                None,
            ),
            Method::Midpoint => (
                fixed_step(
                    cx,
                    problem.rhs,
                    x0f,
                    problem.initial.clone(),
                    x1f,
                    plan.clone(),
                    step_midpoint,
                )?,
                None,
            ),
            Method::Rk4 => (
                fixed_step(
                    cx,
                    problem.rhs,
                    x0f,
                    problem.initial.clone(),
                    x1f,
                    plan.clone(),
                    step_rk4,
                )?,
                None,
            ),
            Method::Rkf45 => (
                adaptive_rkf45(
                    cx,
                    problem.rhs,
                    x0f,
                    problem.initial.clone(),
                    x1f,
                    plan.clone(),
                )?,
                None,
            ),
            Method::Dop853 => {
                let run =
                    adaptive_dop853(cx, problem.rhs, x0f, problem.initial.clone(), x1f, &plan)?;
                (run.points, Some(run.evidence))
            }
        };
        let mut result = solution(points, &plan, evidence);
        locate_events(cx, &problem, &plan, &mut result)?;
        Ok(result)
    }
}

type FixedStepper = fn(&mut Cx, &NumericCallable, f64, &Value, f64, &OdePlan) -> Result<Value>;

fn absolute_tolerance(plan: &OdePlan) -> f64 {
    match &plan.tolerance.absolute {
        AbsoluteTolerance::Scalar(value) => *value,
        AbsoluteTolerance::Components(values) => values.iter().copied().fold(0.0, f64::max),
    }
}

fn validate_admission(
    problem: &OdeProblem<'_>,
    plan: &OdePlan,
    caps: OdeCapabilities,
) -> Result<()> {
    if !plan.tolerance.relative.is_finite()
        || plan.tolerance.relative < 0.0
        || absolute_tolerance(plan) <= 0.0
    {
        return Err(Error::Eval(
            "ode-solve tolerances must be finite, rtol nonnegative, and atol positive".to_owned(),
        ));
    }
    if !problem.events.is_empty() && !caps.events {
        return Err(Error::Eval(
            "ode backend does not support requested events".to_owned(),
        ));
    }
    if problem.jacobian.is_some() && !caps.jacobian {
        return Err(Error::Eval(
            "ode backend does not support requested Jacobian".to_owned(),
        ));
    }
    if let Some(form) = problem.implicit {
        let supported = match form {
            sim_lib_numbers_numeric::ImplicitForm::MassMatrix(_) => caps.mass_matrix,
            sim_lib_numbers_numeric::ImplicitForm::Residual(_) => caps.dae_residual,
        };
        if !supported {
            return Err(Error::Eval(
                "ode backend does not support requested implicit form".to_owned(),
            ));
        }
    }
    if (plan.output.retain_dense || !plan.output.samples.is_empty()) && !caps.dense_path {
        return Err(Error::Eval(
            "ode backend does not support requested dense output".to_owned(),
        ));
    }
    let lo = problem.span.start.min(problem.span.end);
    let hi = problem.span.start.max(problem.span.end);
    if plan
        .output
        .samples
        .iter()
        .any(|time| !time.is_finite() || *time < lo || *time > hi)
    {
        return Err(Error::Eval(
            "ode output sample lies outside span".to_owned(),
        ));
    }
    Ok(())
}

fn solution(
    points: Vec<(Value, Value)>,
    plan: &OdePlan,
    evidence: Option<MethodEvidence>,
) -> OdeSolution {
    let accepted = points
        .into_iter()
        .map(|(time, state)| AcceptedStep {
            time: time
                .object()
                .display(&mut Cx::new(
                    std::sync::Arc::new(sim_kernel::EagerPolicy),
                    std::sync::Arc::new(sim_kernel::DefaultFactory),
                    sim_kernel::HandleSeed::new(0x4f44_4501),
                ))
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(f64::NAN),
            state,
        })
        .collect::<Vec<_>>();
    let dense = accepted
        .windows(2)
        .map(|pair| DenseSegment {
            start: pair[0].clone(),
            end: pair[1].clone(),
        })
        .collect();
    let sizes = accepted
        .windows(2)
        .map(|pair| (pair[1].time - pair[0].time).abs())
        .collect::<Vec<_>>();
    let retained = if sizes.len() <= plan.limits.trace {
        sizes.clone()
    } else {
        sizes[..plan.limits.trace].to_vec()
    };
    let range = sizes.iter().copied().fold(None::<(f64, f64)>, |range, h| {
        Some(match range {
            None => (h, h),
            Some((lo, hi)) => (lo.min(h), hi.max(h)),
        })
    });
    OdeSolution {
        path: Trajectory { accepted, dense },
        events: Vec::new(),
        evidence: evidence.unwrap_or(MethodEvidence {
            accepted_steps: sizes.len(),
            rejected_steps: 0,
            rhs_evaluations: 0,
            jacobian_evaluations: 0,
            step_sizes: retained,
            step_size_range: range,
            achieved_local_error: 0.0,
            event_brackets: Vec::new(),
            termination: OdeTermination::ReachedEnd,
        }),
    }
}

fn event_value(
    cx: &mut Cx,
    event: &sim_lib_numbers_numeric::EventFunction,
    time: f64,
    state: Value,
    work: &mut usize,
    work_limit: usize,
) -> Result<f64> {
    if *work >= work_limit {
        return Err(Error::Eval(
            "ode-solve exceeded :work-limit during event evaluation".to_owned(),
        ));
    }
    *work += 1;
    let time_value = f64_value(cx, time)?;
    let value = event.function.call(cx, vec![time_value, state])?;
    value
        .object()
        .display(cx)?
        .parse()
        .map_err(|_| Error::Eval("ODE event function must return f64".to_owned()))
}

fn interpolate(cx: &mut Cx, segment: &DenseSegment, time: f64) -> Result<Value> {
    let width = segment.end.time - segment.start.time;
    if width == 0.0 {
        return Ok(segment.start.state.clone());
    }
    let delta = cx.apply_value_number_binary_op(
        &Symbol::qualified("math", "sub"),
        segment.end.state.clone(),
        segment.start.state.clone(),
    )?;
    add_scaled(
        cx,
        segment.start.state.clone(),
        delta,
        (time - segment.start.time) / width,
    )
}

fn crosses(direction: EventDirection, left: f64, right: f64) -> bool {
    match direction {
        EventDirection::Rising => left < 0.0 && right >= 0.0,
        EventDirection::Falling => left > 0.0 && right <= 0.0,
        EventDirection::Either => (left < 0.0 && right >= 0.0) || (left > 0.0 && right <= 0.0),
    }
}

fn locate_events(
    cx: &mut Cx,
    problem: &OdeProblem<'_>,
    plan: &OdePlan,
    solution: &mut OdeSolution,
) -> Result<()> {
    // Dense-segment construction and event calls share the plan's hard work
    // budget with RHS evaluations. This prevents output policy from creating
    // an unbounded second computation after stepping succeeds.
    let mut work = solution.evidence.rhs_evaluations + solution.evidence.accepted_steps;
    let mut located = Vec::new();
    for (event_index, event) in problem.events.iter().enumerate() {
        let mut start_seen = false;
        for segment in &solution.path.dense {
            let mut lo = segment.start.time;
            let mut hi = segment.end.time;
            let mut flo = event_value(
                cx,
                event,
                lo,
                segment.start.state.clone(),
                &mut work,
                plan.limits.work,
            )?;
            let fhi = event_value(
                cx,
                event,
                hi,
                segment.end.state.clone(),
                &mut work,
                plan.limits.work,
            )?;
            if flo == 0.0 && !start_seen {
                start_seen = true;
                located.push(LocatedEvent {
                    event: event_index,
                    time: lo,
                    state_before: segment.start.state.clone(),
                    state_after: segment.start.state.clone(),
                    terminal: event.terminal,
                    priority: event.priority,
                });
            }
            if !crosses(event.direction, flo, fhi) {
                continue;
            }
            for _ in 0..53 {
                let mid = lo + (hi - lo) * 0.5;
                let state = interpolate(cx, segment, mid)?;
                let fm = event_value(cx, event, mid, state, &mut work, plan.limits.work)?;
                if fm == 0.0 {
                    lo = mid;
                    hi = mid;
                    break;
                }
                if crosses(event.direction, flo, fm) {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            let time = lo + (hi - lo) * 0.5;
            solution.evidence.event_brackets.push((event_index, lo, hi));
            located.push(LocatedEvent {
                event: event_index,
                time,
                state_before: interpolate(cx, segment, lo)?,
                state_after: interpolate(cx, segment, hi)?,
                terminal: event.terminal,
                priority: event.priority,
            });
        }
    }
    located.sort_by(|a, b| {
        a.time
            .total_cmp(&b.time)
            .then_with(|| b.priority.cmp(&a.priority))
            .then_with(|| a.event.cmp(&b.event))
    });
    located.dedup_by(|a, b| a.event == b.event && a.time == b.time);
    if let Some(terminal) = located
        .iter()
        .filter(|event| event.terminal)
        .min_by(|a, b| {
            a.time
                .total_cmp(&b.time)
                .then_with(|| b.priority.cmp(&a.priority))
        })
        .cloned()
    {
        solution.path.accepted.retain(|step| {
            if problem.span.end >= problem.span.start {
                step.time < terminal.time
            } else {
                step.time > terminal.time
            }
        });
        solution.path.accepted.push(AcceptedStep {
            time: terminal.time,
            state: terminal.state_after.clone(),
        });
        solution.path.dense.retain(|segment| {
            if problem.span.end >= problem.span.start {
                segment.start.time < terminal.time
            } else {
                segment.start.time > terminal.time
            }
        });
        solution.evidence.termination = OdeTermination::TerminalEvent(terminal.event);
        located.retain(|event| {
            if problem.span.end >= problem.span.start {
                event.time <= terminal.time
            } else {
                event.time >= terminal.time
            }
        });
    }
    solution.events = located;
    Ok(())
}

fn fixed_step(
    cx: &mut Cx,
    dy: &NumericCallable,
    x0: f64,
    y0: Value,
    x1: f64,
    opt: OdePlan,
    stepper: FixedStepper,
) -> Result<Vec<(Value, Value)>> {
    let mut h = opt.step.fixed.unwrap_or(0.01);
    if h == 0.0 {
        return Err(Error::Eval(
            "ode-solve :fixed-step must be nonzero".to_owned(),
        ));
    }
    let direction = (x1 - x0).signum();
    h = h.abs() * if direction == 0.0 { 1.0 } else { direction };
    let max_steps = opt.limits.steps;
    let mut out = Vec::new();
    let mut x = x0;
    let mut y = y0;
    out.push((f64_value(cx, x)?, y.clone()));
    let mut steps = 0usize;
    while (x1 - x) * direction > 1.0e-12 {
        if steps >= max_steps {
            return Err(Error::Eval("ode-solve exceeded :step-limit".to_owned()));
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
    _opt: &OdePlan,
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
    opt: &OdePlan,
) -> Result<Value> {
    let x_next = x + h;
    let k1 = rhs_at(cx, dy, x, y.clone())?;
    let mut guess = add_scaled(cx, y.clone(), k1, h)?;
    let tol = absolute_tolerance(opt);
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
    _opt: &OdePlan,
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
    _opt: &OdePlan,
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
    opt: OdePlan,
) -> Result<Vec<(Value, Value)>> {
    let direction = (x1 - x0).signum();
    let mut h = opt
        .step
        .first
        .unwrap_or(((x1 - x0).abs() / 16.0).max(1.0e-3));
    h = h.abs() * if direction == 0.0 { 1.0 } else { direction };
    let tol = opt.tolerance.relative.max(absolute_tolerance(&opt));
    let max_steps = opt.limits.steps;
    let mut out = Vec::new();
    let mut x = x0;
    let mut y = y0;
    out.push((f64_value(cx, x)?, y.clone()));
    let mut steps = 0usize;
    while (x1 - x) * direction > 1.0e-12 {
        if steps >= max_steps {
            return Err(Error::Eval("ode-solve exceeded :step-limit".to_owned()));
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
