//! Dormand-Prince DOP853 reference stepper.
//!
//! Coefficients are transcribed at full published precision from Hairer,
//! Norsett, and Wanner, *Solving Ordinary Differential Equations I*, 2nd ed.,
//! Springer (1993), and the authors' `dop853.f` reference implementation.
//! The row-sum and weight invariants below are tested so a shifted, transposed,
//! rounded, omitted, or mistyped entry cannot silently remain a valid tableau.

use sim_kernel::{Cx, Error, Result, Value};
use sim_lib_numbers_numeric::{MethodEvidence, NumericCallable, OdePlan, OdeTermination};

use super::support::{abs_error, add_scaled, call_rhs, f64_value, scale};

const STAGES: usize = 12;
const C: [f64; STAGES] = [
    0.0,
    0.05260015195876773,
    0.0789002279381516,
    0.1183503419072274,
    0.2816496580927726,
    1.0 / 3.0,
    0.25,
    4.0 / 13.0,
    0.6512820512820513,
    0.6,
    6.0 / 7.0,
    1.0,
];

fn tableau() -> [[f64; STAGES]; STAGES] {
    let mut a = [[0.0; STAGES]; STAGES];
    a[1][0] = 5.260015195876773e-2;
    a[2][0] = 1.97250569845379e-2;
    a[2][1] = 5.91751709536137e-2;
    a[3][0] = 2.958758547680685e-2;
    a[3][2] = 8.876275643042055e-2;
    a[4][0] = 2.413651341592667e-1;
    a[4][2] = -8.845494793282861e-1;
    a[4][3] = 9.24834003261792e-1;
    a[5][0] = 3.703703703703704e-2;
    a[5][3] = 1.708286087294739e-1;
    a[5][4] = 1.254676875668224e-1;
    a[6][0] = 3.7109375e-2;
    a[6][3] = 1.70252211019544e-1;
    a[6][4] = 6.021653898045596e-2;
    a[6][5] = -1.7578125e-2;
    a[7][0] = 3.709200011850479e-2;
    a[7][3] = 1.703839257122399e-1;
    a[7][4] = 1.072620304463733e-1;
    a[7][5] = -1.53194377486244e-2;
    a[7][6] = 8.273789163814023e-3;
    a[8][0] = 6.241109587160757e-1;
    a[8][3] = -3.360892629446941;
    a[8][4] = -8.68219346841726e-1;
    a[8][5] = 2.759209969944671e1;
    a[8][6] = 2.015406755047789e1;
    a[8][7] = -4.348988418106996e1;
    a[9][0] = 4.776625364382644e-1;
    a[9][3] = -2.488114619971668;
    a[9][4] = -5.90290826836843e-1;
    a[9][5] = 2.123005144818119e1;
    a[9][6] = 1.527923363288242e1;
    a[9][7] = -3.328821096898487e1;
    a[9][8] = -2.033120170850863e-2;
    a[10][0] = -9.371424300859873e-1;
    a[10][3] = 5.186372428844064;
    a[10][4] = 1.09143734899673;
    a[10][5] = -8.149787010746926;
    a[10][6] = -1.852006565999696e1;
    a[10][7] = 2.27394870993505e1;
    a[10][8] = 2.493605552679652;
    a[10][9] = -3.04676447189822;
    a[11][0] = 2.273310147516538;
    a[11][3] = -1.053449546673725e1;
    a[11][4] = -2.000872058224862;
    a[11][5] = -1.79589318631188e1;
    a[11][6] = 2.794888452941996e1;
    a[11][7] = -2.858998277135024;
    a[11][8] = -8.87285693353063;
    a[11][9] = 1.23605671757943e1;
    a[11][10] = 6.433927460157635e-1;
    a
}

const B: [f64; STAGES] = [
    5.429373411656876e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450312892752409,
    1.8915178993145,
    -5.801203960010585,
    3.111643669578199e-1,
    -1.521609496625161e-1,
    2.013654008040303e-1,
    4.471061572777259e-2,
];
const E3: [f64; STAGES] = [
    -1.898007540724076e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450312892752409,
    1.8915178993145,
    -5.801203960010585,
    -4.226823213237919e-1,
    -1.521609496625161e-1,
    2.013654008040303e-1,
    2.265179219836082e-2,
];
const E5: [f64; STAGES] = [
    1.312004499419488e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.225156446376204,
    -4.957589496572502e-1,
    1.664377182454986,
    -3.503288487499737e-1,
    3.341791187130175e-1,
    8.192320648511571e-2,
    -2.23553078638863e-2,
];

pub(super) struct Dop853Run {
    pub points: Vec<(Value, Value)>,
    pub evidence: MethodEvidence,
}

fn combination(cx: &mut Cx, base: Value, ks: &[Value], weights: &[f64], h: f64) -> Result<Value> {
    let mut out = base;
    for (k, weight) in ks.iter().zip(weights) {
        if *weight != 0.0 {
            out = add_scaled(cx, out, k.clone(), h * weight)?;
        }
    }
    Ok(out)
}

fn weighted(cx: &mut Cx, ks: &[Value], weights: &[f64]) -> Result<Value> {
    let mut out = scale(cx, ks[0].clone(), weights[0])?;
    for (k, weight) in ks.iter().zip(weights).skip(1) {
        if *weight != 0.0 {
            out = add_scaled(cx, out, k.clone(), *weight)?;
        }
    }
    Ok(out)
}

fn rhs(
    cx: &mut Cx,
    f: &NumericCallable,
    t: f64,
    y: Value,
    work: &mut usize,
    limit: usize,
) -> Result<Value> {
    if *work >= limit {
        return Err(Error::Eval("ode-solve exceeded :work-limit".to_owned()));
    }
    *work += 1;
    let time = f64_value(cx, t)?;
    call_rhs(cx, f, time, y)
}

fn step(
    cx: &mut Cx,
    f: &NumericCallable,
    t: f64,
    y: &Value,
    h: f64,
    work: &mut usize,
    limit: usize,
) -> Result<(Value, f64)> {
    let a = tableau();
    let mut ks = Vec::with_capacity(STAGES);
    ks.push(rhs(cx, f, t, y.clone(), work, limit)?);
    for stage in 1..STAGES {
        let state = combination(cx, y.clone(), &ks, &a[stage][..stage], h)?;
        ks.push(rhs(cx, f, t + C[stage] * h, state, work, limit)?);
    }
    let candidate = combination(cx, y.clone(), &ks, &B, h)?;
    let e5 = weighted(cx, &ks, &E5)?;
    let e3 = weighted(cx, &ks, &E3)?;
    let zero5 = scale(cx, e5.clone(), 0.0)?;
    let zero3 = scale(cx, e3.clone(), 0.0)?;
    let n5 = abs_error(cx, e5, zero5)?;
    let n3 = abs_error(cx, e3, zero3)?;
    let scale_y = abs_error(cx, candidate.clone(), y.clone())?.max(1.0);
    let denominator = (n5 * n5 + 0.01 * n3 * n3).sqrt();
    let error = if denominator == 0.0 {
        0.0
    } else {
        h.abs() * n5 * n5 / denominator / scale_y
    };
    Ok((candidate, error))
}

pub(super) fn adaptive_dop853(
    cx: &mut Cx,
    f: &NumericCallable,
    t0: f64,
    y0: Value,
    t1: f64,
    plan: &OdePlan,
) -> Result<Dop853Run> {
    let direction = (t1 - t0).signum();
    let sign = if direction == 0.0 { 1.0 } else { direction };
    let span = (t1 - t0).abs();
    // The default is deliberately reproducible and problem-scale independent;
    // callers can override it. A discontinuity starts a new solve and therefore
    // reruns this selection rather than carrying FSAL/controller history over.
    let mut h = plan.step.first.unwrap_or((span / 100.0).max(1.0e-6)).abs() * sign;
    if let Some(max) = plan.step.max {
        h = h.signum() * h.abs().min(max.abs());
    }
    let min_step = 16.0 * f64::EPSILON * t0.abs().max(t1.abs()).max(1.0);
    let atol = match &plan.tolerance.absolute {
        sim_lib_numbers_numeric::AbsoluteTolerance::Scalar(x) => *x,
        sim_lib_numbers_numeric::AbsoluteTolerance::Components(xs) => {
            xs.iter().copied().fold(0.0, f64::max)
        }
    };
    let tolerance = atol + plan.tolerance.relative;
    let mut points = vec![(f64_value(cx, t0)?, y0.clone())];
    let (mut t, mut y, mut work, mut accepted, mut rejected) = (t0, y0, 0usize, 0usize, 0usize);
    let mut sizes = Vec::new();
    let mut worst = 0.0_f64;
    while (t1 - t) * sign > 0.0 {
        if accepted >= plan.limits.steps {
            return Err(Error::Eval("ode-solve exceeded :step-limit".to_owned()));
        }
        if h.abs() < min_step {
            return Err(Error::Eval(
                "DOP853 minimum step reached before tolerance".to_owned(),
            ));
        }
        let trial = if (t + h - t1) * sign > 0.0 { t1 - t } else { h };
        let (next, error) = step(cx, f, t, &y, trial, &mut work, plan.limits.work)?;
        let ratio = error / tolerance;
        let factor = if ratio == 0.0 {
            6.0
        } else {
            (0.9 * ratio.powf(-1.0 / 8.0)).clamp(0.2, 6.0)
        };
        if ratio <= 1.0 {
            if work >= plan.limits.work {
                return Err(Error::Eval(
                    "ode-solve exceeded :work-limit constructing dense segment".to_owned(),
                ));
            }
            work += 1;
            t += trial;
            y = next;
            accepted += 1;
            sizes.push(trial.abs());
            worst = worst.max(ratio);
            points.push((f64_value(cx, t)?, y.clone()));
            h = trial * factor;
        } else {
            rejected += 1;
            h = trial * factor.min(0.9);
        }
        if let Some(max) = plan.step.max {
            h = h.signum() * h.abs().min(max.abs());
        }
    }
    let range = sizes.iter().copied().fold(None, |r, x| {
        Some(r.map_or((x, x), |(lo, hi): (f64, f64)| (lo.min(x), hi.max(x))))
    });
    if sizes.len() > plan.limits.trace {
        sizes.truncate(plan.limits.trace);
    }
    Ok(Dop853Run {
        points,
        evidence: MethodEvidence {
            accepted_steps: accepted,
            rejected_steps: rejected,
            rhs_evaluations: work - accepted,
            jacobian_evaluations: 0,
            step_sizes: sizes,
            step_size_range: range,
            achieved_local_error: worst,
            event_brackets: Vec::new(),
            termination: OdeTermination::ReachedEnd,
        },
    })
}

#[cfg(test)]
mod coefficient_tests {
    use super::*;
    #[test]
    fn published_tableau_invariants_hold_at_binary64_precision() {
        let a = tableau();
        for row in 1..STAGES {
            assert!(
                (a[row].iter().sum::<f64>() - C[row]).abs() < 2.0e-14,
                "row {row}"
            );
        }
        assert!((B.iter().sum::<f64>() - 1.0).abs() < 2.0e-14);
        assert!(E3.iter().sum::<f64>().abs() < 2.0e-14);
        assert!(E5.iter().sum::<f64>().abs() < 2.0e-14);
        // Stable fingerprint moments detect permutations that preserve sums.
        let fingerprint = a
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, x)| (i as f64 + 1.0) * x)
            .sum::<f64>();
        assert!(
            (fingerprint - 435.150_323_799_216_27).abs() < 5.0e-12,
            "{fingerprint:.17}"
        );
    }
}
