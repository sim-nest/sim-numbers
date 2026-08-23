//! Lisp adapters for bounded sequential-decision statistics.

use sim_kernel::{Cx, Error, Result, Symbol, Value};

use super::runtime_clustering::{f64_value, u64_value, usize_value, value_to_points};
use super::{
    BootstrapControl, ClusterSample, IsotonicPoint, RegisteredLook, RegisteredLookSequence,
    clustered_bootstrap_interval, exact_binary_interval, fit_isotonic, paired_bootstrap_interval,
    stats_clustered_bootstrap_symbol, stats_exact_binary_interval_symbol, stats_isotonic_symbol,
    stats_paired_bootstrap_symbol, stats_registered_look_symbol,
};

pub(crate) fn is_symbol(symbol: &Symbol) -> bool {
    symbol == &stats_exact_binary_interval_symbol()
        || symbol == &stats_paired_bootstrap_symbol()
        || symbol == &stats_clustered_bootstrap_symbol()
        || symbol == &stats_registered_look_symbol()
        || symbol == &stats_isotonic_symbol()
}

pub(crate) fn call(cx: &mut Cx, symbol: &Symbol, values: Vec<Value>) -> Result<Value> {
    if symbol == &stats_exact_binary_interval_symbol() {
        let rows = one_matrix(cx, &values, "stats/exact-binary-interval")?;
        let row = exact_width(&rows, 3, "(successes trials confidence)")?;
        let interval = exact_binary_interval(
            integer(row[0], "successes")?,
            integer(row[1], "trials")?,
            row[2],
        )
        .map_err(stats_error)?;
        let entries = vec![
            (Symbol::new("successes"), u64_value(cx, interval.successes)?),
            (Symbol::new("trials"), u64_value(cx, interval.trials)?),
            (
                Symbol::new("confidence"),
                f64_value(cx, interval.confidence_level)?,
            ),
            (Symbol::new("lower"), f64_value(cx, interval.lower)?),
            (Symbol::new("upper"), f64_value(cx, interval.upper)?),
        ];
        return cx.factory().table(entries);
    }
    if symbol == &stats_paired_bootstrap_symbol() {
        let [pairs, control_value] = values.as_slice() else {
            return Err(arity("stats/paired-bootstrap", 2));
        };
        let pairs = value_to_points(cx, pairs, "paired rows")?
            .into_iter()
            .map(|row| {
                let row = exact_row(&row, 2, "(baseline candidate)")?;
                Ok((row[0], row[1]))
            })
            .collect::<Result<Vec<_>>>()?;
        let control = control(cx, control_value)?;
        return bootstrap_value(
            cx,
            paired_bootstrap_interval(&pairs, control).map_err(stats_error)?,
        );
    }
    if symbol == &stats_clustered_bootstrap_symbol() {
        let [rows, control] = values.as_slice() else {
            return Err(arity("stats/clustered-bootstrap", 2));
        };
        let rows = value_to_points(cx, rows, "cluster rows")?;
        let mut clusters = Vec::<ClusterSample>::new();
        for row in rows {
            let row = exact_row(&row, 3, "(cluster-id baseline candidate)")?;
            let id = integer(row[0], "cluster-id")?;
            if let Some(c) = clusters.iter_mut().find(|c| c.id == id) {
                c.pairs.push((row[1], row[2]));
            } else {
                clusters.push(ClusterSample {
                    id,
                    pairs: vec![(row[1], row[2])],
                });
            }
        }
        let control_rows = value_to_points(cx, control, "BootstrapControl")?;
        let row = exact_width(
            &control_rows,
            5,
            "(seed resamples confidence work minimum-clusters)",
        )?;
        let bootstrap = BootstrapControl::new(
            integer(row[0], "seed")?,
            usize_integer(row[1], "resamples")?,
            row[2],
            integer(row[3], "work")?,
        )
        .map_err(stats_error)?;
        return bootstrap_value(
            cx,
            clustered_bootstrap_interval(
                &clusters,
                usize_integer(row[4], "minimum-clusters")?,
                bootstrap,
            )
            .map_err(stats_error)?,
        );
    }
    if symbol == &stats_registered_look_symbol() {
        let [observations, looks, budget] = values.as_slice() else {
            return Err(arity("stats/registered-look-interval", 3));
        };
        let observations = value_to_points(cx, observations, "observations")?
            .into_iter()
            .map(|row| Ok(*exact_row(&row, 1, "scalar observation")?.first().unwrap()))
            .collect::<Result<Vec<_>>>()?;
        let looks = value_to_points(cx, looks, "registered looks")?
            .into_iter()
            .map(|row| {
                let row = exact_row(&row, 2, "(samples alpha)")?;
                Ok(RegisteredLook {
                    samples: usize_integer(row[0], "samples")?,
                    alpha: row[1],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let budget_rows = value_to_points(cx, budget, "total budget")?;
        let budget = *exact_width(&budget_rows, 1, "total budget")?
            .first()
            .unwrap();
        let interval = RegisteredLookSequence::new(looks, budget)
            .and_then(|s| s.interval(&observations))
            .map_err(stats_error)?;
        let entries = vec![
            (Symbol::new("samples"), usize_value(cx, interval.samples)?),
            (Symbol::new("mean"), f64_value(cx, interval.mean)?),
            (Symbol::new("lower"), f64_value(cx, interval.lower)?),
            (Symbol::new("upper"), f64_value(cx, interval.upper)?),
            (
                Symbol::new("alpha-spent"),
                f64_value(cx, interval.alpha_spent)?,
            ),
            (
                Symbol::new("total-budget"),
                f64_value(cx, interval.total_budget)?,
            ),
        ];
        return cx.factory().table(entries);
    }
    let rows = one_matrix(cx, &values, "stats/isotonic")?;
    let points = rows
        .into_iter()
        .map(|row| {
            let row = exact_row(&row, 3, "(level value weight)")?;
            Ok(IsotonicPoint {
                level: row[0],
                value: row[1],
                weight: row[2],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let fit = fit_isotonic(&points).map_err(stats_error)?;
    let rows = fit
        .raw
        .iter()
        .zip(&fit.fitted)
        .map(|(p, f)| {
            let values = vec![
                f64_value(cx, p.level)?,
                f64_value(cx, p.value)?,
                f64_value(cx, p.weight)?,
                f64_value(cx, *f)?,
            ];
            cx.factory().list(values)
        })
        .collect::<Result<Vec<_>>>()?;
    let point_values = cx.factory().list(rows)?;
    let area = match fit.normalized_area {
        Some(v) => f64_value(cx, v)?,
        None => cx.factory().nil()?,
    };
    cx.factory().table(vec![
        (Symbol::new("points"), point_values),
        (Symbol::new("normalized-area"), area),
    ])
}

fn one_matrix(cx: &mut Cx, values: &[Value], name: &str) -> Result<Vec<Vec<f64>>> {
    let [value] = values else {
        return Err(arity(name, 1));
    };
    value_to_points(cx, value, name)
}
fn exact_width<'a>(rows: &'a [Vec<f64>], width: usize, shape: &str) -> Result<&'a [f64]> {
    if rows.len() == 1 && rows[0].len() == width {
        Ok(&rows[0])
    } else {
        Err(Error::Eval(format!("expected {shape}")))
    }
}
fn exact_row<'a>(row: &'a [f64], width: usize, shape: &str) -> Result<&'a [f64]> {
    if row.len() == width {
        Ok(row)
    } else {
        Err(Error::Eval(format!("expected {shape}")))
    }
}
fn control(cx: &mut Cx, value: &Value) -> Result<BootstrapControl> {
    let rows = value_to_points(cx, value, "BootstrapControl")?;
    let row = exact_width(&rows, 4, "(seed resamples confidence work)")?;
    BootstrapControl::new(
        integer(row[0], "seed")?,
        usize_integer(row[1], "resamples")?,
        row[2],
        integer(row[3], "work")?,
    )
    .map_err(stats_error)
}
fn bootstrap_value(cx: &mut Cx, v: super::BootstrapEffectInterval) -> Result<Value> {
    let entries = vec![
        (Symbol::new("point-effect"), f64_value(cx, v.point_effect)?),
        (Symbol::new("lower"), f64_value(cx, v.lower)?),
        (Symbol::new("upper"), f64_value(cx, v.upper)?),
        (
            Symbol::new("confidence"),
            f64_value(cx, v.confidence_level)?,
        ),
        (Symbol::new("seed"), u64_value(cx, v.seed)?),
        (Symbol::new("resamples"), usize_value(cx, v.resamples)?),
        (Symbol::new("exclusions"), usize_value(cx, v.exclusions)?),
        (
            Symbol::new("cluster-count"),
            usize_value(cx, v.cluster_count)?,
        ),
        (
            Symbol::new("admitted-work"),
            u64_value(cx, v.admitted_work)?,
        ),
    ];
    cx.factory().table(entries)
}
fn integer(value: f64, name: &str) -> Result<u64> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= u64::MAX as f64 {
        Ok(value as u64)
    } else {
        Err(Error::Eval(format!(
            "{name} must be an exact nonnegative integer"
        )))
    }
}
fn usize_integer(value: f64, name: &str) -> Result<usize> {
    usize::try_from(integer(value, name)?).map_err(|_| Error::Eval(format!("{name} exceeds usize")))
}
fn arity(name: &str, count: usize) -> Error {
    Error::Eval(format!("{name} expects {count} argument(s)"))
}
fn stats_error(error: impl std::fmt::Display) -> Error {
    Error::Eval(error.to_string())
}
