//! Runtime parsing and inspectable result tables for clustering operations.

use std::collections::BTreeMap;

use sim_kernel::{Cx, Error, Expr, NumberLiteral, Result, Symbol, Value, force_list_to_vec};
use sim_lib_numbers_core::domains;

use super::{
    ClusteringError, CovarianceType, GaussianCovariance, GmmControl, GmmReport, GmmSpec,
    GmmTermination, KMeansControl, KMeansReport, KMeansSearchTermination, KMeansTermination,
    SingularComponentPolicy, fit_gmm, fit_kmeans,
};

type Options = BTreeMap<String, Value>;

/// Parses and executes the expression-level `stats/kmeans` call.
pub fn call_kmeans_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    let Some((points, rest)) = args.split_first() else {
        return Err(Error::Eval(
            "stats/kmeans expects points plus :k and optional :control".to_owned(),
        ));
    };
    let points = cx.eval_expr(points.clone())?;
    let options = parse_expr_options(cx, "stats/kmeans", rest)?;
    execute_kmeans(cx, &points, &options)
}

/// Executes `stats/kmeans` from evaluated points and an optional options table.
pub fn call_kmeans_values(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let (points, options) = match values.as_slice() {
        [points, options] => (points, parse_table_options(cx, "stats/kmeans", options)?),
        _ => {
            return Err(Error::Eval(
                "stats/kmeans expects points and an options table".to_owned(),
            ));
        }
    };
    execute_kmeans(cx, points, &options)
}

/// Parses and executes the expression-level `stats/gmm` call.
pub fn call_gmm_exprs(cx: &mut Cx, args: Vec<Expr>) -> Result<Value> {
    let Some((points, rest)) = args.split_first() else {
        return Err(Error::Eval(
            "stats/gmm expects points plus :components and optional model policy".to_owned(),
        ));
    };
    let points = cx.eval_expr(points.clone())?;
    let options = parse_expr_options(cx, "stats/gmm", rest)?;
    execute_gmm(cx, &points, &options)
}

/// Executes `stats/gmm` from evaluated points and an optional options table.
pub fn call_gmm_values(cx: &mut Cx, values: Vec<Value>) -> Result<Value> {
    let (points, options) = match values.as_slice() {
        [points, options] => (points, parse_table_options(cx, "stats/gmm", options)?),
        _ => {
            return Err(Error::Eval(
                "stats/gmm expects points and an options table".to_owned(),
            ));
        }
    };
    execute_gmm(cx, points, &options)
}

fn execute_kmeans(cx: &mut Cx, points: &Value, options: &Options) -> Result<Value> {
    reject_unknown("stats/kmeans", options, &["k", "control"])?;
    let points = value_to_points(cx, points, "stats/kmeans points")?;
    let clusters = required_usize(cx, options, "k", "stats/kmeans")?;
    let control = match options.get("control") {
        Some(value) => kmeans_control(cx, value)?,
        None => KMeansControl::default(),
    };
    let report = fit_kmeans(&points, clusters, control).map_err(clustering_error)?;
    kmeans_report_value(cx, &report)
}

fn execute_gmm(cx: &mut Cx, points: &Value, options: &Options) -> Result<Value> {
    reject_unknown(
        "stats/gmm",
        options,
        &[
            "components",
            "covariance",
            "regularization",
            "singular-policy",
            "minimum-component-weight",
            "control",
        ],
    )?;
    let points = value_to_points(cx, points, "stats/gmm points")?;
    let components = required_usize(cx, options, "components", "stats/gmm")?;
    let covariance = match option_symbol(cx, options, "covariance")?.as_deref() {
        None | Some("diagonal") => CovarianceType::Diagonal,
        Some("full") => CovarianceType::Full,
        Some(name) => {
            return Err(Error::Eval(format!(
                "stats/gmm unsupported covariance {name}; expected diagonal or full"
            )));
        }
    };
    let regularization = option_f64(cx, options, "regularization")?.unwrap_or(1.0e-6);
    let minimum_weight = option_f64(cx, options, "minimum-component-weight")?.unwrap_or(1.0e-8);
    let singular_policy = match option_symbol(cx, options, "singular-policy")?.as_deref() {
        None | Some("reinitialize") => SingularComponentPolicy::Reinitialize { minimum_weight },
        Some("fail") => SingularComponentPolicy::Fail { minimum_weight },
        Some(name) => {
            return Err(Error::Eval(format!(
                "stats/gmm unsupported singular policy {name}; expected reinitialize or fail"
            )));
        }
    };
    let spec = GmmSpec::new(components, covariance, regularization, singular_policy)
        .map_err(clustering_error)?;
    let control = match options.get("control") {
        Some(value) => gmm_control(cx, value)?,
        None => GmmControl::default(),
    };
    let report = fit_gmm(&points, spec, control).map_err(clustering_error)?;
    gmm_report_value(cx, &report, covariance, regularization)
}

fn kmeans_control(cx: &mut Cx, value: &Value) -> Result<KMeansControl> {
    let options = parse_table_options(cx, "stats/kmeans :control", value)?;
    reject_unknown(
        "stats/kmeans :control",
        &options,
        &["seed", "iterations", "tolerance", "work", "results"],
    )?;
    let defaults = KMeansControl::default();
    KMeansControl::new(
        option_u64(cx, &options, "seed")?.unwrap_or(defaults.seed),
        option_usize(cx, &options, "iterations")?.unwrap_or(defaults.max_iterations),
        option_f64(cx, &options, "tolerance")?.unwrap_or(defaults.tolerance),
        option_u64(cx, &options, "work")?.unwrap_or(defaults.max_work),
        option_usize(cx, &options, "results")?.unwrap_or(defaults.restarts),
    )
    .map_err(clustering_error)
}

fn gmm_control(cx: &mut Cx, value: &Value) -> Result<GmmControl> {
    let options = parse_table_options(cx, "stats/gmm :control", value)?;
    reject_unknown(
        "stats/gmm :control",
        &options,
        &["seed", "iterations", "tolerance", "work"],
    )?;
    let defaults = GmmControl::default();
    GmmControl::new(
        option_u64(cx, &options, "seed")?.unwrap_or(defaults.seed),
        option_usize(cx, &options, "iterations")?.unwrap_or(defaults.max_iterations),
        option_f64(cx, &options, "tolerance")?.unwrap_or(defaults.tolerance),
        option_u64(cx, &options, "work")?.unwrap_or(defaults.max_work),
    )
    .map_err(clustering_error)
}

fn kmeans_report_value(cx: &mut Cx, report: &KMeansReport) -> Result<Value> {
    let centroids = point_values(cx, &report.model.centroids)?;
    let assignments = usize_values(cx, &report.model.assignments)?;
    let model = cx.factory().table(vec![
        (Symbol::new("centroids"), centroids),
        (Symbol::new("assignments"), assignments),
    ])?;
    let restart_values = report
        .restarts
        .iter()
        .map(|evidence| {
            let entries = vec![
                (Symbol::new("restart"), usize_value(cx, evidence.restart)?),
                (Symbol::new("seed"), u64_value(cx, evidence.seed)?),
                (Symbol::new("inertia"), f64_value(cx, evidence.inertia)?),
                (
                    Symbol::new("iterations"),
                    usize_value(cx, evidence.iterations)?,
                ),
                (
                    Symbol::new("converged"),
                    cx.factory().bool(evidence.converged)?,
                ),
                (
                    Symbol::new("empty-cluster-repairs"),
                    u64_value(cx, evidence.empty_cluster_repairs)?,
                ),
                (Symbol::new("work"), u64_value(cx, evidence.work)?),
                (
                    Symbol::new("termination"),
                    symbol_value(cx, kmeans_termination(evidence.termination))?,
                ),
            ];
            cx.factory().table(entries)
        })
        .collect::<Result<Vec<_>>>()?;
    let restart_values = cx.factory().list(restart_values)?;
    let selected = &report.restarts[report.selected_restart];
    let evidence_entries = vec![
        (
            Symbol::new("selected-restart"),
            usize_value(cx, report.selected_restart)?,
        ),
        (Symbol::new("inertia"), f64_value(cx, selected.inertia)?),
        (
            Symbol::new("converged"),
            cx.factory().bool(selected.converged)?,
        ),
        (
            Symbol::new("requested-restarts"),
            usize_value(cx, report.requested_restarts)?,
        ),
        (
            Symbol::new("completed-restarts"),
            usize_value(cx, report.restarts.len())?,
        ),
        (Symbol::new("work"), u64_value(cx, report.work)?),
        (
            Symbol::new("termination"),
            symbol_value(cx, kmeans_search_termination(report.termination))?,
        ),
        (Symbol::new("restarts"), restart_values),
    ];
    let evidence = cx.factory().table(evidence_entries)?;
    cx.factory().table(vec![
        (Symbol::new("model"), model),
        (Symbol::new("evidence"), evidence),
    ])
}

fn gmm_report_value(
    cx: &mut Cx,
    report: &GmmReport,
    covariance_type: CovarianceType,
    regularization: f64,
) -> Result<Value> {
    let weights = f64_values(cx, &report.model.weights)?;
    let means = point_values(cx, &report.model.means)?;
    let covariances = covariance_values(cx, &report.model.covariances)?;
    let model_entries = vec![
        (Symbol::new("weights"), weights),
        (Symbol::new("means"), means),
        (
            Symbol::new("covariance"),
            symbol_value(cx, covariance_name(covariance_type))?,
        ),
        (Symbol::new("covariances"), covariances),
        (
            Symbol::new("regularization"),
            f64_value(cx, regularization)?,
        ),
    ];
    let model = cx.factory().table(model_entries)?;
    let selection = report.evidence.model_selection;
    let selection_entries = vec![
        (
            Symbol::new("log-likelihood"),
            f64_value(cx, selection.log_likelihood)?,
        ),
        (
            Symbol::new("parameters"),
            usize_value(cx, selection.parameters)?,
        ),
        (Symbol::new("aic"), f64_value(cx, selection.aic)?),
        (Symbol::new("bic"), f64_value(cx, selection.bic)?),
        (
            Symbol::new("observations"),
            usize_value(cx, selection.observations)?,
        ),
    ];
    let selection = cx.factory().table(selection_entries)?;
    let likelihood_history = f64_values(cx, &report.evidence.likelihood_history)?;
    let evidence_entries = vec![
        (
            Symbol::new("initial-log-likelihood"),
            f64_value(cx, report.evidence.initial_log_likelihood)?,
        ),
        (
            Symbol::new("log-likelihood"),
            f64_value(cx, report.evidence.log_likelihood)?,
        ),
        (Symbol::new("likelihood-history"), likelihood_history),
        (
            Symbol::new("iterations"),
            usize_value(cx, report.evidence.iterations)?,
        ),
        (
            Symbol::new("converged"),
            cx.factory().bool(report.evidence.converged)?,
        ),
        (
            Symbol::new("singular-component-repairs"),
            u64_value(cx, report.evidence.singular_component_repairs)?,
        ),
        (Symbol::new("seed"), u64_value(cx, report.evidence.seed)?),
        (Symbol::new("work"), u64_value(cx, report.evidence.work)?),
        (
            Symbol::new("termination"),
            symbol_value(cx, gmm_termination(report.evidence.termination))?,
        ),
        (Symbol::new("model-selection"), selection),
    ];
    let evidence = cx.factory().table(evidence_entries)?;
    cx.factory().table(vec![
        (Symbol::new("model"), model),
        (Symbol::new("evidence"), evidence),
    ])
}

fn covariance_values(cx: &mut Cx, covariances: &[GaussianCovariance]) -> Result<Value> {
    let values = covariances
        .iter()
        .map(|covariance| match covariance {
            GaussianCovariance::Diagonal(values) => f64_values(cx, values),
            GaussianCovariance::Full(rows) => point_values(cx, rows),
        })
        .collect::<Result<Vec<_>>>()?;
    cx.factory().list(values)
}

fn point_values(cx: &mut Cx, points: &[Vec<f64>]) -> Result<Value> {
    let values = points
        .iter()
        .map(|point| f64_values(cx, point))
        .collect::<Result<Vec<_>>>()?;
    cx.factory().list(values)
}

fn f64_values(cx: &mut Cx, values: &[f64]) -> Result<Value> {
    let values = values
        .iter()
        .map(|&value| f64_value(cx, value))
        .collect::<Result<Vec<_>>>()?;
    cx.factory().list(values)
}

fn usize_values(cx: &mut Cx, values: &[usize]) -> Result<Value> {
    let values = values
        .iter()
        .map(|&value| usize_value(cx, value))
        .collect::<Result<Vec<_>>>()?;
    cx.factory().list(values)
}

fn parse_expr_options(cx: &mut Cx, name: &str, exprs: &[Expr]) -> Result<Options> {
    if !exprs.len().is_multiple_of(2) {
        return Err(Error::Eval(format!(
            "{name} options must be keyword/value pairs"
        )));
    }
    let mut options = Options::new();
    for pair in exprs.chunks(2) {
        let key = keyword(&pair[0], true)?;
        let value = cx.eval_expr(pair[1].clone())?;
        insert_option(&mut options, name, key, value)?;
    }
    Ok(options)
}

fn parse_table_options(cx: &mut Cx, name: &str, value: &Value) -> Result<Options> {
    let Expr::Map(entries) = value.object().as_expr(cx)? else {
        return Err(Error::Eval(format!("{name} options must be a table")));
    };
    let mut options = Options::new();
    for (key, value) in entries {
        let key = keyword(&key, false)?;
        let value = cx.eval_expr(value)?;
        insert_option(&mut options, name, key, value)?;
    }
    Ok(options)
}

fn keyword(expr: &Expr, require_colon: bool) -> Result<String> {
    let Expr::Symbol(symbol) = expr else {
        return Err(Error::Eval("expected keyword option".to_owned()));
    };
    if require_colon && !symbol.name.starts_with(':') {
        return Err(Error::Eval(format!(
            "expected keyword option, found {symbol}"
        )));
    }
    let key = symbol.name.strip_prefix(':').unwrap_or(&symbol.name);
    if key.is_empty() {
        return Err(Error::Eval("keyword option must not be empty".to_owned()));
    }
    Ok(key.to_owned())
}

fn insert_option(options: &mut Options, name: &str, key: String, value: Value) -> Result<()> {
    if options.insert(key.clone(), value).is_some() {
        return Err(Error::Eval(format!("{name}: duplicate option :{key}")));
    }
    Ok(())
}

fn reject_unknown(name: &str, options: &Options, allowed: &[&str]) -> Result<()> {
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::Eval(format!("{name}: unknown option :{key}")));
        }
    }
    Ok(())
}

fn required_usize(cx: &mut Cx, options: &Options, key: &str, name: &str) -> Result<usize> {
    option_usize(cx, options, key)?.ok_or_else(|| Error::Eval(format!("{name} requires :{key}")))
}

fn option_usize(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<usize>> {
    options
        .get(key)
        .map(|value| value_to_literal(cx, value, key).and_then(|value| literal_usize(value, key)))
        .transpose()
}

fn option_u64(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<u64>> {
    options
        .get(key)
        .map(|value| value_to_literal(cx, value, key).and_then(|value| literal_u64(value, key)))
        .transpose()
}

fn option_f64(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<f64>> {
    options
        .get(key)
        .map(|value| value_to_f64(cx, value, key))
        .transpose()
}

fn option_symbol(cx: &mut Cx, options: &Options, key: &str) -> Result<Option<String>> {
    options
        .get(key)
        .map(|value| {
            let Expr::Symbol(symbol) = value.object().as_expr(cx)? else {
                return Err(Error::Eval(format!("expected symbol option :{key}")));
            };
            Ok(symbol.name.to_string())
        })
        .transpose()
}

fn value_to_points(cx: &mut Cx, value: &Value, name: &str) -> Result<Vec<Vec<f64>>> {
    value_to_list(cx, value, name)?
        .iter()
        .enumerate()
        .map(|(point, value)| {
            value_to_list(cx, value, &format!("{name}[{point}]"))?
                .iter()
                .enumerate()
                .map(|(coordinate, value)| {
                    value_to_f64(cx, value, &format!("{name}[{point}][{coordinate}]"))
                })
                .collect()
        })
        .collect()
}

fn value_to_list(cx: &mut Cx, value: &Value, name: &str) -> Result<Vec<Value>> {
    let list = value.object().as_list().ok_or(Error::TypeMismatch {
        expected: "list",
        found: "non-list",
    })?;
    force_list_to_vec(cx, list, name)
}

fn value_to_literal(cx: &mut Cx, value: &Value, name: &str) -> Result<NumberLiteral> {
    value
        .object()
        .as_number_value()
        .ok_or(Error::TypeMismatch {
            expected: "number",
            found: "non-number",
        })?
        .number_literal(cx)?
        .ok_or_else(|| Error::Eval(format!("{name} must have a canonical numeric literal")))
}

fn value_to_f64(cx: &mut Cx, value: &Value, name: &str) -> Result<f64> {
    let value = value_to_literal(cx, value, name)?
        .canonical
        .parse::<f64>()
        .map_err(|_| Error::Eval(format!("{name} must be a finite number")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::Eval(format!("{name} must be a finite number")))
    }
}

fn literal_usize(literal: NumberLiteral, name: &str) -> Result<usize> {
    literal
        .canonical
        .parse::<usize>()
        .map_err(|_| Error::Eval(format!("{name} must be a non-negative integer")))
}

fn literal_u64(literal: NumberLiteral, name: &str) -> Result<u64> {
    literal
        .canonical
        .parse::<u64>()
        .map_err(|_| Error::Eval(format!("{name} must be a non-negative integer")))
}

fn f64_value(cx: &mut Cx, value: f64) -> Result<Value> {
    cx.factory()
        .number_literal(domains::f64(), canonical_f64(value))
}

fn usize_value(cx: &mut Cx, value: usize) -> Result<Value> {
    cx.factory()
        .number_literal(domains::u64(), value.to_string())
}

fn u64_value(cx: &mut Cx, value: u64) -> Result<Value> {
    cx.factory()
        .number_literal(domains::u64(), value.to_string())
}

fn symbol_value(cx: &mut Cx, name: &str) -> Result<Value> {
    cx.factory().symbol(Symbol::new(name))
}

fn canonical_f64(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        value.to_string()
    }
}

fn clustering_error(error: ClusteringError) -> Error {
    Error::Eval(error.to_string())
}

fn kmeans_termination(termination: KMeansTermination) -> &'static str {
    match termination {
        KMeansTermination::Converged => "converged",
        KMeansTermination::IterationLimit => "iteration-limit",
        KMeansTermination::WorkLimit => "work-limit",
    }
}

fn kmeans_search_termination(termination: KMeansSearchTermination) -> &'static str {
    match termination {
        KMeansSearchTermination::Completed => "completed",
        KMeansSearchTermination::WorkLimit => "work-limit",
    }
}

fn gmm_termination(termination: GmmTermination) -> &'static str {
    match termination {
        GmmTermination::Converged => "converged",
        GmmTermination::IterationLimit => "iteration-limit",
        GmmTermination::WorkLimit => "work-limit",
        GmmTermination::LikelihoodDecrease => "likelihood-decrease",
    }
}

fn covariance_name(covariance: CovarianceType) -> &'static str {
    match covariance {
        CovarianceType::Diagonal => "diagonal",
        CovarianceType::Full => "full",
    }
}
