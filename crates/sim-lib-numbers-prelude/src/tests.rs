use std::sync::Arc;

use sim_kernel::{Args, DefaultFactory, EagerPolicy, Error, Expr, Ref, Symbol, Value};
use sim_lib_numbers_func::Func;
use sim_lib_numbers_numeric::{
    integrate_symbol, numeric_compose_symbol, numeric_run_composed_symbol, ode_solve_symbol,
};
use sim_lib_numbers_stats::{
    FairnessClaimValue, StatsClaimValue, stats_claims_symbol, stats_disparate_impact_claim_symbol,
    stats_mean_claim_symbol,
};

use crate::NumbersPreludeLib;

#[test]
fn prelude_loads_everyday_numeric_names() {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();
    for symbol in [
        Symbol::new("+"),
        Symbol::new("fn"),
        Symbol::new("diff"),
        Symbol::new("integrate-sym"),
        Symbol::new("eval-cas"),
        Symbol::new("vec"),
        Symbol::new("numeric-diff"),
        Symbol::new("integrate"),
        Symbol::new("ode-solve"),
        Symbol::qualified("numeric", "compose"),
        Symbol::qualified("numeric", "run-composed"),
        Symbol::qualified("stats", "claims"),
        Symbol::qualified("stats", "mean-claim"),
        Symbol::new("matmul"),
    ] {
        assert!(
            cx.resolve_function(&symbol).is_ok(),
            "missing function {symbol}"
        );
    }
}

#[test]
fn prelude_emits_stats_claims_in_batch() {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();

    let mean_metric = symbol_value(&mut cx, Symbol::qualified("stats", "mean"));
    let mean_samples = f64_list(&mut cx, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mean_pair = list(&mut cx, vec![mean_metric, mean_samples]);

    let entropy_metric = symbol_value(&mut cx, Symbol::qualified("stats", "entropy"));
    let probabilities = f64_list(&mut cx, &[0.5, 0.25, 0.25]);
    let entropy_pair = list(&mut cx, vec![entropy_metric, probabilities]);

    let fairness_metric = symbol_value(&mut cx, Symbol::qualified("stats", "disparate-impact"));
    let reference_selected = u64_value(&mut cx, 80);
    let reference_total = u64_value(&mut cx, 100);
    let comparison_selected = u64_value(&mut cx, 48);
    let comparison_total = u64_value(&mut cx, 80);
    let fairness_pair = list(
        &mut cx,
        vec![
            fairness_metric,
            reference_selected,
            reference_total,
            comparison_selected,
            comparison_total,
        ],
    );

    let pairs = list(&mut cx, vec![mean_pair, entropy_pair, fairness_pair]);
    let claims = cx
        .call_function(&stats_claims_symbol(), Args::new(vec![pairs]))
        .unwrap();
    let claims = claims
        .object()
        .as_list()
        .unwrap()
        .to_vec(&mut cx, None)
        .unwrap();
    assert_eq!(claims.len(), 3);

    let mean = claims[0]
        .object()
        .downcast_ref::<StatsClaimValue>()
        .unwrap();
    assert_eq!(mean.claim().predicate, Symbol::new("stats-result"));
    close(claim_evidence_value(&mut cx, &claims[0]), 3.0);

    let entropy = claims[1]
        .object()
        .downcast_ref::<StatsClaimValue>()
        .unwrap();
    assert_eq!(entropy.claim().predicate, Symbol::new("stats-result"));
    close(claim_evidence_value(&mut cx, &claims[1]), 1.5);

    let fairness = claims[2]
        .object()
        .downcast_ref::<FairnessClaimValue>()
        .unwrap();
    assert_eq!(fairness.claim().predicate, Symbol::new("fairness-result"));
    close(claim_evidence_value(&mut cx, &claims[2]), 0.75);
}

#[test]
fn numeric6_full_smoke() {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();

    let decay = decay_func(&mut cx);
    let ode_compose_args = Args::new(vec![
        decay,
        symbol_value(&mut cx, Symbol::new(":domain")),
        symbol_value(&mut cx, Symbol::new("ode-solve")),
        symbol_value(&mut cx, Symbol::new(":method")),
        symbol_value(&mut cx, Symbol::new("rk4")),
        symbol_value(&mut cx, Symbol::new(":state")),
        symbol_value(&mut cx, Symbol::new("f64")),
    ]);
    let ode_pipeline = cx
        .call_function(&numeric_compose_symbol(), ode_compose_args)
        .unwrap();
    let ode_run_args = Args::new(vec![
        ode_pipeline,
        symbol_value(&mut cx, Symbol::new(":t0")),
        f64_value(&mut cx, 0.0),
        symbol_value(&mut cx, Symbol::new(":t1")),
        f64_value(&mut cx, 1.0),
        symbol_value(&mut cx, Symbol::new(":y0")),
        f64_value(&mut cx, 1.0),
        symbol_value(&mut cx, Symbol::new(":dt")),
        f64_value(&mut cx, 0.001),
    ]);
    let ode_result = cx
        .call_function(&numeric_run_composed_symbol(), ode_run_args)
        .unwrap();
    let y = table_field_f64(&mut cx, &ode_result, "value");
    assert!((y - std::f64::consts::E.recip()).abs() < 1.0e-3);

    let square = square_func(&mut cx);
    let quad_compose_args = Args::new(vec![
        square,
        symbol_value(&mut cx, Symbol::new(":domain")),
        symbol_value(&mut cx, Symbol::new("quadrature")),
        symbol_value(&mut cx, Symbol::new(":method")),
        symbol_value(&mut cx, Symbol::new("simpson")),
        symbol_value(&mut cx, Symbol::new(":state")),
        symbol_value(&mut cx, Symbol::new("f64")),
    ]);
    let quad_pipeline = cx
        .call_function(&numeric_compose_symbol(), quad_compose_args)
        .unwrap();
    let quad_run_args = Args::new(vec![
        quad_pipeline,
        symbol_value(&mut cx, Symbol::new(":a")),
        f64_value(&mut cx, 0.0),
        symbol_value(&mut cx, Symbol::new(":b")),
        f64_value(&mut cx, 1.0),
        symbol_value(&mut cx, Symbol::new(":n")),
        i64_value(&mut cx, 1000),
    ]);
    let quad_result = cx
        .call_function(&numeric_run_composed_symbol(), quad_run_args)
        .unwrap();
    let integral = table_field_f64(&mut cx, &quad_result, "value");
    assert!((integral - 1.0 / 3.0).abs() < 1.0e-5);

    let fairness_args = Args::new(vec![
        u64_value(&mut cx, 80),
        u64_value(&mut cx, 100),
        u64_value(&mut cx, 48),
        u64_value(&mut cx, 80),
    ]);
    let fairness_claim = cx
        .call_function(&stats_disparate_impact_claim_symbol(), fairness_args)
        .unwrap();
    let fairness = fairness_claim
        .object()
        .downcast_ref::<FairnessClaimValue>()
        .unwrap();
    assert_eq!(
        fairness.claim().subject,
        Ref::Symbol(Symbol::qualified("stats", "disparate-impact"))
    );
    assert_eq!(fairness.claim().predicate, Symbol::new("fairness-result"));
    let fairness_evidence = claim_evidence_table(&mut cx, &fairness_claim);
    close(table_field_f64(&mut cx, &fairness_evidence, "value"), 0.75);
    assert!(!table_field_bool(
        &mut cx,
        &fairness_evidence,
        "passes-four-fifths"
    ));

    let mean_samples = f64_list(&mut cx, &[2.0, 4.0, 6.0]);
    let mean_claim = cx
        .call_function(&stats_mean_claim_symbol(), Args::new(vec![mean_samples]))
        .unwrap();
    let mean = mean_claim
        .object()
        .downcast_ref::<StatsClaimValue>()
        .unwrap();
    assert_eq!(mean.claim().predicate, Symbol::new("stats-result"));
    close(claim_evidence_value(&mut cx, &mean_claim), 4.0);

    assert!(cx.resolve_function(&ode_solve_symbol()).is_ok());
    assert!(cx.resolve_function(&integrate_symbol()).is_ok());
}

#[test]
fn numeric6_rk_over_func_composed_pipeline() {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();
    let decay = decay_func(&mut cx);
    let t0 = f64_value(&mut cx, 0.0);
    let t1 = f64_value(&mut cx, 1.0);
    let y0 = f64_value(&mut cx, 1.0);
    let dt = f64_value(&mut cx, 0.01);

    let pipeline = cx
        .call_function(
            &numeric_compose_symbol(),
            Args::new(vec![
                decay.clone(),
                cx.factory().symbol(Symbol::new(":ode-solve")).unwrap(),
                cx.factory().symbol(Symbol::new("rk4")).unwrap(),
                cx.factory().symbol(Symbol::new(":f64")).unwrap(),
            ]),
        )
        .unwrap();

    let result = cx
        .call_function(
            &numeric_run_composed_symbol(),
            Args::new(vec![
                pipeline,
                cx.factory().symbol(Symbol::new(":t0")).unwrap(),
                t0.clone(),
                cx.factory().symbol(Symbol::new(":t1")).unwrap(),
                t1.clone(),
                cx.factory().symbol(Symbol::new(":y0")).unwrap(),
                y0.clone(),
                cx.factory().symbol(Symbol::new(":dt")).unwrap(),
                dt.clone(),
            ]),
        )
        .unwrap();

    let y = table_field_f64(&mut cx, &result, "value");
    assert!(
        (y - std::f64::consts::E.recip()).abs() < 1.0e-3,
        "rk4 composed result {y} not close to exp(-1)"
    );
    assert_eq!(table_field_symbol(&mut cx, &result, "method"), "rk4");
    assert_eq!(table_field_symbol(&mut cx, &result, "domain"), "ode-solve");
    assert_eq!(table_field_symbol(&mut cx, &result, "state-kind"), "f64");
    assert!(table_field_f64(&mut cx, &result, "steps") > 0.0);

    let flat = cx
        .call_function(
            &ode_solve_symbol(),
            Args::new(vec![
                decay,
                cx.factory().symbol(Symbol::new("t")).unwrap(),
                cx.factory().symbol(Symbol::new("y")).unwrap(),
                t0,
                y0,
                t1,
                cx.factory()
                    .table(vec![
                        (
                            Symbol::new(":method"),
                            cx.factory().symbol(Symbol::new("rk4")).unwrap(),
                        ),
                        (Symbol::new(":h"), dt),
                    ])
                    .unwrap(),
            ]),
        )
        .unwrap();
    let flat_y = final_ode_value(&mut cx, &flat);
    assert!((y - flat_y).abs() < 1.0e-12);
}

#[test]
fn numeric6_quad_over_func_composed_pipeline() {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();
    let square = square_func(&mut cx);
    let a = f64_value(&mut cx, 0.0);
    let b = f64_value(&mut cx, 1.0);
    let n = i64_value(&mut cx, 1000);

    let pipeline = cx
        .call_function(
            &numeric_compose_symbol(),
            Args::new(vec![
                square,
                cx.factory().symbol(Symbol::new(":domain")).unwrap(),
                cx.factory().symbol(Symbol::new("quadrature")).unwrap(),
                cx.factory().symbol(Symbol::new(":method")).unwrap(),
                cx.factory().symbol(Symbol::new("simpson")).unwrap(),
                cx.factory().symbol(Symbol::new(":state")).unwrap(),
                cx.factory().symbol(Symbol::new("f64")).unwrap(),
            ]),
        )
        .unwrap();

    let result = cx
        .call_function(
            &numeric_run_composed_symbol(),
            Args::new(vec![
                pipeline,
                cx.factory().symbol(Symbol::new(":a")).unwrap(),
                a,
                cx.factory().symbol(Symbol::new(":b")).unwrap(),
                b,
                cx.factory().symbol(Symbol::new(":n")).unwrap(),
                n,
            ]),
        )
        .unwrap();

    let integral = table_field_f64(&mut cx, &result, "value");
    assert!(
        (integral - 1.0 / 3.0).abs() < 1.0e-5,
        "simpson composed result {integral} not close to 1/3"
    );
    assert_eq!(table_field_symbol(&mut cx, &result, "method"), "simpson");
    assert_eq!(table_field_symbol(&mut cx, &result, "domain"), "quadrature");
    assert_eq!(table_field_symbol(&mut cx, &result, "state-kind"), "f64");
}

#[test]
fn numeric6_tensor_state_guard_errors_cleanly() {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();
    let square = square_func(&mut cx);
    let a = f64_value(&mut cx, 0.0);
    let b = f64_value(&mut cx, 1.0);
    let n = i64_value(&mut cx, 1000);

    let pipeline = cx
        .call_function(
            &numeric_compose_symbol(),
            Args::new(vec![
                square,
                cx.factory().symbol(Symbol::new(":domain")).unwrap(),
                cx.factory().symbol(Symbol::new("quadrature")).unwrap(),
                cx.factory().symbol(Symbol::new(":method")).unwrap(),
                cx.factory().symbol(Symbol::new("simpson")).unwrap(),
                cx.factory().symbol(Symbol::new(":state")).unwrap(),
                cx.factory().symbol(Symbol::new("tensor")).unwrap(),
            ]),
        )
        .unwrap();

    let err = cx
        .call_function(
            &numeric_run_composed_symbol(),
            Args::new(vec![
                pipeline,
                cx.factory().symbol(Symbol::new(":a")).unwrap(),
                a,
                cx.factory().symbol(Symbol::new(":b")).unwrap(),
                b,
                cx.factory().symbol(Symbol::new(":n")).unwrap(),
                n,
            ]),
        )
        .unwrap_err();
    assert!(
        matches!(err, Error::Eval(message) if message.contains("NotYetSupported") && message.contains("tensor state"))
    );
}

#[test]
fn prelude_is_idempotent() {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();
    assert!(
        cx.registry()
            .lib(&Symbol::qualified("numbers", "quad"))
            .is_some()
    );
}

#[test]
fn prelude_registers_typed_tensor_descriptors() {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    NumbersPreludeLib::new().install_all(&mut cx).unwrap();
    for symbol in [
        sim_lib_numbers_tensor_f64::tensor_spec_symbol(),
        sim_lib_numbers_tensor_i64::tensor_spec_symbol(),
        sim_lib_numbers_tensor_rat64::tensor_spec_symbol(),
        sim_lib_numbers_tensor_cmplxf::tensor_spec_symbol(),
        sim_lib_numbers_tensor_bit::tensor_spec_symbol(),
    ] {
        assert!(
            cx.registry().value_by_symbol(&symbol).is_some(),
            "missing typed tensor descriptor {symbol}"
        );
    }
}

fn decay_func(cx: &mut sim_kernel::Cx) -> Value {
    cx.factory()
        .opaque(Arc::new(Func::native(
            vec![Symbol::new("t"), Symbol::new("y")],
            Arc::new(|cx, args| {
                let [_, y] = args else {
                    return Err(Error::Eval("decay function expects t and y".to_owned()));
                };
                let zero = f64_value(cx, 0.0);
                cx.apply_value_number_binary_op(&Symbol::qualified("math", "sub"), zero, y.clone())
            }),
        )))
        .unwrap()
}

fn square_func(cx: &mut sim_kernel::Cx) -> Value {
    cx.factory()
        .opaque(Arc::new(Func::native(
            vec![Symbol::new("x")],
            Arc::new(|cx, args| {
                let [x] = args else {
                    return Err(Error::Eval("square function expects x".to_owned()));
                };
                cx.apply_value_number_binary_op(
                    &Symbol::qualified("math", "mul"),
                    x.clone(),
                    x.clone(),
                )
            }),
        )))
        .unwrap()
}

fn f64_value(cx: &mut sim_kernel::Cx, value: f64) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
        .unwrap()
}

fn u64_value(cx: &mut sim_kernel::Cx, value: u64) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "u64"), value.to_string())
        .unwrap()
}

fn i64_value(cx: &mut sim_kernel::Cx, value: i64) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "i64"), value.to_string())
        .unwrap()
}

fn f64_list(cx: &mut sim_kernel::Cx, values: &[f64]) -> Value {
    let values = values
        .iter()
        .copied()
        .map(|value| f64_value(cx, value))
        .collect::<Vec<_>>();
    list(cx, values)
}

fn symbol_value(cx: &mut sim_kernel::Cx, symbol: Symbol) -> Value {
    cx.factory().symbol(symbol).unwrap()
}

fn list(cx: &mut sim_kernel::Cx, values: Vec<Value>) -> Value {
    cx.factory().list(values).unwrap()
}

fn table_field(cx: &mut sim_kernel::Cx, value: &Value, key: &str) -> Value {
    value
        .object()
        .as_table_impl()
        .unwrap()
        .get(cx, Symbol::new(key))
        .unwrap()
}

fn table_field_f64(cx: &mut sim_kernel::Cx, value: &Value, key: &str) -> f64 {
    let field = table_field(cx, value, key);
    value_to_f64(cx, &field)
}

fn table_field_symbol(cx: &mut sim_kernel::Cx, value: &Value, key: &str) -> String {
    let expr = table_field(cx, value, key).object().as_expr(cx).unwrap();
    let sim_kernel::Expr::Symbol(symbol) = expr else {
        panic!("{key} should be a symbol");
    };
    symbol.to_string()
}

fn table_field_bool(cx: &mut sim_kernel::Cx, value: &Value, key: &str) -> bool {
    let expr = table_field(cx, value, key).object().as_expr(cx).unwrap();
    let Expr::Bool(flag) = expr else {
        panic!("{key} should be a bool");
    };
    flag
}

fn final_ode_value(cx: &mut sim_kernel::Cx, trajectory: &Value) -> f64 {
    let points = trajectory
        .object()
        .as_list()
        .unwrap()
        .to_vec(cx, None)
        .unwrap();
    let last = points.last().unwrap();
    let pair = last.object().as_list().unwrap().to_vec(cx, None).unwrap();
    value_to_f64(cx, &pair[1])
}

fn claim_evidence_value(cx: &mut sim_kernel::Cx, claim_value: &Value) -> f64 {
    let evidence = claim_evidence_table(cx, claim_value);
    let value = table_field(cx, &evidence, "value");
    value_to_f64(cx, &value)
}

fn claim_evidence_table(cx: &mut sim_kernel::Cx, claim_value: &Value) -> Value {
    let table = claim_value.object().as_table(cx).unwrap();
    table_field(cx, &table, "evidence")
}

fn close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1.0e-12,
        "expected {left} to be close to {right}"
    );
}

fn value_to_f64(cx: &mut sim_kernel::Cx, value: &Value) -> f64 {
    value.object().display(cx).unwrap().parse::<f64>().unwrap()
}
