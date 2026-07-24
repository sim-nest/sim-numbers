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

mod numeric_pipeline;

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
        Symbol::qualified("tensor", "cast"),
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
        sim_lib_numbers_tensor_f32::tensor_spec_symbol(),
        sim_lib_numbers_tensor_f64::tensor_spec_symbol(),
        sim_lib_numbers_tensor_half::f16_tensor_spec_symbol(),
        sim_lib_numbers_tensor_half::bf16_tensor_spec_symbol(),
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
