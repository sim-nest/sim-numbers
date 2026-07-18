use super::*;

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
