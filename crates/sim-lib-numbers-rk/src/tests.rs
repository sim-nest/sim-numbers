use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, CapabilityName, ClassRef, Cx, DefaultFactory, EagerPolicy, Error, EvalFabric,
    EvalMode, EvalRequest, Expr, Factory, Object, Symbol, Value,
};
use sim_lib_numbers_cas::CasExpr;
use sim_lib_numbers_func::Func;
use sim_lib_numbers_numeric::NumericNumbersLib;
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorRequest, TensorSite, build_tensor_value, tensor_value_ref,
};

use crate::RkNumbersLib;

fn test_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_tensor::TensorNumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_tensor_bcast::TensorBroadcastLib::new())
        .unwrap();
    cx.load_lib(&NumericNumbersLib::new()).unwrap();
    cx.load_lib(&RkNumbersLib::new()).unwrap();
    cx
}

fn f64_value(cx: &mut Cx, value: f64) -> Value {
    cx.factory()
        .number_literal(Symbol::qualified("numbers", "f64"), value.to_string())
        .unwrap()
}

fn value_to_f64(cx: &mut Cx, value: &Value) -> f64 {
    value.object().display(cx).unwrap().parse::<f64>().unwrap()
}

fn tensor_value(cx: &mut Cx, values: &[f64]) -> Value {
    let cells = values
        .iter()
        .map(|value| f64_value(cx, *value))
        .collect::<Vec<_>>();
    build_tensor_value(
        cx,
        vec![values.len()],
        Some(Symbol::qualified("numbers", "f64")),
        cells,
    )
    .unwrap()
}

fn tensor_f64s(tensor_value: &Value, cx: &mut Cx) -> Vec<f64> {
    tensor_value_ref(tensor_value)
        .expect("tensor value")
        .cells()
        .unwrap()
        .iter()
        .map(|cell| value_to_f64(cx, cell))
        .collect()
}

#[derive(Clone)]
struct PlainBinary {
    name: &'static str,
    f: fn(f64, f64) -> f64,
}

impl Object for PlainBinary {
    fn display(&self, _cx: &mut Cx) -> sim_kernel::Result<String> {
        Ok(format!("#<plain-callable {}>", self.name))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for PlainBinary {
    fn class(&self, _cx: &mut Cx) -> sim_kernel::Result<ClassRef> {
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }

    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for PlainBinary {
    fn call(&self, cx: &mut Cx, args: Args) -> sim_kernel::Result<Value> {
        let values = args.into_vec();
        let [x, y] = values.as_slice() else {
            return Err(Error::Eval("plain binary expected two args".to_owned()));
        };
        let x = value_to_f64(cx, x);
        let y = value_to_f64(cx, y);
        Ok(f64_value(cx, (self.f)(x, y)))
    }
}

fn plain_binary(cx: &mut Cx, name: &'static str, f: fn(f64, f64) -> f64) -> Value {
    cx.factory()
        .opaque(Arc::new(PlainBinary { name, f }))
        .unwrap()
}

fn tensor_identity_rhs() -> Func {
    Func::native(
        vec![Symbol::new("x"), Symbol::new("y")],
        Arc::new(|_cx, args| {
            let [_, y] = args else {
                return Err(Error::Eval("expected two args".to_owned()));
            };
            Ok(y.clone())
        }),
    )
}

fn ode_rhs() -> Func {
    Func::native(
        vec![Symbol::new("x"), Symbol::new("y")],
        Arc::new(|_cx, args| {
            let [_, y] = args else {
                return Err(Error::Eval("expected two args".to_owned()));
            };
            Ok(y.clone())
        }),
    )
}

#[test]
fn ode_accepts_plain_binary_callable() {
    let mut cx = test_cx();
    let rhs = plain_binary(&mut cx, "plain-exp-rhs", |_x, y| y);
    let method = cx.factory().symbol(Symbol::new("rk4")).unwrap();
    let h = f64_value(&mut cx, 0.01);
    let options = cx
        .factory()
        .table(vec![
            (Symbol::new(":method"), method),
            (Symbol::new(":h"), h),
        ])
        .unwrap();
    let x0 = f64_value(&mut cx, 0.0);
    let y0 = f64_value(&mut cx, 1.0);
    let x_end = f64_value(&mut cx, 1.0);

    let out = cx
        .call_function(
            &Symbol::new("ode-solve"),
            Args::new(vec![
                rhs,
                cx.factory().symbol(Symbol::new("x")).unwrap(),
                cx.factory().symbol(Symbol::new("y")).unwrap(),
                x0,
                y0,
                x_end,
                options,
            ]),
        )
        .unwrap();

    let expr = out.object().as_expr(&mut cx).unwrap();
    let last_y = match expr {
        sim_kernel::Expr::List(points) => match points.last().unwrap() {
            sim_kernel::Expr::List(pair) => match &pair[1] {
                sim_kernel::Expr::Number(number) => number.canonical.parse::<f64>().unwrap(),
                other => cx
                    .eval_expr(other.clone())
                    .map(|value| value_to_f64(&mut cx, &value))
                    .unwrap(),
            },
            _ => panic!("expected pair"),
        },
        _ => panic!("expected list"),
    };
    assert!((last_y - std::f64::consts::E).abs() < 1.0e-8);
}

#[test]
fn ode_methods_reach_e_within_tolerance() {
    let mut cx = test_cx();
    let rhs = cx.factory().opaque(Arc::new(ode_rhs())).unwrap();
    for (method, h, tol) in [
        ("forward-euler", 0.01, 2.0e-2),
        ("backward-euler", 0.01, 2.0e-2),
        ("midpoint", 0.01, 3.0e-4),
        ("rk4", 0.01, 1.0e-8),
        ("rkf45", 0.1, 1.0e-6),
    ] {
        let mut entries = vec![
            (
                Symbol::new(":method"),
                cx.factory().symbol(Symbol::new(method)).unwrap(),
            ),
            (Symbol::new(":h"), f64_value(&mut cx, h)),
        ];
        if method == "rkf45" {
            entries.push((Symbol::new(":tol"), f64_value(&mut cx, 1.0e-8)));
        }
        let options = cx.factory().table(entries).unwrap();
        let x0 = f64_value(&mut cx, 0.0);
        let y0 = f64_value(&mut cx, 1.0);
        let x_end = f64_value(&mut cx, 1.0);
        let out = cx
            .call_function(
                &Symbol::new("ode-solve"),
                Args::new(vec![
                    rhs.clone(),
                    cx.factory().symbol(Symbol::new("x")).unwrap(),
                    cx.factory().symbol(Symbol::new("y")).unwrap(),
                    x0,
                    y0,
                    x_end,
                    options,
                ]),
            )
            .unwrap();
        let expr = out.object().as_expr(&mut cx).unwrap();
        let last_y = match expr {
            sim_kernel::Expr::List(points) => match points.last().unwrap() {
                sim_kernel::Expr::List(pair) => match &pair[1] {
                    sim_kernel::Expr::Number(number) => number.canonical.parse::<f64>().unwrap(),
                    other => cx
                        .eval_expr(other.clone())
                        .map(|value| value_to_f64(&mut cx, &value))
                        .unwrap(),
                },
                _ => panic!("expected pair"),
            },
            _ => panic!("expected list"),
        };
        assert!(
            (last_y - std::f64::consts::E).abs() < tol,
            "{method} -> {last_y}"
        );
    }
}

#[test]
// conformance: tensor ODE pipelines run through every registered RK solver.
fn tensor_ode_pipeline_runs_all_rk_methods_and_matches_scalar_cpu() {
    let mut cx = test_cx();
    let rhs = cx
        .factory()
        .opaque(Arc::new(tensor_identity_rhs()))
        .unwrap();
    let y0 = tensor_value(&mut cx, &[1.0, 2.0, -0.5]);

    for (method, h, tol) in [
        ("forward-euler", 0.01, 2.0e-2),
        ("backward-euler", 0.01, 2.0e-2),
        ("midpoint", 0.01, 3.0e-4),
        ("rk4", 0.01, 1.0e-8),
        ("rkf45", 0.1, 1.0e-6),
    ] {
        let pipeline = cx
            .call_function(
                &Symbol::qualified("numeric", "compose"),
                Args::new(vec![
                    rhs.clone(),
                    cx.factory().symbol(Symbol::new("ode-solve")).unwrap(),
                    cx.factory().symbol(Symbol::new(method)).unwrap(),
                    cx.factory().symbol(Symbol::new("tensor")).unwrap(),
                ]),
            )
            .unwrap();
        let t0 = f64_value(&mut cx, 0.0);
        let t1 = f64_value(&mut cx, 1.0);
        let dt = f64_value(&mut cx, h);
        let out = cx
            .call_function(
                &Symbol::qualified("numeric", "run-composed"),
                Args::new(vec![pipeline, t0, t1, y0.clone(), dt]),
            )
            .unwrap();
        let value = out
            .object()
            .as_table_impl()
            .unwrap()
            .get(&mut cx, Symbol::new("value"))
            .unwrap();
        for (actual, initial) in tensor_f64s(&value, &mut cx)
            .into_iter()
            .zip([1.0, 2.0, -0.5])
        {
            let expected = initial * std::f64::consts::E;
            assert!(
                (actual - expected).abs() < tol * expected.abs().max(1.0),
                "{method}"
            );
        }
    }
}

#[derive(Clone)]
struct DeviceExecutor;

impl TensorExecutor for DeviceExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("test", "executor/device"),
            "test-device",
            Symbol::qualified("test", "device"),
            CpuTensorExecutor::new().card().operations.to_vec(),
            Some(CapabilityName::new("test.device")),
        )
    }

    fn execute(
        &self,
        cx: &mut Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        CpuTensorExecutor::new().execute(cx, request)
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        Ok(SubmissionEvidence::new(self.card().symbol, 0))
    }
}

fn number_expr(value: &str) -> Expr {
    Expr::Number(sim_kernel::NumberLiteral {
        domain: Symbol::qualified("numbers", "f64"),
        canonical: value.to_owned(),
    })
}

fn symbol_expr(symbol: impl Into<Symbol>) -> Expr {
    Expr::Symbol(symbol.into())
}

fn call_expr(symbol: impl Into<Symbol>, args: Vec<Expr>) -> Expr {
    Expr::Call {
        operator: Box::new(symbol_expr(symbol)),
        args,
    }
}

fn tensor_pipeline_expr(rhs: Expr) -> Expr {
    call_expr(
        Symbol::qualified("numeric", "run-composed"),
        vec![
            call_expr(
                Symbol::qualified("numeric", "compose"),
                vec![
                    rhs,
                    symbol_expr(Symbol::new("ode-solve")),
                    symbol_expr(Symbol::new("rk4")),
                    symbol_expr(Symbol::new("tensor")),
                ],
            ),
            number_expr("0.0"),
            number_expr("0.1"),
            call_expr(
                Symbol::new("vec"),
                vec![number_expr("1.0"), number_expr("2.0")],
            ),
            number_expr("0.1"),
        ],
    )
}

fn eval_request(expr: Expr) -> EvalRequest {
    EvalRequest {
        expr,
        result_shape: None,
        required_capabilities: Vec::new(),
        deadline: None,
        consistency: sim_kernel::Consistency::LocalFirst,
        mode: EvalMode::Eval,
        answer_limit: None,
        stream_buffer: None,
        stream: false,
        trace: false,
    }
}

#[test]
fn tensor_ode_hardware_refuses_arbitrary_callable_rhs() {
    let mut cx = test_cx();
    let rhs = plain_binary(&mut cx, "native-tensor-rhs", |_x, y| y);
    cx.env_mut().define(Symbol::new("native-rhs"), rhs);
    let site = TensorSite::new(
        Symbol::qualified("test", "site/device"),
        Arc::new(DeviceExecutor),
        Vec::new(),
    );

    let err = match site.realize(
        &mut cx,
        eval_request(tensor_pipeline_expr(symbol_expr(Symbol::new("native-rhs")))),
    ) {
        Ok(_) => panic!("device tensor ODE must reject native RHS"),
        Err(error) => error,
    };
    assert!(err.to_string().contains("pure symbolic Func RHS"), "{err}");
}

#[test]
fn tensor_ode_hardware_refuses_non_tensor_rhs() {
    let mut cx = test_cx();
    let rhs = cx
        .factory()
        .opaque(Arc::new(Func::symbolic(
            vec![Symbol::new("x"), Symbol::new("y")],
            CasExpr::Var(Symbol::new("x")),
        )))
        .unwrap();
    cx.env_mut().define(Symbol::new("scalar-rhs"), rhs);
    let site = TensorSite::new(
        Symbol::qualified("test", "site/device"),
        Arc::new(DeviceExecutor),
        Vec::new(),
    );

    let err = match site.realize(
        &mut cx,
        eval_request(tensor_pipeline_expr(symbol_expr(Symbol::new("scalar-rhs")))),
    ) {
        Ok(_) => panic!("device tensor ODE must reject scalar RHS"),
        Err(error) => error,
    };
    assert!(
        err.to_string().contains("RHS must return a tensor"),
        "{err}"
    );
}
