use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use sim_kernel::{
    Args, CapabilityName, Consistency, Cx, DefaultFactory, EagerPolicy, Error, EvalFabric,
    EvalMode, EvalRequest, Expr, Factory, NumberLiteral, Symbol, read_construct_capability,
};
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorRequest, TensorSite, add_op_symbol, build_tensor_value,
    tensor_value_class_symbol, tensor_value_ref,
};

use crate::TensorBroadcastLib;

fn test_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_rational::RationalNumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_tensor::TensorNumbersLib::new())
        .unwrap();
    cx.load_lib(&TensorBroadcastLib::new()).unwrap();
    cx
}

fn number(canonical: &str) -> sim_kernel::Value {
    DefaultFactory
        .number_literal(Symbol::qualified("numbers", "i64"), canonical.to_owned())
        .unwrap()
}

fn i64_expr(canonical: &str) -> Expr {
    Expr::Number(NumberLiteral {
        domain: Symbol::qualified("numbers", "i64"),
        canonical: canonical.to_owned(),
    })
}

fn call(symbol: &str, args: Vec<Expr>) -> Expr {
    Expr::Call {
        operator: Box::new(Expr::Symbol(Symbol::new(symbol))),
        args,
    }
}

fn symbol(value: Symbol) -> sim_kernel::Value {
    DefaultFactory.symbol(value).unwrap()
}

fn shape_value(dims: &[&str]) -> sim_kernel::Value {
    DefaultFactory
        .list(
            dims.iter()
                .map(|dim| {
                    DefaultFactory
                        .number_literal(Symbol::qualified("citizen", "int"), (*dim).to_owned())
                        .unwrap()
                })
                .collect(),
        )
        .unwrap()
}

fn data_value(cells: Vec<sim_kernel::Value>) -> sim_kernel::Value {
    DefaultFactory.list(cells).unwrap()
}

// conformance: tensor broadcast element-wise execution routes through TensorExecutor.

fn tensor_cells(cx: &mut Cx, value: &sim_kernel::Value) -> Vec<String> {
    tensor_value_ref(value)
        .expect("tensor value")
        .cells()
        .unwrap()
        .iter()
        .map(|cell| match cell.object().as_expr(cx).unwrap() {
            Expr::Number(NumberLiteral { canonical, .. }) => canonical,
            other => panic!("expected number cell, found {other:?}"),
        })
        .collect()
}

fn eval_request(expr: Expr) -> EvalRequest {
    EvalRequest {
        expr,
        result_shape: None,
        required_capabilities: Vec::new(),
        deadline: None,
        consistency: Consistency::LocalFirst,
        mode: EvalMode::Eval,
        answer_limit: None,
        stream_buffer: None,
        stream: false,
        trace: false,
    }
}

#[derive(Clone)]
struct CountingExecutor {
    calls: Arc<AtomicUsize>,
}

impl CountingExecutor {
    fn new(calls: Arc<AtomicUsize>) -> Self {
        Self { calls }
    }
}

impl TensorExecutor for CountingExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("test", "counting-executor"),
            "counting",
            Symbol::qualified("core", "local-fabric"),
            vec![add_op_symbol()],
            None,
        )
    }

    fn execute(
        &self,
        cx: &mut Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request.operation.symbol, add_op_symbol());
        CpuTensorExecutor::new().execute(cx, request)
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        CpuTensorExecutor::new().flush()
    }
}

#[derive(Clone)]
struct DecliningExecutor;

impl TensorExecutor for DecliningExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("test", "declining-executor"),
            "declining",
            Symbol::qualified("test", "device"),
            vec![add_op_symbol()],
            Some(CapabilityName::new("test.device")),
        )
    }

    fn execute(
        &self,
        _cx: &mut Cx,
        _request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        Ok(TensorExecution::Unsupported {
            reason: Arc::from("device declines boxed mixed-number tensors"),
        })
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        Ok(SubmissionEvidence::new(self.card().symbol, 0))
    }
}

#[test]
fn scalar_plus_vector_broadcasts() {
    let mut cx = test_cx();
    let vector = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![number("1"), number("2"), number("3")]),
        )
        .unwrap();
    let out = cx
        .call_function(&Symbol::new("+"), Args::new(vec![number("1"), vector]))
        .unwrap();
    assert_eq!(
        out.object().as_expr(&mut cx).unwrap(),
        Expr::Vector(vec![
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "2".to_owned(),
            }),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "3".to_owned(),
            }),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "4".to_owned(),
            }),
        ])
    );
}

#[test]
fn matrix_plus_vector_broadcasts_trailing_axis() {
    let mut cx = test_cx();
    let rows = cx
        .factory()
        .list(vec![
            cx.factory().list(vec![number("1"), number("2")]).unwrap(),
            cx.factory().list(vec![number("3"), number("4")]).unwrap(),
        ])
        .unwrap();
    let matrix = cx
        .call_function(&Symbol::new("mat"), Args::new(vec![rows]))
        .unwrap();
    let vector = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![number("10"), number("20")]),
        )
        .unwrap();
    let out = cx
        .call_function(&Symbol::new("+"), Args::new(vec![matrix, vector]))
        .unwrap();
    assert_eq!(
        out.object().as_expr(&mut cx).unwrap(),
        Expr::Vector(vec![
            Expr::Vector(vec![
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "11".to_owned(),
                }),
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "22".to_owned(),
                }),
            ]),
            Expr::Vector(vec![
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "13".to_owned(),
                }),
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "24".to_owned(),
                }),
            ]),
        ])
    );
}

#[test]
fn rank_three_broadcast_matches_manual_cells() {
    let mut cx = test_cx();
    let left = build_tensor_value(
        &mut cx,
        vec![2, 1, 3],
        Some(Symbol::qualified("numbers", "i64")),
        ["1", "2", "3", "4", "5", "6"]
            .into_iter()
            .map(number)
            .collect(),
    )
    .unwrap();
    let right = build_tensor_value(
        &mut cx,
        vec![1, 4, 1],
        Some(Symbol::qualified("numbers", "i64")),
        ["10", "20", "30", "40"].into_iter().map(number).collect(),
    )
    .unwrap();

    let out = cx
        .call_function(&Symbol::new("+"), Args::new(vec![left, right]))
        .unwrap();
    let tensor = tensor_value_ref(&out).unwrap();

    assert_eq!(tensor.shape(), &[2, 4, 3]);
    assert_eq!(
        tensor_cells(&mut cx, &out),
        vec![
            "11", "12", "13", "21", "22", "23", "31", "32", "33", "41", "42", "43", "14", "15",
            "16", "24", "25", "26", "34", "35", "36", "44", "45", "46",
        ]
    );
}

#[test]
fn zero_extent_broadcast_preserves_empty_result() {
    let mut cx = test_cx();
    let empty = build_tensor_value(
        &mut cx,
        vec![0],
        Some(Symbol::qualified("numbers", "i64")),
        Vec::new(),
    )
    .unwrap();
    let unit = build_tensor_value(
        &mut cx,
        vec![1],
        Some(Symbol::qualified("numbers", "i64")),
        vec![number("9")],
    )
    .unwrap();

    let out = cx
        .call_function(&Symbol::new("+"), Args::new(vec![empty, unit]))
        .unwrap();
    let tensor = tensor_value_ref(&out).unwrap();

    assert_eq!(tensor.shape(), &[0]);
    assert!(tensor.cells().unwrap().is_empty());
}

#[test]
fn repeated_operand_alias_uses_stable_source_cells() {
    let mut cx = test_cx();
    let vector = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![number("2"), number("3"), number("4")]),
        )
        .unwrap();

    let out = cx
        .call_function(&Symbol::new("*"), Args::new(vec![vector.clone(), vector]))
        .unwrap();

    assert_eq!(tensor_cells(&mut cx, &out), vec!["4", "9", "16"]);
}

#[test]
fn incompatible_broadcast_shapes_error_before_execution() {
    let mut cx = test_cx();
    let left = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![number("1"), number("2")]),
        )
        .unwrap();
    let right = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![number("3"), number("4"), number("5")]),
        )
        .unwrap();

    let err = cx
        .call_function(&Symbol::new("+"), Args::new(vec![left, right]))
        .unwrap_err();

    assert!(err.to_string().contains("cannot broadcast"));
}

#[test]
fn active_env_executor_receives_elementwise_request() {
    let mut cx = test_cx();
    let calls = Arc::new(AtomicUsize::new(0));
    let site = TensorSite::new(
        Symbol::qualified("test", "tensor-counting-site"),
        Arc::new(CountingExecutor::new(calls.clone())),
        Vec::new(),
    );
    let reply = site
        .realize(
            &mut cx,
            eval_request(call(
                "+",
                vec![
                    i64_expr("1"),
                    call("vec", vec![i64_expr("2"), i64_expr("3")]),
                ],
            )),
        )
        .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(tensor_cells(&mut cx, &reply.value), vec!["3", "4"]);
}

#[test]
fn active_device_decline_returns_unsupported_error() {
    let mut cx = test_cx();
    cx.grant(CapabilityName::new("test.device"));
    let site = TensorSite::new(
        Symbol::qualified("test", "tensor-declining-site"),
        Arc::new(DecliningExecutor),
        vec![CapabilityName::new("test.device")],
    );

    let err = match site.realize(
        &mut cx,
        eval_request(call(
            "+",
            vec![
                call("vec", vec![i64_expr("1"), i64_expr("2")]),
                call("vec", vec![i64_expr("3"), i64_expr("4")]),
            ],
        )),
    ) {
        Ok(_) => panic!("declining executor must fail the realized expression"),
        Err(error) => error,
    };

    assert!(matches!(err, Error::Eval(_)));
    assert!(err.to_string().contains("device declines"));
}

#[test]
fn broadcast_ops_accept_tensor_citizen_values() {
    let mut cx = test_cx();
    cx.grant(read_construct_capability());
    let vector = cx
        .read_construct(
            &tensor_value_class_symbol(),
            vec![
                symbol(Symbol::new("v1")),
                shape_value(&["3"]),
                data_value(vec![number("1"), number("2"), number("3")]),
                symbol(Symbol::qualified("numbers", "i64")),
            ],
        )
        .unwrap();
    let out = cx
        .call_function(&Symbol::new("+"), Args::new(vec![number("1"), vector]))
        .unwrap();
    assert_eq!(
        out.object().as_expr(&mut cx).unwrap(),
        Expr::Vector(vec![
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "2".to_owned(),
            }),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "3".to_owned(),
            }),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "4".to_owned(),
            }),
        ])
    );
}

#[test]
fn broadcast_result_over_cell_ceiling_errors_instead_of_oom() {
    let mut cx = test_cx();
    let i64_domain = Symbol::qualified("numbers", "i64");
    // Two individually-legal operands (~1M cells each) whose broadcast is a
    // 1e12-cell shape: it fits in usize but must be rejected before the result
    // is materialized rather than driven into an out-of-memory abort.
    let cell = number("1");
    let column = build_tensor_value(
        &mut cx,
        vec![1_000_000, 1],
        Some(i64_domain.clone()),
        vec![cell.clone(); 1_000_000],
    )
    .unwrap();
    let row = build_tensor_value(
        &mut cx,
        vec![1, 1_000_000],
        Some(i64_domain),
        vec![cell; 1_000_000],
    )
    .unwrap();
    let err = cx
        .call_function(&Symbol::new("+"), Args::new(vec![column, row]))
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("cells") && message.contains("exceeding"),
        "expected a cell-ceiling diagnostic, got: {message}"
    );
}
