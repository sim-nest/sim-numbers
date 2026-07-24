use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use sim_kernel::{
    Args, Consistency, Cx, DefaultFactory, EagerPolicy, EvalFabric, EvalMode, EvalRequest, Expr,
    Factory, NumberLiteral, Symbol, read_construct_capability,
};
use sim_lib_numbers_arith::NumbersArithmeticLib;
use sim_lib_numbers_cas::CasNumbersLib;
use sim_lib_numbers_i64::I64NumbersLib;
use sim_lib_numbers_tensor::{
    CpuTensorExecutor, SubmissionEvidence, TensorExecError, TensorExecution, TensorExecutor,
    TensorExecutorCard, TensorNumbersLib, TensorRequest, TensorSite, matmul_exec_op_symbol,
    tensor_value_class_symbol, tensor_value_ref,
};
use sim_lib_numbers_tensor_bcast::TensorBroadcastLib;

use crate::TensorLinalgLib;

// conformance: tensor linalg executor routing covers reductions and matrix math.

fn cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&TensorNumbersLib::new()).unwrap();
    cx.load_lib(&TensorBroadcastLib::new()).unwrap();
    cx.load_lib(&NumbersArithmeticLib::new()).unwrap();
    cx.load_lib(&I64NumbersLib::new()).unwrap();
    cx.load_lib(&CasNumbersLib::new()).unwrap();
    cx.load_lib(&TensorLinalgLib::new()).unwrap();
    cx
}

fn i64_num(text: &str) -> sim_kernel::Value {
    DefaultFactory
        .number_literal(Symbol::qualified("numbers", "i64"), text.to_owned())
        .unwrap()
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

fn cas_var(cx: &mut Cx, symbol: &str) -> sim_kernel::Value {
    cx.call_function(
        &Symbol::qualified("cas", "var"),
        Args::new(vec![DefaultFactory.symbol(Symbol::new(symbol)).unwrap()]),
    )
    .unwrap()
}

#[derive(Clone)]
struct CountingExecutor {
    calls: Arc<AtomicUsize>,
}

impl TensorExecutor for CountingExecutor {
    fn card(&self) -> TensorExecutorCard {
        TensorExecutorCard::new(
            Symbol::qualified("test", "linalg-counting-executor"),
            "counting",
            Symbol::qualified("core", "local-fabric"),
            vec![matmul_exec_op_symbol()],
            None,
        )
    }

    fn execute(
        &self,
        cx: &mut Cx,
        request: TensorRequest,
    ) -> std::result::Result<TensorExecution, TensorExecError> {
        if request.operation.symbol == matmul_exec_op_symbol() {
            self.calls.fetch_add(1, Ordering::SeqCst);
        }
        CpuTensorExecutor::new().execute(cx, request)
    }

    fn flush(&self) -> std::result::Result<SubmissionEvidence, TensorExecError> {
        CpuTensorExecutor::new().flush()
    }
}

#[test]
fn dot_and_eye_surface_work() {
    let mut cx = cx();
    let left = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![i64_num("1"), i64_num("2"), i64_num("3")]),
        )
        .unwrap();
    let right = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![i64_num("4"), i64_num("5"), i64_num("6")]),
        )
        .unwrap();
    let out = cx
        .call_function(&Symbol::new("dot"), Args::new(vec![left, right]))
        .unwrap();
    assert_eq!(
        out.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "32".to_owned(),
        })
    );

    let eye = cx
        .call_function(&Symbol::new("eye"), Args::new(vec![i64_num("2")]))
        .unwrap();
    let matrix = cx
        .call_function(
            &Symbol::new("mat"),
            Args::new(vec![
                cx.factory()
                    .list(vec![
                        cx.factory().list(vec![i64_num("7"), i64_num("8")]).unwrap(),
                        cx.factory()
                            .list(vec![i64_num("9"), i64_num("10")])
                            .unwrap(),
                    ])
                    .unwrap(),
            ]),
        )
        .unwrap();
    let out = cx
        .call_function(&Symbol::new("matmul"), Args::new(vec![eye, matrix.clone()]))
        .unwrap();
    assert_eq!(
        out.object().as_expr(&mut cx).unwrap(),
        matrix.object().as_expr(&mut cx).unwrap()
    );
}

#[test]
fn matmul_routes_through_active_tensor_executor() {
    let mut cx = cx();
    let calls = Arc::new(AtomicUsize::new(0));
    let site = TensorSite::new(
        Symbol::qualified("test", "linalg-counting-site"),
        Arc::new(CountingExecutor {
            calls: calls.clone(),
        }),
        Vec::new(),
    );
    let expr = Expr::Call {
        operator: Box::new(Expr::Symbol(Symbol::new("matmul"))),
        args: vec![
            Expr::Call {
                operator: Box::new(Expr::Symbol(Symbol::new("mat"))),
                args: vec![Expr::Vector(vec![
                    Expr::Vector(vec![Expr::Number(NumberLiteral {
                        domain: Symbol::qualified("numbers", "i64"),
                        canonical: "1".to_owned(),
                    })]),
                    Expr::Vector(vec![Expr::Number(NumberLiteral {
                        domain: Symbol::qualified("numbers", "i64"),
                        canonical: "2".to_owned(),
                    })]),
                ])],
            },
            Expr::Call {
                operator: Box::new(Expr::Symbol(Symbol::new("vec"))),
                args: vec![Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "3".to_owned(),
                })],
            },
        ],
    };

    let reply = site.realize(&mut cx, eval_request(expr)).unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let tensor = tensor_value_ref(&reply.value).unwrap();
    assert_eq!(tensor.shape(), &[2]);
}

#[test]
fn reductions_and_transcendentals_have_checked_cpu_contracts() {
    let mut cx = cx();
    let vector = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![i64_num("3"), i64_num("4")]),
        )
        .unwrap();
    let sum = cx
        .call_function(&Symbol::new("sum"), Args::new(vec![vector.clone()]))
        .unwrap();
    assert_eq!(
        sum.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "7".to_owned(),
        })
    );

    let norm = cx
        .call_function(&Symbol::new("norm"), Args::new(vec![vector.clone()]))
        .unwrap();
    assert_eq!(
        norm.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "f32"),
            canonical: "5".to_owned(),
        })
    );

    let roots = cx
        .call_function(&Symbol::new("sqrt"), Args::new(vec![vector]))
        .unwrap();
    assert_eq!(
        tensor_value_ref(&roots).unwrap().dtype(),
        &Symbol::qualified("numbers", "f32")
    );
}

#[test]
fn tile_local_phase_arguments_use_tensor_math_not_interference_ops() {
    let mut cx = cx();
    let phase = cx
        .call_function(&Symbol::new("vec"), Args::new(vec![i64_num("0")]))
        .unwrap();

    let sine = cx
        .call_function(&Symbol::new("sin"), Args::new(vec![phase.clone()]))
        .unwrap();
    let cosine = cx
        .call_function(&Symbol::new("cos"), Args::new(vec![phase]))
        .unwrap();

    assert_eq!(
        tensor_value_ref(&sine).unwrap().cells().unwrap()[0]
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "f32"),
            canonical: "0".to_owned(),
        })
    );
    assert_eq!(
        tensor_value_ref(&cosine).unwrap().cells().unwrap()[0]
            .object()
            .as_expr(&mut cx)
            .unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "f32"),
            canonical: "1".to_owned(),
        })
    );
    assert!(
        cx.call_function(
            &Symbol::qualified("interference", "solve"),
            Args::new(Vec::new())
        )
        .is_err()
    );
}

#[test]
fn zeros_with_oversized_shape_errors_instead_of_oom() {
    let mut cx = cx();
    let err = cx
        .call_function(
            &Symbol::new("zeros"),
            Args::new(vec![shape_value(&["1000000000000"])]),
        )
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("cells") && message.contains("exceeding"),
        "expected a cell-ceiling diagnostic, got: {message}"
    );
}

#[test]
fn det_of_large_matrix_uses_elimination_and_returns_promptly() {
    let mut cx = cx();
    // A 20x20 upper-triangular matrix with 2 on the diagonal: determinant is
    // 2^20 = 1048576. Cofactor expansion would need ~20! operations and hang;
    // the Bareiss elimination path returns immediately.
    let n = 20usize;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let entry = if i == j {
                "2"
            } else if j > i {
                "1"
            } else {
                "0"
            };
            row.push(i64_num(entry));
        }
        rows.push(cx.factory().list(row).unwrap());
    }
    let grid = cx.factory().list(rows).unwrap();
    let matrix = cx
        .call_function(&Symbol::new("mat"), Args::new(vec![grid]))
        .unwrap();
    let start = std::time::Instant::now();
    let out = cx
        .call_function(&Symbol::new("det"), Args::new(vec![matrix]))
        .unwrap();
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "det of a 20x20 matrix must return promptly"
    );
    assert_eq!(
        out.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "1048576".to_owned(),
        })
    );
}

#[test]
fn symbolic_matmul_yields_symbolic_cells() {
    let mut cx = cx();
    let a = cas_var(&mut cx, "a");
    let b = cas_var(&mut cx, "b");
    let c = cas_var(&mut cx, "c");
    let d = cas_var(&mut cx, "d");
    let left = cx
        .call_function(
            &Symbol::new("mat"),
            Args::new(vec![
                cx.factory()
                    .list(vec![
                        cx.factory().list(vec![a, b]).unwrap(),
                        cx.factory().list(vec![c, d]).unwrap(),
                    ])
                    .unwrap(),
            ]),
        )
        .unwrap();
    let x = cas_var(&mut cx, "x");
    let y = cas_var(&mut cx, "y");
    let right = cx
        .call_function(
            &Symbol::new("mat"),
            Args::new(vec![
                cx.factory()
                    .list(vec![
                        cx.factory().list(vec![x]).unwrap(),
                        cx.factory().list(vec![y]).unwrap(),
                    ])
                    .unwrap(),
            ]),
        )
        .unwrap();
    let out = cx
        .call_function(&Symbol::new("matmul"), Args::new(vec![left, right]))
        .unwrap();
    let expr = out.object().as_expr(&mut cx).unwrap();
    match expr {
        Expr::Vector(rows) => assert_eq!(rows.len(), 2),
        other => panic!("expected symbolic matrix result, got {other:?}"),
    }
}

#[test]
fn linalg_ops_accept_tensor_citizen_values() {
    let mut cx = cx();
    cx.grant(read_construct_capability());
    let left = cx
        .read_construct(
            &tensor_value_class_symbol(),
            vec![
                symbol(Symbol::new("v1")),
                shape_value(&["3"]),
                data_value(vec![i64_num("1"), i64_num("2"), i64_num("3")]),
                symbol(Symbol::qualified("numbers", "i64")),
            ],
        )
        .unwrap();
    let right = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![i64_num("4"), i64_num("5"), i64_num("6")]),
        )
        .unwrap();
    let out = cx
        .call_function(&Symbol::new("dot"), Args::new(vec![left, right]))
        .unwrap();
    assert_eq!(
        out.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "32".to_owned(),
        })
    );
}
