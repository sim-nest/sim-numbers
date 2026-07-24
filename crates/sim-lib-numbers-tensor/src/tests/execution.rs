use std::sync::Arc;

use sim_kernel::{
    CapabilityName, Consistency, Error, EvalFabric, EvalMode, EvalRequest, Expr, NumberLiteral,
    Symbol, Value, realize_final,
};
use sim_shape::{ExprKind, ExprKindShape, shape_value};

// conformance: tensor execution site

use crate::{
    CpuTensorExecutor, TensorExecution, TensorExecutor, TensorMeta, TensorOp, TensorRequest,
    TensorSite, add_op_symbol, build_tensor_value, reshape_op_symbol, tensor_execute_capability,
    tensor_executor_symbol, tensor_site_symbol, tensor_value_ref,
};

use super::test_cx;

fn int_expr(canonical: &str) -> Expr {
    Expr::Number(NumberLiteral {
        domain: Symbol::qualified("citizen", "int"),
        canonical: canonical.to_owned(),
    })
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

fn reshape_expr() -> Expr {
    call(
        "reshape",
        vec![
            call(
                "vec",
                vec![i64_expr("1"), i64_expr("2"), i64_expr("3"), i64_expr("4")],
            ),
            Expr::Vector(vec![int_expr("2"), int_expr("2")]),
        ],
    )
}

fn request(expr: Expr, required_capabilities: Vec<CapabilityName>) -> EvalRequest {
    EvalRequest {
        expr,
        result_shape: None,
        required_capabilities,
        deadline: None,
        consistency: Consistency::LocalFirst,
        mode: EvalMode::Eval,
        answer_limit: None,
        stream_buffer: None,
        stream: false,
        trace: false,
    }
}

fn tensor_cells_expr(cx: &mut sim_kernel::Cx, value: &Value) -> Vec<Expr> {
    let tensor = tensor_value_ref(value).expect("tensor value");
    tensor
        .cells()
        .unwrap()
        .iter()
        .map(|cell| cell.object().as_expr(cx).unwrap())
        .collect()
}

fn assert_same_tensor(cx: &mut sim_kernel::Cx, left: &Value, right: &Value) {
    let left_tensor = tensor_value_ref(left).expect("left tensor");
    let right_tensor = tensor_value_ref(right).expect("right tensor");
    assert_eq!(left_tensor.shape(), right_tensor.shape());
    assert_eq!(left_tensor.dtype(), right_tensor.dtype());
    assert_eq!(tensor_cells_expr(cx, left), tensor_cells_expr(cx, right));
}

fn tensor_cell_exprs(cx: &mut sim_kernel::Cx, tensor: &crate::Tensor) -> Vec<Expr> {
    tensor
        .cells()
        .unwrap()
        .iter()
        .map(|cell| cell.object().as_expr(cx).unwrap())
        .collect()
}

#[test]
fn cpu_executor_reshapes_using_current_tensor_semantics() {
    let mut cx = test_cx();
    let source_value = cx
        .eval_expr(call("vec", vec![i64_expr("1"), i64_expr("2")]))
        .unwrap();
    let source_tensor = tensor_value_ref(&source_value).unwrap().clone();
    let op = TensorOp::without_attributes(&mut cx, reshape_op_symbol()).unwrap();
    let request = TensorRequest::new(
        op,
        vec![source_tensor],
        TensorMeta::new(vec![2, 1], Symbol::qualified("numbers", "i64")),
    );

    let result = CpuTensorExecutor::new().execute(&mut cx, request).unwrap();

    let TensorExecution::Complete(tensor) = result else {
        panic!("cpu executor must complete reshape");
    };
    assert_eq!(tensor.shape(), &[2, 1]);
    assert_eq!(tensor.dtype(), &Symbol::qualified("numbers", "i64"));
}

#[test]
fn cpu_executor_adds_with_broadcast_metadata() {
    let mut cx = test_cx();
    let left = tensor_value_ref(
        &build_tensor_value(
            &mut cx,
            vec![2, 1],
            Some(Symbol::qualified("numbers", "i64")),
            vec![super::number("i64", "1"), super::number("i64", "2")],
        )
        .unwrap(),
    )
    .unwrap()
    .clone();
    let right = tensor_value_ref(
        &build_tensor_value(
            &mut cx,
            vec![1, 3],
            Some(Symbol::qualified("numbers", "i64")),
            vec![
                super::number("i64", "10"),
                super::number("i64", "20"),
                super::number("i64", "30"),
            ],
        )
        .unwrap(),
    )
    .unwrap()
    .clone();
    let op = TensorOp::without_attributes(&mut cx, add_op_symbol()).unwrap();

    let result = CpuTensorExecutor::new()
        .execute(
            &mut cx,
            TensorRequest::new(
                op,
                vec![left, right],
                TensorMeta::new(vec![2, 3], Symbol::qualified("numbers", "i64")),
            ),
        )
        .unwrap();

    let TensorExecution::Complete(tensor) = result else {
        panic!("cpu executor must complete element-wise add");
    };
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(
        tensor_cell_exprs(&mut cx, &tensor),
        vec![
            i64_expr("11"),
            i64_expr("21"),
            i64_expr("31"),
            i64_expr("12"),
            i64_expr("22"),
            i64_expr("32"),
        ]
    );
}

#[test]
fn tensor_site_realized_cpu_matches_direct_cpu_executor() {
    let mut cx = test_cx();
    let source_value = cx
        .eval_expr(call(
            "vec",
            vec![i64_expr("1"), i64_expr("2"), i64_expr("3"), i64_expr("4")],
        ))
        .unwrap();
    let source_tensor = tensor_value_ref(&source_value).unwrap().clone();
    let op = TensorOp::without_attributes(&mut cx, reshape_op_symbol()).unwrap();
    let direct = match CpuTensorExecutor::new()
        .execute(
            &mut cx,
            TensorRequest::new(
                op,
                vec![source_tensor],
                TensorMeta::new(vec![2, 2], Symbol::qualified("numbers", "i64")),
            ),
        )
        .unwrap()
    {
        TensorExecution::Complete(tensor) => {
            cx.factory().opaque(std::sync::Arc::new(tensor)).unwrap()
        }
        TensorExecution::Unsupported { reason } => panic!("unexpected unsupported: {reason}"),
    };

    let site = TensorSite::local_cpu();
    let realized = realize_final(&mut cx, &site, request(reshape_expr(), Vec::new()))
        .unwrap()
        .value;

    assert_same_tensor(&mut cx, &direct, &realized);
}

#[test]
fn tensor_site_checks_capabilities_and_restores_parent_env() {
    let mut cx = test_cx();
    assert!(cx.env().get(&tensor_executor_symbol()).is_none());
    let site = TensorSite::local_cpu();
    let denied = match site.realize(
        &mut cx,
        request(reshape_expr(), vec![tensor_execute_capability()]),
    ) {
        Ok(_) => panic!("tensor site must deny missing tensor execution capability"),
        Err(error) => error,
    };
    assert!(matches!(
        denied,
        Error::CapabilityDenied { capability } if capability == tensor_execute_capability()
    ));
    assert!(cx.env().get(&tensor_executor_symbol()).is_none());

    cx.grant(tensor_execute_capability());
    let reply = site
        .realize(
            &mut cx,
            request(
                Expr::Symbol(tensor_executor_symbol()),
                vec![tensor_execute_capability()],
            ),
        )
        .unwrap();
    assert!(
        reply
            .value
            .object()
            .display(&mut cx)
            .unwrap()
            .contains("tensor-executor")
    );
    assert!(cx.env().get(&tensor_executor_symbol()).is_none());
}

#[test]
fn tensor_site_checks_result_shape_and_restores_parent_env() {
    let mut cx = test_cx();
    let site = TensorSite::local_cpu();
    let mut request = request(reshape_expr(), Vec::new());
    request.result_shape = Some(shape_value(
        Symbol::qualified("test", "bool-result"),
        Arc::new(ExprKindShape::new(ExprKind::Bool)),
    ));

    let error = match site.realize(&mut cx, request) {
        Ok(_) => panic!("tensor site must reject a mismatched result shape"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::WrongShape { .. }));
    assert!(cx.env().get(&tensor_executor_symbol()).is_none());
}

#[test]
fn tensor_numbers_lib_exports_tensor_site() {
    let cx = test_cx();
    let site = cx
        .registry()
        .site_by_symbol(&tensor_site_symbol())
        .expect("tensor site export");
    assert!(site.object().as_eval_fabric().is_some());
}
