use std::sync::Arc;

use sim_kernel::{
    Args, Cx, DefaultFactory, EagerPolicy, Expr, Factory, NumberLiteral, Symbol,
    read_construct_capability,
};
use sim_lib_numbers_arith::NumbersArithmeticLib;
use sim_lib_numbers_cas::CasNumbersLib;
use sim_lib_numbers_i64::I64NumbersLib;
use sim_lib_numbers_tensor::{TensorNumbersLib, tensor_value_class_symbol};
use sim_lib_numbers_tensor_bcast::TensorBroadcastLib;

use crate::TensorLinalgLib;

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

fn cas_var(cx: &mut Cx, symbol: &str) -> sim_kernel::Value {
    cx.call_function(
        &Symbol::qualified("cas", "var"),
        Args::new(vec![DefaultFactory.symbol(Symbol::new(symbol)).unwrap()]),
    )
    .unwrap()
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
