use std::sync::Arc;

use sim_kernel::{
    Args, Cx, DefaultFactory, EagerPolicy, Expr, Factory, NumberLiteral, Symbol,
    read_construct_capability,
};
use sim_lib_numbers_tensor::tensor_value_class_symbol;

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
