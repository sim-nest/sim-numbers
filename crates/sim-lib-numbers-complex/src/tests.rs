use std::sync::Arc;

use sim_kernel::{Args, DefaultFactory, EagerPolicy, Symbol, read_construct_capability};

use crate::{ComplexNumbersLib, add_symbol, complex_value, complex_value_class_symbol};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&ComplexNumbersLib::new()).unwrap();
    cx
}

#[test]
fn complex_citizen_read_constructor_round_trips() {
    let mut cx = cx();
    let value = complex_value(&mut cx, 1.5, -2.25).unwrap();
    sim_citizen::check_value_fixture_with_wrong_version(
        &mut cx,
        value,
        Some(vec![
            sim_kernel::Expr::Symbol(Symbol::new("v999")),
            sim_kernel::Expr::Number(sim_kernel::NumberLiteral {
                domain: Symbol::qualified("numbers", "f64"),
                canonical: "1.5".to_owned(),
            }),
            sim_kernel::Expr::Number(sim_kernel::NumberLiteral {
                domain: Symbol::qualified("numbers", "f64"),
                canonical: "-2.25".to_owned(),
            }),
        ]),
    )
    .unwrap();
}

#[test]
fn complex_ops_accept_citizen_values() {
    let mut cx = cx();
    cx.grant(read_construct_capability());
    let left = complex_value(&mut cx, 1.0, 2.0).unwrap();
    let right = cx
        .read_construct(
            &complex_value_class_symbol(),
            vec![
                cx.factory().symbol(Symbol::new("v1")).unwrap(),
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "f64"), "3".to_owned())
                    .unwrap(),
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "f64"), "-4".to_owned())
                    .unwrap(),
            ],
        )
        .unwrap();
    let sum = cx
        .call_function(&add_symbol(), Args::new(vec![left, right]))
        .unwrap();
    assert_eq!(sum.object().display(&mut cx).unwrap(), "4-2i");
}
