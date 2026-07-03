use std::sync::Arc;

use sim_kernel::{Args, DefaultFactory, EagerPolicy, Expr, Symbol};

use crate::{ExoticNumbersLib, implementation::number_domain};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_i64::I64NumbersLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_rational::RationalNumbersLib::new())
        .unwrap();
    cx.load_lib(&ExoticNumbersLib::new()).unwrap();
    cx
}

#[test]
fn sqrt2_approximates_to_f64() {
    let mut cx = cx();
    let value = cx
        .call_function(
            &Symbol::new("as-f64"),
            Args::new(vec![
                cx.registry()
                    .value_by_symbol(&Symbol::new("cf-sqrt2"))
                    .unwrap()
                    .clone(),
                cx.factory()
                    .number_literal(Symbol::qualified("numbers", "i64"), "50".to_owned())
                    .unwrap(),
            ]),
        )
        .unwrap();
    let Expr::Number(number) = value.object().as_expr(&mut cx).unwrap() else {
        panic!("expected f64 literal");
    };
    let approx = number.canonical.parse::<f64>().unwrap();
    assert!((approx - std::f64::consts::SQRT_2).abs() < 1e-12);
}

#[test]
fn pi_coefficients_are_lazy_list_values() {
    let mut cx = cx();
    let coeffs = cx
        .registry()
        .value_by_symbol(&Symbol::new("cf-pi"))
        .unwrap()
        .clone();
    let coeffs = coeffs
        .object()
        .as_list()
        .unwrap()
        .to_vec(&mut cx, Some(5))
        .unwrap();
    let coeffs = cx.factory().list(coeffs).unwrap();
    assert_eq!(
        coeffs.object().as_expr(&mut cx).unwrap(),
        Expr::List(vec![
            Expr::Number(sim_kernel::NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "3".to_owned(),
            }),
            Expr::Number(sim_kernel::NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "7".to_owned(),
            }),
            Expr::Number(sim_kernel::NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "15".to_owned(),
            }),
            Expr::Number(sim_kernel::NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "1".to_owned(),
            }),
            Expr::Number(sim_kernel::NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "292".to_owned(),
            }),
        ])
    );
}

#[test]
fn continued_fraction_promotes_to_rational() {
    let mut cx = cx();
    let value = cx
        .apply_value_number_binary_op(
            &Symbol::qualified("math", "add"),
            cx.registry()
                .value_by_symbol(&Symbol::new("cf-sqrt2"))
                .unwrap()
                .clone(),
            cx.factory()
                .number_literal(Symbol::qualified("numbers", "rational"), "1/1".to_owned())
                .unwrap(),
        )
        .unwrap();
    let Expr::Number(number) = value.object().as_expr(&mut cx).unwrap() else {
        panic!("expected rational literal");
    };
    assert_eq!(number.domain, Symbol::qualified("numbers", "rational"));
}

#[test]
fn domain_symbol_is_registered() {
    let cx = cx();
    assert!(
        cx.registry()
            .number_domain_by_symbol(&number_domain())
            .is_some()
    );
}
