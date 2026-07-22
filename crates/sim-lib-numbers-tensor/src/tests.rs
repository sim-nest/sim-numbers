use std::sync::Arc;

use sim_kernel::{
    Args, Cx, DefaultFactory, EagerPolicy, Expr, Factory, NumberLiteral, Symbol,
    read_construct_capability,
};

use crate::{
    Tensor, TensorNumbersLib, build_tensor_value, number_domain, tensor_value_class_symbol,
    tensor_value_ref,
};

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
    cx.load_lib(&TensorNumbersLib::new()).unwrap();
    cx
}

fn number(domain: &str, canonical: &str) -> sim_kernel::Value {
    DefaultFactory
        .number_literal(Symbol::qualified("numbers", domain), canonical.to_owned())
        .unwrap()
}

fn int_expr(canonical: &str) -> Expr {
    Expr::Number(NumberLiteral {
        domain: Symbol::qualified("citizen", "int"),
        canonical: canonical.to_owned(),
    })
}

fn symbol(name: &str) -> sim_kernel::Value {
    DefaultFactory.symbol(Symbol::new(name)).unwrap()
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
fn vec_constructor_and_index_roundtrip() {
    let mut cx = test_cx();
    let tensor = cx
        .call_function(
            &Symbol::new("vec"),
            Args::new(vec![
                number("i64", "1"),
                number("i64", "2"),
                number("i64", "3"),
            ]),
        )
        .unwrap();
    assert_eq!(
        tensor.object().as_expr(&mut cx).unwrap(),
        Expr::Vector(vec![
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "1".to_owned(),
            }),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "2".to_owned(),
            }),
            Expr::Number(NumberLiteral {
                domain: Symbol::qualified("numbers", "i64"),
                canonical: "3".to_owned(),
            }),
        ])
    );
    let cell = cx
        .call_function(
            &Symbol::new("index"),
            Args::new(vec![tensor, number("i64", "1")]),
        )
        .unwrap();
    assert_eq!(
        cell.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "2".to_owned(),
        })
    );
}

#[test]
fn tensor_constructor_infers_join_dtype_for_mixed_cells() {
    let mut cx = test_cx();
    let shape = cx.factory().list(vec![number("i64", "3")]).unwrap();
    let values = cx
        .factory()
        .list(vec![
            number("i64", "1"),
            number("f64", "1.5"),
            number("rational", "1/2"),
        ])
        .unwrap();
    let tensor = cx
        .call_function(
            &Symbol::new("tensor"),
            Args::new(vec![shape, cx.factory().nil().unwrap(), values]),
        )
        .unwrap();
    let table = tensor.object().as_table(&mut cx).unwrap();
    let dtype = match table
        .object()
        .as_table_impl()
        .unwrap()
        .get(&mut cx, Symbol::new("dtype"))
        .unwrap()
        .object()
        .as_expr(&mut cx)
        .unwrap()
    {
        Expr::Symbol(symbol) => symbol,
        other => panic!("expected dtype symbol, found {other:?}"),
    };
    assert!(dtype != Symbol::qualified("numbers", "tensor"));
    assert!(cx.registry().number_domain_by_symbol(&dtype).is_some());
    assert_eq!(number_domain(), Symbol::qualified("numbers", "tensor"));
}

#[test]
fn tensor_constructor_rejects_inferred_impossible_join() {
    let mut cx = test_cx();
    let result = build_tensor_value(
        &mut cx,
        vec![2],
        None,
        vec![
            DefaultFactory
                .number_literal(Symbol::qualified("test", "left"), "1".to_owned())
                .unwrap(),
            DefaultFactory
                .number_literal(Symbol::qualified("test", "right"), "2".to_owned())
                .unwrap(),
        ],
    );
    let err = match result {
        Ok(_) => panic!("unregistered, unrelated domains must not infer a dtype"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("no join domain exists"));
}

#[test]
fn tensor_constructor_rejects_explicit_impossible_dtype() {
    let mut cx = test_cx();
    let result = build_tensor_value(
        &mut cx,
        vec![2],
        Some(Symbol::qualified("numbers", "bool")),
        vec![number("i64", "1"), number("f64", "2.0")],
    );
    let err = match result {
        Ok(_) => panic!("explicit dtype must accept every cell"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("not a valid join"));
}

#[test]
fn checked_tensor_constructor_rejects_invalid_direct_construction() {
    let mut cx = test_cx();
    let missing_scalar_cell = match Tensor::new_checked(
        &mut cx,
        Vec::new(),
        Symbol::qualified("numbers", "i64"),
        Vec::new(),
    ) {
        Ok(_) => panic!("rank-0 tensor without a cell must fail"),
        Err(err) => err,
    };
    assert!(missing_scalar_cell.to_string().contains("expects 1 cells"));

    let mismatched_cell_dtype = match Tensor::new_exact(
        vec![1],
        Symbol::qualified("numbers", "i64"),
        vec![number("f64", "1.0")],
    ) {
        Ok(_) => panic!("exact constructor must reject mismatched cell dtype"),
        Err(err) => err,
    };
    assert!(mismatched_cell_dtype.to_string().contains("does not match"));
}

#[test]
fn tensor_citizen_read_constructor_round_trips() {
    let mut cx = test_cx();
    let tensor = build_tensor_value(
        &mut cx,
        vec![2],
        Some(Symbol::qualified("numbers", "i64")),
        vec![number("i64", "1"), number("i64", "2")],
    )
    .unwrap();
    sim_citizen::check_value_fixture_with_wrong_version(
        &mut cx,
        tensor,
        Some(vec![
            Expr::Symbol(Symbol::new("v999")),
            Expr::List(vec![int_expr("2")]),
            Expr::List(vec![
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "1".to_owned(),
                }),
                Expr::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "i64"),
                    canonical: "2".to_owned(),
                }),
            ]),
            Expr::Symbol(Symbol::qualified("numbers", "i64")),
        ]),
    )
    .unwrap();
}

#[test]
fn tensor_read_constructor_accepts_numeric_shape_dimensions() {
    let mut cx = test_cx();
    cx.grant(read_construct_capability());
    let shape = data_value(vec![number("i64", "2"), number("i64", "2")]);
    let tensor = cx
        .read_construct(
            &tensor_value_class_symbol(),
            vec![
                symbol("v1"),
                shape,
                data_value(vec![
                    number("i64", "1"),
                    number("i64", "2"),
                    number("i64", "3"),
                    number("i64", "4"),
                ]),
                DefaultFactory
                    .symbol(Symbol::qualified("numbers", "i64"))
                    .unwrap(),
            ],
        )
        .unwrap();
    let tensor = tensor_value_ref(&tensor).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.dtype(), &Symbol::qualified("numbers", "i64"));
}

#[test]
fn tensor_read_constructor_rejects_malformed_shape_and_wrong_domain() {
    let mut cx = test_cx();
    cx.grant(read_construct_capability());

    let malformed_shape = cx
        .read_construct(
            &tensor_value_class_symbol(),
            vec![
                symbol("v1"),
                data_value(vec![DefaultFactory.string("bad".to_owned()).unwrap()]),
                data_value(vec![number("i64", "1")]),
                DefaultFactory
                    .symbol(Symbol::qualified("numbers", "i64"))
                    .unwrap(),
            ],
        )
        .unwrap_err();
    assert!(malformed_shape.to_string().contains("shape"));

    let wrong_domain = cx
        .read_construct(
            &tensor_value_class_symbol(),
            vec![
                symbol("v1"),
                shape_value(&["1"]),
                data_value(vec![number("i64", "1")]),
                DefaultFactory
                    .symbol(Symbol::qualified("numbers", "bool"))
                    .unwrap(),
            ],
        )
        .unwrap_err();
    assert!(wrong_domain.to_string().contains("dtype"));

    let negative_dimension = cx
        .read_construct(
            &tensor_value_class_symbol(),
            vec![
                symbol("v1"),
                data_value(vec![number("i64", "-1")]),
                data_value(Vec::new()),
                DefaultFactory
                    .symbol(Symbol::qualified("numbers", "i64"))
                    .unwrap(),
            ],
        )
        .unwrap_err();
    assert!(
        negative_dimension
            .to_string()
            .contains("non-negative integer dimensions")
    );
}

#[test]
fn tensor_ops_accept_citizen_values() {
    let mut cx = test_cx();
    cx.grant(read_construct_capability());
    let tensor = cx
        .read_construct(
            &tensor_value_class_symbol(),
            vec![
                symbol("v1"),
                shape_value(&["2"]),
                data_value(vec![number("i64", "8"), number("i64", "13")]),
                DefaultFactory
                    .symbol(Symbol::qualified("numbers", "i64"))
                    .unwrap(),
            ],
        )
        .unwrap();
    let cell = cx
        .call_function(
            &Symbol::new("index"),
            Args::new(vec![tensor, number("i64", "1")]),
        )
        .unwrap();
    assert_eq!(
        cell.object().as_expr(&mut cx).unwrap(),
        Expr::Number(NumberLiteral {
            domain: Symbol::qualified("numbers", "i64"),
            canonical: "13".to_owned(),
        })
    );
}

#[test]
fn build_tensor_value_overflowing_shape_errors_instead_of_panicking() {
    let mut cx = test_cx();
    // A reshape/tensor shape whose dimension product overflows usize must fail
    // closed rather than wrap or panic on the cell-count computation.
    let result = build_tensor_value(
        &mut cx,
        vec![i64::MAX as usize, i64::MAX as usize],
        Some(Symbol::qualified("numbers", "i64")),
        Vec::new(),
    );
    assert!(result.is_err());
}

#[test]
fn tensor_citizen_fixtures_cover_typed_cell_domains() {
    let mut cx = test_cx();
    for (domain, cells) in [
        ("f64", vec!["1.25"]),
        ("i64", vec!["7"]),
        ("rational", vec!["1/2"]),
        ("bool", vec!["true"]),
        ("complex", vec!["1+2i"]),
    ] {
        let domain = Symbol::qualified("numbers", domain);
        let tensor = build_tensor_value(
            &mut cx,
            vec![cells.len()],
            Some(domain.clone()),
            cells
                .into_iter()
                .map(|cell| {
                    DefaultFactory
                        .number_literal(domain.clone(), cell.to_owned())
                        .unwrap()
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(tensor_value_ref(&tensor).unwrap().dtype(), &domain);
        sim_citizen::check_value_fixture(&mut cx, tensor).unwrap();
    }
}
