use std::sync::Arc;

use sim_kernel::{Args, Cx, DefaultFactory, EagerPolicy, Error, Symbol, Value};
use sim_lib_numbers_func::Func;
use sim_lib_numbers_numeric::NumericNumbersLib;

use crate::RkNumbersLib;

fn test_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
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
