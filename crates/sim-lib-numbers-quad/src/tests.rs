use std::sync::Arc;

use sim_kernel::{Args, Cx, DefaultFactory, EagerPolicy, Error, Symbol, Value};
use sim_lib_numbers_func::Func;
use sim_lib_numbers_numeric::NumericNumbersLib;

use crate::QuadNumbersLib;

fn test_cx() -> Cx {
    let mut cx = Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_arith::NumbersArithmeticLib::new())
        .unwrap();
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&NumericNumbersLib::new()).unwrap();
    cx.load_lib(&QuadNumbersLib::new()).unwrap();
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

fn unary_native<F>(f: F) -> Func
where
    F: Fn(f64) -> f64 + Send + Sync + 'static,
{
    Func::native(
        vec![Symbol::new("x")],
        Arc::new(move |cx, args| {
            let [x] = args else {
                return Err(Error::Eval("expected one arg".to_owned()));
            };
            let x = value_to_f64(cx, x);
            Ok(f64_value(cx, f(x)))
        }),
    )
}

#[test]
fn differentiators_hit_expected_value() {
    let mut cx = test_cx();
    let func = cx
        .factory()
        .opaque(Arc::new(unary_native(f64::sin)))
        .unwrap();
    for method in [
        "forward",
        "backward",
        "central-3",
        "central-5",
        "richardson",
    ] {
        let method_value = cx.factory().symbol(Symbol::new(method)).unwrap();
        let h = f64_value(&mut cx, 1.0e-4);
        let options = cx
            .factory()
            .table(vec![
                (Symbol::new(":method"), method_value),
                (Symbol::new(":h"), h),
            ])
            .unwrap();
        let var = cx.factory().symbol(Symbol::new("x")).unwrap();
        let point = f64_value(&mut cx, 0.0);
        let out = cx
            .call_function(
                &Symbol::new("numeric-diff"),
                Args::new(vec![func.clone(), var, point, options]),
            )
            .unwrap();
        let value = value_to_f64(&mut cx, &out);
        let tol = if method == "forward" || method == "backward" {
            2.0e-4
        } else {
            1.0e-6
        };
        assert!((value - 1.0).abs() < tol, "{method} -> {value}");
    }
}

#[test]
fn quadratures_integrate_sine_to_two() {
    let mut cx = test_cx();
    let func = cx
        .factory()
        .opaque(Arc::new(unary_native(f64::sin)))
        .unwrap();
    for method in [
        "trapezoid",
        "simpson",
        "romberg",
        "gauss-legendre-8",
        "gauss-legendre-16",
        "gauss-legendre-32",
        "adaptive-gauss-kronrod",
    ] {
        let mut entries = vec![(
            Symbol::new(":method"),
            cx.factory().symbol(Symbol::new(method)).unwrap(),
        )];
        if method == "trapezoid" || method == "simpson" || method == "romberg" {
            entries.push((Symbol::new(":n"), f64_value(&mut cx, 128.0)));
        }
        if method == "adaptive-gauss-kronrod" || method == "romberg" {
            entries.push((Symbol::new(":tol"), f64_value(&mut cx, 1.0e-10)));
        }
        let options = cx.factory().table(entries).unwrap();
        let symbol = if method == "adaptive-gauss-kronrod" {
            Symbol::new("integrate-adapt")
        } else {
            Symbol::new("integrate")
        };
        let var = cx.factory().symbol(Symbol::new("x")).unwrap();
        let lo = f64_value(&mut cx, 0.0);
        let hi = f64_value(&mut cx, std::f64::consts::PI);
        let out = cx
            .call_function(&symbol, Args::new(vec![func.clone(), var, lo, hi, options]))
            .unwrap();
        let value = value_to_f64(&mut cx, &out);
        let tol = match method {
            "trapezoid" => 2.0e-4,
            "gauss-legendre-8" => 1.0e-7,
            _ => 1.0e-8,
        };
        assert!((value - 2.0).abs() < tol, "{method} -> {value}");
    }
}
