use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, EagerPolicy, Error, Factory, Object, Symbol,
    Value,
};
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

#[derive(Clone)]
struct PlainBinary {
    name: &'static str,
    f: fn(f64, f64) -> f64,
}

impl Object for PlainBinary {
    fn display(&self, _cx: &mut Cx) -> sim_kernel::Result<String> {
        Ok(format!("#<plain-callable {}>", self.name))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for PlainBinary {
    fn class(&self, _cx: &mut Cx) -> sim_kernel::Result<ClassRef> {
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("core", "Function"),
        )
    }

    fn as_callable(&self) -> Option<&dyn Callable> {
        Some(self)
    }
}

impl Callable for PlainBinary {
    fn call(&self, cx: &mut Cx, args: Args) -> sim_kernel::Result<Value> {
        let values = args.into_vec();
        let [x, y] = values.as_slice() else {
            return Err(Error::Eval("plain binary expected two args".to_owned()));
        };
        let x = value_to_f64(cx, x);
        let y = value_to_f64(cx, y);
        Ok(f64_value(cx, (self.f)(x, y)))
    }
}

fn plain_binary(cx: &mut Cx, name: &'static str, f: fn(f64, f64) -> f64) -> Value {
    cx.factory()
        .opaque(Arc::new(PlainBinary { name, f }))
        .unwrap()
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
fn ode_accepts_plain_binary_callable() {
    let mut cx = test_cx();
    let rhs = plain_binary(&mut cx, "plain-exp-rhs", |_x, y| y);
    let method = cx.factory().symbol(Symbol::new("rk4")).unwrap();
    let h = f64_value(&mut cx, 0.01);
    let options = cx
        .factory()
        .table(vec![
            (Symbol::new(":method"), method),
            (Symbol::new(":h"), h),
        ])
        .unwrap();
    let x0 = f64_value(&mut cx, 0.0);
    let y0 = f64_value(&mut cx, 1.0);
    let x_end = f64_value(&mut cx, 1.0);

    let out = cx
        .call_function(
            &Symbol::new("ode-solve"),
            Args::new(vec![
                rhs,
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
    assert!((last_y - std::f64::consts::E).abs() < 1.0e-8);
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
