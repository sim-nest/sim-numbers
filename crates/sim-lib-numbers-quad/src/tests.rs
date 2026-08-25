use std::{any::Any, sync::Arc};

use sim_kernel::{
    Args, Callable, ClassRef, Cx, DefaultFactory, EagerPolicy, Error, Factory, Object, Symbol,
    Value,
};
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

#[derive(Clone)]
struct PlainUnary {
    name: &'static str,
    f: fn(f64) -> f64,
}

impl Object for PlainUnary {
    fn display(&self, _cx: &mut Cx) -> sim_kernel::Result<String> {
        Ok(format!("#<plain-callable {}>", self.name))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for PlainUnary {
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

impl Callable for PlainUnary {
    fn call(&self, cx: &mut Cx, args: Args) -> sim_kernel::Result<Value> {
        let values = args.into_vec();
        let [x] = values.as_slice() else {
            return Err(Error::Eval("plain unary expected one arg".to_owned()));
        };
        let x = value_to_f64(cx, x);
        Ok(f64_value(cx, (self.f)(x)))
    }
}

fn plain_unary(cx: &mut Cx, name: &'static str, f: fn(f64) -> f64) -> Value {
    cx.factory()
        .opaque(Arc::new(PlainUnary { name, f }))
        .unwrap()
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

#[test]
fn integrate_accepts_plain_callable() {
    let mut cx = test_cx();
    let func = plain_unary(&mut cx, "plain-sin", f64::sin);
    let method = cx.factory().symbol(Symbol::new("simpson")).unwrap();
    let n = f64_value(&mut cx, 128.0);
    let options = cx
        .factory()
        .table(vec![
            (Symbol::new(":method"), method),
            (Symbol::new(":n"), n),
        ])
        .unwrap();
    let var = cx.factory().symbol(Symbol::new("x")).unwrap();
    let lo = f64_value(&mut cx, 0.0);
    let hi = f64_value(&mut cx, std::f64::consts::PI);

    let out = cx
        .call_function(
            &Symbol::new("integrate"),
            Args::new(vec![func, var, lo, hi, options]),
        )
        .unwrap();

    assert!((value_to_f64(&mut cx, &out) - 2.0).abs() < 1.0e-8);
}

#[test]
fn numeric_diff_named_method_accepts_plain_callable() {
    let mut cx = test_cx();
    let func = plain_unary(&mut cx, "plain-quadratic", |x| x * x + x);
    let method = cx.factory().symbol(Symbol::new("central-5")).unwrap();
    let options = cx
        .factory()
        .table(vec![(Symbol::new(":method"), method)])
        .unwrap();
    let var = cx.factory().symbol(Symbol::new("x")).unwrap();
    let point = f64_value(&mut cx, 3.0);

    let out = cx
        .call_function(
            &Symbol::new("numeric-diff"),
            Args::new(vec![func, var, point, options]),
        )
        .unwrap();

    assert!((value_to_f64(&mut cx, &out) - 7.0).abs() < 1.0e-3);
}

#[test]
fn sampled_rules_preserve_orientation_tail_and_vector_evidence() {
    use crate::{
        FinalInterval, SampledPlan, SampledRule, SumMode, cumulative_sampled, integrate_sampled,
    };
    let p = SampledPlan {
        rule: SampledRule::Simpson,
        final_interval: FinalInterval::Trapezoid,
        sum: SumMode::Neumaier,
    };
    let x = [0., 0.5, 2., 3.];
    let y = x.iter().map(|x| vec![x * x, 2. * x]).collect::<Vec<_>>();
    let r = integrate_sampled(&x, &y, p).unwrap();
    assert!((r.value[0] - 55. / 6.).abs() < 1e-12);
    assert!(r.evidence.fallback_used);
    let sx = [0., 0.5, 2.];
    let sy = sx.iter().map(|x| vec![x * x, 2. * x]).collect::<Vec<_>>();
    let sr = integrate_sampled(&sx, &sy, p).unwrap();
    let mut xr = sx;
    xr.reverse();
    let mut yr = sy;
    yr.reverse();
    let rr = integrate_sampled(&xr, &yr, p).unwrap();
    assert!((rr.value[0] + sr.value[0]).abs() < 1e-12);
    assert_eq!(rr.evidence.orientation, -1);
    let c = cumulative_sampled(&x, &y, p).unwrap();
    assert_eq!(c.len(), x.len());
    assert_eq!(c.last().unwrap(), &r.value);
}

#[test]
fn sampled_input_and_tail_policies_fail_closed() {
    use crate::{
        FinalInterval, SampledError, SampledPlan, SampledRule, SumMode, integrate_sampled,
    };
    let simpson = SampledPlan {
        rule: SampledRule::Simpson,
        final_interval: FinalInterval::Error,
        sum: SumMode::Pairwise,
    };
    assert_eq!(
        integrate_sampled(
            &[0., 1., 2., 3.],
            &[vec![0.], vec![1.], vec![4.], vec![9.]],
            simpson
        ),
        Err(SampledError::UnmatchedFinalInterval)
    );
    assert_eq!(
        integrate_sampled(&[0.], &[vec![0.]], SampledPlan::default()),
        Err(SampledError::TooShort)
    );
    assert_eq!(
        integrate_sampled(&[0., 0.], &[vec![0.], vec![0.]], SampledPlan::default()),
        Err(SampledError::NotStrictlyMonotone)
    );
    assert_eq!(
        integrate_sampled(
            &[0., f64::NAN],
            &[vec![0.], vec![1.]],
            SampledPlan::default()
        ),
        Err(SampledError::NonFinite)
    );
}

#[test]
fn nonuniform_simpson_is_quadratic_exact_and_translation_stable() {
    use crate::{FinalInterval, SampledPlan, SampledRule, SumMode, integrate_sampled};
    let plan = SampledPlan {
        rule: SampledRule::Simpson,
        final_interval: FinalInterval::Error,
        sum: SumMode::Neumaier,
    };
    for shift in [0., 1.0e6] {
        let x = [shift, shift + 0.25, shift + 2.0];
        let y = x
            .iter()
            .map(|&x| {
                let u = x - shift;
                vec![1., u, u * u]
            })
            .collect::<Vec<_>>();
        let value = integrate_sampled(&x, &y, plan).unwrap().value;
        assert!((value[0] - 2.).abs() < 1e-12);
        assert!((value[1] - 2.).abs() < 1e-12);
        assert!((value[2] - 8. / 3.).abs() < 1e-12);
    }
}

#[test]
fn event_split_keeps_named_one_sided_samples_separate() {
    use crate::{EventSegment, SampledPlan, integrate_event_segments};
    let segments = [
        EventSegment {
            id: "before-jump".into(),
            x: vec![0., 1.],
            y: vec![vec![0.], vec![1.]],
        },
        EventSegment {
            id: "after-jump".into(),
            x: vec![1., 2.],
            y: vec![vec![10.], vec![10.]],
        },
    ];
    let result = integrate_event_segments(&segments, SampledPlan::default()).unwrap();
    assert_eq!(result[0].0, "before-jump");
    assert_eq!(result[0].1.value, [0.5]);
    assert_eq!(result[1].0, "after-jump");
    assert_eq!(result[1].1.value, [10.]);
}

#[test]
fn vector_kronrod_shares_mesh_and_compensation_keeps_small_terms() {
    use crate::{SumMode, adaptive_vector_gauss_kronrod};
    let r = adaptive_vector_gauss_kronrod(
        |x| vec![x * x, x.sin()],
        0.,
        1.,
        1e-10,
        8,
        SumMode::Neumaier,
    )
    .unwrap();
    assert!((r.value[0] - 1. / 3.).abs() < 1e-10);
    assert!((r.value[1] - (1. - 1f64.cos())).abs() < 1e-10);
    assert!(!r.mesh.is_empty());
    assert_eq!(r.sum, SumMode::Neumaier);
}
