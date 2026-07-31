// conformance: checked Lisp signal operations and finite diagnostic evidence.

use std::sync::Arc;

use sim_codec::{Input, decode_eval_expr_with_codec, encode_value_with_codec};
use sim_codec_lisp::LispCodecLib;
use sim_kernel::{
    CapabilitySet, DefaultFactory, EagerPolicy, EncodeOptions, ReadPolicy, Symbol, TrustLevel,
};

use crate::{RECIPES, SignalNumbersLib};

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(EagerPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&sim_lib_numbers_f64::F64NumbersLib::new())
        .unwrap();
    cx.load_lib(&SignalNumbersLib::new()).unwrap();
    let lisp = LispCodecLib::new(cx.registry_mut().fresh_codec_id()).unwrap();
    cx.load_lib(&lisp).unwrap();
    cx
}

#[test]
fn lisp_surface_runs_an_impulse_fft() {
    let mut cx = cx();
    let recipes = sim_cookbook::recipes_from_embedded(RECIPES).unwrap();
    let recipe = recipes
        .iter()
        .find(|recipe| recipe.id.ends_with("/impulse-fft"))
        .unwrap();
    let expr = decode_eval_expr_with_codec(
        &mut cx,
        &Symbol::qualified("codec", "lisp"),
        Input::Text(String::from_utf8(recipe.setup.clone()).unwrap()),
        ReadPolicy {
            trust: TrustLevel::TrustedSource,
            capabilities: CapabilitySet::new(),
        },
    )
    .unwrap();
    let output = cx.eval_expr(expr).unwrap();
    let encoded = encode_value_with_codec(
        &mut cx,
        &Symbol::qualified("codec", "lisp"),
        &output,
        EncodeOptions::default(),
    )
    .unwrap()
    .into_text()
    .unwrap();
    assert_eq!(recipe.expect.len(), 1);
    assert_eq!(recipe.expect[0].form, 0);
    assert_eq!(encoded, recipe.expect[0].result);
}

#[test]
fn lisp_surface_exposes_costed_convolution_and_guarded_deconvolution() {
    let mut cx = cx();
    let recipes = sim_cookbook::recipes_from_embedded(RECIPES).unwrap();
    let recipe = recipes
        .iter()
        .find(|recipe| recipe.id.ends_with("/convolution-evidence"))
        .unwrap();
    let convolution = eval_lisp(&mut cx, &String::from_utf8(recipe.setup.clone()).unwrap());
    assert_eq!(convolution, recipe.expect[0].result);
    assert!(convolution.contains("algorithm direct"), "{convolution}");
    assert!(convolution.contains("samples (1 1 1 -3)"), "{convolution}");
    assert!(convolution.contains("direct-cost-units 6"), "{convolution}");

    let deconvolution = eval_lisp(
        &mut cx,
        "(signal/deconvolve [1.0 -1.0 0.0] [1.0 -1.0] :regularization {:kind 'tikhonov :lambda 1e-8})",
    );
    assert!(
        deconvolution.contains("regularization tikhonov"),
        "{deconvolution}"
    );
    assert!(
        deconvolution.contains("singular-bins (0)"),
        "{deconvolution}"
    );
    assert!(!deconvolution.contains("inf"), "{deconvolution}");
    assert!(!deconvolution.contains("NaN"), "{deconvolution}");
}

fn eval_lisp(cx: &mut sim_kernel::Cx, source: &str) -> String {
    let expr = decode_eval_expr_with_codec(
        cx,
        &Symbol::qualified("codec", "lisp"),
        Input::Text(source.to_owned()),
        ReadPolicy {
            trust: TrustLevel::TrustedSource,
            capabilities: CapabilitySet::new(),
        },
    )
    .unwrap();
    let output = cx.eval_expr(expr).unwrap();
    encode_value_with_codec(
        cx,
        &Symbol::qualified("codec", "lisp"),
        &output,
        EncodeOptions::default(),
    )
    .unwrap()
    .into_text()
    .unwrap()
}
