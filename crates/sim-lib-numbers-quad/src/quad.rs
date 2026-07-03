//! The quadrature library and its integration backends, registering fixed and
//! adaptive quadrature rules and the finite-difference differentiators.

use std::sync::Arc;

use sim_kernel::{
    AbiVersion, Cx, Dependency, Export, Lib, LibManifest, LibTarget, Linker, Result, Symbol, Value,
    Version,
};
use sim_lib_numbers_codec::{numeric_plugin_descriptor_symbol, numeric_plugin_descriptor_value};
use sim_lib_numbers_core::domains;
use sim_lib_numbers_func::Func;
use sim_lib_numbers_numeric::{
    NumericKind, NumericPlugin, QuadOpts, Quadrature, register_differentiator, register_quadrature,
};

use super::{
    diff::differentiators,
    support::{
        abs_error, add, add_scaled, call_unary_func, f64_value, scale, sub, value_to_f64, zero_like,
    },
};

/// Registered numeric plugin library that installs this crate's quadrature
/// rules and finite-difference differentiators.
///
/// Loading this [`Lib`] registers the fixed and adaptive integration backends
/// (trapezoid, Simpson, Romberg, Gauss-Legendre, adaptive Gauss-Kronrod) used
/// by the numeric `integrate`/`integrate-adapt` surface, and the
/// finite-difference differentiators (forward, backward, central-3, central-5,
/// Richardson) used by `numeric-diff`. It also installs the plugin descriptor
/// values that advertise each backend to the registry.
///
/// # Examples
///
/// ```
/// use sim_kernel::Lib;
/// use sim_lib_numbers_quad::QuadNumbersLib;
///
/// let lib = QuadNumbersLib::new();
/// let manifest = lib.manifest();
/// // One descriptor export per registered backend (5 differentiators plus
/// // 7 quadrature rules).
/// assert_eq!(manifest.exports.len(), 12);
/// ```
pub struct QuadNumbersLib;

impl QuadNumbersLib {
    /// Creates the quadrature/differentiator library. The value is stateless;
    /// all behavior is installed when it is loaded into a [`Cx`].
    pub fn new() -> Self {
        Self
    }
}

impl Default for QuadNumbersLib {
    fn default() -> Self {
        Self::new()
    }
}

impl Lib for QuadNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: domains::quad(),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::<Dependency>::new(),
            capabilities: Vec::new(),
            exports: descriptor_exports(),
        }
    }

    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
        for plugin in differentiators() {
            register_differentiator(plugin)?;
        }
        for plugin in quadratures() {
            register_quadrature(plugin)?;
        }
        install_descriptors(cx, linker)?;
        Ok(())
    }
}

fn descriptor_exports() -> Vec<Export> {
    descriptor_specs()
        .into_iter()
        .map(|(name, _, _adaptive)| Export::Value {
            symbol: numeric_plugin_descriptor_symbol("numbers/quad", name),
        })
        .collect()
}

fn install_descriptors(cx: &sim_kernel::LoadCx, linker: &mut Linker<'_>) -> Result<()> {
    for (name, kind, adaptive) in descriptor_specs() {
        linker.value(
            numeric_plugin_descriptor_symbol("numbers/quad", name),
            numeric_plugin_descriptor_value(
                cx.factory(),
                Symbol::new(name),
                kind,
                adaptive,
                domains::quad(),
            )?,
        )?;
    }
    Ok(())
}

fn descriptor_specs() -> Vec<(&'static str, &'static str, bool)> {
    vec![
        ("forward", "differentiator", false),
        ("backward", "differentiator", false),
        ("central-3", "differentiator", false),
        ("central-5", "differentiator", false),
        ("richardson", "differentiator", false),
        ("trapezoid", "quadrature", false),
        ("simpson", "quadrature", false),
        ("romberg", "quadrature", true),
        ("gauss-legendre-8", "quadrature", false),
        ("gauss-legendre-16", "quadrature", false),
        ("gauss-legendre-32", "quadrature", false),
        ("adaptive-gauss-kronrod", "quadrature", true),
    ]
}

#[derive(Clone, Copy)]
enum Method {
    Trapezoid,
    Simpson,
    Romberg,
    GaussLegendre(usize),
    AdaptiveGaussKronrod,
}

fn quadratures() -> Vec<Arc<dyn Quadrature>> {
    vec![
        Arc::new(QuadPlugin::new(
            "trapezoid",
            NumericKind::QuadratureFixed,
            Method::Trapezoid,
        )),
        Arc::new(QuadPlugin::new(
            "simpson",
            NumericKind::QuadratureFixed,
            Method::Simpson,
        )),
        Arc::new(QuadPlugin::new(
            "romberg",
            NumericKind::QuadratureFixed,
            Method::Romberg,
        )),
        Arc::new(QuadPlugin::new(
            "romberg",
            NumericKind::QuadratureAdaptive,
            Method::Romberg,
        )),
        Arc::new(QuadPlugin::new(
            "gauss-legendre-8",
            NumericKind::QuadratureFixed,
            Method::GaussLegendre(8),
        )),
        Arc::new(QuadPlugin::new(
            "gauss-legendre-16",
            NumericKind::QuadratureFixed,
            Method::GaussLegendre(16),
        )),
        Arc::new(QuadPlugin::new(
            "gauss-legendre-32",
            NumericKind::QuadratureFixed,
            Method::GaussLegendre(32),
        )),
        Arc::new(QuadPlugin::new(
            "adaptive-gauss-kronrod",
            NumericKind::QuadratureAdaptive,
            Method::AdaptiveGaussKronrod,
        )),
    ]
}

struct QuadPlugin {
    name: Symbol,
    kind: NumericKind,
    method: Method,
}

impl QuadPlugin {
    fn new(name: &str, kind: NumericKind, method: Method) -> Self {
        Self {
            name: Symbol::new(name),
            kind,
            method,
        }
    }
}

impl NumericPlugin for QuadPlugin {
    fn name(&self) -> Symbol {
        self.name.clone()
    }

    fn kind(&self) -> NumericKind {
        self.kind
    }
}

impl Quadrature for QuadPlugin {
    fn integrate(
        &self,
        cx: &mut Cx,
        f: &Func,
        _var: &Symbol,
        lo: &Value,
        hi: &Value,
        opt: QuadOpts,
    ) -> Result<Value> {
        let a = value_to_f64(cx, lo, "quadrature lower bound")?;
        let b = value_to_f64(cx, hi, "quadrature upper bound")?;
        match self.method {
            Method::Trapezoid => trapezoid(cx, f, a, b, opt.n.unwrap_or(256)),
            Method::Simpson => simpson(cx, f, a, b, opt.n.unwrap_or(128)),
            Method::Romberg => romberg(cx, f, a, b, opt.n.unwrap_or(6), opt.tol.unwrap_or(1.0e-10)),
            Method::GaussLegendre(n) => gauss_legendre(cx, f, a, b, n),
            Method::AdaptiveGaussKronrod => {
                adaptive_gauss_kronrod(cx, f, a, b, opt.tol.unwrap_or(1.0e-10), 10)
            }
        }
    }
}

fn trapezoid(cx: &mut Cx, f: &Func, a: f64, b: f64, n: usize) -> Result<Value> {
    let n = n.max(1);
    let h = (b - a) / n as f64;
    let fa = sample_at(cx, f, a)?;
    let fb = sample_at(cx, f, b)?;
    let fa = scale(cx, fa, 0.5)?;
    let fb = scale(cx, fb, 0.5)?;
    let mut acc = add(cx, fa, fb)?;
    for i in 1..n {
        let x = a + i as f64 * h;
        let sample = sample_at(cx, f, x)?;
        acc = add(cx, acc, sample)?;
    }
    scale(cx, acc, h)
}

fn simpson(cx: &mut Cx, f: &Func, a: f64, b: f64, n: usize) -> Result<Value> {
    let n = if n < 2 {
        2
    } else if n.is_multiple_of(2) {
        n
    } else {
        n + 1
    };
    let h = (b - a) / n as f64;
    let fa = sample_at(cx, f, a)?;
    let fb = sample_at(cx, f, b)?;
    let mut acc = add(cx, fa, fb)?;
    for i in 1..n {
        let x = a + i as f64 * h;
        let coeff = if i.is_multiple_of(2) { 2.0 } else { 4.0 };
        let sample = sample_at(cx, f, x)?;
        acc = add_scaled(cx, acc, sample, coeff)?;
    }
    scale(cx, acc, h / 3.0)
}

fn romberg(cx: &mut Cx, f: &Func, a: f64, b: f64, levels: usize, tol: f64) -> Result<Value> {
    let levels = levels.max(1);
    let mut table: Vec<Vec<Value>> = Vec::with_capacity(levels);
    for k in 0..levels {
        let panels = 1usize << k;
        let trap = trapezoid(cx, f, a, b, panels)?;
        let mut row = vec![trap];
        for j in 1..=k {
            let factor = 4_f64.powi(j as i32);
            let prev = row[j - 1].clone();
            let coarse = table[k - 1][j - 1].clone();
            let prev = scale(cx, prev, factor)?;
            let numerator = sub(cx, prev, coarse)?;
            row.push(scale(cx, numerator, 1.0 / (factor - 1.0))?);
        }
        if let Some(previous_row) = table.last()
            && let (Some(last), Some(prev)) = (row.last(), previous_row.last())
            && abs_error(cx, last.clone(), prev.clone())? <= tol
        {
            return Ok(last.clone());
        }
        table.push(row);
    }
    Ok(table
        .last()
        .and_then(|row| row.last())
        .cloned()
        .expect("romberg table should contain at least one result"))
}

fn gauss_legendre(cx: &mut Cx, f: &Func, a: f64, b: f64, n: usize) -> Result<Value> {
    let nodes = gauss_legendre_nodes(n);
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let seed = sample_at(cx, f, mid + half * nodes[0].0)?;
    let mut acc = zero_like(cx, seed)?;
    for (x, w) in nodes {
        let sample = sample_at(cx, f, mid + half * x)?;
        acc = add_scaled(cx, acc, sample, w)?;
    }
    scale(cx, acc, half)
}

fn adaptive_gauss_kronrod(
    cx: &mut Cx,
    f: &Func,
    a: f64,
    b: f64,
    tol: f64,
    depth: usize,
) -> Result<Value> {
    let (kronrod, gauss) = gauss_kronrod_15(cx, f, a, b)?;
    if depth == 0 || abs_error(cx, kronrod.clone(), gauss)? <= tol {
        return Ok(kronrod);
    }
    let mid = 0.5 * (a + b);
    let left = adaptive_gauss_kronrod(cx, f, a, mid, tol * 0.5, depth - 1)?;
    let right = adaptive_gauss_kronrod(cx, f, mid, b, tol * 0.5, depth - 1)?;
    add(cx, left, right)
}

fn gauss_kronrod_15(cx: &mut Cx, f: &Func, a: f64, b: f64) -> Result<(Value, Value)> {
    const XGK: [f64; 8] = [
        0.991_455_371_120_812_6,
        0.949_107_912_342_758_5,
        0.864_864_423_359_769_1,
        0.741_531_185_599_394_5,
        0.586_087_235_467_691_1,
        0.405_845_151_377_397_2,
        0.207_784_955_007_898_47,
        0.0,
    ];
    const WGK: [f64; 8] = [
        0.022_935_322_010_529_224,
        0.063_092_092_629_978_56,
        0.104_790_010_322_250_18,
        0.140_653_259_715_525_92,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_42,
        0.204_432_940_075_298_89,
        0.209_482_141_084_727_82,
    ];
    const WG: [f64; 4] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_64,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
    ];
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let seed = sample_at(cx, f, mid)?;
    let mut kronrod = zero_like(cx, seed.clone())?;
    let mut gauss = zero_like(cx, seed)?;
    for (i, x) in XGK.iter().copied().enumerate() {
        let sample = if x == 0.0 {
            sample_at(cx, f, mid)?
        } else {
            let plus = sample_at(cx, f, mid + half * x)?;
            let minus = sample_at(cx, f, mid - half * x)?;
            add(cx, plus, minus)?
        };
        kronrod = add_scaled(cx, kronrod, sample.clone(), WGK[i])?;
        if i == 1 {
            gauss = add_scaled(cx, gauss, sample, WG[0])?;
        } else if i == 3 {
            gauss = add_scaled(cx, gauss, sample, WG[1])?;
        } else if i == 5 {
            gauss = add_scaled(cx, gauss, sample, WG[2])?;
        } else if i == 7 {
            gauss = add_scaled(cx, gauss, sample, WG[3])?;
        }
    }
    Ok((scale(cx, kronrod, half)?, scale(cx, gauss, half)?))
}

fn sample_at(cx: &mut Cx, f: &Func, x: f64) -> Result<Value> {
    let x = f64_value(cx, x)?;
    call_unary_func(cx, f, x)
}

fn gauss_legendre_nodes(n: usize) -> Vec<(f64, f64)> {
    let m = n.div_ceil(2);
    let mut nodes = vec![(0.0, 0.0); n];
    for i in 0..m {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        loop {
            let (pn, pnm1) = legendre(n, z);
            let derivative = (n as f64) * (z * pn - pnm1) / (z * z - 1.0);
            let next = z - pn / derivative;
            if (next - z).abs() < 1.0e-15 {
                z = next;
                break;
            }
            z = next;
        }
        let (pn, pnm1) = legendre(n, z);
        let derivative = (n as f64) * (z * pn - pnm1) / (z * z - 1.0);
        let weight = 2.0 / ((1.0 - z * z) * derivative * derivative);
        nodes[i] = (-z, weight);
        nodes[n - 1 - i] = (z, weight);
    }
    nodes
}

fn legendre(n: usize, x: f64) -> (f64, f64) {
    let mut p0 = 1.0;
    let mut p1 = x;
    if n == 0 {
        return (p0, 0.0);
    }
    if n == 1 {
        return (p1, p0);
    }
    for k in 2..=n {
        let pk = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = pk;
    }
    (p1, p0)
}
