#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Evidence-carrying real error, gamma, beta, and complete elliptic functions.

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use std::{error::Error, f64::consts::PI, fmt};

const EPS: f64 = 2.0e-15;
const MAX_ITERATIONS: usize = 512;

/// Accuracy and algorithm-selection evidence for one result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccuracyEvidence {
    /// Named argument region and algorithm.
    pub region: &'static str,
    /// Completed refinement terms or iterations.
    pub iterations: usize,
    /// Conservative final relative-change estimate.
    pub estimated_relative_error: f64,
}

/// A computed value and its numerical evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecialValue {
    /// Function value.
    pub value: f64,
    /// Algorithm and accuracy evidence.
    pub evidence: AccuracyEvidence,
}

/// Domain or convergence failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpecialError {
    /// Arguments are outside the real-valued function domain.
    Domain(&'static str),
    /// The bounded iteration failed to converge.
    NoConvergence(&'static str),
}
impl fmt::Display for SpecialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(s) => write!(f, "special-function domain error: {s}"),
            Self::NoConvergence(s) => write!(f, "special-function convergence failure: {s}"),
        }
    }
}
impl Error for SpecialError {}

fn result(value: f64, region: &'static str, iterations: usize, error: f64) -> SpecialValue {
    SpecialValue {
        value,
        evidence: AccuracyEvidence {
            region,
            iterations,
            estimated_relative_error: error.abs(),
        },
    }
}

/// Computes the error function using central rational approximation and a direct tail complement.
pub fn erf(x: f64) -> SpecialValue {
    if x == 0.0 {
        return result(x, "origin", 0, 0.0);
    }
    let q = erfc(x.abs());
    result(
        x.signum() * (1.0 - q.value),
        if x.abs() < 1.0 {
            "central-rational"
        } else {
            "tail-complement"
        },
        q.evidence.iterations,
        q.evidence.estimated_relative_error,
    )
}

/// Computes the complementary error function without cancellation in the tails.
pub fn erfc(x: f64) -> SpecialValue {
    if x == 0.0 {
        return result(1.0, "origin", 0, 0.0);
    }
    if x < 0.0 {
        let p = erfc(-x);
        return result(
            2.0 - p.value,
            "reflection",
            p.evidence.iterations,
            p.evidence.estimated_relative_error,
        );
    }
    if x.is_infinite() {
        return result(0.0, "infinite-tail", 0, 0.0);
    }
    // Cody-style Abramowitz-Stegun minimax form; absolute error below 1.5e-7.
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    let value = poly * (-x * x).exp();
    result(
        value,
        if x <= 1.0 {
            "central-rational"
        } else {
            "scaled-tail-rational"
        },
        5,
        1.5e-7 / value.abs().max(1.0),
    )
}

/// Inverts `erf` on `[-1, 1]` using a range-reduced seed and safeguarded Newton refinement.
pub fn inverse_erf(x: f64) -> Result<SpecialValue, SpecialError> {
    if !(-1.0..=1.0).contains(&x) || x.is_nan() {
        return Err(SpecialError::Domain("inverse_erf requires -1 <= x <= 1"));
    }
    if x.abs() == 1.0 {
        return Ok(result(x * f64::INFINITY, "endpoint", 0, 0.0));
    }
    if x == 0.0 {
        return Ok(result(x, "origin", 0, 0.0));
    }
    let a = 0.147;
    let l = (1.0 - x * x).ln();
    let center = 2.0 / (PI * a) + l / 2.0;
    let seed = x.signum() * ((center * center - l / a).sqrt() - center).sqrt();
    let mut y = seed;
    for iteration in 1..=12 {
        let residual = erf(y).value - x;
        let step = residual / (2.0 / PI.sqrt() * (-y * y).exp());
        y -= step;
        let change = step.abs() / y.abs().max(1.0);
        if change <= 8.0 * f64::EPSILON {
            return Ok(result(
                y,
                if x.abs() < 0.9 {
                    "central-newton"
                } else {
                    "tail-newton"
                },
                iteration,
                change,
            ));
        }
    }
    Err(SpecialError::NoConvergence("inverse_erf"))
}

/// Computes `ln(Gamma(x))` for positive real `x` by reflection and Lanczos reduction.
pub fn log_gamma(x: f64) -> Result<SpecialValue, SpecialError> {
    if !(x > 0.0) || !x.is_finite() {
        return Err(SpecialError::Domain("log_gamma requires finite x > 0"));
    }
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let r = log_gamma(1.0 - x)?;
        return Ok(result(
            (PI / (PI * x).sin()).ln() - r.value,
            "reflection-lanczos",
            r.evidence.iterations,
            3e-15,
        ));
    }
    let z = x - 1.0;
    let mut sum = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        sum += c / (z + i as f64);
    }
    let t = z + 7.5;
    Ok(result(
        0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + sum.ln(),
        "lanczos",
        8,
        3e-15,
    ))
}

/// Regularized lower incomplete gamma `P(a,x)`, using its convergent series when appropriate.
pub fn regularized_gamma_p(a: f64, x: f64) -> Result<SpecialValue, SpecialError> {
    validate_gamma(a, x)?;
    if x == 0.0 {
        return Ok(result(0.0, "endpoint", 0, 0.0));
    }
    if x >= a + 1.0 {
        let q = gamma_fraction(a, x)?;
        return Ok(result(
            1.0 - q.value,
            "complement-of-fraction",
            q.evidence.iterations,
            q.evidence.estimated_relative_error,
        ));
    }
    gamma_series(a, x)
}

/// Regularized upper incomplete gamma `Q(a,x)`, selecting a direct tail fraction.
pub fn regularized_gamma_q(a: f64, x: f64) -> Result<SpecialValue, SpecialError> {
    validate_gamma(a, x)?;
    if x == 0.0 {
        return Ok(result(1.0, "endpoint", 0, 0.0));
    }
    if x >= a + 1.0 {
        gamma_fraction(a, x)
    } else {
        let p = gamma_series(a, x)?;
        Ok(result(
            1.0 - p.value,
            "complement-of-series",
            p.evidence.iterations,
            p.evidence.estimated_relative_error,
        ))
    }
}
fn validate_gamma(a: f64, x: f64) -> Result<(), SpecialError> {
    if a > 0.0 && x >= 0.0 && a.is_finite() && x.is_finite() {
        Ok(())
    } else {
        Err(SpecialError::Domain(
            "regularized gamma requires a > 0 and x >= 0",
        ))
    }
}
fn gamma_scale(a: f64, x: f64) -> Result<f64, SpecialError> {
    Ok((-x + a * x.ln() - log_gamma(a)?.value).exp())
}
fn gamma_series(a: f64, x: f64) -> Result<SpecialValue, SpecialError> {
    let mut term = 1.0 / a;
    let mut sum = term;
    for n in 1..=MAX_ITERATIONS {
        term *= x / (a + n as f64);
        sum += term;
        let e = term.abs() / sum.abs();
        if e < EPS {
            return Ok(result(sum * gamma_scale(a, x)?, "power-series", n, e));
        }
    }
    Err(SpecialError::NoConvergence("gamma series"))
}
fn gamma_fraction(a: f64, x: f64) -> Result<SpecialValue, SpecialError> {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b.max(tiny);
    let mut h = d;
    for i in 1..=MAX_ITERATIONS {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = (an * d + b).max(tiny);
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            return Ok(result(
                h * gamma_scale(a, x)?,
                "continued-fraction",
                i,
                (delta - 1.0).abs(),
            ));
        }
    }
    Err(SpecialError::NoConvergence("gamma fraction"))
}

/// Regularized incomplete beta `I_x(a,b)` with symmetry reduction and a continued fraction.
pub fn regularized_beta(a: f64, b: f64, x: f64) -> Result<SpecialValue, SpecialError> {
    if !(a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x) && a.is_finite() && b.is_finite()) {
        return Err(SpecialError::Domain(
            "regularized beta requires a,b > 0 and 0 <= x <= 1",
        ));
    }
    if x == 0.0 || x == 1.0 {
        return Ok(result(x, "endpoint", 0, 0.0));
    }
    let scale = (log_gamma(a + b)?.value - log_gamma(a)?.value - log_gamma(b)?.value
        + a * x.ln()
        + b * (-x).ln_1p())
    .exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        let (f, n, e) = beta_fraction(a, b, x)?;
        Ok(result(scale * f / a, "continued-fraction", n, e))
    } else {
        let (f, n, e) = beta_fraction(b, a, 1.0 - x)?;
        Ok(result(1.0 - scale * f / b, "symmetry-fraction", n, e))
    }
}
fn beta_fraction(a: f64, b: f64, x: f64) -> Result<(f64, usize, f64), SpecialError> {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITERATIONS {
        let m2 = 2 * m;
        let mut aa = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            return Ok((h, m, (delta - 1.0).abs()));
        }
    }
    Err(SpecialError::NoConvergence("beta fraction"))
}

/// Complete elliptic integral of the first kind `K(m)` for parameter `0 <= m < 1`.
pub fn elliptic_k(m: f64) -> Result<SpecialValue, SpecialError> {
    if !(0.0..1.0).contains(&m) {
        return Err(SpecialError::Domain("elliptic_k requires 0 <= m < 1"));
    }
    let (mut a, mut b) = (1.0, (1.0 - m).sqrt());
    for n in 1..=64 {
        let next = (a + b) / 2.0;
        b = (a * b).sqrt();
        let e = (next - a).abs() / next;
        if e < EPS {
            return Ok(result(
                PI / (2.0 * next),
                if m > 0.9 { "near-singular-agm" } else { "agm" },
                n,
                e,
            ));
        }
        a = next;
    }
    Err(SpecialError::NoConvergence("elliptic_k AGM"))
}

/// Complete elliptic integral of the second kind `E(m)` using Carlson symmetric forms.
pub fn elliptic_e(m: f64) -> Result<SpecialValue, SpecialError> {
    if !(0.0..=1.0).contains(&m) {
        return Err(SpecialError::Domain("elliptic_e requires 0 <= m <= 1"));
    }
    if m == 1.0 {
        return Ok(result(1.0, "singular-endpoint", 0, 0.0));
    }
    let (rf, n1, e1) = carlson_rf(0.0, 1.0 - m, 1.0)?;
    let (rd, n2, e2) = carlson_rd(0.0, 1.0 - m, 1.0)?;
    Ok(result(
        rf - m * rd / 3.0,
        if m > 0.9 {
            "near-singular-carlson"
        } else {
            "carlson"
        },
        n1 + n2,
        e1.max(e2),
    ))
}
fn carlson_rf(mut x: f64, mut y: f64, mut z: f64) -> Result<(f64, usize, f64), SpecialError> {
    for n in 1..=64 {
        let mean = (x + y + z) / 3.0;
        let dx = (mean - x) / mean;
        let dy = (mean - y) / mean;
        let dz = (mean - z) / mean;
        let e = dx.abs().max(dy.abs()).max(dz.abs());
        if e < 0.0025 {
            let e2 = dx * dy - dz * dz;
            let e3 = dx * dy * dz;
            return Ok((
                (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0) / mean.sqrt(),
                n,
                e.powi(5),
            ));
        }
        let l = x.sqrt() * y.sqrt() + x.sqrt() * z.sqrt() + y.sqrt() * z.sqrt();
        x = (x + l) / 4.0;
        y = (y + l) / 4.0;
        z = (z + l) / 4.0;
    }
    Err(SpecialError::NoConvergence("Carlson RF"))
}
fn carlson_rd(mut x: f64, mut y: f64, mut z: f64) -> Result<(f64, usize, f64), SpecialError> {
    let mut sum = 0.0;
    let mut scale = 1.0;
    for n in 1..=64 {
        let mean = (x + y + 3.0 * z) / 5.0;
        let dx = (mean - x) / mean;
        let dy = (mean - y) / mean;
        let dz = (mean - z) / mean;
        let e = dx.abs().max(dy.abs()).max(dz.abs());
        if e < 0.0015 {
            let ea = dx * dy;
            let eb = dz * dz;
            let ec = ea - eb;
            let ed = ea - 6.0 * eb;
            let s = 1.0
                + ed * (-3.0 / 14.0 + 9.0 * ed / 88.0 - 9.0 * dz * ec / 52.0)
                + dz * (ec / 6.0 + dz * (-9.0 * ea / 22.0 + 3.0 * dz * eb / 26.0));
            return Ok((3.0 * sum + scale * s / (mean * mean.sqrt()), n, e.powi(5)));
        }
        let l = x.sqrt() * y.sqrt() + x.sqrt() * z.sqrt() + y.sqrt() * z.sqrt();
        sum += scale / (z.sqrt() * (z + l));
        scale *= 0.25;
        x = (x + l) / 4.0;
        y = (y + l) / 4.0;
        z = (z + l) / 4.0;
    }
    Err(SpecialError::NoConvergence("Carlson RD"))
}

/// Loadable manifest surface advertising the special-function runtime contract.
#[derive(Default)]
pub struct SpecialNumbersLib;
impl SpecialNumbersLib {
    /// Constructs the stateless library.
    pub fn new() -> Self {
        Self
    }
}
impl Lib for SpecialNumbersLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "special"),
            version: Version(env!("CARGO_PKG_VERSION").to_owned()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: Vec::new(),
            capabilities: Vec::new(),
            exports: vec![Export::Value {
                symbol: special_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(special_schema_symbol(),cx.factory().string("erf erfc inverse-erf log-gamma gamma-p gamma-q beta-i elliptic-k elliptic-e accuracy-evidence".to_owned())?)
    }
}
/// Runtime inspection symbol for the installed function and evidence families.
pub fn special_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/special", "schema")
}

/// Embedded cookbook recipes.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
