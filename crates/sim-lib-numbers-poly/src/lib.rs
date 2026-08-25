#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Dense coefficient algebra, deliberately separate from symbolic CAS trees.

use sim_kernel::{
    AbiVersion, Export, Lib, LibManifest, LibTarget, Linker, Result as KernelResult, Symbol,
    Version,
};
use sim_lib_numbers_complex::ComplexValue;
use sim_lib_numbers_tensor_decomp::{SchurPlan, SvdPlan, VectorForm, real_schur_f64, svd_f64};
use std::{error::Error, fmt};

/// Reduced exact rational coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    /// Numerator.
    pub numerator: i128,
    /// Positive denominator.
    pub denominator: i128,
}
impl Rational {
    /// Constructs a reduced rational.
    pub fn new(n: i128, d: i128) -> Result<Self, PolynomialError> {
        if d == 0 {
            return Err(PolynomialError::ZeroDenominator);
        }
        let s = if d < 0 { -1 } else { 1 };
        let g = gcd_i(n.unsigned_abs(), d.unsigned_abs()) as i128;
        Ok(Self {
            numerator: s * n / g,
            denominator: s * d / g,
        })
    }
    /// Exact zero.
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };
    fn add(self, o: Self) -> Self {
        Self::new(
            self.numerator * o.denominator + o.numerator * self.denominator,
            self.denominator * o.denominator,
        )
        .expect("nonzero denominator")
    }
    fn sub(self, o: Self) -> Self {
        self.add(Self {
            numerator: -o.numerator,
            denominator: o.denominator,
        })
    }
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.numerator * o.numerator,
            self.denominator * o.denominator,
        )
        .expect("nonzero denominator")
    }
    fn div(self, o: Self) -> Result<Self, PolynomialError> {
        Self::new(
            self.numerator * o.denominator,
            self.denominator * o.numerator,
        )
    }
}
fn gcd_i(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r
    }
    a.max(1)
}

/// Polynomial operation failure.
#[derive(Clone, Debug, PartialEq)]
pub enum PolynomialError {
    /// A rational denominator was zero.
    ZeroDenominator,
    /// Division by the zero polynomial.
    DivisionByZero,
    /// Exponent representation was invalid.
    InvalidExponent,
    /// Input contained a non-finite number.
    NonFinite,
    /// A defining system was rank deficient.
    RankDeficient {
        /// Estimated numerical rank.
        rank: usize,
        /// Required rank.
        required: usize,
    },
    /// A decomposition failed.
    Decomposition,
}
impl fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for PolynomialError {}

/// Canonical ascending-order dense floating polynomial (`c[i] x^i`).
#[derive(Clone, Debug, PartialEq)]
pub struct Polynomial {
    coefficients: Vec<f64>,
}
impl Polynomial {
    /// Constructs a polynomial, rejecting non-finite coefficients and removing high zeros.
    pub fn new(mut c: Vec<f64>) -> Result<Self, PolynomialError> {
        if c.iter().any(|x| !x.is_finite()) {
            return Err(PolynomialError::NonFinite);
        }
        normalize_f64(&mut c);
        Ok(Self { coefficients: c })
    }
    /// Canonical coefficients; zero is exactly `[0.0]`.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    /// Degree, with canonical zero assigned degree zero.
    pub fn degree(&self) -> usize {
        self.coefficients.len() - 1
    }
    /// Horner evaluation with Higham-style forward absolute-error bound.
    pub fn evaluate(&self, x: f64) -> Evaluation {
        let mut y: f64 = 0.;
        let mut mag = 0.;
        for &c in self.coefficients.iter().rev() {
            y = y.mul_add(x, c);
            mag = mag * x.abs() + c.abs()
        }
        let n = self.degree() as f64;
        let u = f64::EPSILON / 2.;
        let gamma = n * u / (1. - n * u);
        Evaluation {
            value: y,
            absolute_error_bound: gamma * mag,
        }
    }
    /// Compensated Horner evaluation using error-free product/sum transforms.
    pub fn evaluate_compensated(&self, x: f64) -> Evaluation {
        let mut s = 0.;
        let mut e = 0.;
        for &c in self.coefficients.iter().rev() {
            let p = s * x;
            let pe = s.mul_add(x, -p);
            let z = p + c;
            let se = (p - (z - c)) + (c - (z - p));
            s = z;
            e = e * x + pe + se
        }
        Evaluation {
            value: s + e,
            absolute_error_bound: e.abs() + 2. * f64::EPSILON * (s.abs() + e.abs()),
        }
    }
    /// Formal derivative.
    pub fn derivative(&self) -> Self {
        Self::new(
            self.coefficients
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, c)| *c * i as f64)
                .collect(),
        )
        .expect("finite derivative")
    }
    /// Antiderivative with the requested constant coefficient.
    pub fn antiderivative(&self, constant: f64) -> Result<Self, PolynomialError> {
        let mut c = vec![constant];
        c.extend(
            self.coefficients
                .iter()
                .enumerate()
                .map(|(i, x)| *x / (i + 1) as f64),
        );
        Self::new(c)
    }
    /// Coefficient-wise sum.
    pub fn add(&self, o: &Self) -> Self {
        let mut c = vec![0.; self.coefficients.len().max(o.coefficients.len())];
        for (i, x) in self.coefficients.iter().enumerate() {
            c[i] += x
        }
        for (i, x) in o.coefficients.iter().enumerate() {
            c[i] += x
        }
        Self::new(c).expect("finite sum")
    }
    /// Convolution product.
    pub fn mul(&self, o: &Self) -> Self {
        let mut c = vec![0.; self.degree() + o.degree() + 1];
        for (i, x) in self.coefficients.iter().enumerate() {
            for (j, y) in o.coefficients.iter().enumerate() {
                c[i + j] += x * y
            }
        }
        Self::new(c).expect("finite product")
    }
    /// Builds the monic product of factors `(x-root)` for real roots.
    pub fn from_roots(roots: &[f64]) -> Result<Self, PolynomialError> {
        roots
            .iter()
            .try_fold(Self::new(vec![1.])?, |p, &r| p.try_mul_linear(r))
    }
    fn try_mul_linear(&self, r: f64) -> Result<Self, PolynomialError> {
        if !r.is_finite() {
            return Err(PolynomialError::NonFinite);
        }
        Ok(self.mul(&Self::new(vec![-r, 1.])?))
    }
    /// Extracts complex roots through a balanced companion and canonical Schur path.
    pub fn roots(&self, plan: RootPlan) -> Result<RootSet, PolynomialError> {
        roots_impl(self, plan)
    }
}
fn normalize_f64(c: &mut Vec<f64>) {
    while c.len() > 1 && c.last() == Some(&0.) {
        c.pop();
    }
    if c.is_empty() {
        c.push(0.)
    }
    if c.len() == 1 && c[0] == 0. {
        c[0] = 0.
    }
}
/// Floating evaluation and conservative error evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evaluation {
    /// Evaluated value.
    pub value: f64,
    /// Absolute forward-error bound.
    pub absolute_error_bound: f64,
}

/// Canonical exact rational polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactPolynomial {
    coefficients: Vec<Rational>,
}
impl ExactPolynomial {
    /// Constructs and high-zero-normalizes exact coefficients.
    pub fn new(mut c: Vec<Rational>) -> Self {
        while c.len() > 1 && c.last() == Some(&Rational::ZERO) {
            c.pop();
        }
        if c.is_empty() {
            c.push(Rational::ZERO)
        }
        Self { coefficients: c }
    }
    /// Canonical ascending coefficients.
    pub fn coefficients(&self) -> &[Rational] {
        &self.coefficients
    }
    /// Exact quotient and remainder.
    pub fn div_rem(&self, d: &Self) -> Result<(Self, Self), PolynomialError> {
        if d.coefficients == [Rational::ZERO] {
            return Err(PolynomialError::DivisionByZero);
        }
        let mut r = self.clone();
        let mut q =
            vec![Rational::ZERO; self.coefficients.len().saturating_sub(d.coefficients.len()) + 1];
        while r.coefficients.len() >= d.coefficients.len() && r.coefficients != [Rational::ZERO] {
            let k = r.coefficients.len() - d.coefficients.len();
            let a = (*r.coefficients.last().unwrap()).div(*d.coefficients.last().unwrap())?;
            q[k] = a;
            for (i, x) in d.coefficients.iter().enumerate() {
                r.coefficients[i + k] = r.coefficients[i + k].sub(a.mul(*x))
            }
            r = Self::new(r.coefficients)
        }
        Ok((Self::new(q), r))
    }
    /// Monic Euclidean GCD over exact rationals.
    pub fn gcd(mut a: Self, mut b: Self) -> Result<Self, PolynomialError> {
        while b.coefficients != [Rational::ZERO] {
            let (_, r) = a.div_rem(&b)?;
            a = b;
            b = r
        }
        let lead = *a.coefficients.last().unwrap();
        Ok(Self::new(
            a.coefficients
                .into_iter()
                .map(|x| x.div(lead))
                .collect::<Result<_, _>>()?,
        ))
    }
}

/// Dense integer-exponent rows spanning a possibly negative minimum exponent.
#[derive(Clone, Debug, PartialEq)]
pub struct LaurentPolynomial {
    /// Exponent of the first coefficient.
    pub minimum_exponent: i64,
    coefficients: Vec<f64>,
}
impl LaurentPolynomial {
    /// Constructs a canonical Laurent row.
    pub fn new(
        mut minimum_exponent: i64,
        mut coefficients: Vec<f64>,
    ) -> Result<Self, PolynomialError> {
        if coefficients.iter().any(|x| !x.is_finite()) {
            return Err(PolynomialError::NonFinite);
        }
        while coefficients.len() > 1 && coefficients.last() == Some(&0.) {
            coefficients.pop();
        }
        while coefficients.len() > 1 && coefficients[0] == 0. {
            coefficients.remove(0);
            minimum_exponent += 1
        }
        if coefficients.is_empty() {
            coefficients.push(0.);
            minimum_exponent = 0
        }
        if coefficients == [0.] {
            minimum_exponent = 0
        }
        Ok(Self {
            minimum_exponent,
            coefficients,
        })
    }
    /// Canonical coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}
/// Rational exponent represented in reduced form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RationalExponent {
    /// Numerator.
    pub numerator: i64,
    /// Positive denominator.
    pub denominator: u64,
}
impl RationalExponent {
    /// Constructs a reduced rational exponent.
    pub fn new(n: i64, d: u64) -> Result<Self, PolynomialError> {
        if d == 0 {
            return Err(PolynomialError::InvalidExponent);
        }
        let g = gcd_i(n.unsigned_abs() as u128, d as u128) as u64;
        Ok(Self {
            numerator: n / g as i64,
            denominator: d / g,
        })
    }
}
/// Dense rational-exponent rows with a fixed positive step.
#[derive(Clone, Debug, PartialEq)]
pub struct PuiseuxPolynomial {
    /// First exponent.
    pub minimum_exponent: RationalExponent,
    /// Positive exponent step.
    pub exponent_step: RationalExponent,
    coefficients: Vec<f64>,
}
impl PuiseuxPolynomial {
    /// Constructs a canonical Puiseux row.
    pub fn new(
        mut min: RationalExponent,
        step: RationalExponent,
        mut c: Vec<f64>,
    ) -> Result<Self, PolynomialError> {
        if step.numerator <= 0 {
            return Err(PolynomialError::InvalidExponent);
        }
        if c.iter().any(|x| !x.is_finite()) {
            return Err(PolynomialError::NonFinite);
        }
        while c.len() > 1 && c.last() == Some(&0.) {
            c.pop();
        }
        while c.len() > 1 && c[0] == 0. {
            c.remove(0);
            let den = (min.denominator as i128) * (step.denominator as i128);
            let num = (min.numerator as i128) * (step.denominator as i128)
                + (step.numerator as i128) * (min.denominator as i128);
            min = RationalExponent::new(num as i64, den as u64)?
        }
        if c.is_empty() {
            c.push(0.);
            min = RationalExponent::new(0, 1)?
        }
        Ok(Self {
            minimum_exponent: min,
            exponent_step: step,
            coefficients: c,
        })
    }
    /// Canonical coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

/// Root extraction policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootPlan {
    /// Relative cluster radius.
    pub cluster_tolerance: f64,
    /// Schur policy.
    pub schur: SchurPlan,
}
impl Default for RootPlan {
    fn default() -> Self {
        Self {
            cluster_tolerance: 1e-7,
            schur: SchurPlan::default(),
        }
    }
}
/// One complex root and its numerical certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct RootEvidence {
    /// Root.
    pub root: ComplexValue,
    /// Estimated multiplicity from clustering.
    pub multiplicity: usize,
    /// Relative coefficient backward error.
    pub backward_error: f64,
}
/// Extracted root collection.
#[derive(Clone, Debug, PartialEq)]
pub struct RootSet {
    /// Root evidence in Schur order.
    pub roots: Vec<RootEvidence>,
    /// Companion balancing scales.
    pub balance_scales: Vec<f64>,
}
fn roots_impl(p: &Polynomial, plan: RootPlan) -> Result<RootSet, PolynomialError> {
    let n = p.degree();
    if n == 0 {
        return Ok(RootSet {
            roots: vec![],
            balance_scales: vec![],
        });
    }
    let lead = p.coefficients[n];
    if lead == 0. {
        return Err(PolynomialError::DivisionByZero);
    }
    let mut a = vec![0.; n * n];
    for i in 1..n {
        a[i * n + i - 1] = 1.
    }
    for i in 0..n {
        a[i * n + n - 1] = -p.coefficients[i] / lead
    }
    let scales = balance_companion(&mut a, n);
    let s = real_schur_f64(&a, n, plan.schur).map_err(|_| PolynomialError::Decomposition)?;
    let vals = s.eigenvalues;
    let norm = p.coefficients.iter().map(|x| x.abs()).sum::<f64>().max(1.);
    let roots = vals
        .iter()
        .map(|z| {
            let mult = vals
                .iter()
                .filter(|w| {
                    complex_distance(z, w) <= plan.cluster_tolerance * (1. + z.re().hypot(z.im()))
                })
                .count();
            let residual = complex_poly_abs(&p.coefficients, z);
            RootEvidence {
                root: z.clone(),
                multiplicity: mult,
                backward_error: residual / norm,
            }
        })
        .collect();
    Ok(RootSet {
        roots,
        balance_scales: scales,
    })
}
fn balance_companion(a: &mut [f64], n: usize) -> Vec<f64> {
    let mut s = vec![1.; n];
    for _ in 0..8 {
        for i in 0..n {
            let row = (0..n)
                .filter(|&j| j != i)
                .map(|j| a[i * n + j].abs())
                .sum::<f64>();
            let col = (0..n)
                .filter(|&j| j != i)
                .map(|j| a[j * n + i].abs())
                .sum::<f64>();
            if row > 0. && col > 0. {
                let f = (col / row).sqrt().clamp(0.125, 8.);
                s[i] *= f;
                for j in 0..n {
                    a[i * n + j] *= f;
                    a[j * n + i] /= f
                }
            }
        }
    }
    s
}
fn complex_distance(a: &ComplexValue, b: &ComplexValue) -> f64 {
    (a.re() - b.re()).hypot(a.im() - b.im())
}
fn complex_poly_abs(c: &[f64], z: &ComplexValue) -> f64 {
    let (mut r, mut i) = (0., 0.);
    for &x in c.iter().rev() {
        (r, i) = (r * z.re() - i * z.im() + x, r * z.im() + i * z.re())
    }
    r.hypot(i)
}

/// Numerator/denominator and defining-system evidence for a Pade approximant.
#[derive(Clone, Debug, PartialEq)]
pub struct PadeApproximant {
    /// Numerator coefficients.
    pub numerator: Polynomial,
    /// Denominator coefficients with constant one.
    pub denominator: Polynomial,
    /// Estimated defining-system rank.
    pub rank: usize,
    /// Smallest retained singular value.
    pub smallest_singular_value: f64,
}
/// Builds `[m/n]` from a Maclaurin coefficient series and rejects rank loss.
pub fn pade(
    series: &[f64],
    m: usize,
    n: usize,
    rank_tolerance: f64,
) -> Result<PadeApproximant, PolynomialError> {
    if series.len() < m + n + 1 || series.iter().any(|x| !x.is_finite()) || rank_tolerance <= 0. {
        return Err(PolynomialError::InvalidExponent);
    }
    if n == 0 {
        return Ok(PadeApproximant {
            numerator: Polynomial::new(series[..=m].to_vec())?,
            denominator: Polynomial::new(vec![1.])?,
            rank: 0,
            smallest_singular_value: f64::INFINITY,
        });
    }
    let mut a = vec![0.; n * n];
    let mut b = vec![0.; n];
    for row in 0..n {
        let k = m + 1 + row;
        b[row] = -series[k];
        for j in 0..n {
            a[row * n + j] = series[k - 1 - j]
        }
    }
    let svd = svd_f64(
        &a,
        n,
        n,
        SvdPlan {
            vectors: VectorForm::Thin,
            ..SvdPlan::default()
        },
    )
    .map_err(|_| PolynomialError::Decomposition)?;
    let max = svd.singular_values.first().copied().unwrap_or(0.);
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&x| x > rank_tolerance * max)
        .count();
    if rank < n {
        return Err(PolynomialError::RankDeficient { rank, required: n });
    }
    let qtail = solve(&a, &b, n)?;
    let mut q = vec![1.];
    q.extend(qtail);
    let mut p = vec![0.; m + 1];
    for k in 0..=m {
        for j in 0..=k.min(n) {
            p[k] += q[j] * series[k - j]
        }
    }
    Ok(PadeApproximant {
        numerator: Polynomial::new(p)?,
        denominator: Polynomial::new(q)?,
        rank,
        smallest_singular_value: *svd.singular_values.last().unwrap(),
    })
}
fn solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, PolynomialError> {
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| x[i * n + k].abs().total_cmp(&x[j * n + k].abs()))
            .unwrap();
        if x[p * n + k].abs() <= f64::EPSILON {
            return Err(PolynomialError::RankDeficient {
                rank: k,
                required: n,
            });
        }
        for j in k..n {
            x.swap(k * n + j, p * n + j)
        }
        y.swap(k, p);
        for i in k + 1..n {
            let f = x[i * n + k] / x[k * n + k];
            for j in k..n {
                x[i * n + j] -= f * x[k * n + j]
            }
            y[i] -= f * y[k]
        }
    }
    let mut z = vec![0.; n];
    for i in (0..n).rev() {
        z[i] = (y[i] - (i + 1..n).map(|j| x[i * n + j] * z[j]).sum::<f64>()) / x[i * n + i]
    }
    Ok(z)
}

/// Loadable coefficient-algebra surface.
#[derive(Default)]
pub struct PolyLib;
impl Lib for PolyLib {
    fn manifest(&self) -> LibManifest {
        LibManifest {
            id: Symbol::qualified("numbers", "poly"),
            version: Version(env!("CARGO_PKG_VERSION").into()),
            abi: AbiVersion { major: 0, minor: 1 },
            target: LibTarget::HostRegistered,
            requires: vec![],
            capabilities: vec![],
            exports: vec![Export::Value {
                symbol: poly_schema_symbol(),
            }],
        }
    }
    fn load(&self, cx: &mut sim_kernel::LoadCx, linker: &mut Linker<'_>) -> KernelResult<()> {
        linker.value(poly_schema_symbol(),cx.factory().string("coefficient polynomial laurent-integer-exponent puiseux-rational-exponent horner compensated-horner exact-gcd div-rem roots backward-error pade rank-evidence".into())?)
    }
}
/// Runtime schema symbol.
pub fn poly_schema_symbol() -> Symbol {
    Symbol::qualified("numbers/poly", "schema")
}
/// Embedded recipes.
pub static RECIPES: sim_cookbook::EmbeddedDir =
    include!(concat!(env!("OUT_DIR"), "/cookbook_recipes.rs"));

#[cfg(test)]
mod tests;
