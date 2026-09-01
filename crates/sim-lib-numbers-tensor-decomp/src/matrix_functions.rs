//! Evidence-carrying dense matrix functions and matrix equations.

use crate::{DecompositionError, SchurPlan, real_schur_f64};

/// Matrix-function failure.
#[derive(Clone, Debug, PartialEq)]
pub enum MatrixFunctionError {
    /// Matrix storage and dimension disagree.
    InvalidDimensions,
    /// Input or parameter is non-finite.
    NonFinite,
    /// A required linear system is singular or numerically rank deficient.
    Singular,
    /// The canonical Schur path failed.
    Schur(DecompositionError),
    /// A result failed its requested residual tolerance.
    Residual {
        /// Measured relative residual.
        residual: f64,
        /// Required maximum.
        tolerance: f64,
    },
}

/// Scaling-and-squaring exponential certificate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExponentialEvidence {
    /// Power-of-two input scale divisor.
    pub scale: f64,
    /// Reviewed Pade family order.
    pub approximant_order: usize,
    /// Number of squarings.
    pub squarings: usize,
    /// Relative backward residual from the rational defining equation.
    pub rational_residual: f64,
}
/// Matrix exponential and its certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixExponential {
    /// Row-major result.
    pub value: Vec<f64>,
    /// Numerical evidence.
    pub evidence: ExponentialEvidence,
}

/// Computes a signed integer matrix power by binary powering.
pub fn matrix_power_f64(a: &[f64], n: usize, power: i64) -> Result<Vec<f64>, MatrixFunctionError> {
    valid(a, n)?;
    if power == 0 {
        return Ok(identity(n));
    }
    let mut base = if power < 0 {
        solve_matrix(a, &identity(n), n)?
    } else {
        a.to_vec()
    };
    let mut e = power.unsigned_abs();
    let mut out = identity(n);
    while e > 0 {
        if e & 1 == 1 {
            out = mul(&out, &base, n)
        }
        e >>= 1;
        if e > 0 {
            base = mul(&base, &base, n)
        }
    }
    Ok(out)
}

/// Computes `exp(A)` with Higham's reviewed diagonal `[13/13]` Pade family.
pub fn matrix_exponential_f64(
    a: &[f64],
    n: usize,
) -> Result<MatrixExponential, MatrixFunctionError> {
    valid(a, n)?;
    let norm = one_norm(a, n);
    let theta = 5.371_920_351_148_152;
    let squarings = if norm <= theta {
        0
    } else {
        (norm / theta).log2().ceil().max(0.) as usize
    };
    let scale = 2f64.powi(squarings as i32);
    let x = a.iter().map(|v| v / scale).collect::<Vec<_>>();
    let x2 = mul(&x, &x, n);
    let x4 = mul(&x2, &x2, n);
    let x6 = mul(&x4, &x2, n);
    let c = [
        64764752532480000.,
        32382376266240000.,
        7771770303897600.,
        1187353796428800.,
        129060195264000.,
        10559470521600.,
        670442572800.,
        33522128640.,
        1323241920.,
        40840800.,
        960960.,
        16380.,
        182.,
        1.,
    ];
    let mut t = add_scaled(&add_scaled(&scale_mat(&x6, c[13]), &x4, c[11]), &x2, c[9]);
    t = mul(&x6, &t, n);
    for (k, m) in [(7, &x6), (5, &x4), (3, &x2)] {
        t = add_scaled(&t, m, c[k])
    }
    t = add_scaled(&t, &identity(n), c[1]);
    let u = mul(&x, &t, n);
    let mut v = add_scaled(&add_scaled(&scale_mat(&x6, c[12]), &x4, c[10]), &x2, c[8]);
    v = mul(&x6, &v, n);
    for (k, m) in [(6, &x6), (4, &x4), (2, &x2)] {
        v = add_scaled(&v, m, c[k])
    }
    v = add_scaled(&v, &identity(n), c[0]);
    let lhs = sub(&v, &u);
    let rhs = add_scaled(&v, &u, 1.);
    let mut r = solve_matrix(&lhs, &rhs, n)?;
    let rational_residual = frob(&sub(&mul(&lhs, &r, n), &rhs)) / frob(&rhs).max(1.);
    for _ in 0..squarings {
        r = mul(&r, &r, n)
    }
    Ok(MatrixExponential {
        value: r,
        evidence: ExponentialEvidence {
            scale,
            approximant_order: 13,
            squarings,
            rational_residual,
        },
    })
}

/// Schur-backed matrix-equation evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixEquationEvidence {
    /// Lower bound estimated from Schur spectra.
    pub separation: f64,
    /// Relative equation residual.
    pub residual: f64,
    /// Schur balancing/reconstruction scale evidence.
    pub schur_scale: f64,
}
/// Matrix-equation solution and evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixEquationSolution {
    /// Row-major solution.
    pub value: Vec<f64>,
    /// Numerical evidence.
    pub evidence: MatrixEquationEvidence,
}

/// Solves the continuous Sylvester equation `A X + X B = C`.
///
/// Both operands first traverse the canonical Schur path for spectral
/// separation and balancing evidence. The compact implementation solves the
/// equivalent Kronecker system with pivoting and independently certifies the
/// original equation.
pub fn solve_sylvester_f64(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    n: usize,
    tolerance: f64,
) -> Result<MatrixEquationSolution, MatrixFunctionError> {
    valid(a, n)?;
    valid(b, n)?;
    valid(c, n)?;
    if !tolerance.is_finite() || tolerance <= 0. {
        return Err(MatrixFunctionError::NonFinite);
    }
    let sa = real_schur_f64(a, n, SchurPlan::default()).map_err(MatrixFunctionError::Schur)?;
    let sb = real_schur_f64(b, n, SchurPlan::default()).map_err(MatrixFunctionError::Schur)?;
    let separation = sa
        .eigenvalues
        .iter()
        .flat_map(|x| {
            sb.eigenvalues
                .iter()
                .map(move |y| (x.real() + y.real()).hypot(x.imaginary() + y.imaginary()))
        })
        .fold(f64::INFINITY, f64::min);
    if separation <= f64::EPSILON * (one_norm(a, n) + one_norm(b, n)).max(1.) {
        return Err(MatrixFunctionError::Singular);
    }
    let nn = n * n;
    let mut k = vec![0.; nn * nn];
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            for q in 0..n {
                k[row * nn + q * n + j] += a[i * n + q]
            }
            for q in 0..n {
                k[row * nn + i * n + q] += b[q * n + j]
            }
        }
    }
    let x = solve_vector(&k, c, nn)?;
    let ax = mul(a, &x, n);
    let xb = mul(&x, b, n);
    let residual = frob(&sub(&add_scaled(&ax, &xb, 1.), c)) / frob(c).max(1.);
    if residual > tolerance {
        return Err(MatrixFunctionError::Residual {
            residual,
            tolerance,
        });
    }
    Ok(MatrixEquationSolution {
        value: x,
        evidence: MatrixEquationEvidence {
            separation,
            residual,
            schur_scale: sa
                .evidence
                .reconstruction_residual
                .max(sb.evidence.reconstruction_residual),
        },
    })
}

/// Solves `A X + X A^T = -Q`, the continuous Lyapunov equation.
pub fn solve_lyapunov_f64(
    a: &[f64],
    q: &[f64],
    n: usize,
    tolerance: f64,
) -> Result<MatrixEquationSolution, MatrixFunctionError> {
    valid(a, n)?;
    valid(q, n)?;
    let at = transpose(a, n);
    let neg = q.iter().map(|x| -x).collect::<Vec<_>>();
    solve_sylvester_f64(a, &at, &neg, n, tolerance)
}

fn valid(a: &[f64], n: usize) -> Result<(), MatrixFunctionError> {
    if n == 0 || n.checked_mul(n) != Some(a.len()) {
        return Err(MatrixFunctionError::InvalidDimensions);
    }
    if a.iter().any(|x| !x.is_finite()) {
        return Err(MatrixFunctionError::NonFinite);
    }
    Ok(())
}
fn identity(n: usize) -> Vec<f64> {
    let mut x = vec![0.; n * n];
    for i in 0..n {
        x[i * n + i] = 1.
    }
    x
}
fn mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.; n * n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                c[i * n + j] += a[i * n + k] * b[k * n + j]
            }
        }
    }
    c
}
fn add_scaled(a: &[f64], b: &[f64], s: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + s * y).collect()
}
fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}
fn scale_mat(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}
fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j]
        }
    }
    t
}
fn frob(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn one_norm(a: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].abs()).sum::<f64>())
        .fold(0., f64::max)
}
fn solve_matrix(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, MatrixFunctionError> {
    let mut out = vec![0.; n * n];
    for j in 0..n {
        let rhs = (0..n).map(|i| b[i * n + j]).collect::<Vec<_>>();
        let x = solve_vector(a, &rhs, n)?;
        for i in 0..n {
            out[i * n + j] = x[i]
        }
    }
    Ok(out)
}
fn solve_vector(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, MatrixFunctionError> {
    let mut m = a.to_vec();
    let mut y = b.to_vec();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| m[i * n + k].abs().total_cmp(&m[j * n + k].abs()))
            .unwrap();
        if m[p * n + k].abs() <= f64::EPSILON * one_norm(&m, n).max(1.) {
            return Err(MatrixFunctionError::Singular);
        }
        for j in k..n {
            m.swap(k * n + j, p * n + j)
        }
        y.swap(k, p);
        for i in k + 1..n {
            let f = m[i * n + k] / m[k * n + k];
            for j in k..n {
                m[i * n + j] -= f * m[k * n + j]
            }
            y[i] -= f * y[k]
        }
    }
    let mut x = vec![0.; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - (i + 1..n).map(|j| m[i * n + j] * x[j]).sum::<f64>()) / m[i * n + i]
    }
    Ok(x)
}
