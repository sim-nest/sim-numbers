//! General real spectra, singular-value decomposition, and derived dense solves.
//!
//! SVD uses cyclic one-sided Jacobi rotations on the original matrix columns;
//! it never forms normal equations. Schur uses balancing, Householder
//! Hessenberg reduction, and bounded paired implicit shifts.

use crate::{
    DecompositionError, identity, matrix_norm, method_evidence, valid_matrix, valid_positive,
};
use sim_kernel::Symbol;
use sim_lib_numbers_complex::ComplexValue;
use sim_lib_numbers_method::{ExecutionIdentity, MethodEvidence, MethodId};

/// Vector materialization policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorForm {
    /// Economical vector matrices.
    Thin,
    /// Complete orthogonal bases.
    Full,
    /// Values only.
    None,
}
/// Whether returned factors are admissible or diagnostic after exhaustion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorStatus {
    /// Fully converged and certified.
    Complete,
    /// Explicitly partial; derived APIs reject it.
    Partial,
}

/// Bounded real-Schur policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchurPlan {
    /// Maximum matrix dimension.
    pub max_dimension: usize,
    /// Maximum paired-shift iterations.
    pub max_iterations: usize,
    /// Relative deflation tolerance.
    pub tolerance: f64,
    /// Relative reconstruction tolerance.
    pub reconstruction_tolerance: f64,
    /// Return labelled partial factors on exhaustion.
    pub return_partial: bool,
}
impl Default for SchurPlan {
    fn default() -> Self {
        Self {
            max_dimension: 1024,
            max_iterations: 512,
            tolerance: 1e-12,
            reconstruction_tolerance: 1e-8,
            return_partial: false,
        }
    }
}
/// Schur certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct SchurEvidence {
    /// Deflated subdiagonals.
    pub deflations: usize,
    /// Paired-shift iterations.
    pub iterations: usize,
    /// Relative reconstruction residual.
    pub reconstruction_residual: f64,
    /// Orthogonality residual.
    pub orthogonality_residual: f64,
}
/// Real quasi-triangular Schur factorization.
#[derive(Clone, Debug)]
pub struct RealSchur {
    /// Row-major quasi-triangular T.
    pub t: Vec<f64>,
    /// Row-major Schur vectors Q.
    pub q: Vec<f64>,
    /// Canonical complex eigenvalues.
    pub eigenvalues: Vec<ComplexValue>,
    /// Completion state.
    pub status: FactorStatus,
    /// Common bounded-method evidence; absent only for labelled partial output.
    pub method: Option<MethodEvidence>,
    /// Certificate.
    pub evidence: SchurEvidence,
}

/// Bounded SVD policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvdPlan {
    /// Maximum row or column count.
    pub max_dimension: usize,
    /// Maximum projected scalar work.
    pub max_work: u64,
    /// Maximum Jacobi sweeps.
    pub max_iterations: usize,
    /// Relative column-correlation tolerance.
    pub tolerance: f64,
    /// Vector policy.
    pub vectors: VectorForm,
    /// Relative reconstruction tolerance.
    pub reconstruction_tolerance: f64,
    /// Return labelled partial factors on exhaustion.
    pub return_partial: bool,
}
impl Default for SvdPlan {
    fn default() -> Self {
        Self {
            max_dimension: 4096,
            max_work: u64::MAX,
            max_iterations: 128,
            tolerance: 1e-12,
            vectors: VectorForm::Thin,
            reconstruction_tolerance: 1e-9,
            return_partial: false,
        }
    }
}
/// SVD certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct SvdEvidence {
    /// Sweeps used.
    pub iterations: usize,
    /// Relative reconstruction residual.
    pub reconstruction_residual: f64,
    /// U orthogonality residual.
    pub left_orthogonality_residual: f64,
    /// V orthogonality residual.
    pub right_orthogonality_residual: f64,
    /// Singular-value ordering check.
    pub ordered: bool,
}
/// `A = U diag(s) V^T` with descending singular values.
#[derive(Clone, Debug)]
pub struct Svd {
    /// Input rows.
    pub rows: usize,
    /// Input columns.
    pub cols: usize,
    /// Descending non-negative singular values.
    pub singular_values: Vec<f64>,
    /// Row-major U.
    pub u: Option<Vec<f64>>,
    /// Columns stored in U.
    pub left_cols: usize,
    /// Row-major V.
    pub v: Option<Vec<f64>>,
    /// Columns stored in V.
    pub right_cols: usize,
    /// Completion state.
    pub status: FactorStatus,
    /// Common bounded-method evidence; absent only for labelled partial output.
    pub method: Option<MethodEvidence>,
    /// Certificate.
    pub evidence: SvdEvidence,
}
/// One caller-declared relative singular cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingularCutoff(pub f64);
impl SingularCutoff {
    /// Validates a finite non-negative cutoff.
    pub fn new(x: f64) -> Result<Self, DecompositionError> {
        if x.is_finite() && x >= 0.0 {
            Ok(Self(x))
        } else {
            Err(DecompositionError::InvalidPlan(
                "singular cutoff must be finite and non-negative",
            ))
        }
    }
}

/// Open provider operation plus expected result shapes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecompositionOp {
    /// Tensor execution symbol.
    pub symbol: Symbol,
    /// Provider result shapes.
    pub output_shapes: Vec<Vec<usize>>,
}
/// Provider execution identity and local admission checks.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderAdmission {
    /// Exact provider invocation.
    pub execution: ExecutionIdentity,
    /// Locally remeasured residual.
    pub reconstruction_residual: f64,
    /// Ordering passed.
    pub ordered: bool,
    /// Shapes passed.
    pub shapes_valid: bool,
}
/// Tensor provider symbol for real Schur.
pub fn schur_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/real-schur")
}
/// Tensor provider symbol for SVD.
pub fn svd_op_symbol() -> Symbol {
    Symbol::qualified("tensor", "op/svd")
}
/// Schur provider descriptor.
pub fn schur_operation(n: usize) -> DecompositionOp {
    DecompositionOp {
        symbol: schur_op_symbol(),
        output_shapes: vec![vec![n, n], vec![n, n], vec![n, 2]],
    }
}
/// SVD provider descriptor.
pub fn svd_operation(r: usize, c: usize, form: VectorForm) -> DecompositionOp {
    let k = r.min(c);
    let (u, v) = match form {
        VectorForm::Thin => (k, k),
        VectorForm::Full => (r, c),
        VectorForm::None => (0, 0),
    };
    DecompositionOp {
        symbol: svd_op_symbol(),
        output_shapes: vec![vec![k], vec![r, u], vec![c, v]],
    }
}

/// Computes balanced real Schur factors with bounded paired shifts.
pub fn real_schur_f64(a: &[f64], n: usize, p: SchurPlan) -> Result<RealSchur, DecompositionError> {
    valid_matrix(a, n, n)?;
    valid_positive(p.tolerance, "Schur tolerance must be positive")?;
    valid_positive(
        p.reconstruction_tolerance,
        "Schur reconstruction tolerance must be positive",
    )?;
    if n > p.max_dimension || p.max_iterations == 0 {
        return Err(DecompositionError::WorkLimit);
    }
    let (balanced, permutation) = permutation_balance(a, n);
    let (mut h, hessenberg_vectors) = hessenberg(&balanced, n);
    let mut q = mul(&permutation, &hessenberg_vectors, n, n, n);
    let (mut it, mut defs) = (0, 0);
    loop {
        for i in 1..n {
            if h[i * n + i - 1].abs()
                <= p.tolerance * (h[(i - 1) * n + i - 1].abs() + h[i * n + i].abs()).max(1.0)
            {
                if h[i * n + i - 1] != 0.0 {
                    defs += 1
                }
                h[i * n + i - 1] = 0.0
            }
        }
        if quasi(&h, n) {
            break;
        }
        if it >= p.max_iterations {
            if !p.return_partial {
                return Err(DecompositionError::NoConvergence);
            }
            break;
        }
        let (a0, b, c, d) = (
            h[(n - 2) * n + n - 2],
            h[(n - 2) * n + n - 1],
            h[(n - 1) * n + n - 2],
            h[(n - 1) * n + n - 1],
        );
        let tr = a0 + d;
        let disc = tr * tr - 4.0 * (a0 * d - b * c);
        let shifts = if disc >= 0.0 {
            let z = disc.sqrt();
            [(tr + z) / 2.0, (tr - z) / 2.0]
        } else {
            [tr / 2.0, tr / 2.0]
        };
        for s in shifts {
            qr_step(&mut h, &mut q, n, s)
        }
        it += 1
    }
    let status = if quasi(&h, n) {
        FactorStatus::Complete
    } else {
        FactorStatus::Partial
    };
    let residual = similarity_residual(a, &q, &h, n) / matrix_norm(a).max(1.0);
    if status == FactorStatus::Complete && residual > p.reconstruction_tolerance {
        return Err(DecompositionError::Reconstruction {
            residual,
            tolerance: p.reconstruction_tolerance,
        });
    }
    let eigenvalues = schur_values(&h, n, p.tolerance);
    let method = (status == FactorStatus::Complete)
        .then(|| {
            method_evidence(
                MethodId::REAL_SCHUR,
                residual,
                p.reconstruction_tolerance,
                (n as u64).saturating_pow(3).saturating_mul(it as u64),
                (n as u64)
                    .saturating_pow(3)
                    .saturating_mul(p.max_iterations as u64),
            )
        })
        .transpose()?;
    Ok(RealSchur {
        t: h,
        q: q.clone(),
        eigenvalues,
        status,
        method,
        evidence: SchurEvidence {
            deflations: defs,
            iterations: it,
            reconstruction_residual: residual,
            orthogonality_residual: orth(&q, n, n),
        },
    })
}

/// Computes one-sided Jacobi SVD without normal equations.
pub fn svd_f64(a: &[f64], r: usize, c: usize, p: SvdPlan) -> Result<Svd, DecompositionError> {
    valid_matrix(a, r, c)?;
    valid_positive(p.tolerance, "SVD tolerance must be positive")?;
    valid_positive(
        p.reconstruction_tolerance,
        "SVD reconstruction tolerance must be positive",
    )?;
    let d = r.max(c);
    if d > p.max_dimension
        || (d as u64)
            .saturating_pow(3)
            .saturating_mul(p.max_iterations as u64)
            > p.max_work
        || p.max_iterations == 0
    {
        return Err(DecompositionError::WorkLimit);
    }
    if r >= c {
        svd_tall(a, r, c, p)
    } else {
        let at = transpose(a, r, c);
        let mut x = svd_tall(&at, c, r, p)?;
        std::mem::swap(&mut x.rows, &mut x.cols);
        std::mem::swap(&mut x.u, &mut x.v);
        std::mem::swap(&mut x.left_cols, &mut x.right_cols);
        // Transposition preserves the Frobenius reconstruction residual.
        Ok(x)
    }
}
fn svd_tall(a: &[f64], m: usize, n: usize, p: SvdPlan) -> Result<Svd, DecompositionError> {
    let mut b = a.to_vec();
    let mut v = identity(n);
    let (mut sweeps, mut done) = (0, false);
    while sweeps < p.max_iterations {
        let mut changed = false;
        for x in 0..n {
            for y in x + 1..n {
                let (mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let (u, w) = (b[i * n + x], b[i * n + y]);
                    aa += u * u;
                    bb += w * w;
                    ab += u * w
                }
                if ab.abs() <= p.tolerance * (aa * bb).sqrt() {
                    continue;
                }
                changed = true;
                let tau = (bb - aa) / (2.0 * ab);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let co = 1.0 / (1.0 + t * t).sqrt();
                rotate(&mut b, m, n, x, y, co, co * t);
                rotate(&mut v, n, n, x, y, co, co * t)
            }
        }
        sweeps += 1;
        if !changed {
            done = true;
            break;
        }
    }
    if !done && !p.return_partial {
        return Err(DecompositionError::NoConvergence);
    }
    let norms = (0..n).map(|j| col_norm(&b, m, n, j)).collect::<Vec<_>>();
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&x, &y| norms[y].total_cmp(&norms[x]));
    let s = order.iter().map(|&j| norms[j]).collect::<Vec<_>>();
    let mut u = vec![0.0; m * n];
    let mut vv = vec![0.0; n * n];
    for (o, &j) in order.iter().enumerate() {
        if norms[j] > 0.0 {
            for i in 0..m {
                u[i * n + o] = b[i * n + j] / norms[j]
            }
        }
        for i in 0..n {
            vv[i * n + o] = v[i * n + j]
        }
    }
    complete_zeros(&mut u, m, n);
    let reconstruction_residual =
        svd_residual_from_factors(a, m, n, &s, &u, &vv) / matrix_norm(a).max(1.0);
    let (uo, uc) = match p.vectors {
        VectorForm::None => (None, 0),
        VectorForm::Thin => (Some(u.clone()), n),
        VectorForm::Full => {
            let mut f = vec![0.0; m * m];
            for i in 0..m {
                for j in 0..n {
                    f[i * m + j] = u[i * n + j]
                }
            }
            complete_basis(&mut f, m, n);
            (Some(f), m)
        }
    };
    let (vo, vc) = if p.vectors == VectorForm::None {
        (None, 0)
    } else {
        (Some(vv), n)
    };
    let mut out = Svd {
        rows: m,
        cols: n,
        singular_values: s,
        u: uo,
        left_cols: uc,
        v: vo,
        right_cols: vc,
        status: if done {
            FactorStatus::Complete
        } else {
            FactorStatus::Partial
        },
        method: None,
        evidence: SvdEvidence {
            iterations: sweeps,
            reconstruction_residual: 0.0,
            left_orthogonality_residual: 0.0,
            right_orthogonality_residual: 0.0,
            ordered: true,
        },
    };
    out.evidence.reconstruction_residual = reconstruction_residual;
    out.evidence.left_orthogonality_residual =
        out.u.as_ref().map_or(0.0, |x| orth(x, m, out.left_cols));
    out.evidence.right_orthogonality_residual =
        out.v.as_ref().map_or(0.0, |x| orth(x, n, out.right_cols));
    if out.status == FactorStatus::Complete
        && out.evidence.reconstruction_residual > p.reconstruction_tolerance
    {
        return Err(DecompositionError::Reconstruction {
            residual: out.evidence.reconstruction_residual,
            tolerance: p.reconstruction_tolerance,
        });
    }
    if out.status == FactorStatus::Complete {
        out.method = Some(method_evidence(
            MethodId::JACOBI_SVD,
            out.evidence.reconstruction_residual,
            p.reconstruction_tolerance,
            (m.max(n) as u64)
                .saturating_pow(3)
                .saturating_mul(sweeps as u64),
            p.max_work,
        )?);
    }
    Ok(out)
}

fn keep(s: &Svd, c: SingularCutoff) -> Result<Vec<bool>, DecompositionError> {
    SingularCutoff::new(c.0)?;
    if s.status != FactorStatus::Complete {
        return Err(DecompositionError::NoConvergence);
    }
    if s.u.is_none() || s.v.is_none() {
        return Err(DecompositionError::InvalidPlan("SVD vectors are required"));
    }
    let z = c.0 * s.singular_values.first().copied().unwrap_or(0.0);
    Ok(s.singular_values.iter().map(|x| *x > z).collect())
}
/// Numerical rank under the shared cutoff.
pub fn numerical_rank(s: &Svd, c: SingularCutoff) -> Result<usize, DecompositionError> {
    Ok(keep(s, c)?.iter().filter(|x| **x).count())
}
/// Two-norm condition estimate; infinity denotes cutoff rank loss.
pub fn condition_2(s: &Svd, c: SingularCutoff) -> Result<f64, DecompositionError> {
    let k = keep(s, c)?;
    if k.iter().any(|x| !*x) {
        Ok(f64::INFINITY)
    } else {
        Ok(s.singular_values[0] / s.singular_values.last().copied().unwrap_or(0.0))
    }
}
/// Moore-Penrose pseudoinverse (`cols x rows`).
pub fn pseudoinverse(s: &Svd, c: SingularCutoff) -> Result<Vec<f64>, DecompositionError> {
    let k = keep(s, c)?;
    let (u, v) = (s.u.as_ref().unwrap(), s.v.as_ref().unwrap());
    let mut z = vec![0.0; s.cols * s.rows];
    for i in 0..s.cols {
        for j in 0..s.rows {
            z[i * s.rows + j] = (0..k.len())
                .filter(|&x| k[x])
                .map(|x| v[i * s.right_cols + x] * u[j * s.left_cols + x] / s.singular_values[x])
                .sum()
        }
    }
    Ok(z)
}
/// Least-squares solution for one right-hand side.
pub fn least_squares(
    s: &Svd,
    b: &[f64],
    c: SingularCutoff,
) -> Result<Vec<f64>, DecompositionError> {
    if b.len() != s.rows {
        return Err(DecompositionError::InvalidDimensions);
    }
    let p = pseudoinverse(s, c)?;
    Ok((0..s.cols)
        .map(|i| (0..s.rows).map(|j| p[i * s.rows + j] * b[j]).sum())
        .collect())
}
/// Orthonormal null-space basis (`cols x nullity`).
pub fn null_space(s: &Svd, c: SingularCutoff) -> Result<Vec<f64>, DecompositionError> {
    let k = keep(s, c)?;
    if s.right_cols < s.cols {
        return Err(DecompositionError::InvalidPlan(
            "full right vectors are required",
        ));
    }
    let v = s.v.as_ref().unwrap();
    let js = (0..s.cols)
        .filter(|&j| j >= k.len() || !k[j])
        .collect::<Vec<_>>();
    let mut z = Vec::new();
    for i in 0..s.cols {
        for &j in &js {
            z.push(v[i * s.right_cols + j])
        }
    }
    Ok(z)
}
/// Admits provider SVD output after shape, ordering, residual, and identity checks.
pub fn admit_provider_svd(
    a: &[f64],
    r: usize,
    c: usize,
    s: &Svd,
    e: ExecutionIdentity,
    t: f64,
) -> Result<ProviderAdmission, DecompositionError> {
    let shapes = s.rows == r && s.cols == c && s.singular_values.len() == r.min(c);
    let ordered = s.singular_values.windows(2).all(|x| x[0] >= x[1])
        && s.singular_values.iter().all(|x| *x >= 0.0);
    let residual = if shapes {
        svd_residual(a, r, c, s) / matrix_norm(a).max(1.0)
    } else {
        f64::INFINITY
    };
    if !shapes || !ordered || s.status != FactorStatus::Complete || residual > t {
        return Err(DecompositionError::Reconstruction {
            residual,
            tolerance: t,
        });
    }
    Ok(ProviderAdmission {
        execution: e,
        reconstruction_residual: residual,
        ordered,
        shapes_valid: shapes,
    })
}
/// Admits provider Schur output after shape, residual, and identity checks.
pub fn admit_provider_schur(
    a: &[f64],
    n: usize,
    s: &RealSchur,
    e: ExecutionIdentity,
    t: f64,
) -> Result<ProviderAdmission, DecompositionError> {
    let shapes = s.t.len() == n * n && s.q.len() == n * n && s.eigenvalues.len() == n;
    let residual = if shapes {
        similarity_residual(a, &s.q, &s.t, n) / matrix_norm(a).max(1.0)
    } else {
        f64::INFINITY
    };
    if !shapes || s.status != FactorStatus::Complete || residual > t {
        return Err(DecompositionError::Reconstruction {
            residual,
            tolerance: t,
        });
    }
    Ok(ProviderAdmission {
        execution: e,
        reconstruction_residual: residual,
        ordered: true,
        shapes_valid: true,
    })
}

include!("advanced_helpers.rs");
