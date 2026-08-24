/// Computes a thin Householder QR factorization without modifying `matrix`.
pub fn qr_f64(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    plan: QrPlan,
) -> Result<QrFactorization, DecompositionError> {
    valid_matrix(matrix, rows, cols)?;
    valid_positive(plan.rank_threshold, "rank threshold must be positive")?;
    valid_positive(
        plan.reconstruction_tolerance,
        "reconstruction tolerance must be positive",
    )?;
    let k = rows.min(cols);
    admit(
        rows.max(cols),
        cubic_work(rows.max(cols)),
        plan.max_dimension,
        plan.max_work,
    )?;
    let mut work = matrix.to_vec();
    let mut reflectors: Vec<(usize, Vec<f64>)> = Vec::with_capacity(k);
    let mut permutation: Vec<usize> = (0..cols).collect();
    for step in 0..k {
        if plan.column_pivoting {
            let pivot = (step..cols)
                .max_by(|&a, &b| {
                    column_tail_norm(&work, rows, cols, a, step)
                        .total_cmp(&column_tail_norm(&work, rows, cols, b, step))
                })
                .unwrap();
            if pivot != step {
                for row in 0..rows {
                    work.swap(row * cols + step, row * cols + pivot);
                }
                permutation.swap(step, pivot);
            }
        }
        let mut v = (step..rows)
            .map(|r| work[r * cols + step])
            .collect::<Vec<_>>();
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            reflectors.push((step, Vec::new()));
            continue;
        }
        v[0] += if v[0] >= 0.0 { norm } else { -norm };
        let vn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= vn;
        }
        for col in step..cols {
            let dot = 2.0
                * v.iter()
                    .enumerate()
                    .map(|(i, x)| x * work[(step + i) * cols + col])
                    .sum::<f64>();
            for (i, x) in v.iter().enumerate() {
                work[(step + i) * cols + col] -= dot * x;
            }
        }
        reflectors.push((step, v));
    }
    let mut q_full = identity(rows);
    for (step, v) in reflectors.iter().rev() {
        if v.is_empty() {
            continue;
        }
        for col in 0..rows {
            let dot = 2.0
                * v.iter()
                    .enumerate()
                    .map(|(i, x)| x * q_full[(step + i) * rows + col])
                    .sum::<f64>();
            for (i, x) in v.iter().enumerate() {
                q_full[(step + i) * rows + col] -= dot * x;
            }
        }
    }
    let mut q = Vec::with_capacity(rows * k);
    for row in 0..rows {
        q.extend((0..k).map(|col| q_full[row * rows + col]));
    }
    let mut r = Vec::with_capacity(k * cols);
    for row in 0..k {
        r.extend((0..cols).map(|col| {
            if col < row {
                0.0
            } else {
                work[row * cols + col]
            }
        }));
    }
    let scale = (0..k).map(|i| r[i * cols + i].abs()).fold(0.0, f64::max);
    let rank = (0..k)
        .filter(|&i| r[i * cols + i].abs() > plan.rank_threshold * scale.max(1.0))
        .count();
    let residual = qr_residual(matrix, rows, cols, &q, &r, &permutation);
    let orth = orthogonality(&q, rows, k);
    if residual > plan.reconstruction_tolerance * matrix_norm(matrix).max(1.0) {
        return Err(DecompositionError::Reconstruction {
            residual,
            tolerance: plan.reconstruction_tolerance,
        });
    }
    let method = method_evidence(
        MethodId::HOUSEHOLDER_QR,
        residual,
        plan.reconstruction_tolerance,
        cubic_work(rows.max(cols)),
        plan.max_work,
    )?;
    Ok(QrFactorization {
        q,
        r,
        permutation,
        rank,
        method,
        evidence: QrEvidence {
            reconstruction_residual: residual,
            orthogonality_residual: orth,
        },
    })
}

/// Solves a real symmetric eigenproblem without modifying `matrix`.
///
/// Householder similarity first reduces the input to symmetric tridiagonal
/// form. Bounded shifted symmetric rotations then diagonalize it; convergence
/// is tested on the complete off-diagonal norm and input is never symmetrized.
pub fn symmetric_eigen_f64(
    matrix: &[f64],
    n: usize,
    plan: EigenPlan,
) -> Result<SymmetricEigen, DecompositionError> {
    valid_matrix(matrix, n, n)?;
    valid_positive(plan.tolerance, "eigen tolerance must be positive")?;
    valid_positive(
        plan.reconstruction_tolerance,
        "reconstruction tolerance must be positive",
    )?;
    let projected = cubic_work(n).saturating_mul(plan.max_iterations as u64);
    admit(n, projected, plan.max_dimension, u64::MAX)?;
    if plan.max_iterations == 0 {
        return Err(DecompositionError::InvalidPlan(
            "iteration limit must be positive",
        ));
    }
    let tolerance = plan.symmetry_tolerance.unwrap_or(0.0);
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(DecompositionError::InvalidPlan(
            "symmetry tolerance must be finite and non-negative",
        ));
    }
    let mismatch = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (matrix[i * n + j] - matrix[j * n + i]).abs()))
        .fold(0.0, f64::max);
    if mismatch > tolerance {
        return Err(DecompositionError::Asymmetric {
            mismatch,
            tolerance,
        });
    }
    let (mut a, mut vectors) = tridiagonalize(matrix, n);
    let scale = matrix_norm(matrix).max(1.0);
    let mut rotations = 0usize;
    loop {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                if a[i * n + j].abs() > largest {
                    largest = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= plan.tolerance * scale {
            break;
        }
        if rotations >= plan.max_iterations {
            return Err(DecompositionError::NoConvergence);
        }
        rotations += 1;
        let tau = (a[q * n + q] - a[p * n + p]) / (2.0 * a[p * n + q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            if k != p && k != q {
                let x = a[k * n + p];
                let y = a[k * n + q];
                a[k * n + p] = c * x - s * y;
                a[p * n + k] = a[k * n + p];
                a[k * n + q] = s * x + c * y;
                a[q * n + k] = a[k * n + q];
            }
        }
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
        for k in 0..n {
            let x = vectors[k * n + p];
            let y = vectors[k * n + q];
            vectors[k * n + p] = c * x - s * y;
            vectors[k * n + q] = s * x + c * y;
        }
    }
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));
    let eigenvalues = order.iter().map(|&i| a[i * n + i]).collect::<Vec<_>>();
    let mut sorted = Vec::with_capacity(n * n);
    for row in 0..n {
        sorted.extend(order.iter().map(|&col| vectors[row * n + col]));
    }
    let pair_residuals = order
        .iter()
        .enumerate()
        .map(|(out, _)| eigen_residual(matrix, n, &sorted, eigenvalues[out], out))
        .collect::<Vec<_>>();
    let residual = pair_residuals.iter().map(|x| x * x).sum::<f64>().sqrt();
    let orth = orthogonality(&sorted, n, n);
    if residual > plan.reconstruction_tolerance * scale {
        return Err(DecompositionError::Reconstruction {
            residual,
            tolerance: plan.reconstruction_tolerance,
        });
    }
    let method = method_evidence(
        MethodId::SYMMETRIC_EIGEN,
        residual,
        plan.reconstruction_tolerance,
        cubic_work(n).saturating_mul(rotations as u64),
        projected.max(1),
    )?;
    let eigenvectors = matches!(plan.vectors, VectorPolicy::Compute).then_some(sorted);
    Ok(SymmetricEigen {
        eigenvalues,
        eigenvectors,
        method,
        evidence: EigenEvidence {
            pair_residuals,
            orthogonality_residual: orth,
            reconstruction_residual: residual,
        },
    })
}

fn method_evidence(
    id: &str,
    residual: f64,
    tolerance: f64,
    charged: u64,
    limit: u64,
) -> Result<MethodEvidence, DecompositionError> {
    let requested = ToleranceSet::new([ErrorMeasure::new(CriterionId::ResidualNorm, tolerance)
        .map_err(|_| DecompositionError::Evidence)?])
    .map_err(|_| DecompositionError::Evidence)?;
    MethodEvidence::new(
        MethodId::new(id).map_err(|_| DecompositionError::Evidence)?,
        Termination::Converged {
            criterion: CriterionId::ResidualNorm,
        },
        WorkReceipt::new(
            charged.min(limit.max(1)),
            WorkLimit::new(limit.max(1)).map_err(|_| DecompositionError::Evidence)?,
            charged >= limit,
        )
        .map_err(|_| DecompositionError::Evidence)?,
        requested,
        vec![
            ErrorMeasure::new(CriterionId::ResidualNorm, residual)
                .map_err(|_| DecompositionError::Evidence)?,
        ],
        PrecisionId::Binary64,
        ExecutionIdentity::new("sim", "sim-lib-numbers-tensor-decomp", "local")
            .map_err(|_| DecompositionError::Evidence)?,
    )
    .map_err(|_| DecompositionError::Evidence)
}
fn valid_matrix(a: &[f64], r: usize, c: usize) -> Result<(), DecompositionError> {
    if r == 0 || c == 0 || r.checked_mul(c) != Some(a.len()) {
        return Err(DecompositionError::InvalidDimensions);
    }
    for (i, x) in a.iter().enumerate() {
        if !x.is_finite() {
            return Err(DecompositionError::NonFinite { index: i });
        }
    }
    Ok(())
}
fn valid_positive(x: f64, msg: &'static str) -> Result<(), DecompositionError> {
    if !x.is_finite() || x <= 0.0 {
        Err(DecompositionError::InvalidPlan(msg))
    } else {
        Ok(())
    }
}
fn admit(d: usize, w: u64, maxd: usize, maxw: u64) -> Result<(), DecompositionError> {
    if d > maxd || w > maxw || maxw == 0 {
        Err(DecompositionError::WorkLimit)
    } else {
        Ok(())
    }
}
fn cubic_work(n: usize) -> u64 {
    u64::try_from(n)
        .unwrap_or(u64::MAX)
        .saturating_pow(3)
        .max(1)
}
fn identity(n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0
    }
    x
}
fn matrix_norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn tridiagonalize(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut vectors = identity(n);
    for column in 0..n.saturating_sub(2) {
        let mut v = vec![0.0; n];
        for row in column + 1..n {
            v[row] = a[row * n + column];
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        v[column + 1] += if v[column + 1] >= 0.0 { norm } else { -norm };
        let vn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= vn;
        }
        let av = (0..n)
            .map(|row| (0..n).map(|col| a[row * n + col] * v[col]).sum::<f64>())
            .collect::<Vec<_>>();
        let alpha = v.iter().zip(&av).map(|(x, y)| x * y).sum::<f64>();
        let w = av
            .iter()
            .zip(&v)
            .map(|(x, y)| 2.0 * (x - alpha * y))
            .collect::<Vec<_>>();
        for row in 0..n {
            for col in 0..n {
                a[row * n + col] -= v[row] * w[col] + w[row] * v[col];
            }
        }
        for row in 0..n {
            let dot = 2.0
                * (0..n)
                    .map(|col| vectors[row * n + col] * v[col])
                    .sum::<f64>();
            for col in 0..n {
                vectors[row * n + col] -= dot * v[col];
            }
        }
        for row in column + 2..n {
            a[row * n + column] = 0.0;
            a[column * n + row] = 0.0;
        }
    }
    (a, vectors)
}
fn column_tail_norm(a: &[f64], r: usize, c: usize, j: usize, start: usize) -> f64 {
    (start..r).map(|i| a[i * c + j] * a[i * c + j]).sum::<f64>()
}
fn orthogonality(q: &[f64], rows: usize, cols: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..cols {
        for j in 0..cols {
            let dot = (0..rows)
                .map(|r| q[r * cols + i] * q[r * cols + j])
                .sum::<f64>()
                - f64::from(i == j);
            s += dot * dot
        }
    }
    s.sqrt()
}
fn qr_residual(a: &[f64], rows: usize, cols: usize, q: &[f64], r: &[f64], p: &[usize]) -> f64 {
    let k = rows.min(cols);
    let mut s = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let got = (0..k).map(|x| q[i * k + x] * r[x * cols + j]).sum::<f64>();
            let e = got - a[i * cols + p[j]];
            s += e * e
        }
    }
    s.sqrt()
}
fn eigen_residual(a: &[f64], n: usize, v: &[f64], value: f64, col: usize) -> f64 {
    (0..n)
        .map(|i| {
            let e =
                (0..n).map(|j| a[i * n + j] * v[j * n + col]).sum::<f64>() - value * v[i * n + col];
            e * e
        })
        .sum::<f64>()
        .sqrt()
}

