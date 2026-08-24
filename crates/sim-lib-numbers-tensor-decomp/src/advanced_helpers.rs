fn hessenberg(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut h = a.to_vec();
    let mut q = identity(n);
    for k in 0..n.saturating_sub(2) {
        let mut v = (k + 1..n).map(|i| h[i * n + k]).collect::<Vec<_>>();
        let z = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if z == 0.0 {
            continue;
        }
        v[0] += if v[0] >= 0.0 { z } else { -z };
        let z = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= z
        }
        for j in k..n {
            let d = 2.0
                * (0..v.len())
                    .map(|i| v[i] * h[(k + 1 + i) * n + j])
                    .sum::<f64>();
            for i in 0..v.len() {
                h[(k + 1 + i) * n + j] -= d * v[i]
            }
        }
        for i in 0..n {
            let d = 2.0
                * (0..v.len())
                    .map(|j| h[i * n + k + 1 + j] * v[j])
                    .sum::<f64>();
            for j in 0..v.len() {
                h[i * n + k + 1 + j] -= d * v[j]
            }
        }
        for i in 0..n {
            let d = 2.0
                * (0..v.len())
                    .map(|j| q[i * n + k + 1 + j] * v[j])
                    .sum::<f64>();
            for j in 0..v.len() {
                q[i * n + k + 1 + j] -= d * v[j]
            }
        }
    }
    (h, q)
}
fn permutation_balance(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut balanced = a.to_vec();
    let mut permutation = identity(n);
    let mut high = n;
    while high > 1 {
        let Some(isolated) = (0..high).find(|&i| {
            (0..high)
                .filter(|&j| j != i)
                .all(|j| balanced[i * n + j] == 0.0)
                || (0..high)
                    .filter(|&j| j != i)
                    .all(|j| balanced[j * n + i] == 0.0)
        }) else {
            break;
        };
        high -= 1;
        if isolated != high {
            for j in 0..n {
                balanced.swap(isolated * n + j, high * n + j);
            }
            for i in 0..n {
                balanced.swap(i * n + isolated, i * n + high);
                permutation.swap(i * n + isolated, i * n + high);
            }
        }
    }
    (balanced, permutation)
}
fn quasi(h: &[f64], n: usize) -> bool {
    (2..n).all(|i| h[i * n + i - 1] == 0.0 || h[(i - 1) * n + i - 2] == 0.0)
}
fn qr_step(h: &mut Vec<f64>, q: &mut Vec<f64>, n: usize, s: f64) {
    let mut x = h.clone();
    for i in 0..n {
        x[i * n + i] -= s
    }
    let (qq, r) = qr(&x, n);
    *h = mul(&r, &qq, n, n, n);
    for i in 0..n {
        h[i * n + i] += s
    }
    *q = mul(q, &qq, n, n, n)
}
fn qr(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0; n * n];
    let mut r = vec![0.0; n * n];
    for j in 0..n {
        let mut v = (0..n).map(|i| a[i * n + j]).collect::<Vec<_>>();
        // Reorthogonalized modified Gram-Schmidt keeps the accumulated Schur
        // similarity orthogonal even when a shifted matrix is nearly singular.
        for _ in 0..2 {
            for k in 0..j {
                let d = (0..n).map(|i| q[i * n + k] * v[i]).sum::<f64>();
                r[k * n + j] += d;
                for i in 0..n {
                    v[i] -= d * q[i * n + k]
                }
            }
        }
        let z = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if z > 1e-300 {
            r[j * n + j] = z;
            for i in 0..n {
                q[i * n + j] = v[i] / z
            }
        } else {
            for e in 0..n {
                v.fill(0.0);
                v[e] = 1.0;
                for k in 0..j {
                    let d = (0..n).map(|i| q[i * n + k] * v[i]).sum::<f64>();
                    for i in 0..n {
                        v[i] -= d * q[i * n + k]
                    }
                }
                let basis_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if basis_norm > 1e-12 {
                    for i in 0..n {
                        q[i * n + j] = v[i] / basis_norm
                    }
                    break;
                }
            }
        }
    }
    (q, r)
}
fn schur_values(h: &[f64], n: usize, t: f64) -> Vec<ComplexValue> {
    let mut z = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 1 < n && h[(i + 1) * n + i].abs() > t {
            let (a, b, c, d) = (
                h[i * n + i],
                h[i * n + i + 1],
                h[(i + 1) * n + i],
                h[(i + 1) * n + i + 1],
            );
            let tr = a + d;
            let disc = tr * tr - 4.0 * (a * d - b * c);
            if disc >= 0.0 {
                let x = disc.sqrt();
                z.push(ComplexValue::new((tr + x) / 2.0, 0.0));
                z.push(ComplexValue::new((tr - x) / 2.0, 0.0))
            } else {
                let x = (-disc).sqrt() / 2.0;
                z.push(ComplexValue::new(tr / 2.0, x));
                z.push(ComplexValue::new(tr / 2.0, -x))
            }
            i += 2
        } else {
            z.push(ComplexValue::new(h[i * n + i], 0.0));
            i += 1
        }
    }
    z
}
fn rotate(a: &mut [f64], r: usize, c: usize, x: usize, y: usize, co: f64, s: f64) {
    for i in 0..r {
        let (u, v) = (a[i * c + x], a[i * c + y]);
        a[i * c + x] = co * u - s * v;
        a[i * c + y] = s * u + co * v
    }
}
fn col_norm(a: &[f64], r: usize, c: usize, j: usize) -> f64 {
    (0..r)
        .map(|i| a[i * c + j] * a[i * c + j])
        .sum::<f64>()
        .sqrt()
}
fn complete_zeros(a: &mut [f64], r: usize, c: usize) {
    for j in 0..c {
        if col_norm(a, r, c, j) < 1e-14 {
            for e in 0..r {
                for i in 0..r {
                    a[i * c + j] = f64::from(i == e)
                }
                for k in 0..j {
                    let d = (0..r).map(|i| a[i * c + k] * a[i * c + j]).sum::<f64>();
                    for i in 0..r {
                        a[i * c + j] -= d * a[i * c + k]
                    }
                }
                let z = col_norm(a, r, c, j);
                if z > 1e-10 {
                    for i in 0..r {
                        a[i * c + j] /= z
                    }
                    break;
                }
            }
        }
    }
}
fn complete_basis(a: &mut [f64], n: usize, start: usize) {
    for j in start..n {
        for e in 0..n {
            for i in 0..n {
                a[i * n + j] = f64::from(i == e)
            }
            for k in 0..j {
                let d = (0..n).map(|i| a[i * n + k] * a[i * n + j]).sum::<f64>();
                for i in 0..n {
                    a[i * n + j] -= d * a[i * n + k]
                }
            }
            let z = col_norm(a, n, n, j);
            if z > 1e-10 {
                for i in 0..n {
                    a[i * n + j] /= z
                }
                break;
            }
        }
    }
}
fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut z = vec![0.0; a.len()];
    for i in 0..r {
        for j in 0..c {
            z[j * r + i] = a[i * c + j]
        }
    }
    z
}
fn mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut z = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            z[i * n + j] = (0..k).map(|x| a[i * k + x] * b[x * n + j]).sum()
        }
    }
    z
}
fn orth(a: &[f64], r: usize, c: usize) -> f64 {
    let mut z = 0.0;
    for i in 0..c {
        for j in 0..c {
            let x = (0..r).map(|k| a[k * c + i] * a[k * c + j]).sum::<f64>() - f64::from(i == j);
            z += x * x
        }
    }
    z.sqrt()
}
fn similarity_residual(a: &[f64], q: &[f64], t: &[f64], n: usize) -> f64 {
    let x = mul(&mul(q, t, n, n, n), &transpose(q, n, n), n, n, n);
    a.iter()
        .zip(x)
        .map(|(u, v)| (u - v) * (u - v))
        .sum::<f64>()
        .sqrt()
}
fn svd_residual(a: &[f64], r: usize, c: usize, s: &Svd) -> f64 {
    let (Some(u), Some(v)) = (&s.u, &s.v) else {
        return f64::INFINITY;
    };
    let k = r.min(c);
    let mut z = 0.0;
    for i in 0..r {
        for j in 0..c {
            let x = (0..k)
                .map(|p| u[i * s.left_cols + p] * s.singular_values[p] * v[j * s.right_cols + p])
                .sum::<f64>();
            z += (a[i * c + j] - x).powi(2)
        }
    }
    z.sqrt()
}
fn svd_residual_from_factors(
    a: &[f64],
    r: usize,
    c: usize,
    singular_values: &[f64],
    u: &[f64],
    v: &[f64],
) -> f64 {
    let mut residual = 0.0;
    for i in 0..r {
        for j in 0..c {
            let reconstructed = (0..c)
                .map(|p| u[i * c + p] * singular_values[p] * v[j * c + p])
                .sum::<f64>();
            residual += (a[i * c + j] - reconstructed).powi(2);
        }
    }
    residual.sqrt()
}
