use super::*;
fn close(a: f64, b: f64, t: f64) {
    assert!((a - b).abs() <= t * b.abs().max(1.0), "{a} != {b}");
}
#[test]
fn error_symmetry_and_inverse() {
    for x in [-0.99, -0.5, 0.0, 0.5, 0.99] {
        let y = erf(x).value;
        close(erf(-x).value, -y, 2e-15);
        close(inverse_erf(y).unwrap().value, x, 2e-7);
        close(y + erfc(x).value, 1.0, 2e-15);
    }
}
#[test]
fn gamma_identities_and_tails() {
    close(log_gamma(5.0).unwrap().value, 24.0f64.ln(), 2e-14);
    for (a, x) in [(0.5, 0.25), (4.0, 20.0), (10.0, 9.0)] {
        let p = regularized_gamma_p(a, x).unwrap().value;
        let q = regularized_gamma_q(a, x).unwrap().value;
        close(p + q, 1.0, 3e-14);
    }
    close(
        regularized_gamma_q(4.0, 20.0).unwrap().value,
        3.203_719_780_476_998e-6,
        2e-14,
    );
}
#[test]
fn beta_symmetry() {
    for x in [0.01, 0.4, 0.99] {
        let p = regularized_beta(2.5, 4.0, x).unwrap().value;
        let q = regularized_beta(4.0, 2.5, 1.0 - x).unwrap().value;
        close(p + q, 1.0, 2e-14);
    }
}
#[test]
fn elliptic_values_and_singular_region() {
    close(elliptic_k(0.0).unwrap().value, PI / 2.0, 2e-15);
    close(elliptic_e(0.0).unwrap().value, PI / 2.0, 2e-14);
    close(elliptic_e(1.0).unwrap().value, 1.0, 0.0);
    close(
        elliptic_k(0.9999).unwrap().value,
        5.991_589_340_507_051,
        2e-13,
    );
    close(
        elliptic_e(0.5).unwrap().value,
        1.350_643_881_047_675_5,
        2e-13,
    );
}
