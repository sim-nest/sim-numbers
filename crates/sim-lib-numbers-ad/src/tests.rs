use crate::{Dual, Scalarish, Tape};

fn quadratic<S: Scalarish>(x: S) -> S {
    x * x + S::from_f64(3.0) * x
}

fn assert_same_bits(expected: f64, actual: Dual<0>) {
    assert_eq!(expected.to_bits(), actual.v.to_bits());
}

#[test]
fn dual_derivative_of_quadratic_at_two_is_seven() {
    let y = quadratic(Dual::<1>::var(2.0, 0));
    assert_eq!(y.v, 10.0);
    assert_eq!(y.d, [7.0]);
}

#[test]
fn dual_derivative_of_sin_at_zero_is_one() {
    let y = Dual::<1>::var(0.0, 0).sin();
    assert_eq!(y.v, 0.0);
    assert_eq!(y.d, [1.0]);
}

#[test]
fn reverse_tape_gradient_matches_ab_plus_a() {
    let mut tape = Tape::new();
    let a = tape.input(0, 2.0);
    let b = tape.input(1, 5.0);
    let product = tape.mul(a, b);
    let out = tape.add(product, a);
    assert_eq!(tape.value(out), 12.0);
    assert_eq!(tape.grad(out, 2), vec![6.0, 2.0]);
}

#[test]
fn dual_zero_matches_plain_f64_bit_for_bit_for_supported_ops() {
    for (lhs, rhs) in [(1.5, -0.0), (-3.25, 2.0), (7.0, -2.0)] {
        assert_same_bits(lhs + rhs, Dual::<0>::cst(lhs) + Dual::<0>::cst(rhs));
        assert_same_bits(lhs - rhs, Dual::<0>::cst(lhs) - Dual::<0>::cst(rhs));
        assert_same_bits(lhs * rhs, Dual::<0>::cst(lhs) * Dual::<0>::cst(rhs));
    }

    assert_same_bits(7.0 / -2.0, Dual::<0>::cst(7.0) / Dual::<0>::cst(-2.0));

    for value in [0.0, 1.25, -0.75] {
        assert_same_bits(value.sin(), Dual::<0>::cst(value).sin());
        assert_same_bits(value.cos(), Dual::<0>::cst(value).cos());
        assert_same_bits(value.exp(), Dual::<0>::cst(value).exp());
    }

    for value in [0.5, 2.0, 9.0] {
        assert_same_bits(value.ln(), Dual::<0>::cst(value).ln());
        assert_same_bits(value.sqrt(), Dual::<0>::cst(value).sqrt());
        assert_same_bits(value.recip(), Dual::<0>::cst(value).recip());
    }
}
