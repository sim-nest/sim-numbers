//! Tests for the scalar-number substrate. These exercise the matcher and spec
//! standalone -- no concrete number-domain crate is involved.

use sim_kernel::{Expr, NumberLiteral, Symbol};

use crate::domains;
use crate::scalar::{ScalarDomainSpec, ScalarLiteralMatcher};

fn spec(canonical: &'static str, family: &'static str, priority: i32) -> ScalarDomainSpec {
    ScalarDomainSpec {
        domain: domains::domain(canonical),
        numeric_family: family,
        canonical_form: canonical,
        parse_priority: priority,
    }
}

fn literal(canonical: &str, domain: &str) -> Expr {
    Expr::Number(NumberLiteral {
        domain: domains::domain(domain),
        canonical: canonical.to_owned(),
    })
}

#[test]
fn int_float_bool_specs_derive_stable_symbols() {
    let int = spec("i64", "integer", 20);
    assert_eq!(int.literal_class_symbol(), domains::literal_class("i64"));
    assert_eq!(
        int.literal_instance_shape_symbol(),
        Symbol::qualified("numbers/i64-literal", "instance-shape")
    );
    assert_eq!(
        crate::value_shape_symbol(&int.domain),
        domains::value_shape(&domains::i64())
    );
}

#[test]
fn domain_matchers_accept_only_their_own_domain() {
    let int = spec("i64", "integer", 20).matcher();
    let float = spec("f64", "real", 10).matcher();
    let boolean = spec("bool", "boolean", 30).matcher();

    assert!(int.matches_expr(&literal("5", "i64")));
    assert!(!int.matches_expr(&literal("5", "f64")));
    assert!(!int.matches_expr(&literal("true", "bool")));

    assert!(float.matches_expr(&literal("1.5", "f64")));
    assert!(!float.matches_expr(&literal("5", "i64")));

    assert!(boolean.matches_expr(&literal("true", "bool")));
    assert!(!boolean.matches_expr(&literal("1.5", "f64")));

    // Non-number expressions never match.
    assert!(!int.matches_expr(&Expr::String("5".to_owned())));
    assert!(!int.matches_expr(&Expr::Symbol(Symbol::new("five"))));
}

#[test]
fn canonical_domains_cover_integer_lattice_names() {
    assert_eq!(domains::fixed_integer_domains().len(), 12);
    assert!(domains::integer_domains().contains(&domains::bigint()));
    assert_eq!(domains::rational_value_class(), domains::domain("Rational"));
}

#[test]
fn sim_value_inspects_matched_literals() {
    // sim-value is the literal-inspection path consumers use after a match.
    let int = spec("i64", "integer", 20).matcher();
    let value = literal("42", "i64");
    assert!(int.matches_expr(&value));
    assert_eq!(sim_value::access::as_i64(&value), Some(42));

    let float = spec("f64", "real", 10).matcher();
    let value = literal("1.5", "f64");
    assert!(float.matches_expr(&value));
    assert_eq!(sim_value::access::as_f64(&value), Some(1.5));
}
