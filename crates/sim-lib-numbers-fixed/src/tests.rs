use std::sync::Arc;

use sim_kernel::{DefaultFactory, NoopEvalPolicy, NumberLiteral};
use sim_lib_numbers_core::domains;

use crate::FixedNumbersLib;

fn cx() -> sim_kernel::Cx {
    let mut cx = sim_kernel::Cx::new(Arc::new(NoopEvalPolicy), Arc::new(DefaultFactory));
    cx.load_lib(&FixedNumbersLib::new()).unwrap();
    cx
}

#[test]
fn fixed_literals_parse_and_round_trip() {
    let mut cx = cx();
    let literal = cx.parse_number_literal("127").unwrap().unwrap();
    assert_eq!(
        literal,
        NumberLiteral {
            domain: domains::i128(),
            canonical: "127".to_owned(),
        }
    );
}

#[test]
fn fixed_registers_value_promotions() {
    let cx = cx();
    let edges = cx.registry().value_promotion_rules();
    assert!(
        edges
            .iter()
            .any(|rule| rule.from_domain == domains::i8() && rule.to_domain == domains::i16())
    );
    assert!(
        edges
            .iter()
            .any(|rule| rule.from_domain == domains::u64() && rule.to_domain == domains::i128())
    );
    assert!(
        edges
            .iter()
            .any(|rule| rule.from_domain == domains::usize() && rule.to_domain == domains::f64())
    );
}
