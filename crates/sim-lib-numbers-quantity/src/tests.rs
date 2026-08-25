use std::sync::Arc;

use sim_codec::{Input, decode_datum_with_codec, encode_datum_with_codec};
use sim_kernel::{Datum, DefaultFactory, Lib, NoopEvalPolicy, NumberLiteral, Symbol};

use crate::*;

fn interval<S: Scalar>(
    scalar: S,
    dimension: Dimension,
    kind: Option<MeasureKind>,
    unit: Option<Unit>,
) -> Quantity<S> {
    Quantity::new(scalar, dimension, kind, unit, MeasureRole::Interval).unwrap()
}

#[test]
fn rational_exponents_are_normalized_bounded_and_canonical() {
    assert_eq!(Exponent::new(2, 4).unwrap(), Exponent::new(1, 2).unwrap());
    assert_eq!(Exponent::new(0, -99).unwrap(), Exponent::ZERO);
    assert_eq!(
        Exponent::new(1025, 1),
        Err(DimensionError::ExponentOutOfBounds)
    );
    let speed = Dimension::base(BaseDimension::Length)
        .quotient(&Dimension::base(BaseDimension::Time))
        .unwrap();
    assert_eq!(speed.canonical(), "time^-1*length^1");
    assert_eq!(
        speed
            .product(&Dimension::base(BaseDimension::Time))
            .unwrap(),
        Dimension::base(BaseDimension::Length)
    );
}

#[test]
fn energy_and_torque_have_equal_dimensions_but_distinct_content_ids() {
    assert_eq!(energy_kind().dimension(), torque_kind().dimension());
    assert_ne!(energy_kind().id(), torque_kind().id());
    let energy = interval(
        ExactScalar::from(1),
        energy_kind().dimension().clone(),
        Some(energy_kind()),
        Some(JOULE()),
    );
    let torque = interval(
        ExactScalar::from(1),
        torque_kind().dimension().clone(),
        Some(torque_kind()),
        Some(NEWTON_METRE()),
    );
    assert!(matches!(
        energy.add(&torque),
        Err(QuantityError::Incompatible {
            reason: "semantic kinds differ"
        })
    ));
}

#[test]
fn affine_points_convert_exactly_and_obey_point_difference_laws() {
    let freezing = Quantity::new(
        ExactScalar::from(0),
        CELSIUS().dimension().clone(),
        None,
        Some(CELSIUS()),
        MeasureRole::Point,
    )
    .unwrap();
    let kelvin = freezing.convert(&KELVIN()).unwrap();
    assert_eq!(*kelvin.scalar(), ExactScalar::new(27315, 100).unwrap());
    assert!(matches!(
        freezing.add(&freezing),
        Err(QuantityError::Incompatible { .. })
    ));
    assert!(matches!(
        freezing.mul(&freezing),
        Err(QuantityError::AffinePointInLinearOperation)
    ));
    let boiling = Quantity::new(
        ExactScalar::from(100),
        CELSIUS().dimension().clone(),
        None,
        Some(CELSIUS()),
        MeasureRole::Point,
    )
    .unwrap();
    let difference = boiling.sub(&freezing).unwrap();
    assert_eq!(*difference.scalar(), ExactScalar::from(100));
    assert_eq!(difference.role(), MeasureRole::Interval);
}

#[test]
fn algebra_derives_dimensions_without_inventing_kinds() {
    let length = interval(
        9.0,
        Dimension::base(BaseDimension::Length),
        None,
        Some(METRE()),
    );
    let area = length.mul(&length).unwrap();
    assert_eq!(
        area.dimension(),
        &Dimension::base(BaseDimension::Length)
            .power(Exponent::new(2, 1).unwrap())
            .unwrap()
    );
    assert!(area.kind().is_none());
    let root = area.pow(Exponent::new(1, 2).unwrap()).unwrap();
    assert_eq!(*root.scalar(), 9.0);
    assert_eq!(root.dimension(), &Dimension::base(BaseDimension::Length));
}

#[test]
fn exact_conversion_preserves_scalar_type_and_dimensionless_kind_rules() {
    let centimetre = Unit::new(
        "cm",
        "si:length",
        (1, 100),
        (0, 1),
        Dimension::base(BaseDimension::Length),
        None,
        MeasureRole::Interval,
    )
    .unwrap();
    let value = interval(
        ExactScalar::from(125),
        Dimension::base(BaseDimension::Length),
        None,
        Some(centimetre),
    );
    let metres: Quantity<ExactScalar> = value.convert(&METRE()).unwrap();
    assert_eq!(*metres.scalar(), ExactScalar::new(5, 4).unwrap());
    let factor = interval(ExactScalar::from(2), Dimension::DIMENSIONLESS, None, None);
    let energy = interval(
        ExactScalar::from(3),
        energy_kind().dimension().clone(),
        Some(energy_kind()),
        None,
    );
    assert_eq!(factor.mul(&energy).unwrap().kind(), Some(&energy_kind()));
}

#[test]
fn shape_constraints_cover_dimension_kind_family_and_role() {
    let quantity = interval(
        ExactScalar::from(3),
        energy_kind().dimension().clone(),
        Some(energy_kind()),
        Some(JOULE()),
    );
    assert!(
        QuantityShape {
            dimension: Some(energy_kind().dimension().clone()),
            kind_id: Some(energy_kind().id().to_owned()),
            unit_family: Some("si:energy".to_owned()),
            role: Some(MeasureRole::Interval)
        }
        .matches(&quantity)
    );
    assert!(
        !QuantityShape {
            kind_id: Some(torque_kind().id().to_owned()),
            ..QuantityShape::default()
        }
        .matches(&quantity)
    );
}

fn i64_datum(value: i64) -> Datum {
    Datum::Number(NumberLiteral {
        domain: Symbol::qualified("numbers", "i64"),
        canonical: value.to_string(),
    })
}
fn runtime_datum(exponents: Vec<(i64, i64)>) -> Datum {
    Datum::Node {
        tag: quantity_class_symbol(),
        fields: vec![
            (
                Symbol::new("scalar"),
                Datum::Number(NumberLiteral {
                    domain: Symbol::qualified("numbers", "rational"),
                    canonical: "25/2".to_owned(),
                }),
            ),
            (
                Symbol::new("dimension"),
                Datum::Vector(
                    exponents
                        .into_iter()
                        .map(|(n, d)| Datum::Vector(vec![i64_datum(n), i64_datum(d)]))
                        .collect(),
                ),
            ),
            (
                Symbol::new("kind"),
                Datum::Symbol(Symbol::qualified("kind", "energy")),
            ),
            (
                Symbol::new("unit"),
                Datum::Symbol(Symbol::qualified("unit", "J")),
            ),
            (
                Symbol::new("role"),
                Datum::Symbol(Symbol::qualified("measure", "interval")),
            ),
        ],
    }
}

#[test]
fn runtime_value_has_canonical_read_construct_and_semantic_data_round_trip() {
    let datum = runtime_datum(vec![(0, 1), (2, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1)]);
    let value = QuantityValue::from_datum(datum.clone()).unwrap();
    assert_eq!(
        Datum::try_from(sim_kernel::Expr::from(value.datum().clone())).unwrap(),
        datum
    );
    let mut cx = sim_kernel::Cx::new(
        Arc::new(NoopEvalPolicy),
        Arc::new(DefaultFactory),
        sim_kernel::HandleSeed::new(7),
    );
    cx.load_lib(&QuantityLib::new()).unwrap();
    assert_eq!(
        value
            .into_value()
            .unwrap()
            .object()
            .snapshot(&mut cx)
            .unwrap(),
        Some(datum)
    );
}

#[test]
fn quantity_data_round_trips_through_installed_general_codecs() {
    let datum = runtime_datum(vec![(0, 1), (2, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1)]);
    let mut cx = sim_kernel::Cx::new(
        Arc::new(NoopEvalPolicy),
        Arc::new(DefaultFactory),
        sim_kernel::HandleSeed::new(8),
    );
    let json = sim_codec_json::JsonCodecLib::new(cx.registry_mut().fresh_codec_id());
    cx.load_lib(&json).unwrap();
    let lisp = sim_codec_lisp::LispCodecLib::new(cx.registry_mut().fresh_codec_id()).unwrap();
    cx.load_lib(&lisp).unwrap();
    for symbol in [
        Symbol::qualified("codec", "json"),
        Symbol::qualified("codec", "lisp"),
    ] {
        let encoded = encode_datum_with_codec(
            &mut cx,
            &symbol,
            &datum,
            sim_kernel::EncodeOptions::default(),
        )
        .unwrap()
        .into_text()
        .unwrap();
        let decoded = decode_datum_with_codec(
            &mut cx,
            &symbol,
            Input::Text(encoded),
            sim_kernel::ReadPolicy::default(),
        )
        .unwrap();
        assert_eq!(
            decoded.content_id().unwrap(),
            datum.content_id().unwrap(),
            "semantic drift through {symbol}"
        );
    }
}

#[test]
fn malicious_read_construct_dimensions_are_bounded() {
    assert!(QuantityValue::from_datum(runtime_datum(vec![(0, 1); 6])).is_err());
    assert!(
        QuantityValue::from_datum(runtime_datum(vec![
            (0, 1),
            (0, 1),
            (1025, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1)
        ]))
        .is_err()
    );
    assert!(
        QuantityValue::from_datum(runtime_datum(vec![
            (0, 1),
            (0, 1),
            (1, 0),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1)
        ]))
        .is_err()
    );
    let manifest = QuantityLib::new().manifest();
    assert_eq!(manifest.id, Symbol::qualified("numbers", "quantity"));
}
